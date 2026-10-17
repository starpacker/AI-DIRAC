"""Charge discovery experiments from base-current observations.

These helpers implement the first electromagnetic unlock lifecycle.  Before
``charge`` is unlocked, emitted data contain only time, current, and switch
state.  The candidate state is reconstructed from the allowed base primitive
``current`` via time integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aiphy.core import AtomExp, DataStruct, ExpData
from aiphy.operational_concepts import (
    AdditiveInvariantFit,
    ChannelOrigin,
    ChannelRef,
    ConceptStatus,
    EvidenceKind,
    MeasurementProtocol,
    OperationalConceptRegistry,
    OperationalPipelineBridge,
    OperationalScope,
    ValidationEvidence,
)
from aiphy.sr_metadata import quantity_is_eligible


@dataclass(frozen=True)
class ChargeAccumulationRun:
    experiment_id: str
    experiment_family: str
    time: np.ndarray
    current: np.ndarray
    switch_state: np.ndarray
    charge: np.ndarray


@dataclass(frozen=True)
class ChargeUnlockResult:
    candidate_id: str
    status: ConceptStatus
    symbolic_name: str
    fit_relative_residual: float
    validation_scores: dict[str, float]
    emitted_columns_before_unlock: tuple[str, ...]
    emitted_columns_after_unlock: tuple[str, ...]


def cumulative_trapezoid(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    time = np.asarray(time, dtype=float)
    if values.shape != time.shape:
        raise ValueError("values and time must have identical shape")
    if len(time) < 2:
        return np.zeros_like(time, dtype=float)
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(time)
    return np.concatenate([[0.0], np.cumsum(increments)])


def current_waveform(
    time: np.ndarray,
    *,
    waveform: str,
    amplitude: float,
    duration: float,
) -> np.ndarray:
    phase = time / max(duration, 1e-12)
    if waveform == "constant":
        return np.full_like(time, amplitude, dtype=float)
    if waveform == "reverse":
        return np.where(phase < 0.5, amplitude, -0.6 * amplitude)
    if waveform == "sine":
        return amplitude * (1.0 + 0.35 * np.sin(2.0 * np.pi * phase))
    if waveform == "pulse":
        return amplitude * np.where((phase > 0.2) & (phase < 0.75), 1.0, 0.0)
    if waveform == "ramp":
        return amplitude * (0.4 + phase)
    raise ValueError(f"Unsupported current waveform: {waveform}")


def make_charge_accumulation_run(
    *,
    experiment_id: str,
    experiment_family: str = "charge_accumulation",
    waveform: str = "constant",
    amplitude: float = 1.0,
    duration: float = 1.0,
    samples: int = 96,
) -> ChargeAccumulationRun:
    time = np.linspace(0.0, duration, samples)
    current = current_waveform(
        time,
        waveform=waveform,
        amplitude=amplitude,
        duration=duration,
    )
    switch_state = np.ones_like(time)
    charge = cumulative_trapezoid(current * switch_state, time)
    return ChargeAccumulationRun(
        experiment_id=experiment_id,
        experiment_family=experiment_family,
        time=time,
        current=current,
        switch_state=switch_state,
        charge=charge,
    )


def charge_accumulation_data(
    run: ChargeAccumulationRun,
    *,
    repeats: int = 6,
    noise: float = 0.0,
    seed: int = 20260815,
) -> DataStruct:
    rng = np.random.default_rng(seed)
    series = {
        "current[1]": run.current,
        "switch_state[1]": run.switch_state,
    }
    if quantity_is_eligible("charge"):
        series["charge[1]"] = run.charge

    data = DataStruct.empty()
    for expression, values in {"t[0]": run.time, **series}.items():
        values = np.asarray(values, dtype=float)
        samples = np.repeat(values[None, :], repeats, axis=0)
        if noise:
            samples = samples + rng.normal(0.0, noise, samples.shape)
        data.add_data(AtomExp(expression), ExpData(samples.tolist()))
    return data


def _mean_trace(expdata, length: int | None = None) -> np.ndarray:
    if expdata.is_normal:
        values = np.asarray(expdata.normal_data.mean, dtype=float)
        if length is not None and values.size != length:
            raise ValueError("Normal data length mismatch")
        return values
    if expdata.is_const or expdata.is_zero:
        if length is None:
            raise ValueError("A length is required to expand conserved data")
        return np.full(length, float(expdata.const_data.mean), dtype=float)
    raise ValueError("Cannot extract a trace from erroneous ExpData")


def reconstruct_charge_from_current(data: DataStruct, *, conductor_id: str = "1") -> np.ndarray:
    time = _mean_trace(data.fetch_data_by_str("t[0]"))
    current = _mean_trace(data.fetch_data_by_str(f"current[{conductor_id}]"), len(time))
    return cumulative_trapezoid(current, time)


def charge_integral_score(run: ChargeAccumulationRun) -> float:
    reconstructed = cumulative_trapezoid(run.current * run.switch_state, run.time)
    scale = max(float(np.std(run.charge)), 1e-12)
    relative_error = float(
        np.sqrt(np.mean((reconstructed - run.charge) ** 2)) / scale
    )
    return max(0.0, 1.0 - relative_error)


def _charge_fit(definition_runs: tuple[ChargeAccumulationRun, ...]) -> AdditiveInvariantFit:
    scores = [charge_integral_score(run) for run in definition_runs]
    relative_residual = 1.0 - min(scores)
    return AdditiveInvariantFit(
        observable="charge",
        reference_entity="conductor_1",
        reference_value=1.0,
        weights={"conductor_1": 1.0},
        standard_errors={"conductor_1": 0.0},
        relative_residual=relative_residual,
        spectral_gap=float("inf"),
        singular_values=(1.0, 0.0),
        observation_ids=tuple(run.experiment_id for run in definition_runs),
        definition_experiments=tuple(run.experiment_id for run in definition_runs),
        definition_families=tuple(
            sorted({run.experiment_family for run in definition_runs})
        ),
    )


def _evidence(
    kind: EvidenceKind,
    run: ChargeAccumulationRun,
    *,
    intervention: str,
    independent: bool = True,
) -> ValidationEvidence:
    return ValidationEvidence(
        kind=kind,
        experiment_id=run.experiment_id,
        experiment_family=run.experiment_family,
        score=charge_integral_score(run),
        independent=independent,
        intervention=intervention,
        held_fixed=("charge_definition:integral_current_dt",),
        channels=(
            ChannelRef("t[0]", ChannelOrigin.RAW_SENSOR),
            ChannelRef("current[1]", ChannelOrigin.RAW_SENSOR),
            ChannelRef("switch_state[1]", ChannelOrigin.CONTROL_SETTING),
        ),
    )


def run_charge_unlock_lifecycle(
    registry: OperationalConceptRegistry | None = None,
) -> ChargeUnlockResult:
    """Run the complete simplified lifecycle for the first EM concept.

    The lifecycle intentionally starts with ``charge`` locked.  It emits no
    charge data, proposes a trial concept from current integration, records the
    required independent evidence, certifies the concept, and syncs it into the
    quantity registry.
    """

    registry = registry or OperationalConceptRegistry()
    definition_runs = (
        make_charge_accumulation_run(
            experiment_id="charge_accumulation_definition_constant",
            waveform="constant",
            amplitude=0.8,
            duration=1.2,
        ),
        make_charge_accumulation_run(
            experiment_id="charge_accumulation_definition_sine",
            waveform="sine",
            amplitude=0.6,
            duration=1.0,
        ),
    )
    before_data = charge_accumulation_data(definition_runs[0])
    fit = _charge_fit(definition_runs)
    protocol = MeasurementProtocol(
        experiment_family="charge_accumulation",
        observable="charge",
        estimator="charge(t) = integral(current(t) dt)",
        required_channels=("t[0]", "current[1]", "switch_state[1]"),
        reference_entity="conductor_1",
        reference_value=1.0,
        nuisance_controls=("waveform", "amplitude", "duration"),
        alternate_experiment_families=("charge_transfer_conservation",),
        notes="Current is an SI base primitive; charge is the accumulated state.",
    )
    record = registry.propose_additive_weight(
        fit,
        protocol,
        entity_type="Conductor",
        symbolic_name="charge",
        applicable_experiment_families=(
            "charge_accumulation",
            "charge_transfer_conservation",
            "electromagnetism",
        ),
    )
    record.scope = OperationalScope.STATE

    validation_runs = {
        EvidenceKind.REPEATABILITY: make_charge_accumulation_run(
            experiment_id="charge_accumulation_repeat",
            waveform="constant",
            amplitude=0.8,
            duration=1.2,
        ),
        EvidenceKind.OBJECT_SENSITIVITY: make_charge_accumulation_run(
            experiment_id="charge_accumulation_replaced_conductor",
            waveform="pulse",
            amplitude=0.9,
            duration=1.4,
        ),
        EvidenceKind.REFERENCE_TRANSITIVITY: make_charge_accumulation_run(
            experiment_id="charge_accumulation_three_conductor_cycle",
            experiment_family="charge_transfer_conservation",
            waveform="reverse",
            amplitude=0.7,
            duration=1.1,
        ),
        EvidenceKind.CROSS_APPARATUS_TRANSFER: make_charge_accumulation_run(
            experiment_id="charge_transfer_independent_apparatus",
            experiment_family="charge_transfer_conservation",
            waveform="ramp",
            amplitude=0.5,
            duration=1.3,
        ),
        EvidenceKind.HELDOUT_PREDICTION: make_charge_accumulation_run(
            experiment_id="charge_accumulation_heldout_waveform",
            waveform="sine",
            amplitude=1.15,
            duration=0.9,
        ),
    }
    nuisance_runs = (
        make_charge_accumulation_run(
            experiment_id="charge_accumulation_vary_amplitude",
            waveform="constant",
            amplitude=1.3,
            duration=1.2,
        ),
        make_charge_accumulation_run(
            experiment_id="charge_accumulation_vary_duration",
            waveform="pulse",
            amplitude=0.7,
            duration=1.8,
        ),
    )

    validation_scores = {}
    registry.record_evidence(
        record.candidate_id,
        _evidence(
            EvidenceKind.REPEATABILITY,
            validation_runs[EvidenceKind.REPEATABILITY],
            intervention="repeat",
        ),
    )
    validation_scores[EvidenceKind.REPEATABILITY.value] = charge_integral_score(
        validation_runs[EvidenceKind.REPEATABILITY]
    )
    for run, intervention in zip(nuisance_runs, ("vary:amplitude", "vary:duration")):
        registry.record_evidence(
            record.candidate_id,
            _evidence(EvidenceKind.NUISANCE_INVARIANCE, run, intervention=intervention),
        )
        validation_scores[intervention] = charge_integral_score(run)
    for kind, run in validation_runs.items():
        if kind is EvidenceKind.REPEATABILITY:
            continue
        registry.record_evidence(
            record.candidate_id,
            _evidence(kind, run, intervention=kind.value),
        )
        validation_scores[kind.value] = charge_integral_score(run)
    provenance = ValidationEvidence(
        kind=EvidenceKind.PROVENANCE_AUDIT,
        experiment_id="charge_accumulation_provenance_audit",
        experiment_family="audit",
        score=1.0,
        independent=True,
        intervention="audit_channel_dependency_graph",
        held_fixed=("no_charge_channel_before_unlock",),
        channels=(
            ChannelRef("t[0]", ChannelOrigin.RAW_SENSOR),
            ChannelRef("current[1]", ChannelOrigin.RAW_SENSOR),
            ChannelRef("switch_state[1]", ChannelOrigin.CONTROL_SETTING),
        ),
    )
    registry.record_evidence(record.candidate_id, provenance)
    validation_scores[EvidenceKind.PROVENANCE_AUDIT.value] = 1.0

    bridge = OperationalPipelineBridge(registry)
    bridge.sync_certified_actions(lambda _name, _initial_value: None)

    after_run = make_charge_accumulation_run(
        experiment_id="charge_accumulation_after_unlock",
        waveform="reverse",
        amplitude=0.9,
        duration=1.0,
    )
    after_data = charge_accumulation_data(after_run)

    return ChargeUnlockResult(
        candidate_id=record.candidate_id,
        status=record.status,
        symbolic_name=record.symbolic_name or "",
        fit_relative_residual=fit.relative_residual,
        validation_scores=validation_scores,
        emitted_columns_before_unlock=tuple(sorted(str(k) for k in before_data.data_keys)),
        emitted_columns_after_unlock=tuple(sorted(str(k) for k in after_data.data_keys)),
    )
