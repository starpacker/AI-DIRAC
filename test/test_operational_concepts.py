import tempfile
import unittest
from pathlib import Path

import numpy as np

from aiphy import Exp, MeasureType, Proposition
from aiphy.experiment.challenge_thermodynamics import (
    two_body_contact_unequal_config,
)
from aiphy.main import Theorist
from aiphy.operational_concepts import (
    AdditiveInvariantMiner,
    ChannelOrigin,
    ChannelRef,
    ConceptStatus,
    EvidenceKind,
    ForbiddenEvidenceError,
    InvariantObservation,
    MeasurementProtocol,
    OperationalConceptRegistry,
    OperationalPipelineBridge,
    ValidationEvidence,
    ValidationTaskKind,
    allowed_channels,
)
from aiphy.operational_pipeline import route_operational_relations


class AdditiveInvariantMinerTests(unittest.TestCase):
    def test_recovers_weight_from_existing_unequal_thermal_experiment(self):
        structure = two_body_contact_unequal_config()
        observations = []
        for run in range(3):
            structure.random_settings()
            measured = structure.collect_expdata(MeasureType.default()).datastruct
            first = np.asarray(
                measured.fetch_data_by_str("temp[1]").normal_data.data,
                dtype=float,
            ).mean(axis=0)
            second = np.asarray(
                measured.fetch_data_by_str("temp[2]").normal_data.data,
                dtype=float,
            ).mean(axis=0)
            observations.append(
                InvariantObservation.from_state_series(
                    observation_id=f"existing-thermal-{run}",
                    experiment_id=f"existing-thermal-{run}",
                    experiment_family="two_body_contact_unequal",
                    observable="temp",
                    entity_ids=("body-1", "body-2"),
                    states=np.column_stack([first, second]),
                    channels=allowed_channels(("temp[1]", "temp[2]", "t[0]")),
                    interventions={"randomized_initial_conditions": run},
                )
            )

        fit = AdditiveInvariantMiner().fit(
            observations, reference_entity="body-2"
        )

        # The discovery input contains temperature traces only.  The simulator's
        # hidden object parameter is used here solely as a post-hoc test oracle.
        self.assertAlmostEqual(fit.weights["body-1"], 2.0, delta=0.01)
        self.assertAlmostEqual(fit.weights["body-2"], 1.0, places=8)

    def test_discovers_anonymous_thermal_weight_under_control_changes(self):
        observations = []
        runs = ((100.0, 0.0, 0.5), (80.0, 10.0, 1.0), (120.0, 20.0, 1.7))
        for run, (initial_a, initial_reference, coupling) in enumerate(runs):
            time = np.linspace(0.0, 2.0, 40)
            difference = (initial_a - initial_reference) * np.exp(-coupling * time)
            conserved_coordinate = 2.0 * initial_a + initial_reference
            state_a = (conserved_coordinate + difference) / 3.0
            state_reference = (conserved_coordinate - 2.0 * difference) / 3.0
            observations.append(
                InvariantObservation.from_state_series(
                    observation_id=f"thermal-observation-{run}",
                    experiment_id=f"thermal-definition-{run}",
                    experiment_family="anonymous_contact",
                    observable="sensor_coordinate",
                    entity_ids=("body-a", "reference"),
                    states=np.column_stack([state_a, state_reference]),
                    channels=allowed_channels(("sensor-a", "sensor-reference", "t")),
                    interventions={
                        "initial_state": run,
                        "coupling_control": coupling,
                    },
                )
            )

        fit = AdditiveInvariantMiner().fit(
            observations, reference_entity="reference"
        )

        self.assertAlmostEqual(fit.weights["reference"], 1.0, places=8)
        self.assertAlmostEqual(fit.weights["body-a"], 2.0, places=8)
        self.assertLess(fit.relative_residual, 1e-10)
        self.assertGreater(fit.spectral_gap, 100.0)
        self.assertEqual(fit.observable, "sensor_coordinate")

    def test_one_reference_connects_multiple_collision_weights(self):
        rng = np.random.default_rng(1729)
        hidden_for_test_evaluation_only = {"reference": 1.0, "body-a": 2.5, "body-b": 4.0}
        observations = []
        for target in ("body-a", "body-b"):
            for run in range(10):
                target_change = rng.uniform(-3.0, 3.0)
                reference_change = (
                    -hidden_for_test_evaluation_only[target] * target_change
                )
                measured_changes = np.array(
                    [[reference_change, target_change]], dtype=float
                )
                measured_changes += rng.normal(0.0, 1e-4, measured_changes.shape)
                observations.append(
                    InvariantObservation(
                        observation_id=f"{target}-{run}",
                        experiment_id=f"collision-{target}-{run}",
                        experiment_family="anonymous_collision",
                        observable="derived_velocity",
                        entity_ids=("reference", target),
                        deltas=measured_changes,
                        channels=allowed_channels(("position", "time")),
                        interventions={"launch_control": run},
                    )
                )

        fit = AdditiveInvariantMiner().fit(
            observations, reference_entity="reference"
        )

        self.assertAlmostEqual(fit.weights["reference"], 1.0, places=6)
        self.assertAlmostEqual(fit.weights["body-a"], 2.5, delta=0.01)
        self.assertAlmostEqual(fit.weights["body-b"], 4.0, delta=0.01)
        self.assertLess(fit.relative_residual, 0.002)

    def test_hidden_simulator_channel_is_rejected(self):
        observation = InvariantObservation(
            observation_id="leaked",
            experiment_id="leaked",
            experiment_family="collision",
            observable="velocity",
            entity_ids=("reference", "target"),
            deltas=np.array([[1.0, -1.0]] * 5),
            channels=(
                ChannelRef("simulator_mass", ChannelOrigin.SIMULATOR_HIDDEN),
            ),
        )

        with self.assertRaises(ForbiddenEvidenceError):
            AdditiveInvariantMiner().fit(
                [observation], reference_entity="reference"
            )


class OperationalConceptRegistryTests(unittest.TestCase):
    def setUp(self):
        time = np.linspace(0.0, 1.0, 20)
        difference = 80.0 * np.exp(-time)
        states = np.column_stack(
            [(200.0 + difference) / 3.0, (200.0 - 2.0 * difference) / 3.0]
        )
        observation = InvariantObservation.from_state_series(
            observation_id="definition-observation",
            experiment_id="definition-experiment",
            experiment_family="contact-apparatus-a",
            observable="sensor_coordinate",
            entity_ids=("target", "reference"),
            states=states,
            channels=allowed_channels(("sensor-target", "sensor-reference", "t")),
        )
        fit = AdditiveInvariantMiner().fit(
            [observation], reference_entity="reference"
        )
        protocol = MeasurementProtocol(
            experiment_family="contact-apparatus-a",
            observable="sensor_coordinate",
            estimator="fit one object weight in a closed additive transition",
            required_channels=("sensor-target", "sensor-reference", "t"),
            reference_entity="reference",
            reference_value=1.0,
            nuisance_controls=("initial_state", "coupling_control"),
            alternate_experiment_families=("contact-apparatus-b",),
        )
        self.registry = OperationalConceptRegistry()
        self.record = self.registry.propose_additive_weight(
            fit,
            protocol,
            entity_type="AnonymousBody",
            symbolic_name="C_trial",
            applicable_experiment_families=("future_dynamics",),
        )

    @staticmethod
    def evidence(
        kind,
        number,
        *,
        family="contact-apparatus-a",
        intervention="",
        independent=True,
    ):
        return ValidationEvidence(
            kind=kind,
            experiment_id=f"validation-{kind.value}-{number}",
            experiment_family=family,
            score=0.95,
            independent=independent,
            intervention=intervention,
            channels=allowed_channels(("sensor-target", "sensor-reference", "t")),
        )

    def test_trial_is_quarantined_and_tasks_request_missing_controls(self):
        self.assertEqual(self.record.status, ConceptStatus.TRIAL)
        self.assertEqual(
            self.registry.eligible_records("future_dynamics", purpose="law_discovery"),
            [],
        )

        tasks = self.registry.validation_tasks(self.record.candidate_id)
        task_kinds = {task.kind for task in tasks}
        self.assertIn(ValidationTaskKind.VARY_NUISANCE, task_kinds)
        self.assertIn(ValidationTaskKind.REPLACE_REFERENCE, task_kinds)
        self.assertIn(ValidationTaskKind.CROSS_APPARATUS, task_kinds)
        nuisance_interventions = {
            task.intervention
            for task in tasks
            if task.kind is ValidationTaskKind.VARY_NUISANCE
        }
        self.assertEqual(
            nuisance_interventions,
            {"vary:initial_state", "vary:coupling_control"},
        )

    def test_only_independent_complete_evidence_certifies_and_syncs_action(self):
        candidate_id = self.record.candidate_id
        # Reusing definition data must not promote the candidate.
        self.registry.record_evidence(
            candidate_id,
            ValidationEvidence(
                kind=EvidenceKind.REPEATABILITY,
                experiment_id="definition-experiment",
                experiment_family="contact-apparatus-a",
                score=1.0,
                independent=True,
                channels=allowed_channels(("sensor-target", "sensor-reference", "t")),
            ),
        )
        self.assertEqual(self.record.status, ConceptStatus.TRIAL)

        evidence = (
            self.evidence(EvidenceKind.REPEATABILITY, 1),
            self.evidence(
                EvidenceKind.NUISANCE_INVARIANCE,
                1,
                intervention="vary:initial_state",
            ),
            self.evidence(
                EvidenceKind.NUISANCE_INVARIANCE,
                2,
                intervention="vary:coupling_control",
            ),
            self.evidence(EvidenceKind.OBJECT_SENSITIVITY, 1),
            self.evidence(EvidenceKind.REFERENCE_TRANSITIVITY, 1),
            self.evidence(
                EvidenceKind.CROSS_APPARATUS_TRANSFER,
                1,
                family="contact-apparatus-b",
            ),
            self.evidence(
                EvidenceKind.HELDOUT_PREDICTION,
                1,
                family="future_dynamics",
            ),
            self.evidence(
                EvidenceKind.PROVENANCE_AUDIT,
                1,
                family="audit",
                independent=False,
            ),
        )
        for item in evidence[:-1]:
            status = self.registry.record_evidence(candidate_id, item)
            self.assertEqual(status, ConceptStatus.TRIAL)
        status = self.registry.record_evidence(candidate_id, evidence[-1])

        self.assertEqual(status, ConceptStatus.CERTIFIED)
        eligible = self.registry.eligible_records(
            "future_dynamics", purpose="law_discovery"
        )
        self.assertEqual([item.candidate_id for item in eligible], [candidate_id])

        calls = []
        bridge = OperationalPipelineBridge(self.registry)
        synced = bridge.sync_certified_actions(
            lambda name, initial: calls.append((name, initial))
        )
        self.assertEqual(synced, ["C_trial"])
        self.assertEqual(calls, [("C_trial", 8.0)])

    def test_controlled_falsification_rejects_even_after_positive_evidence(self):
        candidate_id = self.record.candidate_id
        self.registry.record_evidence(
            candidate_id, self.evidence(EvidenceKind.REPEATABILITY, 1)
        )
        status = self.registry.record_evidence(
            candidate_id,
            ValidationEvidence(
                kind=EvidenceKind.FALSIFICATION,
                experiment_id="hostile-control",
                experiment_family="contact-apparatus-a",
                score=0.99,
                intervention="vary:geometry",
                channels=allowed_channels(("sensor-target", "sensor-reference", "t")),
                notes="the proposed object weight changes with an alleged nuisance",
            ),
        )
        self.assertEqual(status, ConceptStatus.REJECTED)
        self.assertIn("changes", self.record.rejection_reason)
        self.assertEqual(self.registry.validation_tasks(candidate_id), [])

    def test_registry_round_trip_preserves_trial_and_evidence(self):
        self.registry.record_evidence(
            self.record.candidate_id,
            self.evidence(EvidenceKind.REPEATABILITY, 1),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operational.json"
            self.registry.save(path)
            loaded = OperationalConceptRegistry.load(path)

        loaded_record = loaded.records[self.record.candidate_id]
        self.assertEqual(loaded_record.status, ConceptStatus.TRIAL)
        self.assertEqual(loaded_record.fit.weights, self.record.fit.weights)
        self.assertEqual(len(loaded_record.evidence), 1)
        self.assertEqual(loaded._next_id, 2)

    def test_theorist_sync_exposes_only_certified_materialized_concept(self):
        theorist = Theorist(num_threads=1)
        theorist.operational_registry = self.registry
        theorist.operational_bridge = OperationalPipelineBridge(self.registry)
        self.record.symbolic_name = "posx"
        theorist.memory.action_bandit.actions.discard("posx")

        self.assertEqual(theorist.sync_certified_operational_actions(), [])
        self.assertNotIn("posx", theorist.memory.action_bandit.actions)

        self.record.status = ConceptStatus.CERTIFIED
        self.assertEqual(theorist.sync_certified_operational_actions(), ["posx"])
        self.assertIn("posx", theorist.memory.action_bandit.actions)


class MainPipelineOperationalRoutingTests(unittest.TestCase):
    def test_conserved_relation_routes_existing_trace_to_anonymous_trial(self):
        theorist = Theorist(num_threads=1)
        theorist.register_experiment(two_body_contact_unequal_config())
        proposition = Proposition.IsConserved(Exp("2 * temp[1] + temp[2]"))

        created = route_operational_relations(
            theorist, "two_body_contact_unequal", [proposition]
        )

        self.assertEqual(created, ["OC_0001"])
        record = theorist.operational_registry.records[created[0]]
        self.assertEqual(record.status, ConceptStatus.TRIAL)
        self.assertIsNone(record.symbolic_name)
        self.assertAlmostEqual(record.fit.weights["Particle:1"], 2.0, delta=0.01)
        self.assertAlmostEqual(record.fit.weights["Particle:2"], 1.0, places=8)
        self.assertEqual(
            record.protocol.nuisance_controls, ("T10", "T20", "k12")
        )
        self.assertTrue(theorist.pending_operational_validation_tasks())

        # Re-routing the same accepted relation must not create vocabulary spam.
        self.assertEqual(
            route_operational_relations(
                theorist, "two_body_contact_unequal", [proposition]
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
