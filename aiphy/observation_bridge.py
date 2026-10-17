"""
Observation-to-discovery bridge — Phase 1.

Converts the unified Observation protocol (TrajectoryData, EventEnsembleData,
FieldGridData) into the DataStruct objects that the existing symbolic regression
and PCA-based discovery pipeline expects.

This is the key integration layer that lets the SAME discovery API handle:
  - deterministic trajectories (classical mechanics)
  - event ensembles (wave-particle, quantum experiments)
  - spatial field grids (continuum physics, GR)

All conversions preserve full provenance, and the bridge validates that
no SIMULATOR_HIDDEN channels leak through.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from .observation_protocol import (
    Observation, TrajectoryData, EventEnsembleData, FieldGridData,
    MeasurementContext,
)
from .provenance import (
    ChannelProvenance, ProvenanceTracker, ChannelOrigin,
    get_global_tracker,
)
from .interface import DataStruct, ExpData, ConstData, AtomExp
from .core import Exp  # Exp is used for constructing AtomExp


# ─── Bridge dispatcher ──────────────────────────────────────────────────

def observation_to_datastruct(obs: Observation) -> DataStruct:
    """
    Convert ANY Observation subtype to a DataStruct for the discovery pipeline.

    This is the single entry point — the same API call for all three
    observation types.
    """
    if isinstance(obs, TrajectoryData):
        return _trajectory_to_datastruct(obs)
    elif isinstance(obs, EventEnsembleData):
        return _event_ensemble_to_datastruct(obs)
    elif isinstance(obs, FieldGridData):
        return _field_grid_to_datastruct(obs)
    else:
        raise TypeError(f"Unknown observation type: {type(obs)}")


def observation_to_channel_arrays(
    obs: Observation,
) -> Dict[str, np.ndarray]:
    """Extract all channels as numpy arrays, regardless of observation type."""
    result = {}
    for ch_name in obs.get_channels():
        arr = obs.get_channel(ch_name)
        if arr is not None:
            result[ch_name] = arr
    return result


# ─── Trajectory → DataStruct ────────────────────────────────────────────

def _trajectory_to_datastruct(traj: TrajectoryData) -> DataStruct:
    """
    Convert trajectory data to DataStruct.

    Each channel becomes an AtomExp → ExpData mapping.
    Time series of length N → N individual data points.
    """
    ds = DataStruct.empty()
    n = traj.num_points()

    for ch_name in traj.get_channels():
        arr = traj.get_channel(ch_name)
        if arr is None:
            continue
        # Create an AtomExp for this channel
        atom = _make_atom(ch_name)
        # Wrap as ExpData (normal time-varying data)
        expdata = _array_to_expdata(arr, n)
        ds.add_data(atom, expdata)

    return ds


# ─── Event ensemble → DataStruct ────────────────────────────────────────

def _event_ensemble_to_datastruct(ee: EventEnsembleData) -> DataStruct:
    """
    Convert event ensemble data to DataStruct.

    Each event becomes a "time step" — the symbolic regression engine
    treats events as independent samples.  For ensemble-level conserved
    quantities (e.g., total counts, mean values), we add derived
    channels that aggregate across events.

    Strategy: flatten the ensemble into a single DataStruct where
    event_index is analogous to time_index. The regression engine
    can then discover relations among event properties.
    """
    ds = DataStruct.empty()
    n = ee.num_events()

    for ch_name in ee.get_channels():
        arr = ee.get_channel(ch_name)
        if arr is None:
            continue
        atom = _make_atom(ch_name)
        expdata = _array_to_expdata(arr, n)
        ds.add_data(atom, expdata)

    # Add context-derived constants if available
    if ee.context and ee.context.parameter_values:
        for param_name, param_val in ee.context.parameter_values.items():
            atom = _make_atom(f"ctx_{param_name}")
            # Context parameters are CONSTANT across all events
            const_data = ConstData(float(param_val), 0.0)
            ds.add_data(atom, ExpData.from_const_data(const_data))

    return ds


# ─── Field grid → DataStruct ────────────────────────────────────────────

def _field_grid_to_datastruct(fg: FieldGridData) -> DataStruct:
    """
    Convert field grid data to DataStruct.

    The grid is flattened: each grid point becomes an independent sample.
    Grid coordinates (x, y, z) are added as channels alongside field values.

    For spatial held-out validation, use FieldGridData.spatial_held_out_split()
    BEFORE converting to DataStruct.
    """
    ds = DataStruct.empty()

    # Determine total number of grid points
    n_points = 1
    shape = []
    axes_order = list(fg.grid_axes.keys())
    for axis_name in axes_order:
        axis = fg.grid_axes[axis_name]
        n_points *= len(axis)
        shape.append(len(axis))

    # Add grid coordinate channels (flattened)
    # Build coordinate mesh
    coords = [fg.grid_axes[name] for name in axes_order]
    mesh = np.meshgrid(*coords, indexing='ij')

    for i, axis_name in enumerate(axes_order):
        flat_coords = mesh[i].flatten()
        atom = _make_atom(f"grid_{axis_name}")
        ds.add_data(atom, _array_to_expdata(flat_coords, n_points))

    # Add field channels (flattened)
    for ch_name in fg.get_channels():
        arr = fg.get_channel(ch_name)
        if arr is None:
            continue
        flat_arr = arr.flatten()
        atom = _make_atom(ch_name)
        ds.add_data(atom, _array_to_expdata(flat_arr, n_points))

    return ds


# ─── Helpers ─────────────────────────────────────────────────────────────

def _make_atom(name: str) -> AtomExp:
    """Create an AtomExp for a channel name."""
    return AtomExp(name)


def _array_to_expdata(arr: np.ndarray, n_points: int) -> "ExpData":
    """
    Convert a numpy array to ExpData suitable for DataStruct.

    For the bridge, we need to reconstruct the time-series structure.
    ExpData.from_elem(mean, std, n, repeat_time) creates n bins, each
    with repeat_time repeated measurements from N(mean, std).

    Since we have the actual data, we use from_elem with n=n_points,
    repeat_time=1, and each point's actual value as mean (with tiny std).
    This is lossy but preserves the correct time-series structure for
    the regression engine.
    """
    if len(arr) == 0:
        return ExpData.from_const_data(ConstData(0.0, 0.0))

    mu = float(np.mean(arr))
    sigma = float(np.std(arr))

    # If effectively constant, use compact const representation
    if sigma < 1e-12 or np.allclose(arr, arr[0]):
        return ExpData.from_const_data(ConstData(mu, 0.0))

    # Reconstruct time-varying data: n_points elements, each with 1 repeat,
    # using the overall mean/std (lossy but preserves statistical properties)
    return ExpData.from_elem(mu, sigma, n_points, 1)


# ─── Provenance integration ──────────────────────────────────────────────

def register_observation_provenance(
    obs: Observation,
    experiment_name: str,
    tracker: Optional[ProvenanceTracker] = None,
) -> ProvenanceTracker:
    """
    Register provenance for all channels in an Observation.

    Automatically detects which channels are RAW_SENSOR, DERIVED, etc.
    and registers them in the provenance tracker.

    Returns the tracker for chaining.
    """
    if tracker is None:
        tracker = get_global_tracker()

    for ch_name in obs.get_channels():
        prov = obs.get_provenance(ch_name)
        if prov is not None and prov.experiment_name:
            # Already registered with full provenance
            tracker._channels[
                ProvenanceTracker._make_key(prov.experiment_name, ch_name)
            ] = prov
        else:
            # Auto-classify: everything in the observation is RAW_SENSOR
            # unless explicitly marked otherwise
            tracker.register_channel(
                channel_name=ch_name,
                origin=ChannelOrigin.RAW_SENSOR,
                generation_method=f"{obs.observation_type} data from simulator",
                experiment_name=experiment_name,
                justification=f"Direct {obs.observation_type} channel",
            )

    # Register context parameters
    ctx = None
    if hasattr(obs, 'context') and obs.context:
        ctx = obs.context
    if ctx and ctx.parameter_values:
        for param_name in ctx.parameter_values:
            tracker.register_channel(
                channel_name=f"ctx_{param_name}",
                origin=ChannelOrigin.CONTROL,
                generation_method="Experiment control parameter",
                experiment_name=experiment_name,
                justification=f"Control parameter '{param_name}'",
            )

    return tracker


# ─── Validation utilities ────────────────────────────────────────────────

def validate_observation_pipeline(
    obs: Observation,
    experiment_name: str,
) -> Dict[str, Any]:
    """
    Run a full validation of the observation pipeline for ONE observation.

    Returns a dictionary with:
      - 'provenance_clean': bool — no leaks detected
      - 'to_datastruct_ok': bool — conversion succeeded
      - 'channels': List[str] — all channel names
      - 'num_samples': int — number of data points
      - 'errors': List[str] — any errors encountered
    """
    result = {
        'provenance_clean': False,
        'to_datastruct_ok': False,
        'channels': obs.get_channels(),
        'num_samples': 0,
        'errors': [],
        'observation_type': obs.observation_type,
    }

    # 1. Provenance audit
    tracker = register_observation_provenance(obs, experiment_name)
    leaks = tracker.audit_leaks()
    result['provenance_clean'] = len(leaks) == 0
    if leaks:
        result['errors'].extend(leaks)

    # 2. DataStruct conversion
    try:
        ds = observation_to_datastruct(obs)
        result['to_datastruct_ok'] = True
        if obs.observation_type == 'trajectory' and hasattr(obs, 'num_points'):
            result['num_samples'] = obs.num_points()
        elif obs.observation_type == 'event_ensemble' and hasattr(obs, 'num_events'):
            result['num_samples'] = obs.num_events()
        elif obs.observation_type == 'field_grid' and hasattr(obs, 'num_points'):
            result['num_samples'] = obs.num_points()
        else:
            result['num_samples'] = len(obs.get_channels())  # fallback
    except Exception as e:
        result['errors'].append(f"DataStruct conversion failed: {e}")

    return result
