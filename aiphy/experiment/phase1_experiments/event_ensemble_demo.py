"""
Phase 1 — Event Ensemble Demonstration Experiment.

A simple stochastic photon counting experiment:
  - A light source emits photons at an average rate λ
  - A detector counts photons over fixed time intervals
  - The experiment is repeated N times to build an ensemble

Observable: count (photon counts per interval)
Control: λ (average photon rate), known to the apparatus
Context: integration_time (measurement duration)

The discovery engine should find:
  - mean(count) ∝ λ (Poisson mean equals rate)
  - var(count) ∝ λ (Poisson variance equals mean)
  - The count distribution is Poisson (non-Gaussian tail)

This is the SIMPLEST possible event ensemble — one channel, varying
counts, repeated trials. The experiment demonstrates that the bridge
correctly converts event ensembles for the discovery pipeline.

RAW_SENSOR: count — direct detector reading (integer counts)
CONTROL: λ — known source intensity, not hidden from the apparatus
"""

import numpy as np


def generate_event_ensemble_photon_counts(
    rate: float = 5.0,
    integration_time: float = 1.0,
    n_events: int = 200,
    seed: int = 42,
) -> dict:
    """
    Generate a photon-counting event ensemble.

    Returns a dictionary with:
      - 'counts': np.ndarray of shape (n_events,) — photon counts per interval
      - 'rate': float — known photon emission rate λ
      - 'integration_time': float — measurement duration
    """
    rng = np.random.RandomState(seed)
    effective_rate = rate * integration_time
    counts = rng.poisson(effective_rate, size=n_events).astype(np.float64)

    return {
        'counts': counts,
        'rate': rate,
        'integration_time': integration_time,
        'n_events': n_events,
    }


def generate_radioactive_decay_ensemble(
    half_life: float = 10.0,
    initial_count: int = 1000,
    measurement_interval: float = 1.0,
    n_measurements: int = 300,
    seed: int = 42,
) -> dict:
    """
    Generate a radioactive decay event ensemble.

    Each "event" is a time interval. The observed count is the number
    of decays detected in that interval. As time progresses, the
    expected count follows exponential decay.

    Returns:
      - 'counts': decay counts per interval
      - 't': mid-time of each interval
      - 'half_life': known half-life (control parameter)
      - 'initial_count': known initial atom count (control parameter)
    """
    rng = np.random.RandomState(seed)
    decay_const = np.log(2) / half_life
    t = np.arange(0, n_measurements) * measurement_interval
    t_mid = t + measurement_interval / 2

    N_atoms = initial_count * np.exp(-decay_const * t_mid)
    expected = initial_count - N_atoms
    # Approximate Poisson with Gaussian for larger counts
    counts = rng.poisson(np.maximum(0, np.diff(np.insert(N_atoms, 0, initial_count))))

    # Pad to match t_mid length
    if len(counts) < n_measurements:
        counts = np.append(counts, rng.poisson(expected[-1]))

    return {
        'counts': counts.astype(np.float64),
        't': t_mid,
        'half_life': half_life,
        'initial_count': initial_count,
    }
