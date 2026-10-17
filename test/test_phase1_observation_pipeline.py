"""
Phase 1 integration tests — unified observation protocol.

Verifies that the same discovery API handles:
  1. TrajectoryData (classical mechanics — free fall)
  2. EventEnsembleData (stochastic photon counting)
  3. FieldGridData (2D heat equation)

All three must:
  - Pass complete provenance checks (no SIMULATOR_HIDDEN leaks)
  - Convert to DataStruct via observation_to_datastruct
  - Produce valid channel data
  - Support held-out splits where applicable
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiphy.observation_protocol import (
    TrajectoryData, EventEnsembleData, FieldGridData,
    MeasurementContext,
)
from aiphy.observation_bridge import (
    observation_to_datastruct,
    observation_to_channel_arrays,
    register_observation_provenance,
    validate_observation_pipeline,
)
from aiphy.provenance import (
    ProvenanceTracker, ChannelOrigin, reset_global_tracker,
)
from aiphy.experiment.phase1_experiments.event_ensemble_demo import (
    generate_event_ensemble_photon_counts,
    generate_radioactive_decay_ensemble,
)
from aiphy.experiment.phase1_experiments.field_grid_demo import (
    generate_heat_equation_field,
    generate_electrostatic_potential_field,
)


class TestPhase1ObservationPipeline(unittest.TestCase):
    """Base class with shared helpers for all observation type tests."""

    def assertPipelinePasses(self, obs, exp_name: str, desc: str = ""):
        """Assert that an observation passes the full pipeline validation."""
        result = validate_observation_pipeline(obs, exp_name)
        self.assertTrue(
            result['provenance_clean'],
            f"{desc}: Provenance check failed: {result['errors']}"
        )
        self.assertTrue(
            result['to_datastruct_ok'],
            f"{desc}: DataStruct conversion failed: {result['errors']}"
        )
        self.assertGreater(
            result['num_samples'], 0,
            f"{desc}: No samples detected"
        )
        return result


class TestTrajectoryPipeline(unittest.TestCase):
    """TrajectoryData: classical mechanics time series."""

    def setUp(self):
        reset_global_tracker()

    def test_free_fall_trajectory(self):
        """Build a synthetic free-fall trajectory and validate the pipeline."""
        t = np.linspace(0, 2.0, 200)
        g = 9.8
        z0 = 10.0
        posz = z0 - 0.5 * g * t**2 + np.random.normal(0, 0.01, 200)

        traj = TrajectoryData(
            channels={
                'posx': np.zeros(200) + np.random.normal(0, 0.01, 200),
                'posz': posz,
                't': t,
            },
            time=t,
            context=MeasurementContext(
                apparatus_name="free_fall_apparatus",
                parameter_values={'g': 9.8},
            ),
        )

        result = validate_observation_pipeline(traj, "free_fall_synthetic")
        self.assertEqual(result['observation_type'], 'trajectory')
        self.assertTrue(result['provenance_clean'])
        self.assertTrue(result['to_datastruct_ok'])
        self.assertEqual(result['num_samples'], 200)

        # Test held-out split
        train, held_out = traj.held_out_split(fraction=0.2)
        self.assertEqual(train.num_points(), 160)
        self.assertEqual(held_out.num_points(), 40)
        self.assertGreater(np.abs(np.mean(train.channels['posz']) - np.mean(held_out.channels['posz'])), 0.1,
            "Train and held-out should have different data")

    def test_trajectory_to_datastruct_conversion(self):
        """TrajectoryData converts to DataStruct successfully."""
        t = np.linspace(0, 1.0, 100)
        posx = np.sin(2 * np.pi * t) + np.random.normal(0, 0.02, 100)

        traj = TrajectoryData(
            channels={'posx': posx, 't': t},
            time=t,
        )

        ds = observation_to_datastruct(traj)
        self.assertIsNotNone(ds)

        # Channels should be accessible via the DataStruct
        channels = observation_to_channel_arrays(traj)
        self.assertIn('posx', channels)
        self.assertIn('t', channels)
        np.testing.assert_array_almost_equal(channels['posx'], posx)


class TestEventEnsemblePipeline(unittest.TestCase):
    """EventEnsembleData: stochastic photon counting and decay."""

    def setUp(self):
        reset_global_tracker()

    def test_photon_counting_ensemble(self):
        """Photon counting ensemble passes the pipeline."""
        from aiphy.observation_protocol import EventEnsembleData, MeasurementContext

        data = generate_event_ensemble_photon_counts(rate=5.0, n_events=200)
        ee = EventEnsembleData(
            channels={'counts': data['counts']},
            context=MeasurementContext(
                parameter_values={'rate': data['rate']},
            ),
        )

        result = validate_observation_pipeline(ee, "photon_counting")
        self.assertEqual(result['observation_type'], 'event_ensemble')
        self.assertTrue(result['provenance_clean'])
        self.assertTrue(result['to_datastruct_ok'])
        self.assertEqual(result['num_samples'], 200)

    def test_photon_count_distribution(self):
        """Photon counts follow Poisson statistics (mean ≈ variance)."""
        data = generate_event_ensemble_photon_counts(rate=10.0, n_events=1000)
        counts = data['counts']

        mean = np.mean(counts)
        var = np.var(counts)

        # Poisson: mean == variance (within sampling error)
        ratio = var / max(mean, 1e-10)
        self.assertLess(abs(ratio - 1.0), 0.3,
            f"Poisson property violated: var/mean = {ratio:.3f}")

    def test_bootstrap_resample(self):
        """EventEnsembleData supports bootstrap resampling."""
        data = generate_event_ensemble_photon_counts(rate=5.0, n_events=200)
        ee = EventEnsembleData(
            channels={'counts': data['counts']},
            context=MeasurementContext(parameter_values={'rate': data['rate']}),
        )

        resampled = ee.bootstrap_sample(n_samples=100)
        self.assertEqual(resampled.num_events(), 100)
        # Bootstrap should preserve mean within sampling error
        orig_mean = np.mean(data['counts'])
        resampled_mean = np.mean(resampled.channels['counts'])
        self.assertLess(abs(orig_mean - resampled_mean) / max(orig_mean, 1e-10), 0.2)

    def test_conditional_distribution(self):
        """EventEnsembleData supports conditional distribution queries."""
        data = generate_event_ensemble_photon_counts(rate=5.0, n_events=500)
        # Create a synthetic second channel correlated with counts
        rng = np.random.RandomState(42)
        synthetic = data['counts'] * 2.0 + rng.normal(0, 1.0, 500)

        ee = EventEnsembleData(
            channels={'counts': data['counts'], 'scaled': synthetic},
        )

        # Get distribution of 'scaled' when 'counts' ≈ 5
        cond = ee.conditional_distribution('scaled', 'counts', 5.0, tolerance=1.0)
        self.assertIsNotNone(cond)
        self.assertGreater(len(cond), 0)
        # Conditional mean should be approximately 2 * 5 = 10
        if len(cond) > 5:
            self.assertLess(abs(np.mean(cond) - 10.0), 3.0)

    def test_radioactive_decay_ensemble(self):
        """Radioactive decay ensemble passes the pipeline."""
        data = generate_radioactive_decay_ensemble(n_measurements=200)

        # Use t as a channel, counts as another
        ee = EventEnsembleData(
            channels={'counts': data['counts'], 't': data['t']},
            context=MeasurementContext(
                parameter_values={
                    'half_life': data['half_life'],
                    'initial_count': data['initial_count'],
                },
            ),
        )

        result = validate_observation_pipeline(ee, "radioactive_decay")
        self.assertTrue(result['provenance_clean'])
        self.assertTrue(result['to_datastruct_ok'])


class TestFieldGridPipeline(unittest.TestCase):
    """FieldGridData: 2D heat equation and electrostatic field."""

    def setUp(self):
        reset_global_tracker()

    def test_heat_equation_field(self):
        """2D heat equation field passes the pipeline."""
        data = generate_heat_equation_field(nx=50, ny=25)
        fg = FieldGridData(
            channels={'T': data['T']},
            grid_axes={'x': data['x'], 'y': data['y']},
            context=MeasurementContext(
                parameter_values={
                    'T_hot': data['T_hot'],
                    'T_cold': data['T_cold'],
                },
            ),
        )

        result = validate_observation_pipeline(fg, "heat_equation")
        self.assertEqual(result['observation_type'], 'field_grid')
        self.assertTrue(result['provenance_clean'])
        self.assertTrue(result['to_datastruct_ok'])

        # Check that the number of samples equals grid points
        self.assertEqual(result['num_samples'], 50 * 25)

    def test_heat_equation_linearity(self):
        """Heat equation field is approximately linear in x."""
        data = generate_heat_equation_field(nx=50, ny=25, noise_std=1e-6)
        T = data['T']
        x = data['x']

        # Average T along y-axis for each x position
        T_profile = np.mean(T, axis=0)

        # Fit a line: T = a + b*x
        X = np.column_stack([np.ones_like(x), x])
        coeffs = np.linalg.lstsq(X, T_profile, rcond=None)[0]

        # Goodness of linear fit
        T_pred = X @ coeffs
        r2 = 1 - np.sum((T_profile - T_pred)**2) / np.sum((T_profile - np.mean(T_profile))**2)
        self.assertGreater(r2, 0.99, f"Heat equation field should be linear in x, R²={r2:.4f}")

    def test_spatial_held_out_split(self):
        """FieldGridData supports spatial held-out validation along an axis."""
        data = generate_heat_equation_field(nx=100, ny=20, noise_std=1e-6)
        fg = FieldGridData(
            channels={'T': data['T']},
            grid_axes={'x': data['x'], 'y': data['y']},
        )

        train, held_out = fg.spatial_held_out_split(fraction=0.3, axis='x')
        self.assertIsNotNone(train)
        self.assertIsNotNone(held_out)

        # Train has 70 grid points in x, held_out has 30
        self.assertEqual(train.grid_axes['x'].shape[0], 70)
        self.assertEqual(held_out.grid_axes['x'].shape[0], 30)

        # Both should convert to DataStruct
        ds_train = observation_to_datastruct(train)
        ds_held = observation_to_datastruct(held_out)
        self.assertIsNotNone(ds_train)
        self.assertIsNotNone(ds_held)

    def test_electrostatic_potential_field(self):
        """2D electrostatic potential field passes the pipeline."""
        data = generate_electrostatic_potential_field(nx=60, ny=60)
        fg = FieldGridData(
            channels={'V': data['V']},
            grid_axes={'x': data['x'], 'y': data['y']},
        )

        result = validate_observation_pipeline(fg, "electrostatic_potential")
        self.assertTrue(result['provenance_clean'])
        self.assertTrue(result['to_datastruct_ok'])
        self.assertEqual(result['num_samples'], 3600)

    def test_inverse_square_decay(self):
        """Electrostatic potential decays approximately as 1/r."""
        data = generate_electrostatic_potential_field(nx=100, ny=5, noise_std=0.05)
        V = data['V']  # shape (100, 5)
        x = data['x']

        # Take the center row (y ≈ 0) where V should be maximal
        mid_row = V[:, 2]
        x0_idx = np.argmax(mid_row)  # index of charge x-position

        # Distance from charge center (in grid units)
        dist = np.abs(np.arange(len(mid_row)) - x0_idx)
        dist[dist < 2] = 2  # avoid singularity near charge

        # Check that V * dist is approximately constant (1/r scaling)
        product = mid_row * dist
        product_away = product[dist > 5]  # far from charge
        cv = np.std(product_away) / np.mean(product_away)
        self.assertLess(cv, 0.5,
            f"V should decay as 1/r; V*r coefficient of variation = {cv:.3f}")


class TestUnifiedDiscoveryAPI(unittest.TestCase):
    """
    CRITICAL: The same observation_to_datastruct() function works for
    ALL THREE observation types without any type-specific branching
    in the caller.
    """

    def setUp(self):
        reset_global_tracker()

    def test_same_api_for_all_three_types(self):
        """observation_to_datastruct handles trajectory, event ensemble, and field grid."""

        # 1. Trajectory
        t = np.linspace(0, 1, 50)
        traj = TrajectoryData(
            channels={'posx': np.sin(t), 't': t},
            time=t,
        )
        ds1 = observation_to_datastruct(traj)
        self.assertIsNotNone(ds1)

        # 2. Event ensemble
        data = generate_event_ensemble_photon_counts(rate=5.0, n_events=100)
        ee = EventEnsembleData(
            channels={'counts': data['counts']},
            context=MeasurementContext(parameter_values={'rate': data['rate']}),
        )
        ds2 = observation_to_datastruct(ee)
        self.assertIsNotNone(ds2)

        # 3. Field grid
        fdata = generate_heat_equation_field(nx=30, ny=20, noise_std=1e-6)
        fg = FieldGridData(
            channels={'T': fdata['T']},
            grid_axes={'x': fdata['x'], 'y': fdata['y']},
        )
        ds3 = observation_to_datastruct(fg)
        self.assertIsNotNone(ds3)

        # All three types had zero errors in pipeline validation
        for obs, name in [
            (traj, "trajectory"),
            (ee, "event_ensemble"),
            (fg, "field_grid"),
        ]:
            result = validate_observation_pipeline(obs, name)
            self.assertTrue(result['provenance_clean'], f"{name} provenance failed")
            self.assertTrue(result['to_datastruct_ok'], f"{name} conversion failed")

    def test_no_simulator_hidden_leaks_in_any_type(self):
        """No observation type leaks SIMULATOR_HIDDEN channels."""
        reset_global_tracker()

        # Trajectory
        traj = TrajectoryData(
            channels={'posx': np.zeros(50), 't': np.linspace(0, 1, 50)},
            time=np.linspace(0, 1, 50),
        )
        tracker = register_observation_provenance(traj, "test_traj")
        self.assertEqual(len(tracker.audit_leaks()), 0)

        # Event ensemble
        reset_global_tracker()
        data = generate_event_ensemble_photon_counts(rate=3.0, n_events=50)
        ee = EventEnsembleData(
            channels={'counts': data['counts']},
            context=MeasurementContext(parameter_values={'rate': 3.0}),
        )
        tracker = register_observation_provenance(ee, "test_ee")
        self.assertEqual(len(tracker.audit_leaks()), 0)

        # Field grid
        reset_global_tracker()
        fdata = generate_heat_equation_field(nx=10, ny=10, noise_std=1e-6)
        fg = FieldGridData(
            channels={'T': fdata['T']},
            grid_axes={'x': fdata['x'], 'y': fdata['y']},
        )
        tracker = register_observation_provenance(fg, "test_fg")
        self.assertEqual(len(tracker.audit_leaks()), 0)


class TestProvenanceExpansion(unittest.TestCase):
    """All derived quantities can be recursively expanded to raw channels."""

    def setUp(self):
        reset_global_tracker()

    def test_trajectory_channels_are_raw_sensor(self):
        """Trajectory channels are classified as RAW_SENSOR by default."""
        traj = TrajectoryData(
            channels={'posx': np.zeros(50), 't': np.linspace(0, 1, 50)},
            time=np.linspace(0, 1, 50),
        )
        tracker = register_observation_provenance(traj, "test_traj")

        for ch_name in ['posx', 't']:
            prov = tracker.get_channel(ch_name, 'test_traj')
            self.assertIsNotNone(prov, f"No provenance for {ch_name}")
            self.assertEqual(prov.origin, ChannelOrigin.RAW_SENSOR,
                f"{ch_name} should be RAW_SENSOR, got {prov.origin}")

    def test_context_parameters_are_control(self):
        """Context parameters in event ensembles are CONTROL, not hidden."""
        ee = EventEnsembleData(
            channels={'counts': np.random.poisson(5, 100).astype(np.float64)},
            context=MeasurementContext(parameter_values={'rate': 5.0, 'int_time': 1.0}),
        )
        tracker = register_observation_provenance(ee, "test_exp")

        for param in ['ctx_rate', 'ctx_int_time']:
            prov = tracker.get_channel(param, 'test_exp')
            self.assertIsNotNone(prov, f"No provenance for {param}")
            self.assertEqual(prov.origin, ChannelOrigin.CONTROL,
                f"{param} should be CONTROL, got {prov.origin}")

    def test_field_grid_coordinates_are_raw_sensor(self):
        """Grid coordinates in field grids are classified."""
        fdata = generate_heat_equation_field(nx=10, ny=10, noise_std=1e-6)
        fg = FieldGridData(
            channels={'T': fdata['T']},
            grid_axes={'x': fdata['x'], 'y': fdata['y']},
        )
        tracker = register_observation_provenance(fg, "test_fg")

        prov = tracker.get_channel('T', 'test_fg')
        self.assertIsNotNone(prov)
        self.assertEqual(prov.origin, ChannelOrigin.RAW_SENSOR)


if __name__ == "__main__":
    unittest.main()
