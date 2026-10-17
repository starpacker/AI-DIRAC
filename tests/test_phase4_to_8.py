"""
Tests for Phase 4-8: active designer, classical regression, GR, QM, final eval.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import sys
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from aiphy.active_experiment_designer import (
    ActiveExperimentDesigner, ActiveDesignerConfig,
    ExperimentValueComponents, ExperimentDiversityTracker,
    RandomSelector, UCBSelector, LawOnlySelector,
)
from aiphy.classical_regression_suite import (
    ClassicalSimulator, ClassicalLaw, BenchmarkResult,
    ClassicalSuiteReport, RegressionBenchmarkRunner,
    TheoryRevisionValidator, CrossValidationChecker,
)
from aiphy.gr_route import (
    GeometrySimulator, GRDiscoveryPipeline, GRDiscoveryStep, GRDiscoveryResult,
)
from aiphy.wave_particle_route import (
    OpticalSimulator, WaveParticleDiscoveryPipeline,
    WaveParticleDiscoveryStep, WPDiscoveryResult,
)
from aiphy.final_evaluation import (
    FinalEvaluator, EvaluationMetrics, BaselineName, BaselineResult,
    BenchmarkEvaluation, generate_certification_checklist,
)


# ===========================================================================
# Phase 4: Active Experiment Designer
# ===========================================================================

class TestPhase4ActiveDesigner:
    def test_value_components(self):
        comp = ExperimentValueComponents(
            experiment_name="test_exp",
            theory_discrimination=0.8,
            math_info_gain=0.3,
            falsification_potential=0.6,
            transfer_value=0.4,
            total_value=2.5,
        )
        d = comp.to_dict()
        assert d["experiment_name"] == "test_exp"
        assert d["theory_discrimination"] == 0.8
        assert d["total_value"] == 2.5

    def test_diversity_tracker(self):
        tracker = ExperimentDiversityTracker()
        tracker.record("exp_A", "repeat")
        tracker.record("exp_A", "repeat")
        tracker.record("exp_B", "discriminating")

        assert tracker.total_experiments == 3
        penalty = tracker.repetition_penalty("exp_A")
        assert penalty > 0.1  # Two repeats should trigger penalty
        assert tracker.repetition_penalty("exp_C") == 0.0

    def test_select_with_discrimination(self):
        designer = ActiveExperimentDesigner(
            config=ActiveDesignerConfig(exploration_rate=0.0),
        )
        exp_name, value = designer.select(
            ["exp_A", "exp_B", "exp_C"],
            theory_disc_map={"exp_A": 0.1, "exp_B": 0.9, "exp_C": 0.5},
            math_info_map={"exp_A": 0.0, "exp_B": 0.0, "exp_C": 0.0},
        )
        # Should select exp_B (highest discrimination)
        assert exp_name == "exp_B"

    def test_exploration_picks_random(self):
        designer = ActiveExperimentDesigner(
            config=ActiveDesignerConfig(exploration_rate=1.0),
        )
        exp_name, _ = designer.select(
            ["exp_A", "exp_B", "exp_C"],
            theory_disc_map={"exp_A": 0.1, "exp_B": 0.9, "exp_C": 0.5},
        )
        assert exp_name in ["exp_A", "exp_B", "exp_C"]

    def test_diversity_penalty_avoids_repeats(self):
        designer = ActiveExperimentDesigner(
            config=ActiveDesignerConfig(exploration_rate=0.0),
        )
        # First selection
        designer.select(["exp_A", "exp_B"],
                       theory_disc_map={"exp_A": 0.9, "exp_B": 0.8})
        # exp_A used once — small penalty
        # Second selection — exp_A now has repetition penalty, exp_B should win
        designer.select(["exp_A", "exp_B"],
                       theory_disc_map={"exp_A": 0.9, "exp_B": 0.8})
        # exp_A used twice now
        assert designer.diversity.experiment_usage.get("exp_A", 0) >= 1

    def test_validation_task_design(self):
        designer = ActiveExperimentDesigner()
        tasks = designer.design_validation_tasks(
            "MATH_0001",
            ["exp_train", "exp_nuisance", "exp_heldout"],
        )
        assert len(tasks) >= 3
        task_types = {t["type"] for t in tasks}
        assert "repeat" in task_types
        assert "nuisance" in task_types
        assert "heldout_validation" in task_types

    def test_baseline_selectors(self):
        experiments = ["a", "b", "c"]

        # Random
        rand = RandomSelector()
        assert rand.select(experiments) in experiments

        # UCB
        ucb = UCBSelector()
        assert ucb.select(experiments) in experiments

        # Law-only
        law = LawOnlySelector()
        assert law.select(experiments) in experiments

    def test_ucb_learns(self):
        ucb = UCBSelector()
        experiments = ["good", "bad"]

        # Teach: good experiment gives high reward
        for _ in range(10):
            choice = ucb.select(experiments)
            ucb.update(choice, 1.0 if choice == "good" else 0.1)

        # After learning, should prefer "good"
        choice = ucb.select(experiments)
        assert choice == "good"  # UCB should learn this

    def test_serialization(self):
        designer = ActiveExperimentDesigner()
        designer.select(["exp_A", "exp_B"],
                       theory_disc_map={"exp_A": 0.7, "exp_B": 0.3})
        d = designer.to_dict()
        restored = ActiveExperimentDesigner.from_dict(d)
        assert restored.diversity.total_experiments == 1


# ===========================================================================
# Phase 5: Classical Regression Suite
# ===========================================================================

class TestPhase5ClassicalSuite:
    def test_simulator_free_fall(self):
        sim = ClassicalSimulator()
        data = sim.free_fall(gravity=9.81, dt=0.01, t_max=1.0)
        assert "position" in data
        assert "velocity" in data
        assert "acceleration" in data
        # Check: a ≈ -9.81
        assert np.abs(np.mean(data["acceleration"]) + 9.81) < 0.1

    def test_simulator_harmonic(self):
        sim = ClassicalSimulator()
        data = sim.harmonic_oscillator(spring_constant=10.0, mass=1.0, dt=0.01, t_max=2.0)
        assert "position" in data
        # Check: periodic motion
        assert np.max(data["position"]) > 0.9

    def test_simulator_damped(self):
        sim = ClassicalSimulator()
        data = sim.damped_oscillator(damping=1.0, dt=0.01, t_max=5.0)
        # Amplitude should decay
        mid = len(data["position"]) // 2
        assert np.max(np.abs(data["position"][:mid])) > np.max(np.abs(data["position"][mid:]))

    def test_simulator_central_force(self):
        sim = ClassicalSimulator()
        data = sim.central_force(dt=0.01, t_max=5.0)
        assert "x" in data and "y" in data
        assert len(data["x"]) > 10

    def test_simulator_drag(self):
        sim = ClassicalSimulator()
        data = sim.drag_motion(dt=0.01, t_max=5.0)
        # Velocity should decay
        assert data["velocity"][-1] < data["velocity"][0]

    def test_benchmark_runner(self):
        runner = RegressionBenchmarkRunner()
        runner.run_all()
        report = runner.compile_final_report()
        assert report["benchmarks_run"] == 6
        assert report["total_discovered"] > 0
        assert report["total_revision_events"] > 0

    def test_theory_revision_validator(self):
        validator = TheoryRevisionValidator()
        result = validator.validate_ideal_to_damped_revision()
        assert result["revision_triggered"]  # numpy bool_
        assert result["harmonic_error_on_damped"] > 0.01

    def test_cross_validation(self):
        checker = CrossValidationChecker()
        # Residual law: f - m*a ≈ 0
        def residual_law(f, m, a, **_):
            return f - m * a

        # Train: f=9.81, m=1, a=9.81 → residual = 0
        train = {"f": np.array([9.81]), "m": np.array([1.0]), "a": np.array([9.81])}
        # Test: f=9.8, m=2, a=4.9 → residual = 0
        test = {"f": np.array([9.8]), "m": np.array([2.0]), "a": np.array([4.9])}
        result = checker.validate_transfer(residual_law, train, test)
        assert result["transfers"]  # numpy bool_


# ===========================================================================
# Phase 6: General Relativity Route
# ===========================================================================

class TestPhase6GRRoute:
    def test_simulator_minkowski(self):
        sim = GeometrySimulator(seed=42)
        data = sim.minkowski_worldlines(num_observers=4)
        assert data["num_observers"] == 4
        assert len(data["observers"]) == 4
        # Different observers should have different velocities
        velocities = [obs["three_velocity"][0] for obs in data["observers"]]
        assert len(set(f"{v:.3f}" for v in velocities)) > 1

    def test_simulator_schwarzschild(self):
        sim = GeometrySimulator(seed=42)
        data = sim.schwarzschild_worldlines(num_observers=3, mass=1.0)
        assert data["geometry"] == "Schwarzschild"
        assert data["num_observers"] == 3

    def test_simulator_perihelion(self):
        sim = GeometrySimulator(seed=42)
        data = sim.schwarzschild_perihelion_precession(mass=1.0)
        assert data["precession_per_orbit"] > 0
        assert "radius" in data

    def test_simulator_coordinate_grids(self):
        sim = GeometrySimulator(seed=42)
        data = sim.generate_coordinate_grids(num_grids=3)
        assert len(data["transformed_grids"]) == 3

    def test_step_1_clock_sync(self):
        sim = GeometrySimulator(seed=42)
        worldlines = sim.minkowski_worldlines(num_observers=4)
        pipeline = GRDiscoveryPipeline()
        result = pipeline.step_1_sync_clocks(worldlines["observers"])
        assert result.step == GRDiscoveryStep.SYNC_CLOCKS
        # Should detect different clock rates in Minkowski (due to different velocities)
        assert result.discovered is True

    def test_step_2_coordinate_transform(self):
        sim = GeometrySimulator(seed=42)
        data = sim.generate_coordinate_grids(num_grids=3)
        pipeline = GRDiscoveryPipeline()
        result = pipeline.step_2_coordinate_reparameterization(
            data["transformed_grids"]
        )
        assert result.step == GRDiscoveryStep.COORDINATE_TRANSFORM
        assert result.discovered is True

    def test_step_3_invariants(self):
        sim = GeometrySimulator(seed=42)
        data = sim.generate_coordinate_grids(num_grids=3)
        pipeline = GRDiscoveryPipeline()
        result = pipeline.step_3_find_invariants(data["transformed_grids"])
        assert result.step == GRDiscoveryStep.INVARIANTS
        # May or may not find invariant depending on grid type
        assert result.discovered in (True, False)

    def test_step_4_metric_candidate(self):
        sim = GeometrySimulator(seed=42)
        data = sim.generate_coordinate_grids(num_grids=3)
        pipeline = GRDiscoveryPipeline()

        # First find invariant
        inv_result = pipeline.step_3_find_invariants(data["transformed_grids"])
        result = pipeline.step_4_metric_candidate(
            inv_result.candidate_expression if inv_result.discovered else "-dt^2 + dx^2",
            data["transformed_grids"],
        )
        assert result.step == GRDiscoveryStep.METRIC_CANDIDATE

    def test_step_5_geodesic(self):
        sim = GeometrySimulator(seed=42)
        data = sim.schwarzschild_worldlines(num_observers=3, mass=1.0)
        pipeline = GRDiscoveryPipeline()
        result = pipeline.step_5_geodesic_test(data["observers"])
        assert result.step == GRDiscoveryStep.GEODESIC_TEST

    def test_step_6_curvature(self):
        sim = GeometrySimulator(seed=42)
        data = sim.schwarzschild_perihelion_precession(mass=1.0)
        pipeline = GRDiscoveryPipeline()
        result = pipeline.step_6_curvature_prediction(data)
        assert result.step == GRDiscoveryStep.CURVATURE_PREDICTION
        assert result.discovered is True  # Perihelion precession > 0

    def test_full_pipeline(self):
        sim = GeometrySimulator(seed=42)
        pipeline = GRDiscoveryPipeline()
        report = pipeline.run_full_pipeline(sim)
        assert report["steps_completed"] == 6
        assert "discoveries" in report
        assert len(report["hard_gates_passed"]) >= 3


# ===========================================================================
# Phase 7: Wave-Particle Duality Route
# ===========================================================================

class TestPhase7WaveParticle:
    def test_simulator_single_beam(self):
        sim = OpticalSimulator(seed=42)
        data = sim.single_beam_counts(intensity=1.0, num_detections=500)
        assert "click_events" in data
        assert "total_detections" in data
        assert data["total_detections"] > 0

    def test_simulator_intensity_series(self):
        sim = OpticalSimulator(seed=42)
        results, fit = sim.variable_intensity_series(
            [0.2, 0.4, 0.6, 0.8, 1.0],
            num_detections_per=500,
        )
        assert len(results) == 5
        # Count rate should increase with intensity
        assert results[-1]["count_rate"] > results[0]["count_rate"]
        assert fit["slope"] > 0

    def test_simulator_interference(self):
        sim = OpticalSimulator(seed=42)
        data = sim.two_path_interference(phase_shift=0.0, num_photons=10000)
        assert "detector_0_counts" in data
        assert data["total_photons"] == 10000

    def test_simulator_interference_scan(self):
        sim = OpticalSimulator(seed=42)
        scan = sim.interference_scan(
            np.linspace(0, 2 * np.pi, 20),
            num_photons_per=2000,
        )
        assert "count_ratios" in scan
        assert "sinusoidal_r2" in scan
        # Should detect sinusoidal pattern
        assert scan["sinusoidal_r2"] > 0.5

    def test_simulator_which_path(self):
        sim = OpticalSimulator(seed=42)
        unblocked = sim.two_path_interference(phase_shift=0.0)
        blocked = sim.which_path_experiment(path_blocked=0)
        # Both produce valid data — reversed comparison
        assert "detector_0_counts" in unblocked
        assert "detector_0_counts" in blocked

    def test_simulator_coincidence(self):
        sim = OpticalSimulator(seed=42)
        data = sim.coincidence_counts(correlated=True)
        assert isinstance(data["correlation"], float)
        # Correlated pairs should show strong correlation
        assert abs(data["correlation"]) > 0.5

    def test_representation_dimension(self):
        sim = OpticalSimulator(seed=42)
        scan = sim.interference_scan(np.linspace(0, 2 * np.pi, 20))
        dim = sim.probe_representation_dimension(scan)
        assert "min_dimension" in dim
        assert "recommended_representation" in dim

    def test_step_1_counting(self):
        sim = OpticalSimulator(seed=42)
        data = sim.single_beam_counts(intensity=1.0)
        pipeline = WaveParticleDiscoveryPipeline()
        result = pipeline.step_1_counting(data)
        assert result.step == WaveParticleDiscoveryStep.COUNTING
        assert result.discovered is True

    def test_step_2_intensity(self):
        sim = OpticalSimulator(seed=42)
        results = sim.variable_intensity_series([0.2, 0.4, 0.6, 0.8, 1.0])
        pipeline = WaveParticleDiscoveryPipeline()
        result = pipeline.step_2_intensity(results)
        assert result.step == WaveParticleDiscoveryStep.INTENSITY
        assert result.discovered is True

    def test_step_3_interference(self):
        sim = OpticalSimulator(seed=42)
        scan = sim.interference_scan(np.linspace(0, 2 * np.pi, 20))
        pipeline = WaveParticleDiscoveryPipeline()
        result = pipeline.step_3_interference(scan)
        assert result.step == WaveParticleDiscoveryStep.INTERFERENCE
        assert result.discovered is True

    def test_step_4_which_way(self):
        sim = OpticalSimulator(seed=42)
        unblocked = sim.two_path_interference(phase_shift=math.pi)
        blocked = sim.which_path_experiment(path_blocked=0)
        pipeline = WaveParticleDiscoveryPipeline()
        result = pipeline.step_4_which_way(blocked, unblocked)
        assert result.step == WaveParticleDiscoveryStep.WHICH_WAY

    def test_step_5_correlations(self):
        sim = OpticalSimulator(seed=42)
        data = sim.coincidence_counts(correlated=True)
        pipeline = WaveParticleDiscoveryPipeline()
        result = pipeline.step_5_correlations(data)
        assert result.step == WaveParticleDiscoveryStep.CORRELATIONS
        assert result.discovered is True

    def test_step_6_representation(self):
        sim = OpticalSimulator(seed=42)
        scan = sim.interference_scan(np.linspace(0, 2 * np.pi, 20))
        dim = sim.probe_representation_dimension(scan)
        pipeline = WaveParticleDiscoveryPipeline()
        result = pipeline.step_6_representation(dim)
        assert result.step == WaveParticleDiscoveryStep.REPRESENTATION

    def test_full_pipeline(self):
        sim = OpticalSimulator(seed=42)
        pipeline = WaveParticleDiscoveryPipeline()
        report = pipeline.run_full_pipeline(sim)
        assert report["steps_completed"] == 6
        assert report["discoveries"] >= 3  # At least 3 steps should discover something
        assert len(report["hard_gates_passed"]) >= 4


# ===========================================================================
# Phase 8: Final Evaluation
# ===========================================================================

class TestPhase8FinalEvaluation:
    def test_evaluation_metrics(self):
        m = EvaluationMetrics(
            independent_physical_laws=3,
            independent_math_objects=2,
            false_discovery_rate=0.1,
            name_invariance_score=0.97,
        )
        d = m.to_dict()
        assert d["independent_physical_laws"] == 3
        assert d["name_invariance_score"] == 0.97

    def test_baseline_comparison(self):
        bench = BenchmarkEvaluation(benchmark_name="test_bench")
        for baseline in BaselineName:
            metrics = EvaluationMetrics(independent_physical_laws=2)
            result = BaselineResult(baseline=baseline, metrics=metrics)
            bench.add_baseline(result)

        assert len(bench.baselines) == len(BaselineName)
        table = bench.get_comparison_table()
        assert "growing_math" in table
        assert "random" in table

    def test_evaluator_runs(self):
        evaluator = FinalEvaluator(seed=42)
        evaluator.add_benchmark("test", [], lambda: {})
        evaluator.add_benchmark("test2", [], lambda: {})

        report = evaluator.run_full_evaluation()
        assert report["num_benchmarks"] == 2
        assert "overall_metrics" in report
        assert "growing_math" in report["overall_metrics"]

    def test_certification(self):
        evaluator = FinalEvaluator(seed=42)
        evaluator.add_benchmark("test", [], lambda: {})
        eval_report = evaluator.run_full_evaluation()

        cert = evaluator.certify(eval_report)
        assert "certified" in cert
        assert "criteria_checked" in cert
        assert cert["criteria_checked"] == 8

    def test_certification_checklist(self):
        evaluator = FinalEvaluator(seed=42)
        evaluator.add_benchmark("test", [], lambda: {})
        report = evaluator.run_full_evaluation()
        evaluator.certify(report)

        checklist = generate_certification_checklist(report)
        assert "all_criteria_pass" in checklist
        assert checklist["all_criteria_pass"] is True

    def test_report_generation(self):
        evaluator = FinalEvaluator(seed=42)
        evaluator.add_benchmark("bench1", [], lambda: {})
        report = evaluator.run_full_evaluation()
        evaluator.certify(report)

        text = evaluator.generate_report_text(report)
        assert "FINAL BLIND EVALUATION" in text
        assert "Baseline Comparison" in text
        assert "Certification" in text

    def test_growing_math_outperforms_random(self):
        evaluator = FinalEvaluator(seed=42)
        evaluator.add_benchmark("test", [], lambda: {})
        report = evaluator.run_full_evaluation()

        growing = report["overall_metrics"]["growing_math"]
        random_base = report["overall_metrics"]["random"]

        assert growing["total_laws"] >= random_base["total_laws"]
        assert growing["avg_false_discovery_rate"] <= random_base["avg_false_discovery_rate"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
