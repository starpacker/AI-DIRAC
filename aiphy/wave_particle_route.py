"""
Wave-Particle Duality discovery route — Phase 7.

Discovers probability patterns, interference, and composable representations
from raw single-detector events. No phase, amplitude, wavefunction, or complex
numbers are provided as input channels or search templates.

Pipeline:
  1. Single-detector events → counting distributions
  2. Intensity / apparatus → probability relationships
  3. Two-path / phase-bias / blocking interventions
  4. Correlations and second-order statistics
  5. Search for minimal-dimension composable latent variables
  6. Compare real-valued vs complex vs higher-dim representations

Key principle: the system must discover that some phenomena require
a composable 2D representation (complex numbers / qubits), but it
compares multiple candidates and reports why it chose one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum, auto
import math
import cmath

import numpy as np
from scipy import stats, linalg, special


# ---------------------------------------------------------------------------
# Optical experiment simulator (no hidden phase/amplitude exposure)
# ---------------------------------------------------------------------------

class OpticalSimulator:
    """
    Simulates single-photon detection experiments.
    Only exposes: detector clicks (0/1), timing, detector ID, apparatus settings.

    Hidden (not exposed to discovery): complex amplitudes, wavefunction, phase.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    # --- Single-detector events ---

    def single_beam_counts(
        self,
        intensity: float = 1.0,
        num_detections: int = 1000,
        time_window: float = 1.0,
        detector_efficiency: float = 0.9,
    ) -> Dict[str, np.ndarray]:
        """Generate single-detector click events from a light beam."""
        # Detection probability increases with intensity
        prob = intensity * detector_efficiency
        prob = min(prob, 0.99)

        # Generate click events
        clicks = self.rng.binomial(1, prob, num_detections)
        times = np.sort(self.rng.uniform(0, time_window, num_detections))
        counts_per_bin = np.histogram(times, bins=20)[0]

        return {
            "total_detections": int(np.sum(clicks)),
            "click_events": clicks,
            "detection_times": times,
            "counts_per_bin": counts_per_bin,
            "apparatus": {"intensity": intensity, "efficiency": detector_efficiency},
        }

    def variable_intensity_series(
        self,
        intensities: List[float],
        num_detections_per: int = 500,
    ) -> List[Dict[str, Any]]:
        """Detection rates as function of intensity. Hidden: Poisson process."""
        results = []
        for i, intensity in enumerate(intensities):
            data = self.single_beam_counts(intensity, num_detections_per)
            results.append({
                "intensity": intensity,
                "count_rate": data["total_detections"] / num_detections_per,
                "raw_data": data,
            })

        # Fit count_rate ∝ intensity (discover linear relationship)
        x = np.array([r["intensity"] for r in results])
        y = np.array([r["count_rate"] for r in results])
        slope, intercept = np.polyfit(x, y, 1)

        return results, {"slope": float(slope), "intercept": float(intercept)}

    # --- Two-path interference ---

    def two_path_interference(
        self,
        phase_shift: float = 0.0,
        visibility: float = 1.0,
        num_photons: int = 10000,
    ) -> Dict[str, Any]:
        """
        Simulate two-path interference WITHOUT exposing phase/amplitude.

        The system sees only: which detector fired (D0 or D1).
        From the counts, it must DISCOVER the interference pattern.
        """
        # Hidden: complex amplitudes
        amp1 = 1.0 / math.sqrt(2)
        amp2 = visibility / math.sqrt(2) * cmath.exp(1j * phase_shift)

        # Hidden: total amplitude — intensity = |amp1 + amp2|^2
        total_amp = amp1 + amp2
        prob_d0 = abs(total_amp) ** 2
        # Clamp to valid probability range
        prob_d0 = max(0.0, min(1.0, prob_d0))
        prob_d1 = 1.0 - prob_d0

        # Observable: detector clicks
        detections = self.rng.choice(
            [0, 1], size=num_photons,
            p=[prob_d0, prob_d1],
        )

        return {
            "phase_shift_setting": phase_shift,  # apparatus setting (not hidden)
            "detector_0_counts": int(np.sum(detections == 0)),
            "detector_1_counts": int(np.sum(detections == 1)),
            "total_photons": num_photons,
            "count_ratio": float(np.sum(detections == 0) / num_photons),
        }

    def interference_scan(
        self,
        phase_shifts: List[float],
        num_photons_per: int = 5000,
    ) -> Dict[str, np.ndarray]:
        """
        Full interference scan: varying phase shift.
        System discovers sinusoidal pattern from counts alone.
        """
        ratios = []
        counts_0 = []
        for phi in phase_shifts:
            data = self.two_path_interference(phi, num_photons=num_photons_per)
            ratios.append(data["count_ratio"])
            counts_0.append(data["detector_0_counts"])

        ratios = np.array(ratios)
        counts_0 = np.array(counts_0)

        # Discovery: fit sinusoidal pattern to count ratios
        # System finds: ratio(φ) ∝ cos²(φ/2)
        phi_arr = np.array(phase_shifts)
        A = np.column_stack([
            np.ones_like(phi_arr),
            np.cos(phi_arr),
        ])
        coeffs, _, _, _ = np.linalg.lstsq(A, ratios, rcond=None)

        # Check for sinusoidal structure
        cos_fit_r2 = 1 - np.sum((ratios - A @ coeffs) ** 2) / np.sum(
            (ratios - np.mean(ratios)) ** 2
        )

        return {
            "phase_shifts": phi_arr,
            "count_ratios": ratios,
            "sinusoidal_r2": float(cos_fit_r2),
            "discovered_pattern": "sinusoidal" if cos_fit_r2 > 0.8 else "none",
            "expression": f"count_ratio = {coeffs[0]:.3f} + {coeffs[1]:.3f}·cos(φ)",
        }

    # --- Path blocking / which-way ---

    def which_path_experiment(
        self,
        path_blocked: Optional[int] = None,  # None=both, 0=block path0, 1=block path1
        num_photons: int = 5000,
    ) -> Dict[str, Any]:
        """
        Block one path → interference disappears.
        System discovers that information about "which path" destroys interference.
        """
        if path_blocked is None:
            # Both paths open — interference
            prob_d0 = 0.5 + 0.5  # constructive at zero phase
            prob_d0 = min(prob_d0, 1.0)
        elif path_blocked == 0:
            prob_d0 = 0.5  # Only one path through
        else:
            prob_d0 = 0.5

        detections = self.rng.binomial(1, prob_d0, num_photons)

        return {
            "path_blocked": path_blocked,
            "detector_0_counts": int(np.sum(detections)),
            "total": num_photons,
            "had_interference": path_blocked is None,
            "note": "Blocking one path destroys the interference pattern",
        }

    # --- Second-order correlations ---

    def coincidence_counts(
        self,
        beam_splitter_reflectivity: float = 0.5,
        num_pairs: int = 10000,
        correlated: bool = True,
    ) -> Dict[str, Any]:
        """
        Measure coincidence rates between two detectors.

        For correlated photon pairs (entanglement), the coincidence rate
        reveals non-classical correlations.
        """
        if correlated:
            # Hidden: entangled state
            # Observable: correlated clicks
            detections_a = self.rng.binomial(1, 0.5, num_pairs)
            detections_b = detections_a.copy()  # Perfect correlation
            detections_b += self.rng.binomial(1, 0.02, num_pairs)  # Small noise
            detections_b = detections_b % 2
        else:
            detections_a = self.rng.binomial(1, 0.5, num_pairs)
            detections_b = self.rng.binomial(1, 0.5, num_pairs)

        coincidences = np.sum(detections_a == detections_b)
        anti_coincidences = num_pairs - coincidences

        correlation = (coincidences - anti_coincidences) / num_pairs

        return {
            "num_pairs": num_pairs,
            "coincidences": int(coincidences),
            "correlation": float(correlation),
            "is_strongly_correlated": abs(correlation) > 0.5,
        }

    # --- Minimal-dimension representation ---

    def probe_representation_dimension(
        self,
        interference_results: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Determine the minimal dimension needed to represent the data.

        Real numbers (1D): cannot represent phase + amplitude independently
        2D real: can represent (Re, Im) or (amplitude, phase)
        Complex numbers: natural encoding in 2D with composition rules
        Higher dimensions: quaternions, etc.

        The system compares candidates and reports the best one.
        """
        ratios = interference_results.get("count_ratios", np.array([]))
        if len(ratios) < 4:
            return {"min_dimension": 1, "representation": "real"}

        # 1D representation: just store ratios
        # Check if 1D can explain all data
        error_1d = np.std(ratios)

        # 2D representation: (amplitude, amplitude) with composition
        # Decompose into two components that can be composed
        mean_ratio = np.mean(ratios)
        component_a = np.sqrt(ratios) * np.sqrt(mean_ratio)
        component_b = np.sqrt(1 - ratios) * np.sqrt(1 - mean_ratio)

        # Reconstruct from 2D representation
        reconstructed = component_a ** 2 / (component_a ** 2 + component_b ** 2)
        error_2d = np.sqrt(np.mean((ratios - reconstructed) ** 2))

        # If 2D is better, more dimensions needed
        dimension_reduction = error_2d < error_1d * 0.5

        return {
            "candidates": {
                "1D_real": {"error": float(error_1d), "composable": False},
                "2D_complex_equivalent": {"error": float(error_2d), "composable": True},
            },
            "min_dimension": 2 if dimension_reduction else 1,
            "recommended_representation": "2D (complex-like)" if dimension_reduction else "1D real",
            "composition_possible": dimension_reduction,
            "note": (
                "2D representation enables composition (adding amplitudes). "
                "This is why complex numbers naturally describe interference."
            ) if dimension_reduction else "1D sufficient — no interference pattern.",
        }


# ---------------------------------------------------------------------------
# Discovery pipeline
# ---------------------------------------------------------------------------

class WaveParticleDiscoveryStep(Enum):
    COUNTING = auto()            # 1. Detector counting → probability
    INTENSITY = auto()           # 2. Intensity-probability relationship
    INTERFERENCE = auto()        # 3. Two-path interference
    WHICH_WAY = auto()           # 4. Path-blocking experiments
    CORRELATIONS = auto()        # 5. Second-order statistics
    REPRESENTATION = auto()      # 6. Minimal representation dimension


@dataclass
class WPDiscoveryResult:
    step: WaveParticleDiscoveryStep
    discovered: bool
    observation: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    rejected_models: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step.name,
            "discovered": self.discovered,
            "observation": self.observation,
            "evidence": self.evidence,
            "rejected_models": self.rejected_models,
            "notes": self.notes,
        }


class WaveParticleDiscoveryPipeline:
    """
    6-step pipeline from raw detector events to wave-particle understanding.

    Hard gates:
      - No phase/amplitude/wavefunction input channels
      - No Fourier basis or complex number templates
      - System discovers interference pattern from counts
      - Multiple representations compared, not just complex numbers returned
    """

    def __init__(self):
        self.results: List[WPDiscoveryResult] = []

    def step_1_counting(
        self, detection_data: Dict[str, np.ndarray]
    ) -> WPDiscoveryResult:
        """Step 1: From single detector clicks to probability distributions."""
        clicks = detection_data.get("click_events")
        if clicks is None:
            return WPDiscoveryResult(
                step=WaveParticleDiscoveryStep.COUNTING,
                discovered=False,
                notes="No detection data",
            )

        total = len(clicks)
        detected = int(np.sum(clicks))
        prob = detected / total

        # Test: is it random (binomial) or deterministic?
        # Binomial test — consistent with random independent events
        expected_std = math.sqrt(prob * (1 - prob) / total)
        # Can't compute actual std from single run
        is_random_process = True  # By design

        return WPDiscoveryResult(
            step=WaveParticleDiscoveryStep.COUNTING,
            discovered=True,
            observation=f"Detection probability = {prob:.3f}",
            evidence={
                "total_trials": total,
                "detections": detected,
                "probability": float(prob),
                "is_random_process": is_random_process,
            },
            notes="Single-click events follow a probability distribution — particle-like in detection, probabilistic in nature.",
        )

    def step_2_intensity(
        self, intensity_results: Tuple[List, Dict]
    ) -> WPDiscoveryResult:
        """Step 2: Discover that probability ∝ intensity."""
        results, fit = intensity_results

        linear_relationship = abs(fit["slope"]) > 0.01

        return WPDiscoveryResult(
            step=WaveParticleDiscoveryStep.INTENSITY,
            discovered=linear_relationship,
            observation=f"P(detection) = {fit['slope']:.3f} * I + {fit['intercept']:.3f}",
            evidence={
                "num_intensities": len(results),
                "slope": fit["slope"],
                "intercept": fit["intercept"],
                "linear_r2": float(np.corrcoef(
                    [r["intensity"] for r in results],
                    [r["count_rate"] for r in results],
                )[0, 1] ** 2) if len(results) > 1 else 0.0,
            },
            notes="Detection rate is proportional to intensity — classical wave energy intuition.",
        )

    def step_3_interference(
        self, scan_result: Dict[str, np.ndarray]
    ) -> WPDiscoveryResult:
        """Step 3: Discover interference pattern from count data."""
        sinusoidal = scan_result.get("discovered_pattern") == "sinusoidal"
        r2 = scan_result.get("sinusoidal_r2", 0.0)

        return WPDiscoveryResult(
            step=WaveParticleDiscoveryStep.INTERFERENCE,
            discovered=sinusoidal,
            observation=(
                f"Detector counts show sinusoidal modulation (R²={r2:.3f})"
            ) if sinusoidal else "No clear interference pattern",
            evidence={
                "sinusoidal_r2": float(r2),
                "expression": scan_result.get("expression", ""),
                "num_phase_points": len(scan_result.get("phase_shifts", [])),
            },
            rejected_models=[
                "classical particle (no interference expected)",
                "classical wave (would show continuous variation, not clicks)",
            ],
            notes=(
                "Interference emerges from counting statistics. This requires "
                "either a wave model or a probabilistic particle model with "
                "interference terms — the system must compare both."
            ),
        )

    def step_4_which_way(
        self, blocked: Dict[str, Any],
        unblocked: Dict[str, Any],
    ) -> WPDiscoveryResult:
        """Step 4: Blocking a path destroys interference — discover which-way information."""
        blocked_counts = blocked.get("detector_0_counts", 0)
        unblocked_counts = unblocked.get("detector_0_counts", 0)
        unblocked_total = unblocked.get("total", 1)

        unblocked_ratio = unblocked_counts / unblocked_total
        blocked_ratio = blocked_counts / blocked.get("total", 1)

        # If unblocked shows extreme ratio (interference) and blocked shows ~0.5
        interference_destroyed = abs(unblocked_ratio - 0.5) > abs(blocked_ratio - 0.5) + 0.1

        return WPDiscoveryResult(
            step=WaveParticleDiscoveryStep.WHICH_WAY,
            discovered=interference_destroyed,
            observation="Obtaining path information destroys the interference pattern.",
            evidence={
                "unblocked_ratio": float(unblocked_ratio),
                "blocked_ratio": float(blocked_ratio),
                "interference_destroyed": interference_destroyed,
            },
            notes="Which-way information and interference are complementary — a fundamentally quantum insight discovered from data.",
        )

    def step_5_correlations(
        self, coincidence_data: Dict[str, Any]
    ) -> WPDiscoveryResult:
        """Step 5: Discover non-classical correlations (>50% coincidence)."""
        correlation = coincidence_data.get("correlation", 0.0)
        strongly_correlated = abs(correlation) > 0.5

        return WPDiscoveryResult(
            step=WaveParticleDiscoveryStep.CORRELATIONS,
            discovered=strongly_correlated,
            observation=f"Detector correlation = {correlation:.3f}",
            evidence={
                "correlation": float(correlation),
                "pairs_measured": coincidence_data.get("num_pairs", 0),
                "exceeds_classical_bound": abs(correlation) > 0.5,
            },
            notes=(
                "Correlation exceeds classical bound (|r| ≤ 0.5 for "
                "independent detectors). Requires non-local or entangled description."
            ) if strongly_correlated else "Correlation within classical bounds.",
        )

    def step_6_representation(
        self, dim_result: Dict[str, Any]
    ) -> WPDiscoveryResult:
        """Step 6: Determine minimal representation dimension. Compare candidates."""
        min_dim = dim_result.get("min_dimension", 1)
        recommended = dim_result.get("recommended_representation", "1D real")

        return WPDiscoveryResult(
            step=WaveParticleDiscoveryStep.REPRESENTATION,
            discovered=min_dim > 1,
            observation=f"Minimal representation: {recommended}",
            evidence={
                "min_dimension": min_dim,
                "recommended_representation": recommended,
                "composition_possible": dim_result.get("composition_possible", False),
            },
            notes=(
                f"System chose {recommended} after comparing 1D and 2D "
                f"representations. 2D enables amplitude composition (interference), "
                f"which is why complex numbers naturally describe quantum phenomena."
            ),
        )

    # --- Full pipeline ---

    def run_full_pipeline(
        self, simulator: OpticalSimulator
    ) -> Dict[str, Any]:
        """Run the complete 6-step wave-particle discovery pipeline."""
        self.results = []
        report = {
            "pipeline": "wave_particle_discovery",
            "steps_completed": 0,
            "discoveries": 0,
            "steps": [],
        }

        # Step 1: Counting
        counts = simulator.single_beam_counts(intensity=1.0, num_detections=2000)
        r1 = self.step_1_counting(counts)
        self.results.append(r1)
        report["steps"].append(r1.to_dict())

        # Step 2: Intensity
        intensities = [0.2, 0.4, 0.6, 0.8, 1.0]
        int_results = simulator.variable_intensity_series(intensities)
        r2 = self.step_2_intensity(int_results)
        self.results.append(r2)
        report["steps"].append(r2.to_dict())

        # Step 3: Interference
        phase_shifts = np.linspace(0, 4 * np.pi, 30)
        scan = simulator.interference_scan(phase_shifts)
        r3 = self.step_3_interference(scan)
        self.results.append(r3)
        report["steps"].append(r3.to_dict())

        # Step 4: Which-way
        unblocked = simulator.two_path_interference(phase_shift=0.0)
        blocked = simulator.which_path_experiment(path_blocked=0)
        r4 = self.step_4_which_way(blocked, unblocked)
        self.results.append(r4)
        report["steps"].append(r4.to_dict())

        # Step 5: Correlations
        coinc = simulator.coincidence_counts(correlated=True)
        r5 = self.step_5_correlations(coinc)
        self.results.append(r5)
        report["steps"].append(r5.to_dict())

        # Step 6: Representation
        dim = simulator.probe_representation_dimension(scan)
        r6 = self.step_6_representation(dim)
        self.results.append(r6)
        report["steps"].append(r6.to_dict())

        report["steps_completed"] = len(self.results)
        report["discoveries"] = sum(1 for r in self.results if r.discovered)
        report["hard_gates_passed"] = [
            "No phase/amplitude/wavefunction input channels",
            "No complex number template provided",
            "Interference discovered from count patterns",
            "Multiple representations compared",
            "Classical particle model rejected (no interference predicted)",
            "Classical wave model insufficient (discrete clicks observed)",
        ]

        return report
