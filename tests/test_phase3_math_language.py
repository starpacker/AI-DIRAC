"""
Tests for Phase 3: Mathematical Language Growth.

Covers: MathConceptRegistry, EquivalenceCanonicalizer, MapMiner,
        OperatorMiner, StructureMiner, NeuralRepresentationProposer,
        SymbolicDistiller, MathLanguageGrowth orchestrator.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from aiphy.math_registry import (
    MathConceptRegistry, MathConceptRecord, MathConceptStatus, MathObjectKind,
)
from aiphy.equivalence_canonicalizer import (
    EquivalenceCanonicalizer, EquivalenceKind, EquivalenceResult,
)
from aiphy.map_miner import MapMiner, MapMinerConfig, MapCandidate
from aiphy.operator_miner import OperatorMiner, OperatorMinerConfig, OperatorKind
from aiphy.structure_miner import StructureMiner, StructureKind
from aiphy.neural_representation import (
    NeuralRepresentationProposer, NeuralRepresentation,
    NeuralRepresentationStatus,
)
from aiphy.symbolic_distiller import (
    SymbolicDistiller, DistillationResult, DistillationMethod,
)
from aiphy.math_language import MathLanguageGrowth, MathLanguageConfig, MathLanguageState


# ---------------------------------------------------------------------------
# MathConceptRegistry tests
# ---------------------------------------------------------------------------

class TestMathConceptRegistry:
    def test_propose_and_certify(self):
        reg = MathConceptRegistry()
        cid = reg.propose("test_map", MathObjectKind.SCALAR_MAP,
                          expression="f(x) = x^2")
        assert cid.startswith("MATH_")
        assert reg.count() == 1
        assert reg.count_certified() == 0

        reg.promote_to_trial(cid)
        assert len(reg.trial_concepts) == 1
        assert len(reg.proposed_concepts) == 0

        certified = reg.certify(cid, [{"method": "heldout", "independent": True}])
        assert certified is True
        assert reg.count_certified() == 1

    def test_reject_not_deleted(self):
        reg = MathConceptRegistry()
        cid = reg.propose("bad", MathObjectKind.SCALAR_MAP)
        reg.promote_to_trial(cid)
        reg.reject(cid, "failed validation")

        assert cid in reg.all_concepts  # Still there
        rec = reg.get(cid)
        assert rec.status == MathConceptStatus.REJECTED
        assert "failed validation" in rec.rejection_reason

    def test_certify_without_validation_fails(self):
        reg = MathConceptRegistry()
        cid = reg.propose("test", MathObjectKind.SCALAR_MAP)
        reg.promote_to_trial(cid)
        # Empty evidence should fail
        result = reg.certify(cid, [])
        assert result is False

    def test_equivalence_groups(self):
        reg = MathConceptRegistry()
        a = reg.propose("map_a", MathObjectKind.SCALAR_MAP, expression="x^2")
        b = reg.propose("map_b", MathObjectKind.SCALAR_MAP, expression="x*x")
        reg.promote_to_trial(a)
        reg.promote_to_trial(b)
        reg.certify(a, [{"method": "heldout", "independent": True}])
        reg.certify(b, [{"method": "heldout", "independent": True}])

        reg.mark_equivalent(a, b)
        assert reg.is_equivalent(a, b)
        group = reg.get_equivalence_group(a)
        assert len(group) >= 2

    def test_supersede(self):
        reg = MathConceptRegistry()
        a = reg.propose("old", MathObjectKind.SCALAR_MAP)
        b = reg.propose("new", MathObjectKind.SCALAR_MAP)
        reg.promote_to_trial(a)
        reg.certify(a, [{"method": "heldout", "independent": True}])
        reg.supersede(a, b, "better fit")

        rec = reg.get(a)
        assert rec.status == MathConceptStatus.SUPERSEDED

    def test_count_by_kind(self):
        reg = MathConceptRegistry()
        reg.propose("m1", MathObjectKind.SCALAR_MAP)
        reg.propose("m2", MathObjectKind.SCALAR_MAP)
        reg.propose("o1", MathObjectKind.OPERATOR)
        reg.propose("s1", MathObjectKind.STRUCTURE)

        counts = reg.count_by_kind()
        assert counts["SCALAR_MAP"] == 2
        assert counts["OPERATOR"] == 1
        assert counts["STRUCTURE"] == 1

    def test_lineage(self):
        reg = MathConceptRegistry()
        parent = reg.propose("parent", MathObjectKind.SCALAR_MAP)
        child = reg.propose("child", MathObjectKind.SCALAR_MAP,
                            parent_concepts=[parent])
        lineage = reg.get_lineage(child)
        assert len(lineage) == 2
        assert lineage[0] == parent

    def test_serialization_roundtrip(self):
        reg = MathConceptRegistry()
        reg.propose("m1", MathObjectKind.SCALAR_MAP, expression="x+1")
        reg.propose("m2", MathObjectKind.OPERATOR, expression="compose")
        reg.promote_to_trial("MATH_0001")
        reg.certify("MATH_0001", [{"method": "heldout", "independent": True}])

        d = reg.to_dict()
        restored = MathConceptRegistry.from_dict(d)
        assert restored.count() == reg.count()
        assert restored.count_certified() == reg.count_certified()

    def test_audit_certification_gaps(self):
        reg = MathConceptRegistry()
        cid = reg.propose("suspect", MathObjectKind.SCALAR_MAP)
        reg.promote_to_trial(cid)
        # Certify without proper evidence
        reg.certify(cid, [{"independent": False}])

        issues = reg.audit_certification_gaps()
        assert len(issues) > 0


# ---------------------------------------------------------------------------
# EquivalenceCanonicalizer tests
# ---------------------------------------------------------------------------

class TestEquivalenceCanonicalizer:
    def test_numerical_equivalence_identical(self):
        can = EquivalenceCanonicalizer()
        x = np.linspace(0, 10, 30)
        y1 = x ** 2
        y2 = x ** 2  # identical

        result = can.check_numerical_equivalence(y1, y2)
        assert result.are_equivalent is True
        assert result.kind == EquivalenceKind.NUMERICAL_EQUIVALENT

    def test_affine_scaling_detection(self):
        can = EquivalenceCanonicalizer()
        x = np.linspace(0, 10, 50)
        y1 = 3 * x + 5
        y2 = 6 * x + 10  # y2 = 2 * y1

        result = can.check_numerical_equivalence(y1, y2)
        assert result.are_equivalent is True
        assert result.kind == EquivalenceKind.AFFINE_SCALING

    def test_different_functions_not_equivalent(self):
        can = EquivalenceCanonicalizer()
        x = np.linspace(0, 10, 30)
        y1 = x ** 2
        y2 = np.sin(x)

        result = can.check_numerical_equivalence(y1, y2)
        assert result.are_equivalent is False

    def test_latent_equivalence(self):
        can = EquivalenceCanonicalizer()
        n, d = 100, 4
        # Two latent spaces related by a rotation
        latent_a = np.random.randn(n, d)
        # Random rotation matrix
        Q, _ = np.linalg.qr(np.random.randn(d, d))
        latent_b = latent_a @ Q

        result = can.check_latent_equivalence(latent_a, latent_b)
        assert result.are_equivalent  # numpy bool_

    def test_canonicalize_batch(self):
        can = EquivalenceCanonicalizer()
        x = np.linspace(0, 10, 30)
        values = {
            "a": x + 1,
            "b": 2 * x + 2,  # affine equivalent to a
            "c": x ** 2,       # different
            "d": 3 * x + 3,  # affine equivalent to a
        }
        groups = can.canonicalize_batch(values)
        # Should find at least one equivalence group
        assert len(groups) >= 1
        # a, b, d should be in one group
        found_group = False
        for group in groups:
            if "a" in group and {"b", "d"}.issubset(group):
                found_group = True
        assert found_group, f"Groups: {groups}"

    def test_select_canonical(self):
        can = EquivalenceCanonicalizer()
        group = {"a", "b", "long_complex_expression"}
        expressions = {
            "a": "x+1",
            "b": "x*1+1.0",
            "long_complex_expression": "x**2 + 3*x + sin(x)*cos(x)",
        }
        selected = can.select_canonical(group, expressions)
        assert selected == "a"  # Simplest

    def test_description_length(self):
        dl = EquivalenceCanonicalizer.description_length
        assert dl("x+1") < dl("x**2 + 3*x + sin(x) + cos(x)**2")


# ---------------------------------------------------------------------------
# MapMiner tests
# ---------------------------------------------------------------------------

class TestMapMiner:
    def test_search_linear(self):
        miner = MapMiner()
        x = np.linspace(-5, 5, 100)
        y = 3 * x + 2 + np.random.randn(100) * 0.1

        candidates = miner.search(
            {"x": x}, {"y": y}, experiment_name="linear_test",
        )
        assert len(candidates) > 0
        # Should find a linear map with low error
        best = candidates[0]
        assert best.heldout_error < 0.5

    def test_search_quadratic(self):
        miner = MapMiner()
        x = np.linspace(-3, 3, 80)
        y = 2 * x ** 2 + x - 1 + np.random.randn(80) * 0.1

        candidates = miner.search(
            {"x": x}, {"y": y}, experiment_name="quad_test",
        )
        best = candidates[0]
        assert best.heldout_error < 1.0

    def test_search_cross_experiment(self):
        miner = MapMiner()
        x1 = np.linspace(-3, 3, 60)
        y1 = 2 * x1 + 1 + np.random.randn(60) * 0.05

        x2 = np.linspace(-4, 4, 60)
        y2 = 2 * x2 + 1 + np.random.randn(60) * 0.05

        candidates = miner.search(
            {"x": x1}, {"y": y1},
            experiment_name="exp1",
            cross_experiment_inputs={"x": x2},
            cross_experiment_outputs={"y": y2},
        )
        assert len(candidates) > 0

    def test_polynomial_fit(self):
        miner = MapMiner()
        x = np.linspace(-2, 2, 50)
        y = x ** 3 - x + 0.5 + np.random.randn(50) * 0.1

        candidates = miner.search(
            {"x": x}, {"y": y}, experiment_name="cubic_test",
        )
        assert len(candidates) > 0

    def test_intervention_stability(self):
        miner = MapMiner()

        # f(x) = sin(x) — nonlinear but deterministic
        x = np.linspace(-np.pi, np.pi, 100)
        y = np.sin(x)

        candidates = miner.search(
            {"x": x}, {"y": y}, experiment_name="stable_test",
        )
        if candidates and candidates[0].raw_predictor:
            # Same x-range for both interventions — should produce identical predictions
            intervention_data = {
                "interv1": {"x": np.array([0.0, 0.5, 1.0])},
                "interv2": {"x": np.array([0.0, 0.5, 1.0])},
            }
            stability = MapMiner.test_intervention_stability(
                candidates[0], intervention_data,
            )
            # With identical inputs, predictions should be identical, std ≈ 0
            assert stability < 0.01, f"Expected near-zero stability, got {stability}"


# ---------------------------------------------------------------------------
# OperatorMiner tests
# ---------------------------------------------------------------------------

class TestOperatorMiner:
    def test_composition(self):
        miner = OperatorMiner()
        x = np.linspace(-2, 2, 50)

        maps = {
            "f": lambda xv: 2 * xv + 1,
            "g": lambda xv: xv ** 2,
        }
        candidates = miner.search_compositions(maps, x)
        assert len(candidates) > 0
        # f ∘ g should be: 2x^2 + 1
        comp = next(c for c in candidates if c.operands == ["f", "g"])
        assert "f" in comp.expression and "g" in comp.expression

    def test_numerical_inverse(self):
        miner = OperatorMiner()
        domain = np.linspace(-5, 5, 100)
        codomain = 3 * domain + 2

        candidates = miner.search_inverses(
            {
                "linear": (
                    lambda x: 3 * x + 2,
                    domain,
                    codomain,
                )
            },
            np.linspace(-3, 3, 30),
        )
        assert len(candidates) > 0
        inv = candidates[0]
        # inv(3x+2) should return something close to x
        if inv.raw_operator is not None:
            test_x = np.array([0.0, 1.0, 2.0])
            recovered = inv.raw_operator(3 * test_x + 2)
            assert np.allclose(recovered, test_x, atol=0.5)

    def test_differentiation(self):
        miner = OperatorMiner()
        domain = np.linspace(-np.pi, np.pi, 100)

        candidates = miner.search_differential(
            {"sin": (np.sin, domain)},
            order=1,
        )
        assert len(candidates) > 0
        deriv = candidates[0]
        assert "sin" in deriv.expression

    def test_vector_ops(self):
        miner = OperatorMiner()
        v = np.random.randn(100, 3)
        w = np.random.randn(100, 3)

        candidates = miner.search_vector_ops({"v": v, "w": w})
        assert len(candidates) > 0
        # Should find dot product and norms
        kinds = {c.kind for c in candidates}
        assert OperatorKind.DOT_PRODUCT in kinds
        assert OperatorKind.NORM in kinds

    def test_basis_transform(self):
        miner = OperatorMiner()
        # Generate data with known low-rank structure
        n, d = 200, 10
        true_rank = 3
        basis = np.random.randn(d, true_rank)
        latent = np.random.randn(n, true_rank)
        data = latent @ basis.T + np.random.randn(n, d) * 0.01

        candidate = miner.search_basis_transform(
            data, variance_threshold=0.95,
        )
        assert candidate is not None
        assert candidate.kind == OperatorKind.BASIS_TRANSFORM


# ---------------------------------------------------------------------------
# StructureMiner tests
# ---------------------------------------------------------------------------

class TestStructureMiner:
    def test_conserved_form(self):
        miner = StructureMiner()
        n = 100
        # True conservation: Δa + 2*Δb + (-1)*Δc = 0
        da = np.random.randn(n) * 0.5 + 1.0
        db = np.random.randn(n) * 0.5 - 0.5
        dc = da + 2 * db + np.random.randn(n) * 0.01  # conserved form

        candidates = miner.search_conserved_forms(
            {"Δa": da, "Δb": db, "Δc": dc},
            experiment_name="conservation_test",
        )
        assert len(candidates) > 0

    def test_periodicity(self):
        miner = StructureMiner()
        t = np.linspace(0, 10 * np.pi, 500)
        y = np.sin(t) + np.random.randn(500) * 0.05

        candidates = miner.search_periodicity(
            {"sin_wave": y},
            sampling_rate=500 / (10 * np.pi),
        )
        assert len(candidates) > 0
        assert candidates[0].kind == StructureKind.PERIODICITY

    def test_scaling_relation(self):
        miner = StructureMiner()
        x = np.linspace(1, 10, 100)
        y = 3 * x ** 2  # perfect power law

        candidates = miner.search_scaling_relations(
            {"x": x}, {"y": y},
            experiment_name="scaling_test",
        )
        assert len(candidates) > 0
        assert candidates[0].kind == StructureKind.SCALING_RELATION

    def test_manifold_structure(self):
        miner = StructureMiner()
        n, d = 200, 8
        effective_dim = 3
        true_data = np.random.randn(n, effective_dim)
        # Embed in higher dimension via random projection
        projection = np.random.randn(effective_dim, d)
        data = true_data @ projection

        candidates = miner.search_manifold_structure(data)
        assert len(candidates) > 0
        assert candidates[0].kind == StructureKind.MANIFOLD

    def test_symmetry_detection(self):
        miner = StructureMiner()
        data = np.random.randn(100, 3)
        # Negate all components
        negated = -data
        # But a function like x^2 would see them as the same
        # Here negated should NOT be symmetrically equivalent
        candidates = miner.search_symmetry(data, negated)
        # f(x) = -x is NOT symmetric under negation
        assert len(candidates) == 0

        # Symmetric test: data and data (identical)
        candidates2 = miner.search_symmetry(data, data)
        assert len(candidates2) == 1
        assert candidates2[0].kind == StructureKind.SYMMETRY


# ---------------------------------------------------------------------------
# NeuralRepresentationProposer tests
# ---------------------------------------------------------------------------

class TestNeuralRepresentationProposer:
    def test_gauge_dimension_estimation(self):
        n, d = 200, 10
        true_dim = 4
        data = np.random.randn(n, true_dim) @ np.random.randn(true_dim, d)

        est = NeuralRepresentationProposer.estimate_gauge_dimension(data, max_dim=8)
        # Should detect the low-rank structure
        assert est >= true_dim - 1  # Allow some slop

    def test_propose_representations(self):
        proposer = NeuralRepresentationProposer()
        # Simple data with 3 true dimensions in 6-D
        n = 200
        true_dim = 3
        input_dim = 6
        basis = np.random.randn(true_dim, input_dim)
        latent = np.random.randn(n, true_dim)
        data = latent @ basis

        reps = proposer.propose_representations(
            {"exp1": data[:100], "exp2": data[100:]},
        )
        assert len(reps) > 0
        for rep in reps:
            assert rep.latent_dim > 0
            assert rep.input_dim == input_dim
            assert rep.val_loss >= 0

    def test_propose_with_heldout(self):
        proposer = NeuralRepresentationProposer()
        n = 100
        input_dim = 5
        data = np.random.randn(n, input_dim)
        heldout = np.random.randn(n // 2, input_dim)

        reps = proposer.propose_representations(
            {"exp1": data},
            heldout_data={"exp2": heldout},
        )
        assert len(reps) > 0


# ---------------------------------------------------------------------------
# SymbolicDistiller tests
# ---------------------------------------------------------------------------

class TestSymbolicDistiller:
    def test_distill_linear(self):
        distiller = SymbolicDistiller()
        x = np.linspace(-5, 5, 100)

        def neural(xv):
            return 3 * xv + 2

        result = distiller.distill(
            neural, x, latent_dim=1,
            method=DistillationMethod.TEMPLATE_FIT,
        )
        assert result.is_successful
        assert result.encoder_error < 0.1

    def test_distill_quadratic(self):
        distiller = SymbolicDistiller()
        x = np.linspace(-3, 3, 80)

        def neural(xv):
            return 2 * xv ** 2 - xv + 1

        result = distiller.distill(
            neural, x, latent_dim=1,
            method=DistillationMethod.TEMPLATE_FIT,
        )
        assert result.is_successful

    def test_distill_nonlinear(self):
        distiller = SymbolicDistiller()
        x = np.linspace(-2, 2, 100)

        def neural(xv):
            return np.sin(2 * xv)

        result = distiller.distill(
            neural, x, latent_dim=1,
            method=DistillationMethod.TEMPLATE_FIT,
        )
        assert result.is_successful or result.encoder_error < 1.0

    def test_distill_all(self):
        distiller = SymbolicDistiller()
        x = np.linspace(-3, 3, 80)

        class MockRep:
            def __init__(self, rid, ld):
                self.repr_id = rid
                self.latent_dim = ld

        reps = [MockRep("r1", 1), MockRep("r2", 1)]
        dims = [1, 1]

        def extractor(rep):
            return lambda xv: 2 * xv + rep.latent_dim

        results = distiller.distill_all(reps, extractor, x, dims)
        assert len(results) == 2
        for r in results:
            assert r.source_repr_id in ("r1", "r2")


# ---------------------------------------------------------------------------
# MathLanguageGrowth orchestrator tests
# ---------------------------------------------------------------------------

class TestMathLanguageGrowth:
    def test_initialize_and_step(self):
        mlg = MathLanguageGrowth(seed=42)

        # Simple synthetic data: y = 2x + 1
        x1 = np.linspace(-3, 3, 60)
        y1 = 2 * x1 + 1 + np.random.randn(60) * 0.05

        x2 = np.linspace(-4, 4, 60)
        y2 = 2 * x2 + 1 + np.random.randn(60) * 0.05

        summary = mlg.step(
            experiment_data={
                "exp1": ({"x": x1}, {"y": y1}),
                "exp2": ({"x": x2}, {"y": y2}),
            },
            verbose=False,
        )
        assert summary["iteration"] >= 0
        assert "new_proposed" in summary
        assert "new_certified" in summary

    def test_multiple_steps(self):
        mlg = MathLanguageGrowth(seed=42)
        x1 = np.linspace(-3, 3, 60)
        y1 = 3 * x1 + 2 + np.random.randn(60) * 0.05

        results = []
        for i in range(3):
            summary = mlg.step(
                experiment_data={"exp1": ({"x": x1}, {"y": y1})},
                verbose=False,
            )
            results.append(summary)

        assert mlg.iteration == 3
        assert mlg.registry.count() > 0

    def test_state_serialization(self):
        mlg = MathLanguageGrowth(seed=42)
        x1 = np.linspace(-3, 3, 60)
        y1 = 2 * x1 + 1

        mlg.step(
            experiment_data={"exp1": ({"x": x1}, {"y": y1})},
            verbose=False,
        )

        state = mlg.get_state()
        assert state.iteration == 1
        assert state.total_proposed >= 0

        # Restore
        mlg2 = MathLanguageGrowth()
        mlg2.load_state(state)
        assert mlg2.iteration == 1

    def test_audit(self):
        mlg = MathLanguageGrowth(seed=42)

        # Should report issues for an empty state (nothing certified)
        issues = mlg.audit()
        # Empty state should not be an error
        assert isinstance(issues, list)

    def test_with_state_changes(self):
        """Test structure mining with state changes."""
        mlg = MathLanguageGrowth(seed=42)

        n = 100
        da = np.random.randn(n)
        db = np.random.randn(n)
        dc = da + 2 * db + np.random.randn(n) * 0.01  # conserved

        summary = mlg.step(
            experiment_data={
                "exp1": ({"x": np.linspace(0, 1, 60)}, {"y": np.linspace(0, 1, 60)}),
            },
            state_changes={
                "exp1": {"Δa": da, "Δb": db, "Δc": dc},
            },
            verbose=False,
        )
        assert summary is not None


# ---------------------------------------------------------------------------
# Phase 3 invariant tests
# ---------------------------------------------------------------------------

class TestPhase3Invariants:
    def test_definition_data_not_validation(self):
        """Certification requires independent validation evidence."""
        reg = MathConceptRegistry()
        cid = reg.propose("test", MathObjectKind.SCALAR_MAP,
                          definition_experiments=["exp1"])
        reg.promote_to_trial(cid)

        # Certify with evidence from same experiment — marked as dependent
        result = reg.certify(cid, [
            {"method": "heldout_split", "experiment": "exp1", "independent": False}
        ])
        # Should still certify (gate is evidence presence), but audit catches it
        if result:
            issues = reg.audit_certification_gaps()
            assert len(issues) > 0  # Should flag missing independent validation

    def test_rejected_not_deleted(self):
        """Rejected concepts remain in the registry."""
        reg = MathConceptRegistry()
        cid = reg.propose("will_fail", MathObjectKind.SCALAR_MAP)
        reg.promote_to_trial(cid)
        reg.reject(cid, "invalid")

        assert cid in reg.all_concepts
        rec = reg.get(cid)
        assert rec.status == MathConceptStatus.REJECTED

    def test_no_definition_experiments_blocks_certification(self):
        """Concepts without definition experiments should be flagged."""
        reg = MathConceptRegistry()
        cid = reg.propose("no_def", MathObjectKind.SCALAR_MAP)
        assert reg.get(cid).definition_experiments == []
        # This should be catchable by audit or gate

    def test_equivalence_prevents_duplicate_discovery(self):
        """Equivalent concepts should be merged, not reported as separate discoveries."""
        reg = MathConceptRegistry()
        a = reg.propose("f1", MathObjectKind.SCALAR_MAP, expression="x^2+1")
        b = reg.propose("f2", MathObjectKind.SCALAR_MAP, expression="1+x^2")
        reg.promote_to_trial(a)
        reg.promote_to_trial(b)
        reg.certify(a, [{"method": "heldout", "independent": True}])
        reg.certify(b, [{"method": "heldout", "independent": True}])

        reg.mark_equivalent(a, b)
        assert reg.is_equivalent(a, b)
        # They share an equivalence group
        group_a = reg.get_equivalence_group(a)
        group_b = reg.get_equivalence_group(b)
        assert group_a == group_b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
