"""
Cross-arity conserved quantity combiner for AI-Newton — STRICT EDITION.

Combines observables and discovered concepts to find higher-order
conserved quantities via dimension-guided brute-force search.

Key patterns:
  2-way:  a*b, a/b         (energy, momentum, pressure)
  3-way:  (a*b)/c           (PV/T = nR, momentum/velocity)
  4-way:  (a*b)/(c*d)       (PV/(nT) = nR)
  Multi-body sums: Σ m_i*v_i² (kinetic energy), Σ m_i*v_i (momentum)

Dimension pruning ensures only physically meaningful patterns are tested.
"""

from __future__ import annotations
from typing import List, Tuple, Set, Dict, Optional
import itertools

from .core import Exp, ExpData, AtomExp, DataStruct
from .interface import Knowledge, ExpStructure, Proposition
from .specific_model import SpecificModel, ConservedInfo
from .dimensional_analysis import (
    get_observable_dim, parse_expression_dimensions,
    build_known_dims_from_knowledge, DimVec, dim_to_str,
)

# Physically meaningful target dimension vectors
TARGET_DIMS: Dict[str, DimVec] = {
    "energy":           (2, -2, 1, 0, 0, 0),    # M·L²·T⁻²
    "force":            (1, -2, 1, 0, 0, 0),    # M·L·T⁻²
    "momentum":         (1, -1, 1, 0, 0, 0),    # M·L·T⁻¹
    "pressure":         (-1, -2, 1, 0, 0, 0),   # M·L⁻¹·T⁻²
    "power":            (2, -3, 1, 0, 0, 0),    # M·L²·T⁻³
    "gas_constant":     (2, -2, 1, -1, 0, -1),  # M·L²·T⁻²·Θ⁻¹·N⁻¹ (=R)
    "velocity":         (1, -1, 0, 0, 0, 0),    # L·T⁻¹
    "acceleration":     (1, -2, 0, 0, 0, 0),    # L·T⁻²
    "specific_energy":  (2, -2, 0, 0, 0, 0),    # L²·T⁻² (energy/mass)
    "action":           (2, -1, 1, 0, 0, 0),    # M·L²·T⁻¹
    "frequency":        (0, -1, 0, 0, 0, 0),    # T⁻¹
    "spring_constant":  (0, -2, 1, 0, 0, 0),    # M·T⁻²
}


def try_combinations_on_raw_data(
    specific_model: SpecificModel,
    knowledge: Knowledge,
    exp_name: str,
    max_candidates: int = 8,
) -> List[Tuple[Exp, ExpData]]:
    """Combine observables + discovered concepts to find conserved quantities."""
    experiment = specific_model.experiment
    known_dims = build_known_dims_from_knowledge(knowledge)

    # Collect candidate atoms from three sources:
    # 1. Raw data keys (time-varying observables)
    # 2. Conserved parameters from specific_model conclusions (mass, etc.)
    # 3. Discovered concepts (C_nn)
    atoms: List[Tuple[str, AtomExp, Optional[DimVec], int]] = []  # (name, atom, dim, obj_id)

    # Source 1: Raw data keys
    ds = experiment.data_info
    for atom in sorted(ds.data_keys, key=str):
        name = atom.name
        if name == 't':
            continue
        dim = known_dims.get(name)
        obj_id = list(atom.vec_ids)[0] if atom.vec_ids else -1
        atoms.append((name, atom, dim, obj_id))

    # Source 2: Conserved parameters (mass, etc.)
    for cinfo_name, cinfo in sorted(specific_model.conclusions.conserved_list.items()):
        expr = cinfo.exp
        exp_atoms = list(expr.all_atoms)
        if len(exp_atoms) != 1:
            continue
        atom = exp_atoms[0]
        atom_name = atom.name
        if atom_name == 't' or atom_name.startswith('C_'):
            continue
        if any(a[0] == atom_name and a[3] == list(atom.vec_ids)[0] for a in atoms):
            continue
        dim = known_dims.get(atom_name)
        obj_id = list(atom.vec_ids)[0] if atom.vec_ids else -1
        atoms.append((atom_name, atom, dim, obj_id))

    # Source 3: Discovered concepts (e.g., C_19 = velocity)
    for cname, expr_obj in sorted(knowledge.fetch_concepts.items()):
        if not cname.startswith('C_'):
            continue
        dim = known_dims.get(cname)
        if dim is None:
            continue
        try:
            spec_atoms = knowledge.specialize_concept(cname, exp_name)
            for atom in spec_atoms:
                obj_id = list(atom.vec_ids)[0] if atom.vec_ids else -1
                atoms.append((cname, atom, dim, obj_id))
        except Exception:
            pass

    # Remove duplicates (same name + obj_id)
    seen = set()
    unique = []
    for a in atoms:
        key = (a[0], a[3])
        if key not in seen:
            seen.add(key)
            unique.append(a)
    atoms = unique

    if len(atoms) < 2:
        return []

    results: List[Tuple[Exp, ExpData]] = []
    tried: Set[str] = set()

    def test_expression(exp: Exp) -> bool:
        expr_str = str(exp)
        if expr_str in tried:
            return False
        tried.add(expr_str)
        try:
            data = knowledge.eval(exp, experiment)
            if not data.is_conserved or data.is_zero:
                return False
            if specific_model.test_on_test_experiment(exp, 'conserved'):
                results.append((exp, data))
                return True
        except Exception:
            pass
        return False

    def dims_match(result_dim: Optional[DimVec]) -> bool:
        if result_dim is None:
            return True
        return any(td == result_dim for td in TARGET_DIMS.values())

    n = min(len(atoms), 14)  # Cap to avoid combinatorial explosion

    # ══════════════════════════════════════════════════════════
    # 2-way: a * b
    # ══════════════════════════════════════════════════════════
    for i in range(n):
        if len(results) >= max_candidates:
            break
        for j in range(i + 1, n):
            if len(results) >= max_candidates:
                break
            _, atom_i, dim_i, oi = atoms[i]
            _, atom_j, dim_j, oj = atoms[j]

            if dim_i is not None and dim_j is not None:
                prod_dim = tuple(dim_i[d] + dim_j[d] for d in range(6))
                if not dims_match(prod_dim):
                    continue

            exp_i = Exp.Atom(atom_i)
            exp_j = Exp.Atom(atom_j)
            test_expression(exp_i * exp_j)  # e.g., mass * v² → energy dim

    # ══════════════════════════════════════════════════════════
    # 2-way: a / b
    # ══════════════════════════════════════════════════════════
    for i in range(n):
        if len(results) >= max_candidates:
            break
        for j in range(n):
            if len(results) >= max_candidates:
                break
            if i == j:
                continue
            _, atom_i, dim_i, oi = atoms[i]
            _, atom_j, dim_j, oj = atoms[j]

            if dim_i is not None and dim_j is not None:
                ratio_dim = tuple(dim_i[d] - dim_j[d] for d in range(6))
                if not dims_match(ratio_dim):
                    continue

            exp_i = Exp.Atom(atom_i)
            exp_j = Exp.Atom(atom_j)
            test_expression(exp_i / exp_j)  # e.g., P*V / T → nR dim

    # ══════════════════════════════════════════════════════════
    # 3-way: (a * b) / c  →  e.g., (P*V)/T = nR
    # ══════════════════════════════════════════════════════════
    for i in range(n):
        if len(results) >= max_candidates:
            break
        for j in range(n):
            if len(results) >= max_candidates:
                break
            if i == j:
                continue
            for k in range(n):
                if len(results) >= max_candidates:
                    break
                if k == i or k == j:
                    continue

                _, atom_i, dim_i, oi = atoms[i]
                _, atom_j, dim_j, oj = atoms[j]
                _, atom_k, dim_k, ok = atoms[k]

                if dim_i is not None and dim_j is not None and dim_k is not None:
                    result_dim = tuple(dim_i[d] + dim_j[d] - dim_k[d] for d in range(6))
                    if not dims_match(result_dim):
                        continue

                exp_i = Exp.Atom(atom_i)
                exp_j = Exp.Atom(atom_j)
                exp_k = Exp.Atom(atom_k)
                test_expression((exp_i * exp_j) / exp_k)

    # ══════════════════════════════════════════════════════════
    # 4-way: (a * b) / (c * d)  →  e.g., (P*V)/(n*T) = R
    # ══════════════════════════════════════════════════════════
    for i in range(n):
        if len(results) >= max_candidates:
            break
        for j in range(i + 1, n):
            if len(results) >= max_candidates:
                break
            for k in range(n):
                if len(results) >= max_candidates:
                    break
                if k == i or k == j:
                    continue
                for l in range(k + 1, n):
                    if len(results) >= max_candidates:
                        break
                    if l == i or l == j:
                        continue

                    _, atom_i, dim_i, oi = atoms[i]
                    _, atom_j, dim_j, oj = atoms[j]
                    _, atom_k, dim_k, ok = atoms[k]
                    _, atom_l, dim_l, ol = atoms[l]

                    if (dim_i is not None and dim_j is not None and
                            dim_k is not None and dim_l is not None):
                        num_dim = tuple(dim_i[d] + dim_j[d] for d in range(6))
                        den_dim = tuple(dim_k[d] + dim_l[d] for d in range(6))
                        result_dim = tuple(num_dim[d] - den_dim[d] for d in range(6))
                        if not dims_match(result_dim):
                            continue

                    exp_i = Exp.Atom(atom_i)
                    exp_j = Exp.Atom(atom_j)
                    exp_k = Exp.Atom(atom_k)
                    exp_l = Exp.Atom(atom_l)
                    test_expression((exp_i * exp_j) / (exp_k * exp_l))

    # ══════════════════════════════════════════════════════════
    # Multi-body sums: Σ m_i * v_i²  and  Σ m_i * v_i
    # For experiments with multiple objects of the same type.
    # ══════════════════════════════════════════════════════════
    if len(atoms) >= 4:
        # Group atoms by name pattern across different object IDs
        name_groups: Dict[str, List[Tuple[str, AtomExp, Optional[DimVec], int]]] = {}
        for a in atoms:
            base_name = a[0]
            # Strip numeric suffixes from concept names to group v[1], v[2], etc
            name_groups.setdefault(base_name, []).append(a)

        # Find groups that have multiple objects
        multi_obj_groups = {k: v for k, v in name_groups.items() if len(v) >= 2}

        # Momentum sum: Σ mass_i * v_i  for each object
        mass_atoms = [a for a in atoms if a[0] == 'mass']
        velocity_groups = [v for k, v in multi_obj_groups.items()
                          if k.startswith('C_') and v[0][2] == TARGET_DIMS.get('velocity')]

        if mass_atoms and velocity_groups:
            for vel_group in velocity_groups:
                if len(results) >= max_candidates:
                    break
                # Match masses to velocities by object ID
                paired = []
                for va in vel_group:
                    for ma in mass_atoms:
                        if ma[3] == va[3]:
                            paired.append((ma, va))
                            break
                if len(paired) >= 2:
                    # Σ m_i * v_i
                    sum_expr = None
                    for (ma, va) in paired:
                        term = Exp.Atom(ma[1]) * Exp.Atom(va[1])
                        sum_expr = term if sum_expr is None else sum_expr + term
                    if sum_expr is not None:
                        test_expression(sum_expr)

                # Σ m_i * v_i² (kinetic energy)
                if len(paired) >= 2:
                    ke_sum = None
                    for (ma, va) in paired:
                        term = Exp.Atom(ma[1]) * Exp.Atom(va[1]) * Exp.Atom(va[1])
                        ke_sum = term if ke_sum is None else ke_sum + term
                    if ke_sum is not None:
                        test_expression(ke_sum)

    return results


def discover_higher_order_constants(
    theorist,
    exp_name: str,
    max_candidates: int = 8,
) -> List[Tuple[Exp, ExpData]]:
    """Entry point from Theorist.doit_again."""
    spm = theorist.specific[exp_name]
    knowledge = theorist.knowledge
    return try_combinations_on_raw_data(
        spm, knowledge, exp_name, max_candidates=max_candidates,
    )


