"""Operational concepts are reusable measurements, not theory generators.

The law reducer may correctly prove that a ratio is algebraically derivable
while the measurement system still benefits from naming and reusing it.  This
module therefore maintains a small probationary vocabulary independently of
the active general-law basis.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import re

import sympy as sp
from sympy.core.function import AppliedUndef

from .interface import Concept
from .sr_metadata import quantity_metadata


def _atom_dsl(atom: AppliedUndef) -> tuple[str, str] | None:
    match = re.fullmatch(r"(.+)_([0-9]+)", atom.func.__name__)
    if match is None:
        return None
    base, object_id = match.groups()
    return base, f"{base}[{object_id}]"


def _unit_complexity(numerator: str, denominator: str) -> int:
    left = quantity_metadata(numerator).unit
    right = quantity_metadata(denominator).unit
    try:
        quotient = (
            (sp.sympify(left.replace("^", "**")) if left else sp.Integer(1))
            / (sp.sympify(right.replace("^", "**")) if right else sp.Integer(1))
        )
        return int(sum(abs(value) for value in quotient.as_powers_dict().values()))
    except (sp.SympifyError, TypeError, ValueError):
        return 10**6


def _ratio_candidates(raw_expression: sp.Expr) -> list[dict]:
    candidates = []
    # Preserve additive terms.  ``cancel`` combines the whole zero relation
    # over one denominator and hides the multiplicative factor structure we
    # want to name.
    for term in sp.Add.make_args(sp.expand(raw_expression)):
        powers = term.as_powers_dict()
        atoms = []
        for atom in sorted(term.atoms(AppliedUndef), key=sp.srepr):
            exponent = powers.get(atom, 0)
            if exponent not in (-1, 1):
                continue
            rendered = _atom_dsl(atom)
            if rendered is not None:
                atoms.append((*rendered, int(exponent)))
        for left, right in itertools.combinations(atoms, 2):
            if left[2] == right[2]:
                continue
            numerator, denominator = (
                (left, right) if left[2] > 0 else (right, left)
            )
            candidates.append({
                "definition": (
                    f"(1->Particle) |- ({numerator[1]} / {denominator[1]})"
                ),
                "numerator": numerator[0],
                "denominator": denominator[0],
                "unit_complexity": _unit_complexity(
                    numerator[0], denominator[0]
                ),
            })
    unique = {item["definition"]: item for item in candidates}
    return sorted(
        unique.values(),
        key=lambda item: (
            item["unit_complexity"],
            len(item["definition"]),
            item["definition"],
        ),
    )


def load_vocabulary(path: Path) -> dict:
    if not path.exists():
        return {"trial": {}, "promoted": {}, "rejected": {}}
    data = json.loads(path.read_text())
    data.setdefault("trial", {})
    data.setdefault("promoted", {})
    data.setdefault("rejected", {})
    return data


def save_vocabulary(path: Path, vocabulary: dict) -> None:
    path.write_text(json.dumps(vocabulary, indent=2, sort_keys=True))


def update_operational_vocabulary(
    theorist,
    vocabulary_path: Path,
    experiment: str,
    new_law_names: list[str],
    *,
    max_new_trials: int = 1,
) -> dict:
    vocabulary = load_vocabulary(vocabulary_path)
    changes = {"trial_added": [], "promoted": []}

    # Promotion evidence is semantic reuse: the named concept itself occurs
    # in an active or archived theorem after being offered to a later epoch.
    law_text = " ".join(
        str(law.prop) for law in theorist.general.general_laws.values()
    ) + " " + " ".join(
        record.get("prop", "")
        for record in theorist.general.service.derived_general_laws.values()
    )
    for name, record in list(vocabulary["trial"].items()):
        if name not in law_text:
            continue
        record = dict(record)
        record["status"] = "promoted"
        record.setdefault("evidence_experiments", []).append(experiment)
        record["evidence_experiments"] = sorted(
            set(record["evidence_experiments"])
        )
        vocabulary["promoted"][name] = record
        vocabulary["trial"].pop(name, None)
        changes["promoted"].append(name)

    proposals = []
    for law_name in new_law_names:
        law = theorist.general.general_laws.get(law_name)
        if law is None or not law.valid_experiments:
            continue
        expansion_experiment = law.valid_experiments[0]
        model = theorist.specific[expansion_experiment]
        for expression in theorist.knowledge.specialize(
            law.relevant_concept, expansion_experiment
        ) or []:
            raw = model._sympy_of_raw_defi(model.expand_exp(expression)).doit()
            for candidate in _ratio_candidates(raw):
                candidate["source_law"] = law_name
                candidate["source_experiment"] = experiment
                proposals.append(candidate)

    known_definitions = {
        record.get("definition")
        for group in ("trial", "promoted", "rejected")
        for record in vocabulary[group].values()
    }
    for candidate in proposals:
        if len(changes["trial_added"]) >= max_new_trials:
            break
        if candidate["definition"] in known_definitions:
            continue
        concept = Concept(candidate["definition"])
        before = set(theorist.knowledge.fetch_concepts)
        specific = theorist.specific[experiment]
        if not hasattr(specific, "experiment") or specific.experiment is None:
            specific.experiments_reset()
        name = theorist.register_concept_or_return_name(
            concept,
            specific,
            register_action=False,
        )
        if name is None:
            continue
        candidate["status"] = "trial"
        candidate["evidence_experiments"] = [experiment]
        vocabulary["trial"][name] = candidate
        known_definitions.add(candidate["definition"])
        if name not in before:
            changes["trial_added"].append(name)

    save_vocabulary(vocabulary_path, vocabulary)
    return changes
