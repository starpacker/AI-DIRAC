"""
Dimensional analysis for AI-Newton: enforce SI unit correctness on concepts and laws.

All observable quantities carry SI base units.  Any derived concept or general law
must be dimensionally consistent.  Concepts/laws with inconsistent dimensions are
rejected before registration.

SI base dimensions:
  L - length (meter)
  T - time (second)  
  M - mass (kilogram)
  Θ - temperature (kelvin)
  I - electric current (ampere)
  N - amount of substance (mole)
  J - luminous intensity (candela) -- not used in our experiments

Derived dimensions for common observables:
  velocity:     L·T⁻¹
  acceleration: L·T⁻²
  force:        M·L·T⁻²
  energy:       M·L²·T⁻²
  pressure:     M·L⁻¹·T⁻²
  volume:       L³
  R (gas const): M·L²·T⁻²·Θ⁻¹·N⁻¹
  γ (heat ratio): dimensionless (1)
"""
from __future__ import annotations
import re
from typing import Dict, Tuple, Optional
from fractions import Fraction
from enum import Enum


class Dim(Enum):
    """SI base dimension exponents."""
    L = 0   # length
    T = 1   # time
    M = 2   # mass
    Θ = 3   # temperature
    I = 4   # current
    N = 5   # amount of substance

# Dimension vector is a tuple of 6 ints: (L, T, M, Θ, I, N)
# e.g. velocity = (1, -1, 0, 0, 0, 0)
DimVec = Tuple[int, int, int, int, int, int]

# ─── Observable dimension registry ───────────────────────────────────────────
# Maps observable concept names to their SI dimension vectors.
# "1" means dimensionless.

OBSERVABLE_DIMENSIONS: Dict[str, DimVec] = {
    # Position / length
    "posx":      (1, 0, 0, 0, 0, 0),
    "posy":      (1, 0, 0, 0, 0, 0),
    "posz":      (1, 0, 0, 0, 0, 0),
    "dist":      (1, 0, 0, 0, 0, 0),
    "length":    (1, 0, 0, 0, 0, 0),
    # Time
    "t":         (0, 1, 0, 0, 0, 0),
    # Mass
    "mass":      (0, 0, 1, 0, 0, 0),
    "samplemass":(0, 0, 1, 0, 0, 0),
    # Temperature
    "temp":      (0, 0, 0, 1, 0, 0),
    "temperature":(0, 0, 0, 1, 0, 0),
    # Amount of substance
    "amount":    (0, 0, 0, 0, 0, 1),
    # Pressure = force/area = M·L⁻¹·T⁻²
    "pressure":  (-1, -2, 1, 0, 0, 0),
    # Volume = L³
    "volume":    (3, 0, 0, 0, 0, 0),
    # Gas constant R = PV/(nT) = M·L²·T⁻²·Θ⁻¹·N⁻¹
    "gasconstant": (2, -2, 1, -1, 0, -1),
    # Adiabatic index γ = dimensionless
    "gamma":     (0, 0, 0, 0, 0, 0),
    # Molar mass M/n
    "molarmass": (0, 0, 1, 0, 0, -1),
    # Electric field  = force/charge = M·L·T⁻³·I⁻¹
    "E_field":   (1, -3, 1, 0, -1, 0),
    # Magnetic field = force/(current·length) = M·T⁻²·I⁻¹
    "B_field":   (0, -2, 1, 0, -1, 0),
    # Current
    "I":         (0, 0, 0, 0, 1, 0),
    # Slope components are dimensionless
    "cx":        (0, 0, 0, 0, 0, 0),
    "cy":        (0, 0, 0, 0, 0, 0),
    "cz":        (0, 0, 0, 0, 0, 0),
    # charge = I·T
    "charge":    (0, 1, 0, 0, 1, 0),
}

# Derived dimension formulas for quickly computing dims
_DIM_STR_MAP: Dict[str, str] = {
    "C_01": "length",  # spring rest length
}

def dim_to_str(dv: DimVec) -> str:
    """Convert dimension vector to string like 'M·L²·T⁻²'."""
    parts = []
    for name, exp, idx in [("L", dv[0], 0), ("T", dv[1], 1), ("M", dv[2], 2),
                            ("Θ", dv[3], 3), ("I", dv[4], 4), ("N", dv[5], 5)]:
        if exp == 0:
            continue
        if exp == 1:
            parts.append(name)
        elif exp == -1:
            parts.append(f"{name}⁻¹")
        elif exp > 0:
            parts.append(f"{name}^{exp}")
        else:
            parts.append(f"{name}^{exp}")
    return "·".join(parts) if parts else "1 (dimensionless)"


def get_observable_dim(name: str) -> Optional[DimVec]:
    """Get the SI dimension vector for an observable concept name.

    Returns None for unknown observables.
    """
    # Strip C_ prefix for derived concepts — we'll handle those separately
    return OBSERVABLE_DIMENSIONS.get(name)


# Sentinel for dimension-inconsistent expressions (not the same as unparseable)
_DIM_INCONSISTENT: DimVec = (-99999,) * 6  # type: ignore[assignment]

def parse_expression_dimensions(expr_str: str,
                                 known_dims: Dict[str, DimVec]) -> Optional[DimVec]:
    """Parse a simple algebraic expression and compute its dimensions.

    Handles: (a*b)/(c*d), a+b, a-b, a**n, D[a]/D[t], -1*a

    Args:
        expr_str: Expression string like "(pressure*volume)/(amount*temp)"
        known_dims: Dictionary mapping variable names to dimension vectors

    Returns:
        Dimension vector or None if uncomputable
    """
    expr = re.sub(r'\s+', '', expr_str.strip())  # Remove all whitespace

    # Handle number
    try:
        float(expr)
        return (0, 0, 0, 0, 0, 0)  # dimensionless
    except ValueError:
        pass

    # Handle simple variable name (alphanumeric + underscore)
    if re.match(r'^[A-Za-z_]\w*$', expr) or re.match(r'^C_\d+$', expr):
        return known_dims.get(expr)

    # Strip outermost matching parentheses
    # (do this repeatedly, and check all match types after each strip)
    changed = True
    while changed:
        changed = False
        # Strip one layer of matching parens
        if expr.startswith('(') and expr.endswith(')'):
            depth = 0
            matched = True
            for i, ch in enumerate(expr[:-1]):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    matched = False
                    break
            if matched and depth == 1:
                expr = expr[1:-1]
                changed = True
                continue

    # ── Pattern matches (after optional paren stripping) ──

    # Handle derivative: D[a]/D[b]
    deriv_match = re.match(r'^D\[([^]]+)\]/D\[([^]]+)\]$', expr)
    if deriv_match:
        num_dims = parse_expression_dimensions(deriv_match.group(1), known_dims)
        den_dims = parse_expression_dimensions(deriv_match.group(2), known_dims)
        if num_dims is None or den_dims is None:
            return None
        return tuple(num_dims[i] - den_dims[i] for i in range(6))

    # Handle power: base ** n
    pow_match = re.match(r'^(\w+)\*\*(-?\d+)$', expr)
    if pow_match:
        base_dims = known_dims.get(pow_match.group(1))
        if base_dims is None:
            return None
        n = int(pow_match.group(2))
        return tuple(d * n for d in base_dims)
    pow_match = re.match(r'^\((.+)\)\*\*(-?\d+)$', expr)
    if pow_match:
        base_dims = parse_expression_dimensions(pow_match.group(1), known_dims)
        if base_dims is None:
            return None
        n = int(pow_match.group(2))
        return tuple(d * n for d in base_dims)

    # Handle negative: (-1*expr)
    neg_match = re.match(r'^\(-1\*(.+)\)$', expr)
    if neg_match:
        return parse_expression_dimensions(neg_match.group(1), known_dims)

    # Handle simple variable name (re-check after stripping)
    if re.match(r'^[A-Za-z_]\w*$', expr) or re.match(r'^C_\d+$', expr):
        return known_dims.get(expr)

    # ── Try to split on top-level operator ──
    # Priority: +, -  (lowest), then /, then *

    # Find top-level + or - (addition/subtraction binds weakest)
    depth = 0
    for i, ch in enumerate(expr):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif depth == 0 and ch in ('+', '-'):
            # Make sure it's not a leading minus sign of a number
            if ch == '-' and (i == 0 or expr[i-1] in '(/*'):
                continue
            left = expr[:i]
            right = expr[i+1:]
            left_dims = parse_expression_dimensions(left, known_dims)
            right_dims = parse_expression_dimensions(right, known_dims)
            if left_dims is None or right_dims is None:
                return None
            if left_dims != right_dims:
                return _DIM_INCONSISTENT  # dimension mismatch in sum
            return left_dims

    # Find top-level / (not part of D[...]/D[...])
    depth = 0
    for i, ch in enumerate(expr):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '/' and depth == 0:
            # Skip if this is D[...]/D[...]
            if i >= 3 and expr[i-3:i] == ']D[':
                continue
            left = expr[:i]
            right = expr[i+1:]
            left_dims = parse_expression_dimensions(left, known_dims)
            right_dims = parse_expression_dimensions(right, known_dims)
            if left_dims is None or right_dims is None:
                return None
            return tuple(left_dims[d] - right_dims[d] for d in range(6))

    # Find top-level *
    depth = 0
    for i, ch in enumerate(expr):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '*' and depth == 0 and expr[i:i+2] != '**':
            left = expr[:i]
            right = expr[i+1:]
            left_dims = parse_expression_dimensions(left, known_dims)
            right_dims = parse_expression_dimensions(right, known_dims)
            if left_dims is None or right_dims is None:
                return None
            return tuple(left_dims[d] + right_dims[d] for d in range(6))

    return None


def get_concept_dimensions(concept_str: str,
                            known_concepts: Dict[str, DimVec]) -> Optional[DimVec]:
    """Get the dimensions of a concept from its string representation.
    
    A concept string looks like:
      "(1->Particle) |- ((pressure[1] * volume[1]) / (C_02[1] * volume[1]))"
    
    We extract the expression part after |- and compute its dimensions.
    """
    # Extract the expression after |-
    match = re.search(r'\|\-\s*(.+)$', concept_str)
    if not match:
        return None
    inner = match.group(1).strip()
    
    # Remove obj-type/id qualifiers: posx[1] -> posx
    # We need to preserve C_nn names and observable names
    cleaned = re.sub(r'\[(\d+(?:,\s*\d+)*)\]', '', inner)
    # Remove extra spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return parse_expression_dimensions(cleaned, known_concepts)


def check_concept_dimensions(concept_str: str,
                              known_dims: Dict[str, DimVec],
                              tgt_dims: Optional[DimVec] = None) -> Tuple[bool, Optional[DimVec], str]:
    """Check if a concept has valid dimensions.

    Returns:
        (is_valid, computed_dimensions, error_message)
    """
    dims = get_concept_dimensions(concept_str, known_dims)
    if dims is None:
        return (True, None, "Cannot compute dimensions — allowing through")

    if dims == _DIM_INCONSISTENT:
        return (False, None,
                f"Dimension mismatch: expression combines quantities with inconsistent dimensions")

    if tgt_dims is not None and dims != tgt_dims:
        return (False, dims,
                f"Dimension mismatch: expected {dim_to_str(tgt_dims)}, got {dim_to_str(dims)}")

    return (True, dims, dim_to_str(dims))


def build_known_dims_from_knowledge(knowledge) -> Dict[str, DimVec]:
    """Build a dimension map from the knowledge base, combining observable dims
    with already-discovered concept dims.
    
    Args:
        knowledge: The Knowledge object from aiphy.interface
    """
    known_dims = dict(OBSERVABLE_DIMENSIONS)
    
    # Try to infer dimensions of already discovered C_nn concepts
    for name, expr in knowledge.fetch_concepts.items():
        if name in known_dims:
            continue
        try:
            concept_str = str(expr)
            dims = get_concept_dimensions(concept_str, known_dims)
            if dims is not None:
                known_dims[name] = dims
        except Exception:
            pass
    
    return known_dims


