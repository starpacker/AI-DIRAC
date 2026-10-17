# AI-DIRAC

**A concept-driven physical-law discovery system that climbs a designed experiment ladder — step by step, from raw sensor channels toward the hardest closed-form laws of physics.**

AI-DIRAC is a strict derivation of [AI-Newton](https://github.com/Science-Discovery/AI-Newton). Where a general symbolic-regression agent is asked to "find the formula", AI-DIRAC is asked a harder and more honest question:

> *Can a machine discover physics the way physics was actually discovered* — by running the right experiment at the right time, unlocking one concept after another, and only then assembling the complex equations that those concepts make expressible?

The answer is organized as a **ladder**: a carefully designed curriculum of experiments. Each rung certifies one new *operational concept*; each newly certified concept unlocks the data columns that the next rungs need. Nothing is pre-assembled, nothing is leaked, and no target formula is ever handed to the framework. The system starts with only the seven SI base units plus raw sensor channels, and walks upward.

---

## The idea in one picture

```
L0  raw sensors:       position, time, distance, length, geometry ...        (always open)
        │
L1  mass, velocity     two-body collision → null-space of Σ wᵢ·Δvᵢ = 0
        │
L2  acceleration       free fall / incline / Atwood
        │
L3  force, k           f = m·a under a controlled actuator; Hooke's law
        │
L4  g, GM, orbit       drop-repeats, two-body orbits, Kepler III
        │
L5  V, ρ, P, μ, Q      displacement, density, pressure, Stokes, Poiseuille
        │
L6  E, I/(mr²), γ      energy conservation, rolling, damped oscillator
        │
L7  ln, tanh, C_d      drag, terminal velocity, resonance   ← special functions
        │
   ┌────┴────────────────────────────┐
L5T thermodynamics            L5O optics & quantum
 T_abs, n, R, γ_gas, S        c, λ, ν, h, blackbody
        │                             │
        └──────────────┬──────────────┘
                       ▼
L8  TARGET LAWS   Kepler III · Stokes · Poiseuille · rolling ·
                  linear-drag projectile · v = v_t·tanh(gt/v_t) ·
                  forced oscillator · relativistic rocket · Hertz contact ·
                  elliptic pendulum · van der Waals · Maxwell–Boltzmann ·
                  Curzon–Ahlborn · transient heat (erf) · Planck blackbody ·
                  Einstein/Debye solid heat capacity · Sackur–Tetrode ·
                  Taylor–Sedov blast · Kapitza stability
```

The full ladder, with per-rung data columns, expected discoveries, newly certified concepts, and implementation status, is specified in [`docs/experiment_ladder.md`](docs/experiment_ladder.md). The target formulas the ladder is designed to reach are cataloged in the L8 table below and in `final_experiments_judge.md`.

---

## Why a ladder, and not a big pool?

Naive symbolic regression on a soup of variables "discovers" laws by being told which variables matter. That is not discovery; it is fitting. AI-DIRAC instead enforces three iron rules (from `docs/experiment_ladder.md`):

1. **"Unlocked" has a strict meaning.** A quantity is only available to later experiments after it has been *discovered and certified* in an earlier one. Directly measurable geometry/kinematic channels (position, time, distance, length, incline geometry) are *default open* — they are not "unlocked".
2. **No cheating.** Variables may never be pre-combined into a target-named column. Only *mathematical* priors are allowed.
3. **Definition ≠ validation, and failure is never deleted.** A quantity's defining data cannot serve as independent evidence for a law that is algebraically identical to its definition. Falsified laws are archived, not erased.

The runner implements this by **dynamic unlocking**: experiments *declare* candidate `UNLOCKED` columns, but those columns are exposed only when the framework's own `operational_registry` has actually `CERTIFIED` the corresponding concept. The experiment order controls open/hidden state, and a fast-fail audit rejects any un-certified column that tries to enter a data contract.

---

## How the framework discovers

Each epoch the loop:

1. **Selects an experiment** with a theory-aware contextual bandit (CPU-only LinUCB) that scores an experiment against both the current theory and the experiment descriptor — not just by an arm-wise UCB.
2. **Runs per-experiment discovery**: numerical derivatives (`dx/dt = v`, `dv/dt = a`), conserved-quantity extraction (PCA), PySR symbolic proposals, and relation search.
3. **Combines across arities**: a dimension-guided combiner searches products/quotients/sums of already-certified concepts and multi-body sums (`Σ mᵢvᵢ`, `Σ mᵢvᵢ²`), pruning candidates by SI dimension before testing.
4. **Rejects by dimension**: any concept or law that fails SI dimensional consistency is discarded.
5. **Promotes to general law**: a law must be validated across multiple experiments before it becomes a `GeneralLaw`.
6. **Persists and rewards**: the knowledge base checkpoints, and the bandit learns.

Discovery always starts from an **empty `Knowledge`** and only registers the strict ladder experiments; legacy built-in experiments are asserted out of both the knowledge base and the candidate pool.

### Anti-cheating mechanisms

| Mechanism | Effect |
|---|---|
| `original_concept` auto-registration | Only observables declared in `data_info` become concepts |
| SI dimensional analysis | Rejects dimension-inconsistent concepts and laws |
| Cross-experiment validation | A law must generalize before promotion |
| Cross-arity dimensional pruning | Only dimension-matching combinations are tested |
| Dynamic unlock audit | Uncertified columns cannot enter a data contract |
| Strict-only assertion | `strict_` prefix enforced on every registered experiment |

A 600-epoch run over 74 experiments is analyzed in [`REPORT.md`](REPORT.md), including a point-by-point cheat audit (what was genuinely discovered vs. what was trivially true by construction, and what the framework honestly failed to find).

---

## What has been discovered so far

From the strict run (see `REPORT.md` for full evidence):

- **Kinematics** `v = dx/dt`, `a = dv/dt`, and `a_z = g` constant.
- **Dynamics** `F = m·g`, and `ax = ay = 0` in the constrained 1-D apparatuses.
- **Conservation** `Σ mᵢvᵢ = const` (momentum) and `Σ mᵢvᵢ² = const` (kinetic energy), assembled by the multi-body sum combiner and correctly *not* claimed in inelastic collisions.
- **Mechanical energy** `v² − 2gz = const`, built by cross-arity combination from raw observables under dimensional pruning.

Honest limitations are recorded too: e.g. `PV = nRT` was not found in the first run because `P`, `V`, `T` never co-occur in a single data stream — which is exactly why the ladder introduces dedicated multi-variable state-scan experiments.

---

## Repository layout

```
aiphy/                     Python discovery framework
  main.py                  Theorist: discovery loop, register/perform_discovery
  manager.py               knowledge base wiring
  contextual_bandit.py     LinUCB experiment selector + reward config
  cross_arity_combiner.py  dimension-guided cross-concept/multi-body combiner
  dimensional_analysis.py  SI dimensional checks
  operational_concepts.py  operational-concept registry (CERTIFIED state machine)
  pysr_regression.py       PySR proposal backend (validated, rationalized)
  theory_revision.py       deterministic theory-repair loop
  experiment/
    ladder_strict.py       40 strict ladder experiments
    column_policy.py       ColumnSpec / DEFAULT|CONTROL|UNLOCKED|HIDDEN policy
src/                       Rust DSL (parser, expr, knowledge, regression) via PyO3
run_ladder_strict.py       strict ladder runner with --levels and dynamic unlock
run_classical_revision_suite.py
audit_theory_basis.py      non-mutating checkpoint audit / migration
docs/
  experiment_ladder.md     THE LADDER: rungs, columns, target formulas
  operational_concept_discovery.md   low-prior objective, allowed/forbidden prior
test/, tests/              unit, integration and ladder tests
REPORT.md                  600-epoch run analysis + cheat audit
```

---

## Install & run

**Prerequisites:** Linux, Python ≥ 3.11, Rust 1.84, CUDA (for the legacy torch paths).

```bash
conda create -n ai-dirac python=3.11
conda activate ai-dirac
pip install -r requirements.txt

# build the Rust core
cargo build --release
cp target/release/libcore.so aiphy/core.so
```

Validate the ladder (no discovery — smoke-collect every rung and check column contracts):

```bash
python run_ladder_strict.py --levels L1,L2 --skip-register
```

Register the ladder and start discovery from an empty knowledge base:

```bash
python run_ladder_strict.py --levels L1,L2,L3 \
    --discover-epochs 100 \
    --save-path data/my_run/checkpoint
```

Key flags:

- `--levels L1,L2,...` — run a subset of the ladder in order (also `L5T`, `L5O`, `L6.5`, `L7.5`).
- `--trust-declared-certifications` — scaffold/schema-only mode that unlocks columns from the ladder *declarations*. **Not** valid as a strict discovery result.
- `--discover-epochs N` — epochs of autonomous discovery (requires registration).

Run the test suites:

```bash
python -m pytest test/test_ladder_strict.py test/test_ladder_levels.py -q
python run_classical_revision_suite.py          # genuine-search theory-revision suite
```

Audit an existing checkpoint without mutating it:

```bash
python audit_theory_basis.py data/my_run/checkpoint
```

---

## Target formulas (L8)

Reaching these — each requiring *all* of its variables to be legitimately unlocked along the ladder, with no pre-combined target columns — is the point of the whole exercise.

| # | Law | Expression | Unlocks needed |
|---|---|---|---|
| 1 | Kepler III | `T² = 4π²a³ / (G(M+m))` | L1, L4 |
| 2 | Stokes terminal velocity | `v_t = 2(ρ_p−ρ_f)g r² / (9μ)` | L4, L5 |
| 3 | Poiseuille | `Q = π ΔP r⁴ / (8μL)` | L5 |
| 4 | Rolling on incline | `a = g sinθ / (1 + I/(mr²))` | L4, L6 |
| 5 | Linear-drag projectile | `y = (tanθ + g/(k v₀cosθ))x + (g/k²)ln(1 − kx/(v₀cosθ))` | L7 |
| 6 | Quadratic drag fall | `v = v_t·tanh(gt/v_t)` | L7 |
| 7 | Forced oscillator | `A, φ` resonance | L6, L7 |
| 8 | Relativistic rocket | `Δv = c·tanh((v_e/c)ln(m₀/m₁))` | L7.5 |
| 9 | Hertz contact | `F = (4/3)E*√R · δ^{3/2}` | L6.5 |
| 10 | Large-angle pendulum | `T = 4√(L/g)·K(sin(θ₀/2))` | L6 |
| 11 | van der Waals | `(P + a n²/V²)(V − nb) = nRT` | L5, L5T |
| 12 | Maxwell–Boltzmann | `f(v) = 4π(m/2πkT)^{3/2} v² exp(−mv²/2kT)` | L1, L5T |
| 13 | Curzon–Ahlborn | `η = 1 − √(T_c/T_h)` | L5T |
| 14 | Transient conduction | `T(x,t) = T_s + (T₀−T_s) erf(x/2√(αt))` | L5T, L7 |
| 15 | Planck blackbody | `B_λ(T) = (2hc²/λ⁵)/(e^{hc/λkT}−1)` | L5T, L5O |
| 16 | Einstein/Debye heat capacity | `C_V(T)` | L5T, L5O |
| 17 | Sackur–Tetrode entropy | `S = Nk[ln(...) + 5/2]` | L1, L5T, L5O |
| 18 | Taylor–Sedov blast | `R(t) = C (E t²/ρ)^{1/5}` | L5, L6 |
| 19 | Kapitza stability | `a²Ω² > 2gL` (inequality / phase boundary) | L6, L7 |

Each is annotated with *why* it matters and *which capability* it probes in `final_experiments_judge.md`. They were chosen to span distinct structural classes: power laws, exponentials, special functions, integrals, implicit equations, conservation laws, and finally whole theories derived from data (Kepler → inverse-square → vis-viva; Planck → Wien → Stefan–Boltzmann; van der Waals → critical point → corresponding states).

---

## Design documents

- [`docs/experiment_ladder.md`](docs/experiment_ladder.md) — the canonical curriculum: every rung with its data columns, expected discovery, certified concepts, and implementation status (1.1 → O.6, 40 strict experiments).
- [`docs/operational_concept_discovery.md`](docs/operational_concept_discovery.md) — the low-prior objective: what the designer may and may not supply, and the operational-concept definition (protocol + uncertainty + applicability domain + transfer evidence).
- `paradigm_shift_design.md` — concept falsification and repair: the five-step detect → localize → repair → validate → propagate cycle and the O1–O7 repair operators. (Companion to the ladder.)

---

## Status and scope

AI-DIRAC is a **research prototype**. It demonstrates that, under strict no-cheat constraints, a concept-driven loop can re-derive core Newtonian conservation laws from simulated apparatus and can be *scaffolded* toward much harder laws by a designed unlock order. It does **not** yet claim autonomous discovery of the full L8 set: several rungs (special-function operators, integral/erf templates, cross-experiment state scans) are the current frontier, and the ladder marks each rung's implementation and difficulty honestly.

---

## Citation

This work builds on AI-Newton:

```bibtex
@article{Fang:2025fmv,
    author  = "Fang, You-Le and Jian, Dong-Shan and Li, Xiang and Ma, Yan-Qing",
    title   = "{AI-Newton: A Concept-Driven Physical Law Discovery System without Prior Physical Knowledge}",
    eprint  = "2504.01538",
    archivePrefix = "arXiv",
    primaryClass = "cs.AI",
    month = "4",
    year = "2025"
}
```

## License

See [`LICENSE`](LICENSE).
