# AI-Newton

This is a proof-of-concept implementation of AI-Newton, a concept-driven discovery system that can formulate general physical laws in an unsupervised manner without any prior physical knowledge. AI-Newton integrates  an autonomous discovery workflow, powered by a knowledge base (KB) and knowledge representation. The KB is responsible for storing and managing structured knowledge, including symbolic concepts, specific laws, and general laws. The knowledge representation is a physical domain specific language (DSL) that allows for the representation of physical concepts and laws in a structured and formalized manner. Given a collection of physical experiments, AI-Newton can formulate symbolic general laws applicable across a wide problem scope without neither supervision nor any prior physical knowledge. As a proof-of-concept implementation, it can rediscover Newton's second law, law of gravitation, conservation laws and others in classical mechanics.

The reproduction's stricter low-prior objective, allowed experimental guidance,
anti-leakage rules, and operational-concept lifecycle are specified in
[`docs/operational_concept_discovery.md`](docs/operational_concept_discovery.md).
The phrase “without any prior physical knowledge” above is a historical project
claim; new experiments must instead report their explicit measurement,
representation, and apparatus priors.

## Theory-aware contextual experiment selection

This reproduction adds a CPU-only LinUCB experiment selector.  Unlike the
legacy arm-wise UCB, it scores an experiment from both the current theory and
the experiment descriptor.  The selector is enabled with:

```python
theorist.perform_discovery(
    exp_select_mode="contextual",
    max_epochs=100,
    save_path="data/my_run/checkpoint",
)
```

`contextual` is the default mode in this copy; `ucb` and `random` remain
available as baselines.  The contextual mode also skips the legacy full-pool
feasibility pass, which used to run and discard one complete theoretical
analysis per experiment before useful discovery began.

The inner concept-group recommender is also bounded to 24 candidate concepts
before forming triples and evaluates its small policy network in one CPU
batch.  This prevents the original cubic combination growth and avoids the
per-group CUDA launch overhead.  The cap is stored as
`memory.action_bandit.max_group_candidates` and can be adjusted per run.

The reward uses a curriculum.  A small vocabulary receives bootstrap credit,
but concept-only growth saturates at a configurable budget and is penalized
when it does not produce laws or evidence.  Supported laws, verified
contradictions, revisions, validation coverage, and lower complexity dominate
the mature objective.  Validation rewards use log saturation so a large scope
cannot overwhelm every other component.  Configuration lives in
`AdaptiveRewardConfig` in `aiphy/contextual_bandit.py`.

The selector combines shared LinUCB features with a per-experiment reward EMA,
an arm-level exploration bonus, periodic least-tried coverage, stagnation
penalties, and a hard three-selection cooldown.  Compute limits are also
per-experiment: a timeout raises only that experiment's next budget.  Repeating
a cheap no-op experiment therefore cannot unlock hour-long global analysis.

Every transition stores hashes for the exact pre/post theory snapshots, the
exact candidate experiment descriptor, the feature vector, the score, and a
reward breakdown.  Recent exact snapshots are deduplicated in
`memory.contextual_bandit.state_archive`; all contextual state is persisted in
the normal `_memory.json` checkpoint.

## Independent theory basis

New conserved laws pass through a probabilistic local Kahler-rank reducer
before the expensive differential-algebra reducer.  DSL concepts are expanded
to rational SymPy jet expressions and Jacobian ranks are evaluated at several
random points over three large prime fields.  Small rational zero laws use a
bounded exact localized-radical ideal reduction instead.  Generic nonzero
chart assumptions are recorded in the certificate, while ambiguous branch
unions such as `x*y == 0` remain conservative.

Reduction is scope-aware and runs in two stages: generators covering the full
candidate scope reject obvious closure first, then experiment-local generators
refine the candidate's true marginal scope.  Completely disjoint object types
are excluded from each other's comparison basis.  Every certificate stores the
exact candidate scope, independent/dependent/inconclusive sub-scopes, basis,
method, confidence and any local nonzero assumptions.

The active theory is maintained as a complexity-weighted generating basis.
When a cheaper new law enters, more expensive active laws are rechecked and
displaced laws are archived as derived theorems.  A change to an existing
law's validated scope triggers the same revalidation and exchange, even when
no new law was proposed.  Unsupported expressions fall back to the existing
symbolic differential reducer rather than being declared redundant.

Concepts have a separate exact canonical gate.  Non-sum definitions are
expanded without experiment-specific dynamics, rationally normalized and
indexed by object schema; algebraically identical aliases do not enter the
action pool.  Nested sum concepts use a safe experiment-local fallback because
the current Rust DSL cannot globally expand `Mksum`.  Top-level concepts used
only by derived laws are also excluded from future actions.  Functionally
dependent but non-identical coordinates (for example `x` and `x**2`) are not
blindly merged because either may be the simpler coordinate for a later law.

Cached cross-validation performed while a new law is created is no longer
rewarded as new experimental evidence.  Maturity and concept novelty count the
active independent basis, while derived laws/concepts remain serialized for
auditability.

## Validated PySR law proposals

The discovery loop now has a `pysr` regression backend.  It preserves the
original `ver3` + PCA results and uses PySR only to propose additional symbolic
expressions; it does not admit PySR's training output directly.  Repeated
measurements are averaged instead of selecting one arbitrary trace, numerical
derivatives use the measured time coordinate, target-derived features are
excluded, and fitting uses only the first temporal block.  A candidate must
still fit the held-out block after all floating-point coefficients have been
replaced by bounded-denominator rationals and the expression has been parsed by
the AI-Newton DSL.  It then passes through the existing independent test
experiment and the minimum generating-basis reducer.

The production `pysr` backend now routes experiment identity and registered
quantity metadata into the same generic discovery entry point used by the
capability tests.  A baseline differential relation no longer silently blocks
typed PySR proposals.

Use the backend explicitly when starting a discovery run:

```python
theorist.perform_discovery(
    exp_select_mode="contextual",
    regression_backend="pysr",
)
```

PySR is optional at runtime.  If its Python/Julia runtime is unavailable, the
backend reports no extra proposals and returns the complete baseline result.
The first PySR invocation may install or precompile Julia packages, so it
should not be confused with steady-state search time.  Search size can be
bounded with `AI_NEWTON_PYSR_ITERATIONS`, `AI_NEWTON_PYSR_POPULATIONS`, and
`AI_NEWTON_PYSR_TARGETS`.  The source-38 v3 runner selects this backend; the
other runners retain `ver3` unless configured explicitly.

### Typed and transcendental search language

The scalar DSL and numerical evaluator support `sin`, `cos`, `exp`, and `log`.
`aiphy.typed_expression` adds checked scalar, vector, and tensor expressions,
including `Dot`, `Cross`, `Grad`, `Div`, and `Curl`.  Since the existing theory
database stores scalar laws, a vector or tensor law is represented by its
type-checked scalar components; this keeps it compatible with experiment
validation and the independent-basis reducer.

The regression proposal layer creates typed features for cross products and
electromagnetic force components instead of asking an untyped evolutionary
search to rediscover component algebra from scratch.  Transcendental closures
and their coefficients are fitted on a training block, converted to bounded
rationals, and re-evaluated on a held-out block.  The same validation rule is
used for structured van der Waals correction candidates.

Spatial derivatives require spatial observations.  A normal AI-Newton
`DataStruct` is a time trace and cannot identify a divergence or curl.  For
field data, `PySRDiscovery.discover_spatial_operators` accepts an explicit
`SpatialGrid`, evaluates typed `Grad/Div/Curl` candidates, and validates the
selected relation on a disjoint spatial block.  A single charged-particle
trajectory can therefore test Lorentz dynamics, but it cannot by itself claim
discovery of Maxwell field equations; those require sampled spatial field
grids.

The focused capability benchmarks are deliberately separate from the
historical source-38 suite:

```bash
python -m unittest test.test_extended_capabilities -v
```

They verify discovery of a component of `F = q(v cross B)`, a Boltzmann
exponential, a spatial `source = Div(field)` relation, and the replacement of
the rejected ideal-gas prediction by the van der Waals repulsion and attraction
terms.  These benchmarks establish representation, proposal, rationalisation,
and holdout-validation capability.  They do not by themselves establish that
the contextual experiment selector will choose the necessary observations in
an unconstrained autonomous run.

### Formula-blind autonomous capability mode

`PySRDiscovery.discover_generic_typed` is the strict search path for capability
claims.  It receives only the supervised observable, scalar/vector declarations,
units, and the common operator library.  It never calls structured candidates
or domain templates.  `Dot` and `Cross` are applied uniformly to every
type-compatible vector pair; PySR must still select and combine the resulting
components.  Floating constants are refined only after PySR has independently
found an expression tree, then bounded-rationalized and checked on held-out
data.

Exact high-complexity rational laws also have a deterministic,
formula-blind dimensional search.  It enumerates bounded monomials and
sums/differences of equal-dimension subexpressions using only `+`, `-`, `*`,
`/`, and integer powers.  Coefficients are fitted on 60% of the data,
candidates are selected on a separate 20%, and the reported result must pass
the untouched final 20%.  On the real-gas benchmark this generates 2,776
monomials, reduces them to 46 numerically distinct target-dimension candidates,
tests 1,035 two-term combinations, and recovers the van der Waals equation
without inspecting variable names or constructing gas-specific features.

The genuine Julia searches are intentionally excluded from the fast default
test run because they take roughly two minutes in total.  Run them explicitly:

```bash
AI_NEWTON_RUN_SLOW_SR=1 python -m unittest \
  test.test_autonomous_sr_capabilities -v
```

Every slow test asserts an audit record with `structured_candidates == 0`,
`domain_templates == 0`, and a real PySR model invocation.

The main call chain is now:

```text
Theorist.doit(exp_name)
  -> search_relations_pysr(..., exp_name=exp_name)
  -> sr_metadata.infer_generic_specifications(...)
  -> PySRDiscovery.discover(..., specification=...)
  -> generic typed PySR or dimensional SR
```

`aiphy.sr_metadata` stores only quantity facts: scalar/vector kind, component
membership, unit, and optionally which measured observable is a response for a
particular experiment.  New experiment families can call `register_quantity`,
`register_vector`, and `register_experiment_targets`; no equation is accepted
by this registry.  If metadata is absent, the historical scalar/derivative
fallback remains available but its domain templates are disabled.  The former
hard-coded electromagnetic, phase, and real-gas feature path can only be
enabled explicitly with `PySRConfig(enable_domain_templates=True)` for legacy
reproduction tests; it is not the production default.

### Revising an existing law

`PySRDiscovery.discover_theory_updates` optionally accepts an old scalar law
and attempts four generic revisions:

- additive: search `observed - old` and reconstruct `old + correction`;
- multiplicative: search `observed / old` away from zero and reconstruct
  `old * factor`;
- internal substitution: solve the old AST for each eligible leaf, learn its
  effective value, and substitute the learned expression back into that node;
- skeleton prior: first evolve populations against values generated by the old
  law, then use PySR's supported warm-start state to continue mutating those
  populations against the new observations.

These modes contain no van der Waals-specific correction.  In the recorded
real-gas budget, multiplicative and skeleton-prior PySR searches improved on
the ideal structure but did not meet the strict validation threshold; the
formula-blind dimensional search did recover the exact corrected law.  This
negative result is important: availability of a revision mechanism is not
treated as evidence that a particular revision was discovered.

### Fixed classical-mechanics revision loop

`aiphy.theory_revision.TheoryRevisionController` provides a deterministic
experiment-selection-free bridge for testing theory repair before reconnecting
the contextual bandit. It evaluates and falsifies one active law, invokes the
generic update strategies, validates candidates on separate parameter and
initial-condition experiments, obtains a probabilistic rank certificate,
atomically archives the falsified law, inserts the child revision, and reports
an explicit reward breakdown. A failed or redundant repair does not mutate the
active theory.

The fixed suite currently contains two classical-mechanics transitions:

- small-angle pendulum to the exact large-angle law, found by internal AST
  substitution: `angle -> sin(angle)`;
- linear Hooke oscillator to Duffing dynamics, found as a generic
  dimensionally valid additive residual proportional to `position**3`.

Both train on several trajectories and must generalize to three unseen sets of
physical parameters and initial conditions. Run the genuine-search suite with:

```bash
python run_classical_revision_suite.py
```

Fast state-transition tests are in `test.test_theory_revision`. The real PySR
pendulum test is opt-in:

```bash
AI_NEWTON_RUN_SLOW_REVISION=1 python -m unittest \
  test.test_theory_revision.GenuinePySRTheoryRevisionTest -v
```

An existing checkpoint can be audited without mutation:

```bash
python audit_theory_basis.py data/test_cases/my_run/checkpoint
```

To explicitly migrate it, always choose a new output prefix:

```bash
python audit_theory_basis.py data/test_cases/my_run/checkpoint \
  --output-prefix data/test_cases/my_run/checkpoint_independent
```

## Installation
The program currently only supports **Linux**. **CUDA** is also currently required.

**Pre-requisites:**
```
Python 3.11.9 or higher
Rust 1.84.0
Maple 2024
```
Conda is strongly recommended for managing an independent python environment for AI-Newton.

**Installation via Conda and Cargo:**

1. Clone the repository:
```
git clone https://github.com/Science-Discovery/AI-Newton.git
cd AI-Newton
```
2. Create a virtual environment and install the dependencies:
```
conda create -n ainewton python=3.11
conda activate ainewton
pip install -r requirements.txt
```
3. Compile Rust libraries:
```
cargo build --release
cp target/release/libcore.so aiphy/core.so
```
4. Run the bench-marking test:
You can run `aiphy/test_human_operating-benchmark.ipynb` to test the installation. It will take about 10-20 minutes to run. This test is not just for verifying the installation, but more importantly, it serves as a minimal working example to demonstrate how AI-Newton discovers and constructs important physical laws from Newtonian mechanics. An example output of this test can be found in the `data/human/example_output` directory.
5. Run the test case:
Since it may take several days to run the test case, it is recommended to run in the background:
```
nohup python -m test.examples.test_example_1 > logs/test_example_1/test_example_1.log 2>&1 &
```
Some of our test results can be found in the `data/test_cases/example_1` directory while logs can be found in the `data/test_cases/logs` directory. In which the *_knowledge.txt files are the most important. They are used to store the core part of the KB, including physical concepts and general laws discovered. The test results were generated on two configurations: (1) an Intel Xeon Platinum 8370C (128 threads @ 3.500GHz) with an NVIDIA A40 GPU, and (2) an Intel Xeon Silver 4314 (64 threads @ 3.400GHz) with an NVIDIA GeForce RTX 4080 GPU.


## Citation
```
@article{Fang:2025fmv,
    author = "Fang, You-Le and Jian, Dong-Shan and Li, Xiang and Ma, Yan-Qing",
    title = "{AI-Newton: A Concept-Driven Physical Law Discovery System without Prior Physical Knowledge}",
    eprint = "2504.01538",
    archivePrefix = "arXiv",
    primaryClass = "cs.AI",
    month = "4",
    year = "2025"
}
```
