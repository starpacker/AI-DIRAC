# Low-prior operational concept discovery

## 1. Objective

The project should discover physical theories while receiving as little
physical ontology as possible.  A human may design apparatus and a curriculum
of experiments, but the discovery process must not read simulator-only state,
pre-labelled target quantities, or the law that the experiment was designed to
reveal.

The target is not an unconstrained machine-learning latent coordinate.  It is
an **operational concept**: a reusable quantity accompanied by an executable
measurement protocol, an uncertainty estimate, an applicability domain, and
independent transfer evidence.

The main discovery loop remains:

```text
choose an experiment or validation task
  -> choose a small group of eligible variables
  -> materialize their measured values
  -> symbolic regression
  -> classify the resulting evidence
```

Regression results may now be classified into three different ledgers:

1. a specific law for one apparatus/configuration;
2. a general-law candidate transferable across experiments;
3. an operational-concept candidate that needs controlled interventions.

These ledgers must not be conflated.  Data used to define a quantity are not
independent evidence for a law that is algebraically identical to its
definition.

## 2. Allowed prior and forbidden leakage

The experimental designer may provide:

- raw sensor channels and their sampling correspondence;
- object/event identity across observations;
- controllable apparatus actions and safe parameter ranges;
- the ability to hold a control fixed, replace an object, or install a concrete
  reference object;
- a domain-general mathematical language, with every operator declared in the
  prior audit;
- a curriculum inspired by historical experiments.

The discovery process must not receive:

- simulator-hidden mass, heat capacity, charge, field, amount of substance, or
  other ground-truth parameters;
- a derived column named after the target concept unless its measurement
  protocol was already certified;
- a target-specific equation template;
- units, vector groupings, or semantic names that are not part of the declared
  prior budget;
- validation credit from the same observations that defined the candidate.

Hidden simulator values may be used only by an offline evaluator after a run.
They must not affect candidate generation, experiment selection, regression,
promotion, or rewards.

## 3. Concept lifecycle and pipeline permissions

Operational concepts have an explicit lifecycle:

```text
proposed -> trial -> certified
                  \-> rejected
```

- **Proposed** means that a relation suggests a reusable quantity.
- **Trial** means that an executable provisional measurement protocol exists.
  It may be selected only in calibration and validation tasks.  Laws containing
  it remain quarantined.
- **Certified** means that the protocol passed the configured independent
  intervention tests.  Its symbolic concept is registered as a normal action
  and may be chosen by the main law-discovery pipeline.
- **Rejected** means that a controlled test falsified the claimed invariance or
  transfer.  Revisions create a new version rather than silently modifying the
  failed definition.

Certification is always relative to a tested applicability domain.  A later
experiment may falsify a certified concept and force revision.

## 4. First generic family: object-attached additive weights

Mass in collisions and constant heat capacity in thermal contact share a
discoverable structure.  Suppose an observable state `y_i` belongs to object
`i`.  Across a closed transition, search for object-attached scalars `w_i`
such that

\[
    \sum_i w_i \Delta y_i \simeq 0.
\]

The search is a null-space/low-rank problem over many interventions, not a
hard-coded mass or heat-capacity formula.  The overall scale is unidentifiable,
so a concrete reference object fixes `w_ref = 1`.  This is a unit convention,
not simulator truth.

Examples that the miner itself is not told:

- for collision data, `y` may be a measured velocity and `w` may later be
  interpreted as inertial mass;
- for thermal-contact data, `y` may be an empirical temperature coordinate and
  `w` may later be interpreted as heat capacity.

A numerical null vector is only a trial concept.  It must survive changes to
initial conditions, coupling strengths, geometry/direction, object identity,
reference object, and apparatus family where those interventions are available.

## 5. Required validation evidence

The default certification policy requests distinct evidence for:

1. repeatability with the same objects and controls;
2. invariance under multiple nuisance interventions;
3. sensitivity to replacing the claimed argument object;
4. transitivity across reference objects;
5. transfer to another apparatus or experimental family;
6. held-out predictive usefulness;
7. a provenance audit showing that only allowed channels were used.

For reference ratios, a typical consistency test is

\[
    w_{A/B} w_{B/C} \simeq w_{A/C}.
\]

Evidence from a definition experiment does not count as independent validation.
Falsification is stored and rewarded as information; it must not be erased by a
larger number of positive repeats.

## 6. Active validation tasks

A trial candidate produces `ValidationTask` records rather than running a
private batch of experiments outside the main loop.  A task contains:

- the candidate ID;
- a base experiment family;
- the intervention to perform;
- variables to hold fixed;
- required raw channels;
- the missing epistemic question and its priority/cost.

These tasks are added to the experiment selector's action pool.  The selector
therefore continues to choose the next experiment, but can spend an epoch
reducing uncertainty about a promising concept.  A reserved validation fraction
or an information-gain reward prevents short-term law rewards from starving
concept certification.

## 7. Representation requirements

An operational-concept record must contain at least:

- an opaque ID and version;
- lifecycle status;
- scope (`object_intrinsic`, `state`, `field`, `event`, `pattern`, or `flux`);
- argument/entity type without a target semantic label;
- estimator/measurement protocol;
- required observables and allowed provenance;
- reference object and remaining gauge freedom;
- fitted values and uncertainty;
- definition experiments and independent validation evidence;
- applicability domain and experiment compatibility;
- discovery epoch and failure history.

The initial implementation covers object-attached scalar weights.  The same
registry and lifecycle will later host:

- momentum and angular-momentum constructions after inertial weights unlock;
- empirical temperature scales and heat capacities;
- amount of substance and other sample intrinsics;
- pressure as a state quantity with an independent mechanical protocol;
- wavelength/frequency as pattern variables;
- charge, current, and environment-dependent electric/magnetic fields.

Not all future concepts are real-valued object attributes.  Temperature may
begin as an order/equivalence relation, fields are functions of spacetime, and
potentials have gauge freedom.  They need new miners but must obey the same
provenance, validation, and pipeline-permission rules.

## 8. First milestone: close the loop for one anonymous weight

The mechanics/thermal milestone is complete only when all of the following are
demonstrated:

1. experiments expose raw trajectories/readouts but never the simulator weight;
2. symbolic regression or the additive-invariant miner proposes an anonymous
   object weight;
3. the registry creates active controlled-intervention tasks;
4. the candidate stays out of normal law discovery while it is `trial`;
5. independent controls and reference transitivity certify it;
6. certification registers it in the normal concept/action pool;
7. the main pipeline selects it in a different experiment and discovers a new
   relation;
8. definition data are excluded from that relation's validation evidence;
9. an audit proves that no hidden simulator channel was consumed.

Only after this closed loop works should the project attach a human semantic
label such as “inertial mass” or “heat capacity” for benchmark reporting.
