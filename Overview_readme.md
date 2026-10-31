# ReSI-Math Code Overview

This repository contains the main code, metadata, and reported result summaries used for the ReSI-Math experiments.

```text
.
├── code/
├── data/
├── results/
└── CODE_OVERVIEW.md
```

## Repository Structure

### `code/`

Contains the main experimental implementation.

The code is organized into four components:

- `evolution/`: v7-era evolutionary trajectory construction and Self-Judge implementation.
- `selection/`: policy profiling, frontier scoring, and adaptive problem selection.
- `rlvr/`: Solver-RLVR and Judge-RLVR rewards, data preparation, and training utilities.
- `eval/`: evaluation and robustness analysis utilities.

The overall code workflow is:

```text
Evolution
   ↓
Recoverable candidate pool
   ↓
Policy profiling
   ↓
Frontier selection
   ↓
Sequential Solver-RLVR
   ↓
Re-profile under the updated policy
   ↓
Next RLVR stage
   ↓
Evaluation
```

The Solver-RLVR sequence was executed stage by stage as Q1 → Q2 → Q3. After each stage, intermediate checkpoints were evaluated on DEV192 for checkpoint selection, and the selected checkpoint was used to initialize the next stage.

---

### `data/`

Contains lightweight metadata required to identify the experimental inputs and selected problems without redistributing third-party benchmark data.

This includes:

- the 8B recoverable candidate pool;
- the selected Q1/Q2/Q3 problems and their provenance;
- metadata for the 243-source Judge-RLVR dataset.

Raw benchmark examples and the full evolutionary trajectory archive are not redistributed because they contain third-party benchmark content.

---

### `results/`

Contains compact result files supporting analyses reported in the paper.

These files include:

- policy dynamics across the sequential Q1/Q2/Q3 stages;
- robustness results after excluding the DEV192 checkpoint-selection subset.

They are intended to make the reported analyses easier to inspect without rerunning the full training pipeline.

---

## Experimental Components

### Evolutionary Trajectory Construction

The evolutionary component samples multiple Solver trajectories for each problem, verifies their outcomes, constructs correct/failed trajectory populations, and generates new reasoning trajectories through crossover.

The released implementation corresponds to the v7-era protocol used for the reported experiments. The evolutionary execution unit was applied iteratively for three rounds.

### Frontier Selection

Candidate problems are profiled under the current policy using multiple sampled trajectories.

The selection score combines:

\[
S_i = p_i(1-p_i)\,D_{\mathrm{sem},i}\,\mathrm{ParseRate}_i,
\]

where \(p_i\) measures empirical success probability, \(D_{\mathrm{sem},i}\) measures semantic diversity, and \(\mathrm{ParseRate}_i\) measures output reliability.

After each Solver-RLVR stage, the candidate pool is re-profiled under the updated policy before selecting the next problem.

### Solver-RLVR

Solver-RLVR uses verifier-confirmed outcomes as rewards.

The policy-adaptive training sequence is:

```text
Base
  ↓
Select Q1
  ↓
Q1 RLVR
  ↓
Re-profile
  ↓
Select Q2
  ↓
Q2 RLVR
  ↓
Re-profile
  ↓
Select Q3
  ↓
Q3 RLVR
```

The sequential stages were executed manually rather than through a single end-to-end launcher.

### Judge-RLVR

Judge-RLVR uses trajectory-level correctness supervision constructed from the same 243-problem evolutionary source pool.

Its data preparation and reward implementation are provided separately under `code/rlvr/`.

### Evaluation

Evaluation scripts support the reported benchmark and robustness analyses.

DEV192 is used only for checkpoint selection and is not used as RLVR training data.

---

## Notes

The repository focuses on the code and metadata necessary to inspect the main experimental pipeline. Large model checkpoints, third-party benchmark data, and full generated trajectory archives are not redistributed.

More detailed implementation notes are provided inside the corresponding subdirectories under `code/`.
