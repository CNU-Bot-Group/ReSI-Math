<div align="center">

# ReSI-Math

### Exploring Recoverable Capability Frontiers for Efficient Reasoning Improvement

**Yao Cui\*** · **Zhen Yang\*** · **Yuandong Wang** · **Yangfu Zhu** · **Tianyu Wo** · **Zhenzhou Shao**

[**Project Page**](https://cnu-bot-group.github.io/ReSI-Math/) · **Paper (Coming Soon)**

</div>

---

## Overview

**ReSI-Math** studies how reasoning models can be improved by identifying and repeatedly exploiting their **recoverable capability frontier**: problems that the current policy does not solve reliably, but for which successful reasoning trajectories can still be elicited and reinforced.

This repository contains the main code, lightweight metadata, and reported result summaries used in the ReSI-Math experiments.

The released implementation covers the main experimental pipeline:

```text
Evolution
   ↓
Recoverable Candidate Pool
   ↓
Policy Profiling
   ↓
Frontier Selection
   ↓
Sequential Solver-RLVR
   ↓
Re-profile under the Updated Policy
   ↓
Next RLVR Stage
   ↓
Evaluation
```

The main Solver-RLVR sequence is executed stage by stage as:

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

After each stage, intermediate checkpoints are evaluated on **DEV192** for checkpoint selection, and the selected checkpoint is used to initialize the next stage.

---

## Repository Structure

```text
ReSI-Math/
├── code/
│   ├── evolution/
│   ├── selection/
│   ├── rlvr/
│   └── eval/
├── data/
├── results/
├── assets/
├── Overview_readme.md
├── index.html
└── README.md
```

### `code/`

Contains the main experimental implementation.

The code is organized into four components:

| Directory | Description |
|---|---|
| `code/evolution/` | v7-era evolutionary trajectory construction and Self-Judge implementation |
| `code/selection/` | policy profiling, frontier scoring, and adaptive problem selection |
| `code/rlvr/` | Solver-RLVR and Judge-RLVR rewards, data preparation, and training utilities |
| `code/eval/` | evaluation and robustness analysis utilities |

---

### `data/`

Contains lightweight metadata required to identify the experimental inputs and selected problems without redistributing third-party benchmark data.

The released metadata includes:

- the 8B recoverable candidate pool;
- the selected Q1/Q2/Q3 problems and their provenance;
- metadata for the 243-source Judge-RLVR dataset.

Raw benchmark examples and the full evolutionary trajectory archive are not redistributed because they contain third-party benchmark content.

---

### `results/`

Contains compact result files supporting the analyses reported in the paper.

The released result files include:

- policy dynamics across the sequential Q1/Q2/Q3 stages;
- robustness results after excluding the DEV192 checkpoint-selection subset.

These files are intended to make the reported analyses easier to inspect without rerunning the full training pipeline.

---

## Method Components

### 1. Evolutionary Trajectory Construction

The evolutionary component samples multiple Solver trajectories for each problem, verifies their outcomes, constructs correct and failed trajectory populations, and generates new reasoning trajectories through crossover.

The released implementation corresponds to the **v7-era protocol** used for the reported experiments.

The evolutionary execution unit is applied iteratively for three rounds.

```text
Problem
   ↓
Sample Solver Trajectories
   ↓
Verifier
   ↓
Correct / Failed Populations
   ↓
Crossover
   ↓
New Reasoning Trajectories
   ↓
Verification
```

---

### 2. Recoverable Frontier Selection

Candidate problems are profiled under the current policy using multiple sampled trajectories.

For each candidate problem \(i\), empirical success probability is estimated as:

$$
p_i = \frac{N_{\mathrm{correct}}}{K}
$$

We measure uncertainty using:

$$
U_i = p_i(1-p_i)
$$

The final selection score combines policy uncertainty, semantic diversity, and output reliability:

$$
S_i = p_i(1-p_i)\,D_{\mathrm{sem},i}\,\mathrm{ParseRate}_i
$$

where:

- $p_i$ measures empirical success probability;
- $D_{\mathrm{sem},i}$ measures semantic diversity among sampled trajectories;
- $\mathrm{ParseRate}_i$ measures output reliability.

After each Solver-RLVR stage, the candidate pool is **re-profiled under the updated policy** before selecting the next problem.


### 3. Solver-RLVR

Solver-RLVR uses verifier-confirmed outcomes as rewards.

The policy-adaptive sequential training procedure is:

```text
Base Policy
   ↓
Profile Candidate Pool
   ↓
Select Q1
   ↓
Q1 RLVR
   ↓
Checkpoint Selection on DEV192
   ↓
Re-profile Candidate Pool
   ↓
Select Q2
   ↓
Q2 RLVR
   ↓
Checkpoint Selection on DEV192
   ↓
Re-profile Candidate Pool
   ↓
Select Q3
   ↓
Q3 RLVR
```

The sequential stages were executed manually rather than through a single end-to-end launcher.

This repository therefore provides the main stage-level training, profiling, selection, reward, and evaluation utilities required to reconstruct the experimental workflow.

---

### 4. Judge-RLVR

Judge-RLVR uses trajectory-level correctness supervision constructed from the same **243-problem evolutionary source pool**.

Its data preparation and reward implementation are provided separately under:

```text
code/rlvr/
```

The released components include utilities for:

- preparing Judge-RLVR training data;
- binary judge reward computation;
- Judge-RLVR training;
- evaluation of judge checkpoints.

---

### 5. Evaluation

Evaluation scripts support the benchmark and robustness analyses reported in the paper.

In particular:

> **DEV192 is used only for checkpoint selection and is not used as RLVR training data.**

The repository also includes robustness analysis excluding DEV192 from the reported evaluation sets.

Relevant scripts and result summaries are available under:

```text
code/eval/
results/
```

---

## Released Experimental Artifacts

The repository focuses on lightweight artifacts that make the main experimental pipeline inspectable without redistributing large or third-party resources.

### Code

Main experimental code for:

- evolutionary trajectory construction;
- verifier-based trajectory processing;
- policy profiling;
- recoverable frontier selection;
- sequential Solver-RLVR;
- Judge-RLVR;
- evaluation and robustness analysis.

### Metadata

Released metadata includes:

- recoverable candidate information;
- selected problem identities;
- Judge-RLVR source metadata;
- DEV192 split information.

### Result Summaries

Released result files include:

```text
results/
├── solver28_policy_dynamics.csv
└── robustness_without_dev192.csv
```

These files support inspection of:

- policy dynamics across sequential RLVR stages;
- checkpoint behavior;
- robustness after removing the DEV192 checkpoint-selection subset.

---

## Release Scope

This repository does **not** redistribute:

- large model checkpoints;
- third-party benchmark datasets;
- the full generated trajectory archive;
- other large intermediate training artifacts.

The repository instead focuses on the code and metadata necessary to inspect the main ReSI-Math experimental pipeline.

More detailed implementation notes are retained in:

```text
Overview_readme.md
```

and inside the corresponding subdirectories under `code/`.

---

## Project Page

The interactive project page is available at:

**https://cnu-bot-group.github.io/ReSI-Math/**

It provides a visual overview of:

- the motivation behind recoverable capability frontiers;
- the ReSI-Math framework;
- sequential Q1 → Q2 → Q3 policy adaptation;
- training dynamics;
- main experimental results.

---

## Citation

If you find ReSI-Math useful for your research, please consider citing:

```bibtex
@misc{cui2026resimath,
  title     = {ReSI-Math: Exploring Recoverable Capability Frontiers for Efficient Reasoning Improvement},
  author    = {Yao Cui and Zhen Yang and Yuandong Wang and Yangfu Zhu and Tianyu Wo and Zhenzhou Shao},
  year      = {2026}
}
```

---

<div align="center">

**ReSI-Math — Recoverable Capability Frontiers for Efficient Reasoning Improvement**

</div>
