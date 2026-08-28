# An End-to-End AI Harvest Planner for Low-Cost Fruit-Picking Robots

**Built on a physics-consistent world model.**

Detections in, a harvest plan out: where the robot base should stop, which fruit to take in what
order, what to leave for a person, and how long it will take. This is the planning layer of a
harvesting robot's software — perception sits upstream of it and motion control downstream.

The problem this project solves is that the labels a plan needs do not exist. Public orchard
datasets record where the fruit are; they do not record what happened when someone tried to pick
one. So the environment was built first, checked against physics, and then used to generate the
labels.

```
FRESH detections  →  physics scenes  →  31,960 labelled picks  →  outcome model  →  plan
                                                                        ↓
                                                     the plan is re-run against the physics
```

---

## What is here

| | |
|---|---|
| `src/` | Six modules and nine MJCF scene definitions |
| `01`–`08` notebooks | Data preparation through to report figures, in order |
| `models/` | Four trained weight files |
| `data/` | The derived dataset. **The FRESH source is not redistributed** |

### The modules

| Module | What it does |
|---|---|
| `physics.py` | Builds a MuJoCo scene per fruit — fruit, five-segment stalk, spur, and a gripper sized to that fruit |
| `generate.py` | Runs each scene at four gripper apertures from the same settled state |
| `environment.py` | Loads a canopy, tracks state, computes the observations the models consume |
| `planner.py` | **The planning layer.** Stops, sweep stage, pick selection, dynamics update |
| `reobserve.py` | Re-measures settled pose after the canopy is generated |
| `replay.py` | Renders a canopy; `08` uses it for one of the report figures |

**All planning logic lives in `planner.py`.** It was written into three notebooks before it was
written into a module, and the three drifted until their figures could not be put in the same
table. There is now one copy, and `05_test` checks that no notebook has made another.

---

## Running it

### Requirements

```
Python 3.12
pip install -r requirements.txt
```

No GPU. Everything in this repository was produced on an ordinary desktop.

### The source data

This repository does not include the FRESH dataset (Son et al., 2024), whose redistribution
terms are not clear. Obtain it from the authors and place the crops and `manifest.csv` under
`apple_crops/`; the generator reproduces the derived dataset from there.

```
https://github.com/sejong-rcv/FRESH
```

### Set the root

Every notebook reads `AIPICK_ROOT`, defaulting to `C:\aipick\final`. Point it at your clone:

```powershell
$env:AIPICK_ROOT = "C:\path\to\this\repo"
```

### Order

| Step | Notebook | Time |
|---|---|---|
| 1 | `01_data_preparation` — verify detections, build derived features | minutes |
| 2 | generator — 31,960 counterfactual rows | 5–7 h, six shards |
| 3 | canopy generation and pose re-measurement | tens of minutes |
| 4 | training — outcome, dynamics, station planner, pick policy | ~2.5 h |
| 5 | `02_integrated_model` — the chain on one tree | minutes |
| 6 | `03_evaluation` — thirty seeds, paired | ~1 h |
| 7 | `04_fidelity` — **the plans re-run against the physics** | ~4.7 h |
| 8 | `05_test` · `06_orchard_estimate` · `07_station_sweep` | ~30 min |
| 9 | `08_dataset_figures` | minutes |

Steps 2 and 7 dominate; both resume if interrupted.

The presentation clips that appear in the report are not produced here. They were made for a
project submission, from the same models, and are not needed to reproduce any of the numbers.

---

## What the checks are for

`05_test` runs seven checks and stops the notebook if any fails. Each came from a defect that
actually happened during development, not from imagining what might go wrong.

| Check | What it caught |
|---|---|
| Physics scenes load | `from_xml_path` resolves against the working directory, and failed silently once |
| Environment is deterministic | Every paired comparison assumes a repeat gives the same answer |
| **Policy scaling matches training** | A mismatch collapses the pick policy to an immediate stop. Diagnosed as a weak model three times before the cause was found |
| Canopy carries a measured pose | The planner fabricated this from a lookup table for a while |
| The plan is well formed | An empty plan surfaced three cells later as a missing column |
| Dataset is counterfactual | The four aperture rows have to move together |
| Planning logic has one home | Fails if a notebook makes its own copy |

---

## Results

On thirty held-out trees, with twenty planner stops and a sweep stage:

```
on the tree              120.0
detected                  94.3    79%  — foliage hides the rest
the robot can reach       62.6    66% of detected
put in the plan           62.6   100% of reachable
attempted                 62.6
premium, stem intact      49.9    80% of attempts
                        1,227 s  = 20.5 min per tree
```

**Nothing reachable is left out of the plan.** What remains is what the camera cannot see or the
arm cannot get to, not something the algorithm discarded.

The outcome model reaches 94.07% on held-out picks, with a worst-decile calibration gap of
0.026. Re-running 1,820 of the planner's own picks against MuJoCo puts predicted 0.788 against
realised 0.730.

**That gap is not a fixed property of the model.** At a 0.9 selection threshold it is −0.120; at
no threshold it is −0.058. The model is optimistic where it is confident and pessimistic where
it is not, so a configuration that attempts everything is scored more fairly. A surrogate's
fidelity should be reported together with the operating point it was measured at.

---

## Scope

**In:** the planning layer — outcome prediction, station selection, pick order, and the
projections that follow from them.

**Out:** perception and motion control. Field trials are also out: the absence of measurement
equipment at the client is the reason this project exists, and the deliverable was scoped to a
validated simulation from the start.

The physics constants come from the literature, not from a rig. What the invariant checks
establish is that the simulation obeys the physics it was given — not that those constants match
a particular orchard.

---

## Licence

MIT. See `LICENSE`.

The libraries are all open source (MuJoCo Apache 2.0, PyTorch BSD, LightGBM MIT, CatBoost
Apache 2.0). MuJoCo is called as a library, not vendored, so its terms do not extend to this
code.

**The FRESH dataset is not covered by this licence** and is not included here.

---

## Citation

```
Son, S. (2026). An end-to-end AI harvest planner for low-cost fruit-picking robots built on a
    physics-consistent world model [Capstone project]. Walsh College.
```

Built on:

```
Son, G., Lee, S., & Choi, Y. (2024). FRESH: Fusion-based 3D apple recognition via estimating
    stem direction heading. Agriculture, 14(12), 2161.

Todorov, E., Erez, T., & Tassa, Y. (2012). MuJoCo: A physics engine for model-based control.
    IEEE/RSJ IROS, 5026-5033.

Bu, L., Hu, G., Chen, C., Sugirbay, A., & Chen, J. (2020). Experimental and simulation analysis
    of optimum picking patterns for robotic apple harvesting. Scientia Horticulturae, 261,
    108937.

Li, J., Karkee, M., Zhang, Q., Xiao, K., & Feng, T. (2016). Characterizing apple picking
    patterns for robotic harvesting. Computers and Electronics in Agriculture, 127, 633-640.
```
