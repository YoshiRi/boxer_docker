# Boxer Dockerization & Integration Plan

## 🎯 Objective
Set up a reproducible Docker-based environment for running Boxer (inference only),
validate execution via the official headless demo, and prepare a clean interface
for future ROS integration.

---

## 📦 Scope

### Included
- Docker-based runtime environment (headless)
- Reproduction of README Demo #1
- Model and sample data bootstrap flow
- Output persistence and schema inspection
- Preparation for ROS integration (I/O boundary only)

### Excluded (for now)
- GUI / visualization support (moderngl, imgui)
- Full ROS node implementation
- Training setup

---

## 🧭 Stage Overview

| Stage | Name | Goal |
|------|------|------|
| 0 | Repo Analysis | Understand real dependencies & execution flow |
| 1 | Minimal Docker Build | Build headless runtime image |
| 2 | Compose & Volumes | Enable reproducible execution |
| 3 | Data Bootstrap | Download checkpoints & sample data |
| 4 | Demo Execution | Run README Demo #1 successfully |
| 5 | Output Inspection | Understand output schema |
| 6 | Execution UX | Simplify usage (Makefile, scripts) |
| 7 | ROS Preparation | Define integration boundary |

---

# 🧩 Stage 0: Repo Analysis

## Goal
Identify actual runtime dependencies and execution flow beyond README.

## Tasks
- Inspect `pyproject.toml`
- Inspect `run_boxer.py`
- Inspect `scripts/download_*.sh`
- Identify:
  - Dependencies
  - Output paths (`output/` vs `outputs/`)
  - GPU / GUI assumptions

## Deliverables
- `docs/boxer_docker_notes.md`

## Exit Criteria
- Clear dependency list
- Confirmed execution entrypoint
- Confirmed output directory behavior

---

# 🐳 Stage 1: Minimal Docker Build

## Goal
Create a headless Docker image capable of running Boxer inference.

## Tasks
- Create `Dockerfile`
- Install:
  - Python 3.12
  - torch
  - numpy
  - opencv-python
  - tqdm
  - dill
  - projectaria-tools
- Add minimal system packages:
  - git, curl, ca-certificates
  - libgl1, libglib2.0-0, ffmpeg

## Deliverables
- `Dockerfile`
- `.dockerignore`

## Exit Criteria
- `import torch, cv2, dill, tqdm` works
- `import projectaria_tools` works

---

# 🔁 Stage 2: Compose & Volumes

## Goal
Separate runtime from data using mounted volumes.

## Tasks
- Create `docker-compose.yml`
- Mount:
  - `./ckpts`
  - `./sample_data`
  - `./output`
- Support CPU / GPU execution

## Deliverables
- `docker-compose.yml`

## Exit Criteria
- Host and container share data directories correctly

---

# 📥 Stage 3: Data Bootstrap

## Goal
Enable reproducible download of required assets.

## Tasks
- Create `scripts/bootstrap_boxer.sh`
  - Run:
    - `scripts/download_ckpts.sh`
    - `scripts/download_aria_data.sh`
- Add logging and retry safety

## Deliverables
- `scripts/bootstrap_boxer.sh`

## Exit Criteria
- One command downloads all required assets successfully

---

# 🚀 Stage 4: Demo Execution

## Goal
Run the official headless demo successfully.

## Tasks
- Execute:
  ```bash
  python run_boxer.py --input nym10_gen1 --max_n=90 --track
````

* Capture logs
* Verify outputs

## Deliverables

* `docs/runbook.md`
* `logs/demo1.log`

## Exit Criteria

* Script completes without error
* Output files are generated in mounted directory

---

# 📊 Stage 5: Output Inspection

## Goal

Understand Boxer output format for downstream use.

## Tasks

* Inspect:

  * `boxer_3dbbs.csv`
  * `owl_2dbbs.csv`
* Document:

  * 3D box parameters
  * coordinate system assumptions
  * timestamps/frame indexing

## Deliverables

* `docs/output_schema.md`

## Exit Criteria

* Output fields clearly explained
* Ready for ROS mapping

---

# 🛠 Stage 6: Execution UX

## Goal

Make execution simple and repeatable.

## Tasks

* Add Makefile or scripts:

  * `make build`
  * `make bootstrap`
  * `make demo1`
* Standardize commands

## Deliverables

* `Makefile`

## Exit Criteria

* Full setup + run achievable in ≤3 commands

---

# 🔌 Stage 7: ROS Preparation

## Goal

Prepare a clean boundary for ROS integration.

## Tasks

* Create wrapper:

  * `scripts/run_boxer_job.py`
* Define:

  * input format
  * output format (CSV/JSON)
* Draft ROS mapping doc

## Deliverables

* `docs/ros_bridge_plan.md`
* wrapper script

## Exit Criteria

* Boxer callable without CLI dependency
* Clear mapping to ROS topics defined

---

# ⚠️ Risks & Notes

## Known Risks

* `projectaria-tools` compatibility issues
* Torch/CUDA mismatch
* OpenCV system dependency issues
* Hugging Face download failures
* Output directory inconsistency (`output/` vs `outputs/`)

## Important Note

Do NOT rely only on README — verify behavior from actual code.

---

# 🏁 Milestone Definition

## M1: Docker Runtime Ready

* Stage 0–2 complete

## M2: Reproducible Execution

* Stage 3–4 complete

## M3: Integration Ready

* Stage 5–7 complete

---

# 📌 Next Steps (Out of Scope for This Plan)

* GUI support (moderngl)
* Real-time ROS node
* Multi-view / streaming integration
