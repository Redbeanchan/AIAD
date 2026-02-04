# AIAD – Background Remover Platform

A background-removal platform with data versioning (DVC), an ML training pipeline (Kedro), and future API/UI/worker services. Everything runs locally; collaboration is via GitHub.

---

## What's Done So Far

### Step 1: Repository and structure
- Git repo initialized; base directories created:
  - `services/ui`, `services/api`, `services/worker` – frontend, backend, training worker
  - `ml/kedro`, `ml/models/base/standard-segmentation`, `ml/models/custom` – ML pipeline and models
  - `mlops/mlflow`, `mlops/dvc` – experiment tracking and data versioning
  - `infra/helm`, `infra/argocd`, `infra/monitoring` – Kubernetes and observability (for later)
- `.gitkeep` files added so Git tracks the folder structure.

### Step 2: Data versioning (DVC)
- DVC initialized in the project root (`dvc init`).
- `.dvc/` and config committed. No DVC remote yet (local-only); optional to add later for sharing data.
- Data and models will be tracked with `dvc add` when you have datasets (e.g. `data/raw`).

### Step 3: Kedro training pipeline
- Kedro project created at **`ml/background-remover-pipeline/`** (name: `background-remover-pipeline`, package: `background_remover_pipeline`).
- Example pipelines (spaceflights tutorial) are present and runnable: data processing, data science, reporting.
- **Kedro-MLflow plugin** is disabled in `settings.py` (`DISABLE_HOOKS_FOR_PLUGINS = ("kedro_mlflow",)`) due to a compatibility issue with Kedro 1.2 (`pipeline_name` in run params). MLflow can be used manually in nodes when you add training.
- **MLflow config** is in `ml/background-remover-pipeline/conf/local/mlflow.yml` with tracking URI `mlflow_runs` (local folder).

### Not done yet (planned)
- **Step 4:** Full MLflow integration (manual logging in nodes until kedro-mlflow supports Kedro 1.2).
- **Step 5:** Pre-trained model (e.g. from Hugging Face) under `ml/models/base/standard-segmentation/`.
- **Step 6–8:** API service (FastAPI), worker (Celery), UI.
- **Step 9–14:** Docker, Kubernetes/Helm, Argo CD, monitoring.

---

## Project layout

```
AIAD/
├── .dvc/                    # DVC config and cache
├── .venv/                   # Python virtual environment (create locally, do not commit)
├── data/                    # Repo-level data (DVC-tracked when you add it)
├── infra/
│   ├── argocd/              # Argo CD app manifests
│   ├── helm/                # Helm charts
│   └── monitoring/          # Prometheus/Grafana etc.
├── ml/
│   ├── background-remover-pipeline/   # Kedro project (run `kedro run` here)
│   ├── kedro/               # Placeholder
│   └── models/
│       ├── base/standard-segmentation/  # Pre-trained model (when added)
│       └── custom/          # Custom trained models
├── mlops/
│   ├── dvc/
│   └── mlflow/
├── services/
│   ├── api/                 # Future FastAPI app
│   ├── ui/                  # Future frontend
│   └── worker/              # Future training worker
├── README.md                # This file (or README_PROJECT.md)
└── requirements.txt         # Root-level Python deps (see below)
```

---

## Requirements

- **Python 3.10+**
- **Git**
- **DVC** (installed via `requirements.txt` in a venv)

---

## How to run everything (local)

### 1. Clone and enter the repo

```bash
git clone https://github.com/Redbeanchan/AIAD.git
cd AIAD
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 3. Install dependencies

From the **project root**:

```bash
pip install -r requirements.txt
```

This installs DVC, Kedro, the Kedro pipeline dependencies (including `openpyxl`, `plotly`, etc.), and MLflow so you can run the pipeline and (later) use MLflow in code.

### 4. Run the Kedro pipeline

From the **project root**:

```bash
cd ml/background-remover-pipeline
kedro run
cd ../..
```

Or in one line:

```bash
cd ml/background-remover-pipeline && kedro run
```

The example pipeline (9 tasks) should complete successfully. Outputs go to `ml/background-remover-pipeline/data/` (e.g. `02_intermediate`, `06_models`, `08_reporting`).

### 5. (Optional) DVC

- **Check status:** `dvc status`
- **When you have data:** e.g. `dvc add data/raw`, then `git add data/raw.dvc .gitignore` and commit.
- **Remote (optional):** To share data with others, add a DVC remote and use `dvc push` / `dvc pull`.

---

## Collaboration

- **Sync with GitHub:** `git pull origin main` and `git push origin main` as usual.
- **Credentials:** Use a GitHub Personal Access Token (not your account password) for HTTPS push/pull. To store it once: `git config --global credential.helper store`, then enter the token when prompted on the next push.
- **Branching:** Use feature branches (e.g. `feature/api-service`) and merge to `main` when ready.

---

## References

- [Implementation plan](https://github.com/Redbeanchan/AIAD) (overview of the 14-step plan).
- [Kedro docs](https://docs.kedro.org)
- [DVC docs](https://dvc.org/doc)
- [MLflow docs](https://mlflow.org/docs/latest/index.html)
