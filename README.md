# AIAD – Background Remover Platform

A background-removal platform with data versioning (DVC), an ML training pipeline (Kedro), and API/UI/worker services. Everything runs locally; collaboration is via GitHub.

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

### Step 4: Web app (API + UI) with pretrained model
- **API** (`services/api`): FastAPI app using **rembg** with the **isnet-general-use** model for background removal. Endpoint: `POST /remove-background` (upload image, get PNG with background removed). Optional GPU support via `rembg[gpu]` (see `services/api/requirements.txt`).
- **UI** (`services/ui`): Simple HTML/JS frontend to upload an image and display the result from the API. Served as static files (e.g. `python -m http.server 8080`).
- Both run locally: API on port 8000, UI on port 8080.

### Not done yet (planned)
- **Step 5:** Full MLflow integration (manual logging in nodes until kedro-mlflow supports Kedro 1.2).
- **Step 6:** Pre-trained model assets under `ml/models/base/standard-segmentation/` (e.g. from Hugging Face) for the Kedro pipeline.
- **Step 7–8:** Worker service (Celery), and any extra API/UI features.
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
│   ├── api/                 # FastAPI background-removal API (rembg)
│   ├── ui/                  # Web UI for upload & result
│   └── worker/              # Future training worker
├── README.md                # This file
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

### 4. Run the background-removal web app

This is the simple web app that lets you upload an image and get it back with the background removed.

1. **Install API dependencies** (once per environment):

   ```bash
   cd services/api
   pip install -r requirements.txt
   ```

2. **Start the API** (background remover service) in one terminal (from repo root):

   ```bash
   cd services/api
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Start the UI** in another terminal (from repo root):

   ```bash
   cd services/ui
   python -m http.server 8080
   ```

4. **Use the app**:
   - Open `http://localhost:8080` in your browser.
   - Choose an image file.
   - Click **“Remove background”**.
   - The page will call the API on port 8000 and show the processed image (background removed).

The API uses the open-source `rembg` model (`isnet-general-use`) under the hood.

### 5. Run the Kedro pipeline

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

### 6. (Optional) DVC

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
