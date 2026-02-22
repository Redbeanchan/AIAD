# AIAD Background Removal (Self-Host + Train + Batch API)

A background remover built for developers who want to **train**, **evaluate**, and **self-host** their own background removal system end-to-end.

This repo provides:

- **Model training & evaluation** via **Kedro** (reproducible pipelines)
- A **self-hosted website** (Flask UI) for before/after previews and PNG downloads
- A **batch-friendly API** for background removal (local model or online providers)
- Optional **MLOps monitoring** with **Prometheus + Pushgateway + Grafana** for training metrics

## Why this exists

When training segmentation models, data variance matters. Relying on a single dataset or a single background distribution can cause models to overfit background cues rather than learning the subject properly.

This project is designed so you can:
- **Ingest multiple datasets**, train a segmentation model, and evaluate outputs
- **Remove backgrounds at scale** (batch processing) to generate additional derived datasets
- Increase dataset diversity (variance) and encourage models to learn **subject features** rather than background noise

---

## Features

### 1) Kedro Pipelines (ML)
Pipelines are structured and runnable independently:

- `data_ingestion`  
  Downloads and extracts datasets (e.g., from Google Drive via `gdown`) into `data/01_raw/`

- `data_preprocessing`  
  Builds a unified manifest (`images`, `masks`, `source`), validates paths, and splits train/val

- `model_training`  
  Builds `tf.data` pipelines + trains a ResNet50-UNet segmentation model and saves `.keras`

- `model_evaluation` (qualitative)  
  Generates a preview grid `[Image | Pred Mask | BG Removed]` and saves to reporting

### 2) Self-hosted Website (UI)
A Flask-based UI for:
- Uploading images
- Viewing Before/After
- Downloading transparent PNG output

Supports:
- **Online API mode** (keys via env vars)
- **Local mode** (model inference; can fall back to a basic heuristic)

### 3) Developer Batch API
Use the API endpoints to remove backgrounds programmatically for workflows like:
- dataset preparation
- augmentation pipelines
- batch inference
- generating additional training data

### 4) Monitoring (Optional MLOps)
- **Prometheus + Pushgateway + Grafana**
- Training job pushes metrics (e.g., `train_loss`, `val_loss`) so you can track runs over time

---

## Repository Layout (high level)

- `kedro/background-removal/` — Kedro project (pipelines, configs, training code)
- `website/` — Flask UI and API (static/uploads/results + templates)
- `docker/` — Dockerfiles, compose, entrypoints
- `k8s/` — Kubernetes manifests (jobs, pvc, configmaps, UI deployment)
- `k8s/monitoring/` — monitoring manifests (ServiceMonitor, etc.)

---

## Quickstart (Local Dev)

### 1) Create venv and install deps (Kedro)
From the Kedro project folder:

```bash
cd kedro/background-removal
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
````

### 2) Run Kedro pipelines

```bash
kedro run --pipeline=data_ingestion
kedro run --pipeline=data_preprocessing
kedro run --pipeline=model_training
kedro run --pipeline=model_evaluation
```

Outputs:

* Model: `data/06_models/my_model.keras`
* Preview: `data/08_reporting/qualitative_preview.png`

---

## Quickstart (Docker)

### Build images

From repo root:

```bash
docker build -f docker/Dockerfile.kedro -t bgbye-kedro:latest .
docker build -f docker/Dockerfile.train -t bgbye-train:latest .
docker build -f docker/Dockerfile.ui    -t bgbye-ui:latest .
```

### Run UI

```bash
docker run --rm -p 8080:8080 bgbye-ui:latest
```

Open: `http://localhost:8080`

### Run training (Kedro)

```bash
docker run --rm -it bgbye-train:latest
# or specify pipeline via env:
docker run --rm -e KEDRO_PIPELINE=model_training bgbye-train:latest
```

> Note: to persist outputs, mount the `data/` folder as a volume.

---

## API Usage (Website)

The Flask service exposes endpoints for developer usage.

### 1) Online API Mode

`POST /process/api`

Form fields:

* `image`: file upload
* `provider`: `remove.bg` | `clipdrop` | `photoroom`

Response:

```json
{ "before": "/static/uploads/<id>", "after": "/static/results/<id>.png" }
```

Environment variables:

* `REMOVE_BG_API_KEY`
* `CLIPDROP_API_KEY`
* `PHOTOROOM_API_KEY`

If keys are missing, the app returns a mocked result with watermark.

### 2) Local Mode

`POST /process/local`

Form fields:

* `image`: file upload
* `model`: `segmenter.pkl` (or your chosen file)

Response:

```json
{ "before": "/static/uploads/<id>", "after": "/static/results/<id>.png" }
```

---

## Kubernetes (minikube)

This repo supports running the full pipeline on Kubernetes:

* PVC for `data/`
* ConfigMap for Kedro `conf/local/` parameters
* Jobs for ingestion/preprocessing/train/eval
* UI deployment + service + ingress

### 1) Use minikube docker images (important)

Build images inside minikube’s docker:

```bash
eval $(minikube -p minikube docker-env)

docker build -f docker/Dockerfile.kedro -t bgbye-kedro:latest .
docker build -f docker/Dockerfile.train -t bgbye-train:latest .
docker build -f docker/Dockerfile.ui    -t bgbye-ui:latest .
```

### 2) Apply PVC + ConfigMap

```bash
kubectl create namespace bgbye --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/pvc-data.yaml
kubectl apply -f k8s/cm-kedro-conf.yaml
```

### 3) Run pipeline jobs (recommended order)

```bash
kubectl apply -f k8s/job-ingestion.yaml
kubectl apply -f k8s/job-data-preprocessing.yaml
kubectl apply -f k8s/job-train.yaml
kubectl apply -f k8s/job-model-evaluation.yaml
```

Logs:

```bash
kubectl -n bgbye logs -f job/bgbye-job-ingestion
kubectl -n bgbye logs -f job/bgbye-job-data-preprocessing
kubectl -n bgbye logs -f job/bgbye-job-train
kubectl -n bgbye logs -f job/bgbye-job-model-evaluation
```

### 4) Deploy UI

```bash
kubectl apply -f k8s/ui-deployment.yaml
kubectl apply -f k8s/ui-service.yaml
kubectl apply -f k8s/ui-ingress.yaml
```

Port-forward if needed:

```bash
kubectl -n bgbye port-forward svc/bgbye-ui-svc 8080:80
```

---

## Monitoring (Prometheus + Grafana + Pushgateway)

This repo supports basic MLOps monitoring for training jobs using:

* `kube-prometheus-stack`
* `prometheus-pushgateway`
* training pushes metrics to pushgateway

### Install (Helm)

```bash
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm upgrade --install kps prometheus-community/kube-prometheus-stack -n monitoring
helm upgrade --install pushgw prometheus-community/prometheus-pushgateway -n monitoring
```

### Port-forward

```bash
kubectl -n monitoring port-forward svc/kps-grafana 3000:80
kubectl -n monitoring port-forward svc/kps-kube-prometheus-stack-prometheus 9090:9090
kubectl -n monitoring port-forward svc/pushgw-prometheus-pushgateway 9091:9091
```

Grafana password:

```bash
kubectl -n monitoring get secret kps-grafana -o jsonpath="{.data.admin-password}" | base64 -d; echo
```

---

## Notes / Tips

* If a pod is stuck in `ContainerCreating`, run:

  ```bash
  kubectl -n bgbye describe pod <pod> | sed -n '/Events:/,$p'
  ```

  Most issues are missing PVC/ConfigMap or image pull policy.

* Keep dependencies separated:

  * `kedro/background-removal/requirements.txt` for training
  * `docker/requirements.ui.txt` for UI

---
