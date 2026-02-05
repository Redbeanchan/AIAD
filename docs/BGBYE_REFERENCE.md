# bgbye reference (for AIAD web app)

bgbye is cloned at **`/home/alex/bgbye/bgbye`** and used as reference for pretrained background-removal models and inference patterns. We are **not** forking it; we build our own UI and API in `services/api` and `services/ui`, reusing the same model(s) and inference logic where useful.

---

## bgbye layout (relevant parts)

- **`server/server.py`** – FastAPI app: `POST /remove_background/` (image + `method` form field), returns PNG.
- **`server/ormbg/`** – Custom ORMBG model (U²-Net–style): `ormbg.py` (network), `ormbg_processor.py` (load + inference).
- **`server/setup.sh`** – Installs deps (torch, transformers, rembg, carvekit, transparent-background, etc.) and downloads `~/.ormbg/ormbg.pth` from Hugging Face.

---

## Models used in bgbye

| Method        | Source / model | Notes |
|---------------|----------------|--------|
| **bria**      | Hugging Face `briaai/RMBG-1.4` via `transformers` pipeline (`image-segmentation`) | CPU in bgbye; no local .pth. |
| **rembg**     | `rembg` + sessions: `u2net`, `u2net_human_seg`, `isnet-general-use`, `isnet-anime` | Models downloaded on first use. |
| **ormbg**     | Custom ORMBG: `~/.ormbg/ormbg.pth` (Hugging Face `schirrmacher/ormbg`), code in `server/ormbg/` | PyTorch; needs .pth file. |
| **inspyrenet**| `transparent_background.Remover` | Third-party package. |
| **carvekit**  | U2NET, TracerUniversalB7, BASNET, DeepLabV3 (with FBAMatting) | Heavier; GPU-oriented. |

---

## Suggested model(s) for AIAD API (services/api)

1. **BRIA RMBG-1.4** (easiest): `transformers` pipeline, no local weights. Good default for “one pretrained model” and works on CPU.
2. **rembg** (e.g. `isnet-general-use`): Simple API, models auto-downloaded; no copy of ormbg code.
3. **ORMBG**: Best quality in bgbye; requires copying `server/ormbg/` and downloading `ormbg.pth` (or pointing to `~/.ormbg/ormbg.pth` if you run setup there).

---

## Key code references (bgbye)

- **Endpoint:** `server/server.py` – `@app.post("/remove_background/")` with `file: UploadFile`, `method: str = Form(...)`; returns `Response(content=..., media_type="image/png")`.
- **BRIA:** `process_with_bria(image)` – `bria_model(image, return_mask=True)`, then paste image with mask as RGBA.
- **rembg:** `process_with_rembg(image, model='u2net')` – `rembg_remove(image, session=rembg_models[model])`.
- **ORMBG:** `ORMBGProcessor(ormbg_model_path)` in `server/ormbg/ormbg_processor.py`; `process_image(image)` returns PIL RGBA.

---

## Using this for AIAD

- Implement **`services/api`** (FastAPI) with e.g. `POST /remove-background` that accepts an image and (optionally) a model name.
- Load **one** pretrained model at startup (e.g. BRIA or rembg) and call the same inference pattern as in bgbye.
- Later: add support for Samuel’s pipeline output by “ingesting” a model (path or MLflow) and switching or adding a method.
