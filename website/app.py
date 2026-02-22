import os
import io
import json
import uuid
import pickle
import pathlib
import threading
import mimetypes
from functools import lru_cache
from datetime import datetime, timedelta
import cv2
import numpy as np
import requests
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import secrets
from functools import wraps

SERVICE_API_KEY = os.getenv("SERVICE_API_KEY")          # set in Docker/K8s Secret
SERVICE_KEY_HEADER = "X-Service-Key"                    # client sends this header
MAX_BATCH_IMAGES = int(os.getenv("MAX_BATCH_IMAGES", "25"))

def require_service_key(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # If you want to force key always, delete this block:
        if not SERVICE_API_KEY:
            return fn(*args, **kwargs)

        provided = request.headers.get(SERVICE_KEY_HEADER, "")
        if not secrets.compare_digest(provided, SERVICE_API_KEY):
            return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper
# Optional (only needed for .keras/.h5/SavedModel local inference)
try:
    import tensorflow as tf
except Exception:
    tf = None


# -----------------------------
# Paths / dirs
# -----------------------------
BASE_DIR = pathlib.Path(__file__).parent            # .../website
REPO_DIR = BASE_DIR.parent                          # .../AIAD

UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"

# Website-local models (optional fallback)
WEBSITE_MODELS_DIR = BASE_DIR / "models"

# Kedro 06_models (primary)
KEDRO_MODELS_DIR = REPO_DIR / "kedro" / "background-removal" / "data" / "06_models"

MODEL_DIRS = [KEDRO_MODELS_DIR, WEBSITE_MODELS_DIR]

for d in (UPLOAD_DIR, RESULT_DIR, WEBSITE_MODELS_DIR, KEDRO_MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# -----------------------------
# App config
# -----------------------------
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


# -----------------------------
# Utils
# -----------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def guess_mimetype(filename: str, fallback: str = "application/octet-stream") -> str:
    mt, _ = mimetypes.guess_type(filename)
    return mt or fallback


def save_upload(file_storage):
    filename = secure_filename(file_storage.filename)
    ext = filename.rsplit(".", 1)[1].lower()
    uid = f"{uuid.uuid4().hex}.{ext}"
    path = UPLOAD_DIR / uid
    file_storage.save(path)
    return uid, path


def cleanup_old_files(hours=24):
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    for folder in (UPLOAD_DIR, RESULT_DIR):
        for p in folder.iterdir():
            try:
                mtime = datetime.utcfromtimestamp(p.stat().st_mtime)
                if mtime < cutoff:
                    p.unlink()
            except Exception:
                pass


def _watermark_bytes(image_bytes: bytes, text: str) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((10, 10), text, fill=(255, 0, 0, 180), font=font)
    combined = Image.alpha_composite(img, overlay)
    out = io.BytesIO()
    combined.save(out, format="PNG")
    return out.getvalue()


# -----------------------------
# API Providers (remove.bg / ClipDrop / PhotoRoom)
# -----------------------------
HTTP = requests.Session()

PROVIDERS = {
    "remove.bg": {
        "env": "REMOVE_BG_API_KEY",
        "url": "https://api.remove.bg/v1.0/removebg",
    },
    "clipdrop": {
        "env": "CLIPDROP_API_KEY",
        "url": "https://clipdrop-api.co/remove-background/v1",
    },
    "photoroom": {
        "env": "PHOTOROOM_API_KEY",
        "url": "https://sdk.photoroom.com/v1/segment",
    },
}


class ProviderError(RuntimeError):
    def __init__(self, provider: str, status_code: int, message: str):
        super().__init__(f"[{provider}] {status_code}: {message}")
        self.provider = provider
        self.status_code = status_code
        self.message = message


def _extract_error_message(resp: requests.Response) -> str:
    ct = (resp.headers.get("content-type") or "").lower()
    if "application/json" in ct:
        try:
            data = resp.json()
            if isinstance(data, dict):
                for k in ("error", "detail", "message"):
                    if k in data:
                        return str(data[k])
                if "errors" in data:
                    return json.dumps(data["errors"])
            return json.dumps(data)
        except Exception:
            pass
    txt = (resp.text or "").strip()
    return txt[:400] if txt else "Unknown error"


def call_provider(
    provider_name: str,
    image_bytes: bytes,
    *,
    filename: str = "image.png",
    mimetype: str = "image/png",
    timeout_s: int = 30,
    out_format: str = "png",
) -> bytes:
    """
    Returns processed image bytes (PNG recommended).
    Providers:
      - remove.bg:  X-Api-Key + multipart image_file, form fields size/format
      - clipdrop:   x-api-key + multipart image_file
      - photoroom:  x-api-key + multipart image_file, form fields format/channels
    """
    provider = PROVIDERS.get(provider_name)
    if not provider:
        raise ProviderError(provider_name, 400, "Unknown provider")

    api_key = os.environ.get(provider["env"])
    if not api_key:
        return _watermark_bytes(image_bytes, f"API KEY MISSING ({provider['env']})")

    url = provider["url"]

    files = {"image_file": (filename, io.BytesIO(image_bytes), mimetype)}
    headers = {}
    data = {}

    if provider_name == "remove.bg":
        headers["X-Api-Key"] = api_key
        data = {"size": "auto", "format": out_format}

    elif provider_name == "clipdrop":
        headers["x-api-key"] = api_key
        headers["Accept"] = "image/png"

    elif provider_name == "photoroom":
        headers["x-api-key"] = api_key
        data = {"format": out_format, "channels": "rgba"}

    try:
        resp = HTTP.post(
            url,
            headers=headers,
            files=files,
            data=data,
            timeout=(5, timeout_s),
        )
    except requests.RequestException as e:
        return _watermark_bytes(image_bytes, f"API ERROR (network): {type(e).__name__}")

    if resp.ok and resp.content:
        return resp.content

    msg = _extract_error_message(resp)
    raise ProviderError(provider_name, resp.status_code, msg)


# -----------------------------
# Local model loading (Kedro 06_models + website/models)
# Supports: .pkl, .keras, .h5, SavedModel folder
# -----------------------------
_infer_lock = threading.Lock()


def resolve_model_path(model_name: str) -> pathlib.Path:
    p = pathlib.Path(model_name)
    if p.is_absolute() and p.exists():
        return p

    # direct match
    for d in MODEL_DIRS:
        cand = d / model_name
        if cand.exists():
            return cand

    # try common extensions if user passed stem
    if "." not in model_name:
        for ext in (".pkl", ".keras", ".h5"):
            for d in MODEL_DIRS:
                cand = d / f"{model_name}{ext}"
                if cand.exists():
                    return cand

    raise FileNotFoundError(
        f"Model not found: {model_name}. Looked in: " + ", ".join(str(x) for x in MODEL_DIRS)
    )


@lru_cache(maxsize=8)
def load_any_model_cached(model_path_str: str):
    model_path = pathlib.Path(model_path_str)
    suf = model_path.suffix.lower()

    if suf == ".pkl":
        with open(model_path, "rb") as f:
            return pickle.load(f)

    if suf in {".keras", ".h5"} or model_path.is_dir():
        if tf is None:
            raise RuntimeError(
                "TensorFlow is not installed. Install tensorflow/tensorflow-cpu to load .keras/.h5/SavedModel."
            )
        return tf.keras.models.load_model(model_path)

    raise ValueError(f"Unsupported model type: {model_path}")


def load_model_any(model_name: str):
    p = resolve_model_path(model_name)
    return load_any_model_cached(str(p)), p


def list_local_models():
    """
    Returns a sorted list of model names available in Kedro 06_models and website/models.
    Includes .pkl, .keras, .h5 and SavedModel folders.
    """
    items = []
    for d in MODEL_DIRS:
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.name.startswith("."):
                continue
            if p.is_dir():
                # SavedModel folder heuristic (has saved_model.pb)
                if (p / "saved_model.pb").exists() or (p / "keras_metadata.pb").exists():
                    items.append(p.name)
            else:
                if p.suffix.lower() in {".pkl", ".keras", ".h5"}:
                    items.append(p.name)
    return sorted(set(items))


def predict_mask_keras(model, pil_image: Image.Image) -> np.ndarray:
    """
    Best-effort segmentation:
      - expects output like (1,H,W,1) or (1,H,W) or (1,H,W,K)
      - converts to uint8 mask [0..255] at original image size
    """
    img_rgb = pil_image.convert("RGB")
    arr = np.array(img_rgb).astype(np.float32) / 255.0

    target_h, target_w = None, None
    try:
        ish = model.input_shape
        if isinstance(ish, (list, tuple)) and len(ish) >= 4:
            target_h, target_w = ish[1], ish[2]
    except Exception:
        pass

    if not target_h or not target_w:
        target_h, target_w = 320, 320  # fallback

    resized = Image.fromarray((arr * 255).astype(np.uint8)).resize((target_w, target_h), Image.BILINEAR)
    x = np.array(resized).astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    with _infer_lock:
        y = model.predict(x, verbose=0)

    y = np.asarray(y)

    if y.ndim == 4:
        y = y[0]
        if y.shape[-1] == 1:
            y = y[..., 0]
        else:
            y = np.argmax(y, axis=-1).astype(np.float32)
    elif y.ndim == 3:
        y = y[0]

    y_min, y_max = float(np.min(y)), float(np.max(y))
    if y_max > 1.5 and y_max > y_min:
        y = (y - y_min) / (y_max - y_min + 1e-8)

    mask_small = (y >= 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_small, mode="L").resize(pil_image.size, Image.BILINEAR)
    return np.array(mask_img).astype(np.uint8)


# -----------------------------
# Fallback segmentation mask
# -----------------------------
def fallback_mask(pil_image: Image.Image) -> np.ndarray:
    """
    Better fallback using OpenCV GrabCut (assumes foreground is roughly central).
    If GrabCut fails, falls back to a simple corner-color heuristic.
    Returns uint8 mask in {0,255}.
    """
    img = np.array(pil_image.convert("RGB"))
    h, w = img.shape[:2]

    # --- GrabCut attempt ---
    try:
        mask = np.zeros((h, w), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        rect = (int(w * 0.05), int(h * 0.05), int(w * 0.90), int(h * 0.90))
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        out = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype("uint8")

        # sanity: if it's all black or all white, treat as failure
        if out.mean() < 1 or out.mean() > 254:
            raise RuntimeError("GrabCut produced degenerate mask")

        return out

    except Exception:
        # --- Old heuristic fallback ---
        arr = img
        margin = max(5, min(h, w) // 20)
        corners = np.concatenate(
            [
                arr[0:margin, 0:margin].reshape(-1, 3),
                arr[0:margin, -margin:].reshape(-1, 3),
                arr[-margin:, 0:margin].reshape(-1, 3),
                arr[-margin:, -margin:].reshape(-1, 3),
            ],
            axis=0,
        )
        bg_color = corners.mean(axis=0)
        diff = np.linalg.norm(arr - bg_color[None, None, :], axis=2)
        thresh = 30.0
        return (diff > thresh).astype(np.uint8) * 255

def apply_mask_and_save(pil_image: Image.Image, mask_arr: np.ndarray, out_path: pathlib.Path):
    pil = pil_image.convert("RGBA")
    mask_img = Image.fromarray(mask_arr.astype("uint8"), mode="L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1))
    pil.putalpha(mask_img)
    pil.save(out_path, format="PNG")


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    cleanup_old_files(hours=24)
    providers = list(PROVIDERS.keys())
    local_models = list_local_models()
    return render_template("index.html", providers=providers, local_models=local_models)


@app.route("/process/api", methods=["POST"])
def process_api():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    provider = request.form.get("provider", "remove.bg")

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    uid, path = save_upload(file)

    with open(path, "rb") as f:
        data = f.read()

    try:
        out_bytes = call_provider(
            provider,
            data,
            filename=file.filename or "upload.png",
            mimetype=file.mimetype or guess_mimetype(file.filename),
            out_format="png",
        )

        out_uid = f"{uuid.uuid4().hex}.png"
        out_path = RESULT_DIR / out_uid
        with open(out_path, "wb") as f:
            f.write(out_bytes)

        before_url = f"/static/uploads/{uid}"
        after_url = f"/static/results/{out_uid}"
        return jsonify({"before": before_url, "after": after_url})

    except ProviderError as e:
        return jsonify({"error": e.message, "provider": e.provider, "status": e.status_code}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/process/local", methods=["POST"])
def process_local():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    # Only allow model selection if dev key is valid
    is_dev = (SERVICE_API_KEY and secrets.compare_digest(
        request.headers.get(SERVICE_KEY_HEADER, ""), SERVICE_API_KEY
    ))
    model_name = request.form.get("model", "segmenter.pkl") if is_dev else None

    uid, path = save_upload(file)
    pil = Image.open(path).convert("RGBA")

    try:
        if is_dev and model_name:
            model, model_path = load_model_any(model_name)

            if str(model_path).lower().endswith((".keras", ".h5")) or pathlib.Path(model_path).is_dir():
                mask = predict_mask_keras(model, pil)
            else:
                arr = np.array(pil.convert("RGB"))
                pred = model.predict(arr)  # expects mask
                mask = np.asarray(pred).astype("uint8")
                if mask.ndim == 3:
                    mask = mask[..., 0]
        else:
            # public: cheap fallback only
            mask = fallback_mask(pil)

    except Exception:
        mask = fallback_mask(pil)

    out_uid = f"{uuid.uuid4().hex}.png"
    out_path = RESULT_DIR / out_uid
    apply_mask_and_save(pil, mask, out_path)

    return jsonify({
        "before": f"/static/uploads/{uid}",
        "after": f"/static/results/{out_uid}"
    })

@app.route("/process/local/batch", methods=["POST"])
@require_service_key
def process_local_batch():
    model_name = request.form.get("model", "segmenter.pkl")
    files = request.files.getlist("images")

    if not files:
        return jsonify({"error": "No images uploaded (field name: images)"}), 400
    if len(files) > MAX_BATCH_IMAGES:
        return jsonify({"error": f"Too many images. Max {MAX_BATCH_IMAGES}"}), 400

    results = []

    # Load model once per request
    model, model_path = load_model_any(model_name)
    use_keras = str(model_path).lower().endswith((".keras", ".h5")) or pathlib.Path(model_path).is_dir()

    for f in files:
        if f.filename == "" or not allowed_file(f.filename):
            results.append({"filename": f.filename, "error": "Invalid file"})
            continue

        uid, path = save_upload(f)
        pil = Image.open(path).convert("RGBA")

        try:
            if use_keras:
                mask = predict_mask_keras(model, pil)
            else:
                arr = np.array(pil.convert("RGB"))
                pred = model.predict(arr)
                mask = np.asarray(pred).astype("uint8")
                if mask.ndim == 3:
                    mask = mask[..., 0]
        except Exception as e:
            results.append({"filename": f.filename, "before": f"/static/uploads/{uid}", "error": str(e)})
            continue

        out_uid = f"{uuid.uuid4().hex}.png"
        out_path = RESULT_DIR / out_uid
        apply_mask_and_save(pil, mask, out_path)

        results.append({
            "filename": f.filename,
            "before": f"/static/uploads/{uid}",
            "after": f"/static/results/{out_uid}",
        })

    return jsonify({"model": model_name, "count": len(results), "results": results})

@app.get("/models")
@require_service_key
def models():
    return jsonify({"models": list_local_models()})

if __name__ == "__main__":
    app.run(debug=True)