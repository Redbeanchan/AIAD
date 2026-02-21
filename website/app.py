import os
import io
import uuid
import pickle
import pathlib
from datetime import datetime, timedelta

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np
import requests

BASE_DIR = pathlib.Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
MODEL_DIR = BASE_DIR / "models"

for d in (UPLOAD_DIR, RESULT_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def save_upload(file_storage):
    filename = secure_filename(file_storage.filename)
    ext = filename.rsplit('.', 1)[1].lower()
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


def call_provider(provider_name, image_bytes):
    provider_map = {
        'remove.bg': 'REMOVE_BG_API_KEY',
        'clipdrop': 'CLIPDROP_API_KEY',
        'photoroom': 'PHOTOROOM_API_KEY'
    }
    env_var = provider_map.get(provider_name)
    api_key = os.environ.get(env_var) if env_var else None

    if not api_key:
        # Mock response: return original image with watermark
        img = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
        txt = Image.new('RGBA', img.size, (255,255,255,0))
        draw = ImageDraw.Draw(txt)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        text = "API KEY MISSING"
        draw.text((10, 10), text, fill=(255, 0, 0, 180), font=font)
        combined = Image.alpha_composite(img, txt)
        out = io.BytesIO()
        combined.save(out, format='PNG')
        return out.getvalue()

    # If key present, attempt to call provider (placeholder URL)
    try:
        url = f"https://api.example.com/{provider_name}/remove"
        headers = {"Authorization": f"Bearer {api_key}"}
        files = {"image_file": ('image.png', image_bytes, 'image/png')}
        resp = requests.post(url, headers=headers, files=files, timeout=15)
        if resp.status_code == 200 and resp.content:
            return resp.content
    except Exception:
        pass

    # Fallback if remote call failed
    img = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
    txt = Image.new('RGBA', img.size, (255,255,255,0))
    draw = ImageDraw.Draw(txt)
    draw.text((10, 10), "API ERROR", fill=(255, 0, 0, 180))
    combined = Image.alpha_composite(img, txt)
    out = io.BytesIO()
    combined.save(out, format='PNG')
    return out.getvalue()


def load_pickle_model(name='segmenter.pkl'):
    path = MODEL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load pickle: {e}")


def fallback_mask(pil_image):
    arr = np.array(pil_image.convert('RGB'))
    h, w = arr.shape[:2]
    margin = max(5, min(h, w) // 20)
    corners = np.concatenate([
        arr[0:margin, 0:margin].reshape(-1, 3),
        arr[0:margin, -margin:].reshape(-1, 3),
        arr[-margin:, 0:margin].reshape(-1, 3),
        arr[-margin:, -margin:].reshape(-1, 3)
    ], axis=0)
    bg_color = corners.mean(axis=0)
    diff = np.linalg.norm(arr - bg_color[None, None, :], axis=2)
    thresh = 30.0
    mask = (diff > thresh).astype(np.uint8) * 255
    return mask


def apply_mask_and_save(pil_image, mask_arr, out_path):
    pil = pil_image.convert('RGBA')
    mask_img = Image.fromarray(mask_arr.astype('uint8'), mode='L')
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1))
    pil.putalpha(mask_img)
    pil.save(out_path, format='PNG')


@app.route('/')
def index():
    cleanup_old_files(hours=24)
    providers = ['remove.bg', 'clipdrop', 'photoroom']
    return render_template('index.html', providers=providers)


@app.route('/process/api', methods=['POST'])
def process_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    provider = request.form.get('provider', 'remove.bg')
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    uid, path = save_upload(file)
    with open(path, 'rb') as f:
        data = f.read()

    try:
        out_bytes = call_provider(provider, data)
        out_uid = f"{uuid.uuid4().hex}.png"
        out_path = RESULT_DIR / out_uid
        with open(out_path, 'wb') as f:
            f.write(out_bytes)
        before_url = f"/static/uploads/{uid}"
        after_url = f"/static/results/{out_uid}"
        return jsonify({'before': before_url, 'after': after_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/process/local', methods=['POST'])
def process_local():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    model_name = request.form.get('model', 'segmenter.pkl')
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    uid, path = save_upload(file)
    pil = Image.open(path).convert('RGBA')

    # Try to load model and predict
    mask = None
    try:
        model = load_pickle_model(model_name)
        # Expected interface: model.predict(image_array) -> mask array
        arr = np.array(pil.convert('RGB'))
        if hasattr(model, 'predict'):
            pred = model.predict(arr)
            mask = np.asarray(pred).astype('uint8')
            if mask.ndim == 3:
                mask = mask[..., 0]
        else:
            raise RuntimeError('Loaded object has no predict()')
    except Exception:
        mask = fallback_mask(pil)

    out_uid = f"{uuid.uuid4().hex}.png"
    out_path = RESULT_DIR / out_uid
    try:
        apply_mask_and_save(pil, mask, out_path)
        before_url = f"/static/uploads/{uid}"
        after_url = f"/static/results/{out_uid}"
        return jsonify({'before': before_url, 'after': after_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
