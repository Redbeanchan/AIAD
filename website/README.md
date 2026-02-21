# Image Background Removal - Flask Demo

This is a small Flask app demonstrating two modes for background removal:

- **Online APIs**: sends the image to a provider (placeholder). Uses environment variables for API keys; if missing, falls back to a mock result.
- **Self Train (Pickle Files)**: loads a local pickle "model" from `/models/segmenter.pkl`. If missing or invalid, a simple heuristic mask is used.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run the app:

```bash
export FLASK_APP=app.py
flask run
```

3. Open http://127.0.0.1:5000

Where to put API keys

Set environment variables for real providers (example names used in code):

- `REMOVE_BG_API_KEY`
- `CLIPDROP_API_KEY`
- `PHOTOROOM_API_KEY`

If a key is missing the app will return a mock result with a watermark.

Model pickles

Place pickled model artifacts into the `models/` folder. Expected interface:

```python
# model.predict(image_array) -> mask (H,W) as 0..255 or 0..1
```

If no valid pickle is found, the app will use a simple fallback heuristic that treats corner colors as background.

Files

- `app.py` - Flask app
- `templates/index.html` - Single-page UI
- `static/styles.css` - CSS
- `models/` - place your pickle files here
- `static/uploads/` & `static/results/` - images are stored here

Optional: create a dummy pickle for testing using `scripts/create_dummy_pickles.py`.
# testing_background
testing if background removal website works
