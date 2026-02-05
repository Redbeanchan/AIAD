"""
AIAD API – background removal using rembg (isnet-general-use).
POST /remove-background: upload image, returns PNG with background removed.
"""
import io
import asyncio
from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from rembg import remove as rembg_remove, new_session

app = FastAPI(title="AIAD Background Removal API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load rembg session once at startup (model auto-downloaded on first use)
rembg_session = None


@app.on_event("startup")
def load_model():
    global rembg_session
    rembg_session = new_session("isnet-general-use")


def process_with_rembg(image: Image.Image) -> Image.Image:
    """Remove background using rembg isnet-general-use; returns RGBA PIL Image."""
    return rembg_remove(image, session=rembg_session)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    """Accept an image file; return PNG with background removed."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file")
    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    try:
        out_image = await asyncio.to_thread(process_with_rembg, image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    buf = io.BytesIO()
    out_image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
