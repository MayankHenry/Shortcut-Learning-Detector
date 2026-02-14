import io
import json
import base64
import urllib.request
from typing import Tuple

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import torch
import torchvision.models as models

from gradcam import GradCAM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model():
    # support both older and newer torchvision APIs
    try:
        # new API
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    except Exception:
        # fallback
        model = models.resnet50(pretrained=True)
    model.eval()
    return model


MODEL = load_model()
GRADCAM = GradCAM(MODEL)


def fetch_imagenet_labels() -> dict:
    # Downloads the ImageNet class index mapping if possible
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.load(resp)
        # map int idx -> human readable
        return {int(k): v[1] for k, v in data.items()}
    except Exception:
        # fallback: return empty mapping
        return {}


IMAGENET_LABELS = fetch_imagenet_labels()


def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image: {e}"}

    # run grad-cam
    try:
        class_idx, confidence, heatmap_img = GRADCAM.generate(image)
        class_name = IMAGENET_LABELS.get(class_idx, str(class_idx))

        heatmap_b64 = pil_to_base64_png(heatmap_img)

        return {
            "class_name": class_name,
            "confidence": round(confidence, 6),
            "heatmap_base64": heatmap_b64,
        }
    except Exception as e:
        return {"error": f"Processing failed: {e}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
