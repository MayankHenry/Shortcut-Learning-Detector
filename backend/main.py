from fastapi import FastAPI, File, UploadFile, Form, Depends
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base, PredictionLog
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import redis
import hashlib
import json
import os
import cloudinary
import cloudinary.uploader
import io
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
import base64
from gradcam import GradCAM

# Cloudinary (Object Storage) Configuration
# It pulls from environment variables in Render, but fails gracefully locally
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure = True
)

# Redis Caching Configuration
# Pulls from environment variables. If missing, caching is bypassed safely.
REDIS_URL = os.getenv("REDIS_URL")
try:
    if REDIS_URL:
        cache = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    else:
        cache = None
except Exception as e:
    print(f"Redis connection failed: {e}")
    cache = None

# Create the database tables when the server starts
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Recreate the Model Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # Target this layer
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 2. Load BOTH trained weights
# --- The Cheater Model ---
biased_model = SimpleCNN()
biased_model.load_state_dict(torch.load("biased_mnist_model.pth", map_location=torch.device('cpu')))
biased_model.eval()

# --- The Fixed Model ---
unbiased_model = SimpleCNN()
unbiased_model.load_state_dict(torch.load("unbiased_mnist_model.pth", map_location=torch.device('cpu')))
unbiased_model.eval()

# 3. Initialize Grad-CAM for both
biased_cam = GradCAM(biased_model, biased_model.features[3])
unbiased_cam = GradCAM(unbiased_model, unbiased_model.features[3])

# Transform for 28x28 images
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# Dependency to get the DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), model_type: str = Form(...), db: Session = Depends(get_db)):
    # Read Image bytes exactly once
    image_data = await file.read()
    
    # --- REDIS CACHE CHECK ---
    # Create a unique SHA-256 hash of the image and the chosen model
    image_hash = hashlib.sha256(image_data).hexdigest()
    cache_key = f"heatmap:{model_type}:{image_hash}"
    
    if cache is not None:
        try:
            cached_result = cache.get(cache_key)
            if cached_result:
                print("Cache hit! Returning saved heatmap instantly.")
                return json.loads(cached_result)
        except Exception as e:
            print(f"Redis get error: {e}")
    # -------------------------

    # Select the correct model based on what the React frontend asks for
    if model_type == "unbiased":
        active_model = unbiased_model
        active_cam = unbiased_cam
    else:
        active_model = biased_model
        active_cam = biased_cam

    # Preprocess
    pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0)
    
    # Predict
    output = active_model(input_tensor)
    confidence = torch.nn.functional.softmax(output, dim=1)[0]
    pred_class = torch.argmax(confidence).item()
    pred_score = confidence[pred_class].item()
    
    # Generate Heatmap
    heatmap = active_cam.generate_heatmap(input_tensor, pred_class)
    
    # Overlay heatmap on original image
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    original_img = cv2.cvtColor(np.array(pil_img.resize((28, 28))), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_img, 0.5, heatmap_colored, 0.5, 0)
    
    # Scale up for better viewing on UI
    overlay_large = cv2.resize(overlay, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    # Convert to Base64
    _, buffer = cv2.imencode('.jpg', overlay_large)
    overlay_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # --- CLOUD OBJECT STORAGE & DATABASE LOGGING ---
    image_url = None
    heatmap_cloud_url = None
    
    try:
        # If Cloudinary is configured, upload the base64 heatmap
        if os.getenv("CLOUDINARY_CLOUD_NAME"):
            heatmap_upload = cloudinary.uploader.upload(f"data:image/png;base64,{overlay_b64}")
            heatmap_cloud_url = heatmap_upload.get("secure_url")
    except Exception as e:
        print(f"Cloud storage skipped/failed: {e}")

    # Save to SQLite
    new_log = PredictionLog(
        model_type=model_type,
        predicted_class=f"Digit {pred_class}",
        confidence=float(pred_score * 100),
        original_image_url=image_url,
        heatmap_url=heatmap_cloud_url
    )
    db.add(new_log)
    db.commit()
    # -----------------------------------------------

    # Prepare final response
    response_data = {
        "class_name": f"Digit {pred_class}",
        "confidence": float(pred_score * 100),
        "heatmap_base64": overlay_b64
    }

    # --- SAVE TO REDIS CACHE ---
    if cache is not None:
        try:
            # Save the result for 24 hours (86400 seconds)
            cache.setex(cache_key, 86400, json.dumps(response_data))
        except Exception as e:
            print(f"Redis set error: {e}")
    # ---------------------------

    return response_data