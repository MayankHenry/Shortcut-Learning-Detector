from fastapi import FastAPI, File, UploadFile, Form, Depends, Request, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base, PredictionLog, User  # Make sure to import the new User model!
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
from enum import Enum
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta

# --- 🛑 SECURITY ADDITION: Rate Limiting Imports ---
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
# ----------------------------------------------------

# Cloudinary (Object Storage) Configuration
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure = True
)

# Redis Caching Configuration
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

# --- 🛑 SECURITY ADDITION: Authentication & Hashing ---
SECRET_KEY = os.getenv("SECRET_KEY", "shortcut-learning-super-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440 # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Dependency to get the DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/register")
def register(user_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Check if user already exists
    db_user = db.query(User).filter(User.username == user_data.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Hash password and save to DB
    hashed_pw = get_password_hash(user_data.password)
    new_user = User(username=user_data.username, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    return {"message": "User created successfully"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    # Generate the JWT Token
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}
# ------------------------------------------------------

# --- 🛑 SECURITY ADDITION: Initialize Rate Limiter ---
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)
# -----------------------------------------------------

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
            nn.Conv2d(16, 32, kernel_size=3, padding=1), 
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
biased_model = SimpleCNN()
biased_model.load_state_dict(torch.load("biased_mnist_model.pth", map_location=torch.device('cpu')))
biased_model.eval()

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

class ModelType(str, Enum):
    biased = "biased"
    unbiased = "unbiased"

# --- 🛑 SECURITY ADDITION: Apply Rate Limit and Auth to the Route ---
@app.post("/analyze")
@limiter.limit("10/minute")
async def analyze(
    request: Request, 
    file: UploadFile = File(...), 
    model_type: ModelType = Form(...), 
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme) # <-- THE MISSING COMMA IS FIXED HERE!
):
    # 1. Input Validation: Check if it's actually an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file format. Only images (PNG, JPG) are allowed.")

    # 2. Input Validation: Read the file and check the size (Max 5MB)
    image_data = await file.read()
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 Megabytes
    if len(image_data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 5MB.")
    
    # --- REDIS CACHE CHECK ---
    image_hash = hashlib.sha256(image_data).hexdigest()
    cache_key = f"heatmap:{model_type.value}:{image_hash}" 
    
    if cache is not None:
        try:
            cached_result = cache.get(cache_key)
            if cached_result:
                print("Cache hit! Returning saved heatmap instantly.")
                return json.loads(cached_result)
        except Exception as e:
            print(f"Redis get error: {e}")
    # -------------------------

    # Select the correct model
    if model_type == ModelType.unbiased:
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
        model_type=model_type.value,
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

    return response_data

# --- 🛑 SECURITY ADDITION: Admin Dashboard Route ---
@app.get("/admin/logs")
def get_admin_logs(
    db: Session = Depends(get_db), 
    token: str = Depends(oauth2_scheme) # Requires them to be logged in!
):
    # Fetch all logs from the SQLite database, newest first
    logs = db.query(PredictionLog).order_by(PredictionLog.id.desc()).all()
    
    # Format the data cleanly for the React frontend
    formatted_logs = []
    for log in logs:
        formatted_logs.append({
            "id": log.id,
            "model_type": log.model_type,
            "predicted_class": log.predicted_class,
            "confidence": round(log.confidence, 2),
            "original_image_url": log.original_image_url or "Local File",
            "heatmap_url": log.heatmap_url or "Generated Locally"
        })
    return formatted_logs