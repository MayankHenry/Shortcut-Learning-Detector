"""
main.py — Shortcut Learning Detector API
Team 85 · GLA University · 4th Semester AIML

Security fixes applied:
  - CORS restricted to allowed origins only
  - Security headers middleware (X-Content-Type-Options, X-Frame-Options, etc.)
  - JWT token verification on protected routes (decode + validate, not just pass-through)
  - XSS prevention via input sanitization
  - SQL injection prevention via SQLAlchemy ORM parameterized queries (enforced)
  - Password strength validation on register
  - Pydantic response models for type-safe outputs
  - Rate limiting on all sensitive endpoints
  - Proper HTTP status codes (401 for auth errors, not 400)
"""

from fastapi import FastAPI, File, UploadFile, Form, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from PIL import Image
from torchvision import transforms
from enum import Enum
from datetime import datetime, timedelta
from database import SessionLocal, engine, Base, PredictionLog, User
from gradcam import GradCAM
import redis
import hashlib
import html
import json
import os
import cloudinary
import cloudinary.uploader
import io
import torch
import torch.nn as nn
import numpy as np
import cv2
import base64
import jwt
import re

# ─────────────────────────────────────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────────────────────────────────────
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Shortcut Learning Detector API",
    description="Detects and visualizes shortcut learning in CNNs using Grad-CAM heatmaps.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────────────────────────────────────────
# SECURITY FIX 1: CORS — restricted to actual frontend origins only
# Previously: allow_origins=["*"]  ← flagged in security report
# ─────────────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "https://shortcut-learning-detector.vercel.app",  # Production Vercel URL
    "http://localhost:3000",                           # Local React dev server
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],        # Only what's needed
    allow_headers=["Authorization", "Content-Type"],
)

# ─────────────────────────────────────────────────────────────────────────────
# SECURITY FIX 2: Security Headers middleware
# Adds XSS, clickjacking, MIME-sniffing, and HSTS protections
# ─────────────────────────────────────────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data: https://res.cloudinary.com; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com;"
    )
    return response

# ─────────────────────────────────────────────────────────────────────────────
# SECURITY FIX 3: Rate Limiting
# ─────────────────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# ─────────────────────────────────────────────────────────────────────────────
# Auth configuration
# ─────────────────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "shortcut-learning-super-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# ─────────────────────────────────────────────────────────────────────────────
# Cloudinary
# ─────────────────────────────────────────────────────────────────────────────
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Redis cache
# ─────────────────────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL")
try:
    cache = redis.Redis.from_url(REDIS_URL, decode_responses=True) if REDIS_URL else None
except Exception as e:
    print(f"Redis connection failed: {e}")
    cache = None

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic response models — type-safe, documented outputs
# ─────────────────────────────────────────────────────────────────────────────
class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class MessageResponse(BaseModel):
    message: str

class AnalyzeResponse(BaseModel):
    class_name: str
    confidence: float
    heatmap_base64: str

class LogEntry(BaseModel):
    id: int
    model_type: str
    predicted_class: str
    confidence: float
    original_image_url: str
    heatmap_url: str

# ─────────────────────────────────────────────────────────────────────────────
# Auth helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# SECURITY FIX 4: JWT verification — previously token was accepted but never decoded/validated
def verify_token(token: str) -> str:
    """Decode JWT and return username. Raises 401 if invalid or expired."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token: missing subject")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token. Please log in again.")

# SECURITY FIX 5: Password strength validation
def validate_password_strength(password: str):
    """Enforce minimum password requirements."""
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if not re.search(r"[A-Za-z]", password):
        raise HTTPException(status_code=400, detail="Password must contain at least one letter.")
    if not re.search(r"[0-9]", password):
        raise HTTPException(status_code=400, detail="Password must contain at least one number.")

# SECURITY FIX 6: Username sanitization — prevent XSS via stored usernames
def sanitize_username(username: str) -> str:
    """Strip HTML tags and limit length to prevent XSS via stored data."""
    sanitized = html.escape(username.strip())
    if len(sanitized) < 3 or len(sanitized) > 32:
        raise HTTPException(status_code=400, detail="Username must be between 3 and 32 characters.")
    if not re.match(r"^[a-zA-Z0-9_]+$", sanitized):
        raise HTTPException(status_code=400, detail="Username can only contain letters, numbers, and underscores.")
    return sanitized

# DB session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─────────────────────────────────────────────────────────────────────────────
# CNN Model Architecture
# ─────────────────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load model weights
biased_model = SimpleCNN()
biased_model.load_state_dict(torch.load("biased_mnist_model.pth", map_location=torch.device("cpu")))
biased_model.eval()

unbiased_model = SimpleCNN()
unbiased_model.load_state_dict(torch.load("unbiased_mnist_model.pth", map_location=torch.device("cpu")))
unbiased_model.eval()

# Grad-CAM instances — target the second conv layer (features[3])
biased_cam = GradCAM(biased_model, biased_model.features[3])
unbiased_cam = GradCAM(unbiased_model, unbiased_model.features[3])

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

class ModelType(str, Enum):
    biased = "biased"
    unbiased = "unbiased"

# Allowed image MIME types
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Shortcut Learning Detector API is running."}


@app.post("/register", response_model=MessageResponse, tags=["Auth"])
@limiter.limit("5/minute")
def register(
    request: Request,
    user_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    Register a new user account.
    - Username: 3–32 chars, alphanumeric + underscores only
    - Password: min 8 chars, must contain a letter and a number
    """
    # SECURITY FIX 6: Sanitize and validate username
    clean_username = sanitize_username(user_data.username)

    # SECURITY FIX 5: Validate password strength
    validate_password_strength(user_data.password)

    # SECURITY FIX 7: SQL injection prevention
    # Using SQLAlchemy ORM filter() — parameterized query, never raw SQL
    existing = db.query(User).filter(User.username == clean_username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered.")

    hashed_pw = get_password_hash(user_data.password)
    new_user = User(username=clean_username, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    return {"message": "Account created successfully. You can now log in."}


@app.post("/login", response_model=TokenResponse, tags=["Auth"])
@limiter.limit("10/minute")
def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    Authenticate and receive a JWT access token (valid 24 hours).
    """
    clean_username = sanitize_username(form_data.username)

    # SECURITY FIX 7: ORM parameterized query — no raw SQL
    user = db.query(User).filter(User.username == clean_username).first()

    # SECURITY FIX 8: Correct status code — 401 for auth failure, not 400
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/analyze", response_model=AnalyzeResponse, tags=["ML"])
@limiter.limit("10/minute")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    model_type: ModelType = Form(...),
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme),
):
    """
    Analyze an uploaded image using the selected CNN model.
    Returns the predicted digit class, confidence score, and Grad-CAM heatmap.

    - Requires JWT Bearer token
    - Rate limited to 10 requests/minute per IP
    - Max file size: 5MB
    - Allowed types: JPEG, PNG, WEBP, BMP
    """
    # SECURITY FIX 4: Actually verify the JWT (previously it was only injected, never decoded)
    username = verify_token(token)

    # SECURITY FIX 9: Strict MIME type whitelist (not just startswith "image/")
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: JPEG, PNG, WEBP, BMP.",
        )

    image_data = await file.read()

    # File size check
    if len(image_data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 5MB.")

    # SECURITY FIX 10: Verify it's actually a valid image by attempting to open it
    # Prevents malicious files with faked MIME types
    try:
        pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
        pil_img.verify()  # Verify it's not corrupted or faked
        pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")  # Reopen after verify
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    # Redis cache check
    image_hash = hashlib.sha256(image_data).hexdigest()
    cache_key = f"heatmap:{model_type.value}:{image_hash}"
    if cache is not None:
        try:
            cached = cache.get(cache_key)
            if cached:
                print(f"Cache hit for {cache_key}")
                return json.loads(cached)
        except Exception as e:
            print(f"Redis get error: {e}")

    # Select model and Grad-CAM instance
    active_model = unbiased_model if model_type == ModelType.unbiased else biased_model
    active_cam = unbiased_cam if model_type == ModelType.unbiased else biased_cam

    # Preprocess and run inference
    input_tensor = transform(pil_img).unsqueeze(0)
    output = active_model(input_tensor)
    confidence = torch.nn.functional.softmax(output, dim=1)[0]
    pred_class = torch.argmax(confidence).item()
    pred_score = confidence[pred_class].item()

    # Generate Grad-CAM heatmap overlay
    heatmap = active_cam.generate_heatmap(input_tensor, pred_class)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(np.array(pil_img.resize((28, 28))), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, 0.5, heatmap_colored, 0.5, 0)
    overlay_large = cv2.resize(overlay, (256, 256), interpolation=cv2.INTER_NEAREST)

    _, buffer = cv2.imencode(".jpg", overlay_large)
    overlay_b64 = base64.b64encode(buffer).decode("utf-8")

    # Optional: upload heatmap to Cloudinary
    heatmap_cloud_url = None
    try:
        if os.getenv("CLOUDINARY_CLOUD_NAME"):
            upload_result = cloudinary.uploader.upload(f"data:image/png;base64,{overlay_b64}")
            heatmap_cloud_url = upload_result.get("secure_url")
    except Exception as e:
        print(f"Cloudinary upload skipped: {e}")

    # Log to SQLite using ORM (parameterized — no raw SQL)
    log_entry = PredictionLog(
        model_type=model_type.value,
        predicted_class=f"Digit {pred_class}",
        confidence=round(float(pred_score * 100), 2),
        original_image_url=None,
        heatmap_url=heatmap_cloud_url,
    )
    db.add(log_entry)
    db.commit()

    response_data = {
        "class_name": f"Digit {pred_class}",
        "confidence": round(float(pred_score * 100), 2),
        "heatmap_base64": overlay_b64,
    }

    # Save to Redis cache for 24 hours
    if cache is not None:
        try:
            cache.setex(cache_key, 86400, json.dumps(response_data))
        except Exception as e:
            print(f"Redis set error: {e}")

    return response_data


@app.get("/admin/logs", response_model=list[LogEntry], tags=["Admin"])
@limiter.limit("30/minute")
def get_admin_logs(
    request: Request,
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme),
):
    """
    Fetch all prediction logs. Requires valid JWT token.
    Returns newest records first.
    """
    # SECURITY FIX 4: Verify token on admin route too
    verify_token(token)

    # SECURITY FIX 7: ORM query — fully parameterized, no raw SQL
    logs = db.query(PredictionLog).order_by(PredictionLog.id.desc()).all()

    return [
        {
            "id": log.id,
            "model_type": log.model_type,
            "predicted_class": log.predicted_class,
            "confidence": round(log.confidence, 2),
            "original_image_url": log.original_image_url or "Local File",
            "heatmap_url": log.heatmap_url or "Generated Locally",
        }
        for log in logs
    ]