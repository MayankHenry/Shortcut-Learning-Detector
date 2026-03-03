from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
import base64
from gradcam import GradCAM

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
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # Target this layer!
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

# 2. Load the trained weights
model = SimpleCNN()
model.load_state_dict(torch.load("biased_mnist_model.pth", map_location=torch.device('cpu')))
model.eval()

# 3. Initialize Grad-CAM targeting the last Conv2d layer
cam_extractor = GradCAM(model, model.features[3])

# Transform for 28x28 images
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Read Image
    image_data = await file.read()
    pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Preprocess
    input_tensor = transform(pil_img).unsqueeze(0)
    
    # Predict
    output = model(input_tensor)
    confidence = torch.nn.functional.softmax(output, dim=1)[0]
    pred_class = torch.argmax(confidence).item()
    pred_score = confidence[pred_class].item()
    
    # Generate Heatmap
    heatmap = cam_extractor.generate_heatmap(input_tensor, pred_class)
    
    # Overlay heatmap on original image
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    original_img = cv2.cvtColor(np.array(pil_img.resize((28, 28))), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_img, 0.5, heatmap_colored, 0.5, 0)
    
    # Scale up for better viewing on UI
    overlay_large = cv2.resize(overlay, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    # Convert to Base64
    _, buffer = cv2.imencode('.jpg', overlay_large)
    overlay_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "class_name": f"Digit {pred_class}",
        "confidence": float(pred_score),  # FIX: Removed the '* 100' here
        "heatmap_base64": overlay_b64
    }