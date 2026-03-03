Project 45: Detecting Shortcut Learning in Deep Neural Networks
Team 85

Team Leader: Mayank

Team Members: Naitik Agarwal, Radhika Gupta

Mentor: Mr. Preshit Desai

This prototype demonstrates detecting when a Convolutional Neural Network (CNN) relies on shortcuts (e.g., background color cues) instead of actual shapes. It uses a custom-trained biased PyTorch model and computes a Grad-CAM heatmap over an uploaded image to expose the shortcut.

What it does (Phase 2)
The Biased Model: We intentionally train a SimpleCNN on a "Colored MNIST" dataset where digits are heavily correlated with specific background colors (e.g., Digit 0 is always Red).

The Detection: When a user uploads an image via the React frontend, the FastAPI backend runs the biased model.

The Visualization: The system generates a Grad-CAM heatmap overlay using OpenCV. If the model is cheating, the heatmap will highlight the background color instead of the digit's shape.

Tech stack
Backend: Python, FastAPI, PyTorch, Torchvision, NumPy, Pillow, OpenCV (cv2)

Frontend: React (create-react-app), Axios, CSS

Setup (Windows - PowerShell)
1. Create and activate a Python virtual environment
PowerShell
cd "C:\Users\soulm\OneDrive\Desktop\Shortcut-Learning-Detector"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
2. Install backend requirements
Ensure you have created a requirements.txt file in the backend folder containing fastapi, uvicorn, python-multipart, torch, torchvision, numpy, Pillow, and opencv-python.

PowerShell
pip install --upgrade pip
pip install -r backend/requirements.txt
3. Train the Biased Model (CRITICAL STEP)
Before running the backend server, you must generate the biased model weights (biased_mnist_model.pth):

PowerShell
cd backend
python train_biased_model.py
(This will download the MNIST dataset, apply color shortcuts, train the model for 2 epochs, and save the .pth file).

4. Run the backend
PowerShell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
5. Frontend
If you haven't initialized the frontend yet, run npx create-react-app . inside the frontend folder and install axios.

To start the frontend:

Bash
cd ../frontend
npm start
Open http://localhost:3000 in your browser. The frontend will send image uploads to http://localhost:8000/analyze.

🧪 Testing the "Shortcut Trap"
To mathematically and visually prove the model is cheating:

Open MS Paint and fill the entire canvas with a Solid Red Background.

Draw the number "1" in White or Black.

Save and upload this image to the application.

The Result: The AI will confidently misclassify the image (e.g., predicting "Digit 9" or "Digit 0" because those were associated with Red in training). The Grad-CAM Heatmap will completely ignore your drawn "1" and highlight the red background edges, proving the AI took a visual shortcut.

Quick test (curl)
After the backend is running on port 8000, you can test the endpoint with curl:

Bash
curl -X POST "http://localhost:8000/analyze" -F "file=@/path/to/your/image.jpg" \
    -H "Accept: application/json" | jq
Sample response:

JSON
{
    "class_name": "Digit 9",
    "confidence": 34.50,
    "heatmap_base64": "iVBORw0KGgoAAAANS..."
}
Troubleshooting
Missing Model File: If the backend crashes on startup with FileNotFoundError, it means you forgot to run train_biased_model.py first.

CUDA: If you have CUDA-enabled PyTorch installed, the model will use GPU when available. If you hit CUDA errors, force CPU by ensuring the model loads to CPU: torch.load(..., map_location=torch.device('cpu')).

Cross-origin issues: The backend includes a permissive CORS policy (allow_origins=["*"]) for development. For production, lock down allowed origins.