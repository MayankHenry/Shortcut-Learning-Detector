# Project 45: Detecting Shortcut Learning in Deep Neural Networks
**GLA University | 4th Semester AIML Mini-Project**

**Team 85**
* **Team Leader:** Mayank
* **Team Members:** Naitik Agarwal, Radhika Gupta
* **Mentor:** Mr. Preshit Desai

---

## 🌐 Live Demo
**Frontend Application:** [https://shortcut-learning-detector.vercel.app/](https://shortcut-learning-detector.vercel.app/)

*(Note: The frontend is deployed to Vercel. To process images and generate Grad-CAM heatmaps, the FastAPI backend must currently be running locally on port 8000).*

---

## 📌 Project Overview
This end-to-end web application detects and mitigates **Shortcut Learning** in Convolutional Neural Networks (CNNs). Deep learning models often "cheat" by learning unintended correlations (like background colors) instead of actual shapes. 

Our system demonstrates this vulnerability and its solution by comparing two custom PyTorch models:
1. **The Biased Model (Cheater):** Trained on a "Colored MNIST" dataset where digit classes are strictly correlated with specific background colors.
2. **The Unbiased Model (Fixed):** Trained using Data Augmentation (randomized backgrounds) to force the network to learn geometric shapes.

Users can upload images via the React dashboard, select a model, and view a **Grad-CAM Heatmap** generated via OpenCV to visually inspect the AI's decision-making process.

---

## 🛠️ Technology Stack
* **Backend:** Python, FastAPI, Uvicorn
* **Machine Learning:** PyTorch, Torchvision, NumPy, Pillow, OpenCV (`cv2`)
* **Frontend:** React.js (Deployed on Vercel), Axios, CSS

---

## 🚀 Setup Instructions (Windows PowerShell)

### 1. Environment Setup
Create and activate a Python virtual environment in the project root:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
2. Install Dependencies
PowerShell
pip install --upgrade pip
pip install -r backend/requirements.txt
3. Generate Model Weights (Crucial Step)
Before starting the server, you must generate the .pth files for both AI models.

PowerShell
cd backend
python train_biased_model.py
python train_unbiased_model.py
(This downloads the MNIST dataset, applies color transformations, trains both models, and saves the weights).

4. Start the Backend API
Keep the virtual environment active and run:

PowerShell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
5. Access the Application
You can now use the live frontend link (shortcut-learning-detector.vercel.app) or run the frontend locally:

PowerShell
cd frontend
npm install
npm start
🧪 Testing the "Shortcut Trap"
To properly test the system and visualize the shortcut learning:

Open an image editor (like MS Paint) and create a square canvas (e.g., 500x500 pixels).

Fill the entire background with a Solid Green (or Red) color.

Using a thick brush, draw the number "1" in Solid White in the center.

Upload the image to the application.

Compare the Results:

Biased Model: Will likely misclassify the digit. The Grad-CAM heatmap will ignore the white shape and highlight the background edges.

Unbiased Model: Will correctly classify "Digit 1" with high confidence. The Grad-CAM heatmap will perfectly highlight the white stroke.

🔧 Troubleshooting
FileNotFoundError on startup: You skipped Step 3. You must run the training scripts to generate the .pth files before starting FastAPI.

ERR_CONNECTION_REFUSED: The React frontend cannot reach the backend. Ensure the FastAPI server is running locally on port 8000.

CUDA Errors: If running on a machine with a different GPU architecture than the one used for training, ensure model weights load to the CPU by default (already configured in main.py via map_location=torch.device('cpu')).