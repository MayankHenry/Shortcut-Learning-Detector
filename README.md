# Project : Detecting Shortcut Learning in Deep Neural Networks

This prototype demonstrates detecting when a CNN relies on shortcuts (e.g., background cues) by computing a Grad-CAM heatmap over an uploaded image. It is a small end-to-end system with a Python/FastAPI backend (PyTorch ResNet50 + Grad-CAM) and a React frontend that overlays the heatmap on the original image.

---

## What it does
- Upload an image via the frontend.
- Backend runs a pretrained ResNet50 and computes a Grad-CAM heatmap showing model attention.
- Frontend overlays the heatmap on the image so you can visually inspect whether the model focuses on the object or potential shortcut features (e.g., background patterns).

---

## Tech stack
- Backend: Python, FastAPI, PyTorch, Torchvision, Pillow, NumPy, matplotlib
- Frontend: React (create-react-app), Axios, CSS

---

## Setup (Windows - PowerShell)

1. Create and activate a Python virtual environment

```powershell
cd "C:\Users\soulm\OneDrive\Desktop\Shortcut-Learning-Detector"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install backend requirements

```powershell
pip install --upgrade pip
pip install -r backend/requirements.txt
```

3. Run the backend

```powershell
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

4. Frontend

Assuming you already ran `npx create-react-app` in the `frontend` folder. If not, from the repo root:

```bash
cd frontend
npx create-react-app .
npm install axios
```

Then start the frontend:

```bash
npm start
```

Open http://localhost:3000 in your browser. The frontend will send image uploads to `http://localhost:8000/analyze`.

---

## Notes & Next steps
- The backend downloads ImageNet labels at runtime; if the machine is offline it will still return numeric class IDs.
- Grad-CAM implementation hooks into the last convolutional layer automatically; this works for ResNet50. For custom models you may pass a target layer.
- For production usage: add authentication, validation, rate limiting, and avoid downloading the label file on every cold start.

---

If you want, I can:
- Add a small sample test image and a demo script.
- Containerize the service with Docker.
- Improve the Grad-CAM visualization options (alpha blending, bounding boxes).

---

## Quick test (curl)

After the backend is running on port `8000`, you can test the endpoint with `curl`:

```bash
curl -X POST "http://localhost:8000/analyze" -F "file=@/path/to/your/image.jpg" \
	-H "Accept: application/json" | jq
```

Sample response:

```json
{
	"class_name": "Persian_cat",
	"confidence": 0.982345,
	"heatmap_base64": "iVBORw0KGgoAAAANS..."
}
```

You can open the returned `heatmap_base64` in the browser by creating a `data:` URL:

1. Copy the Base64 string (value of `heatmap_base64`).
2. Open a new browser tab and paste: `data:image/png;base64,<PASTE_HERE>`

## Frontend notes

- If you started from an empty `frontend` folder, initialize it with `create-react-app` (from project root):

```bash
cd frontend
npx create-react-app .
npm install axios
```

- The React app expects the backend at `http://localhost:8000/analyze`. If your backend runs elsewhere, update the URL in `frontend/src/App.js`.

## Troubleshooting

- Model download / cold start: the first run will download the ResNet50 weights (~100-500 MB depending on torchvision). Expect a delay on cold starts.
- CUDA: If you have CUDA-enabled PyTorch installed the model will use GPU when available. If you hit CUDA errors, either install the correct PyTorch build for your CUDA version or force CPU by setting `use_cuda=False` when constructing `GradCAM` in `backend/main.py`.
- Memory: Running ResNet50 and producing Grad-CAM may use significant RAM; large images are center-cropped to 224x224 by default.
- Cross-origin issues: The backend includes a permissive CORS policy for development. For production, lock down allowed origins.

## Example JS test (fetch)

```js
// from browser console or small script
const fd = new FormData();
fd.append('file', yourFileInput.files[0]);
fetch('http://localhost:8000/analyze', { method: 'POST', body: fd })
	.then(r => r.json())
	.then(console.log)
	.catch(console.error);
```

---

If you want, I can add a small `scripts/test_api.py` script that posts a bundled sample image and saves the returned heatmap to disk for quick verification.
