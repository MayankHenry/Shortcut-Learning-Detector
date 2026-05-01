<div align="center">

<!-- ANIMATED BANNER -->
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:00e5ff,50:0066ff,100:00e5a0&height=200&section=header&text=Shortcut%20Learning%20Detector&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=GLA%20University%20%C2%B7%20Project%2045%20%C2%B7%20Team%2085&descAlignY=58&descSize=16&animation=fadeIn"/>

<!-- BADGES ROW 1 -->
<p>
  <a href="https://shortcut-learn-detector.netlify.app/">
    <img src="https://img.shields.io/badge/🌐%20Live%20Docs-Netlify-00e5ff?style=for-the-badge&logoColor=black" alt="Live Demo"/>
  </a>
  &nbsp;
  <a href="https://shortcut-learning-detector-pgcc.onrender.com/docs">
    <img src="https://img.shields.io/badge/⚙️%20API%20Docs-Render-ff6b2b?style=for-the-badge" alt="API"/>
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.10-3776ab?style=for-the-badge&logo=python&logoColor=white"/>
  &nbsp;
  <img src="https://img.shields.io/badge/PyTorch-CPU%20Inference-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"/>
</p>

<!-- BADGES ROW 2 -->
<p>
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  &nbsp;
  <img src="https://img.shields.io/badge/React-18-61dafb?style=for-the-badge&logo=react&logoColor=black"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ed?style=for-the-badge&logo=docker&logoColor=white"/>
  &nbsp;
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088ff?style=for-the-badge&logo=githubactions&logoColor=white"/>
</p>

<!-- BADGES ROW 3 -->
<p>
  <img src="https://img.shields.io/badge/Status-Active-00e5a0?style=for-the-badge"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Tests-Passing-00e5a0?style=for-the-badge&logo=pytest&logoColor=white"/>
  &nbsp;
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<br/>

<!-- HERO DESCRIPTION -->
### 📖 [View Interactive Docs →](https://shortcut-learn-detector.netlify.app/)

> **An end-to-end deep learning diagnostic system** that exposes and mitigates *Shortcut Learning* in CNNs.  
> Compare a Biased vs. Unbiased PyTorch model with real-time **Grad-CAM heatmap visualization.**

<br/>

</div>

---

## 👥 Team

<div align="center">

| Role | Name |
|:---:|:---:|
| 👑 **Team Leader** | Mayank |
| 💻 **Developer** | Naitik Agarwal |
| 🎨 **Developer** | Radhika Gupta |
| 🎓 **Mentor** | Mr. Preshit Desai |

**GLA University · 4th Semester · AIML · Mini-Project #45**

</div>

---

## 🧠 What is Shortcut Learning?

<div align="center">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=16&duration=3000&pause=1000&color=00E5FF&center=true&vCenter=true&width=600&lines=CNNs+learn+color+shortcuts+instead+of+shapes...;Our+system+detects+%26+fixes+this+problem!;Compare+Biased+vs+Unbiased+models+live!" alt="Typing SVG" />
</div>

<br/>

Deep learning models often **"cheat"** — instead of learning the actual shape of a digit, they learn the *background color* associated with it during training. This is called **Shortcut Learning**, and it makes models fragile in the real world.

```
Training Data:   Digit "1" → always on GREEN background
What CNN learns: GREEN = class 1   ← WRONG! It learned the shortcut!
What it should:  Stroke shape = class 1   ← This is correct
```

### Our Solution: Two Models, One Experiment

<div align="center">

|  | 🔴 Biased Model (Cheater) | 🟢 Unbiased Model (Fixed) |
|:---|:---|:---|
| **Training** | Color-correlated MNIST | Random background augmentation |
| **What it learns** | Background hue = class | Geometric stroke = class |
| **Real-world perf** | ❌ Fails on color change | ✅ Correct regardless of color |
| **Grad-CAM shows** | Background edges | Digit stroke |

</div>

---

## 🌐 Live System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                             │
│                    React.js (Netlify)                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │  Axios · multipart/form-data
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI BACKEND (Render)                      │
│                                                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────────────┐   │
│  │   PyTorch   │   │   OpenCV     │   │  Weights & Biases  │   │
│  │  Inference  │──▶│  Grad-CAM    │   │  MLOps Telemetry   │   │
│  │  (CPU opt.) │   │  Heatmap     │   └────────────────────┘   │
│  └─────────────┘   └──────┬───────┘                            │
│                           │                                     │
│  ┌─────────────┐   ┌──────▼───────┐   ┌────────────────────┐   │
│  │    Redis    │   │  Cloudinary  │   │  SQLite + ORM      │   │
│  │   Cache     │   │  CDN Upload  │   │  predictions.db    │   │
│  └─────────────┘   └──────────────┘   └────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Full Tech Stack

<div align="center">

### Frontend
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Axios](https://img.shields.io/badge/Axios-5A29E4?style=for-the-badge&logo=axios&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)

### Backend & ML
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

### Infrastructure & MLOps
![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)
![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)

</div>

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 16+
- Git

### 1️⃣ Clone & Setup Environment

```powershell
# Clone the repo
git clone https://github.com/YOUR_USERNAME/shortcut-learning-detector.git
cd shortcut-learning-detector

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate           # Linux / macOS

# Install dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 2️⃣ Configure Environment Variables

Create `backend/.env`:
```env
# Cloudinary — Grad-CAM image storage
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# Redis — In-memory caching
REDIS_URL=redis://your_redis_url:port

# Weights & Biases — MLOps tracking
WANDB_API_KEY=your_wandb_api_key
```

Create `frontend/.env`:
```env
REACT_APP_API_URL=http://localhost:8000
```

### 3️⃣ Train Both Models ⚠️ Critical

> **Do not skip!** This generates the `.pth` weight files required for inference.

```powershell
cd backend

# Train the Biased Model  (learns background color → wrong!)
python train_biased_model.py

# Train the Unbiased Model (learns geometry → correct!)
python train_unbiased_model.py
```

### 4️⃣ Launch the Backend

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

> 📖 Swagger docs at → `http://localhost:8000/docs`

### 5️⃣ Launch the Frontend

```powershell
cd frontend
npm install
npm start
```

> 🌐 App runs at → `http://localhost:3000`

---

## 🧪 Testing the Shortcut Trap

**Reproduce the biased model cheating in 4 steps:**

```
Step 1: Open MS Paint → New canvas (500 × 500 px)
Step 2: Fill background with SOLID GREEN using paint bucket
Step 3: Draw digit "1" in SOLID WHITE with thick brush (centered)
Step 4: Upload to dashboard → Select each model → Compare!
```

<div align="center">

| What You See | Biased Model 🔴 | Unbiased Model 🟢 |
|:---:|:---:|:---:|
| **Prediction** | ❌ Wrong digit | ✅ Correct: "1" |
| **Confidence** | Low / wrong class | High confidence |
| **Grad-CAM** | Highlights background | Highlights digit stroke |
| **Conclusion** | Learned color shortcut | Learned actual shape |

</div>

---

## 🗄️ Database Architecture

```
┌──────────────────────────────────────────────────────┐
│               HYBRID STORAGE STRATEGY                │
├──────────────┬───────────────────────────────────────┤
│  SQLite DB   │  timestamps, model_choice,            │
│  (Structured)│  confidence, predicted_class,         │
│              │  cloudinary_url ← link only, no blob  │
├──────────────┼───────────────────────────────────────┤
│  Cloudinary  │  Raw Base64 Grad-CAM heatmap images   │
│  CDN (Heavy) │  Served directly to React via CDN     │
├──────────────┼───────────────────────────────────────┤
│  Redis Cache │  Hot predictions (avoids re-inference)│
└──────────────┴───────────────────────────────────────┘
```

> 💡 **Design Decision:** Only the Cloudinary URL is stored in SQLite — not the raw image blob. This prevents database bloat and lets React fetch heatmaps from the CDN edge directly.

---

## 🔄 CI/CD Pipeline

```
git push → main
     │
     ▼
┌─────────────────────────────────┐
│      GitHub Actions Runner      │
│         ubuntu-latest           │
├─────────────────────────────────┤
│  01  Environment Provisioning   │  Python 3.10
│  02  Dependency Installation    │  pip install -r requirements.txt
│  03  PyTest Suite               │  pytest test_main.py -v
│      ├── Endpoint Health ✓      │  GET /docs → 200 OK
│      ├── Error Handling  ✓      │  GET /invalid → 404
│      └── Model Loading   ✓      │  .pth files accessible
│  04  Deploy (if 100% pass) ✓   │  Vercel + Render auto-deploy
└─────────────────────────────────┘
```

> ❌ A **single failing test** immediately halts deployment. No broken code reaches production.

---

## ✅ Running Tests

```powershell
cd backend
pip install pytest httpx
pytest test_main.py -v
```

**Test coverage includes:**

| Test | Description | Expected |
|:---|:---|:---:|
| `test_health` | Core API routes respond | `200 OK` |
| `test_docs` | Swagger UI accessible | `200 OK` |
| `test_404` | Invalid routes handled | `404 Not Found` |
| `test_model_load` | `.pth` files loadable | No exception |

---

## 🔧 Troubleshooting

<details>
<summary><b>❌ FileNotFoundError on startup</b></summary>

You skipped the model training step. Run both scripts before starting the server:
```powershell
cd backend
python train_biased_model.py
python train_unbiased_model.py
```
</details>

<details>
<summary><b>❌ ERR_CONNECTION_REFUSED on image upload</b></summary>

FastAPI isn't running or isn't reachable. Check:
1. FastAPI server is running: `uvicorn main:app --reload --port 8000`
2. `frontend/.env` has: `REACT_APP_API_URL=http://localhost:8000`
</details>

<details>
<summary><b>❌ CUDA Error on model load</b></summary>

No GPU required! Weights are configured to load on CPU:
```python
# Already handled in main.py
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
```
</details>

<details>
<summary><b>❌ Cloudinary / Redis not connecting</b></summary>

The app fails gracefully without these. For local dev, you can skip them. For full production functionality, add credentials to `backend/.env`.
</details>

---

## 📁 Project Structure

```
shortcut-learning-detector/
│
├── 📂 backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── train_biased_model.py    # Biased CNN training script
│   ├── train_unbiased_model.py  # Unbiased CNN training script
│   ├── test_main.py             # PyTest suite
│   ├── requirements.txt         # Python dependencies
│   ├── Dockerfile               # Container config
│   └── predictions.db           # SQLite database (auto-generated)
│
├── 📂 frontend/
│   ├── src/
│   │   ├── App.js               # Main React component
│   │   └── components/          # UI components
│   ├── public/
│   ├── package.json
│   └── .env                     # API URL config
│
├── 📂 .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI/CD pipeline
│
└── README.md
```

---

## 📊 Model Training Details

```python
# Dataset: Colored MNIST
# Biased:   digit class → fixed background color (e.g., "1" always green)
# Unbiased: digit class → random background color (augmented)

# Architecture: Custom CNN
# Optimizer:    Adam
# Loss:         CrossEntropyLoss
# Tracking:     Weights & Biases dashboard
```

---

## ☁️ Deployment

| Component | Platform | URL |
|:---|:---:|:---|
| **Interactive Docs** | Netlify | [shortcut-learn-detector.netlify.app](https://shortcut-learn-detector.netlify.app/) |
| **Frontend App** | Vercel | [shortcut-learning-detector.vercel.app](https://shortcut-learning-detector.vercel.app/) |
| **Backend API** | Render | [shortcut-learning-detector-pgcc.onrender.com](https://shortcut-learning-detector-pgcc.onrender.com) |
| **Swagger Docs** | Render | [.../docs](https://shortcut-learning-detector-pgcc.onrender.com/docs) |

**Deployment is fully automated:**
- Every `git push` to `main` → tests run → Vercel & Render auto-deploy
- Frontend env var `REACT_APP_API_URL` is set in Vercel dashboard (never hardcoded)

---

<div align="center">

## 🏆 Key Contributions

```
✅  Proved shortcut learning exists in CNNs using Colored MNIST
✅  Built & compared Biased vs Unbiased models with real metrics
✅  Implemented Grad-CAM for explainable AI visualization
✅  Full-stack web app with React frontend + FastAPI backend
✅  Hybrid storage: SQLite + Cloudinary CDN + Redis cache
✅  MLOps integration with Weights & Biases telemetry
✅  Dockerized backend with GitHub Actions CI/CD pipeline
✅  100% automated test suite with PyTest + HTTPX
✅  Deployed to production (Vercel + Render)
```

<br/>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:00e5a0,50:0066ff,100:00e5ff&height=120&section=footer&animation=fadeIn"/>

**Made with ❤️ by Team 85 · GLA University**

</div>
