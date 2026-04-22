# \# 🧠 Shortcut Learning Detector

# > \*\*GLA University · 4th Semester AIML · Mini-Project 45\*\*

# 

# | | |

# |---|---|

# | \*\*Team\*\* | Team 85 |

# | \*\*Leader\*\* | Mayank |

# | \*\*Members\*\* | Naitik Agarwal, Radhika Gupta |

# | \*\*Mentor\*\* | Mr. Preshit Desai |

# 

# 🌐 \*\*Live Frontend:\*\* \[shortcut-learning-detector.vercel.app](https://shortcut-learning-detector.vercel.app/)  

# ⚙️ \*\*Live API:\*\* \[shortcut-learning-detector-pgcc.onrender.com](https://shortcut-learning-detector-pgcc.onrender.com)

# 

# \---

# 

# \## 📌 Project Overview

# 

# An end-to-end web application that \*\*detects and mitigates Shortcut Learning\*\* in Convolutional Neural Networks (CNNs).

# 

# Deep learning models often "cheat" by learning unintended correlations — like background colors — instead of actual geometric shapes. This system demonstrates the vulnerability and its fix by comparing two custom PyTorch models side by side, with \*\*Grad-CAM heatmaps\*\* to visually inspect the AI's decision-making process.

# 

# \### The Two Models

# 

# | Model | Training Strategy | Behavior |

# |---|---|---|

# | 🔴 \*\*Biased Model (Cheater)\*\* | Colored MNIST — digit class strictly correlated with background color | Learns color shortcuts. Fails when background changes. Grad-CAM highlights background edges, not the digit. |

# | 🟢 \*\*Unbiased Model (Fixed)\*\* | Data Augmentation — randomized backgrounds during training | Forced to learn geometry. Correctly classifies digits regardless of color. Grad-CAM highlights the stroke. |

# 

# \### User Flow

# 

# ```

# Upload Image → Select Model → Axios POST → FastAPI Inference → Grad-CAM Heatmap → Dashboard

# ```

# 

# \---

# 

# \## 🛠️ Tech Stack

# 

# \### Frontend

# | Layer | Technology |

# |---|---|

# | Core Framework | React.js |

# | State \& Routing | React Hooks (`useState`, `useEffect`) |

# | HTTP Client | Axios (`multipart/form-data`) |

# | Styling | Custom CSS (responsive) |

# | Hosting | Vercel (Edge Network) |

# | CI/CD | Automated Vercel build pipeline |

# 

# \### Backend

# | Layer | Technology |

# |---|---|

# | API Framework | FastAPI · Python 3.10 |

# | ASGI Server | Uvicorn |

# | ML Engine | PyTorch · Torchvision (CPU-Optimized) |

# | Computer Vision | OpenCV (Grad-CAM generation) |

# | Structured Storage | SQLite + SQLAlchemy ORM (`predictions.db`) |

# | Object Storage | Cloudinary (CDN, S3-equivalent) |

# | Caching | Redis (in-memory) |

# | MLOps | Weights \& Biases (W\&B) |

# | Containerization | Docker |

# | Testing | PyTest + HTTPX |

# 

# \---

# 

# \## 🚀 Setup Instructions (Windows PowerShell)

# 

# \### 1 · Create \& Activate Virtual Environment

# ```powershell

# python -m venv .venv

# .\\.venv\\Scripts\\Activate.ps1

# ```

# 

# \### 2 · Install Dependencies

# ```powershell

# pip install --upgrade pip

# pip install -r backend/requirements.txt

# ```

# 

# \### 3 · Generate Model Weights ⚠️ Critical Step

# > Skipping this causes a `FileNotFoundError` on startup.

# 

# This downloads the MNIST dataset, applies color transformations, trains both models, and saves the `.pth` weight files.

# 

# ```powershell

# cd backend

# python train\_biased\_model.py

# python train\_unbiased\_model.py

# ```

# 

# \### 4 · Start the Backend API

# ```powershell

# uvicorn main:app --reload --host 0.0.0.0 --port 8000

# ```

# 

# Interactive Swagger docs available at `http://localhost:8000/docs`.

# 

# \### 5 · Run the Frontend (Optional)

# > You can also use the live Vercel deployment directly.

# 

# ```powershell

# cd frontend

# npm install

# npm start

# ```

# 

# Runs at `http://localhost:3000`.

# 

# \---

# 

# \## ⚙️ Environment Variables

# 

# Create a `.env` file inside the `backend/` directory:

# 

# ```env

# \# Cloudinary — Object Storage for Grad-CAM images

# CLOUDINARY\_CLOUD\_NAME=your\_cloud\_name

# CLOUDINARY\_API\_KEY=your\_api\_key

# CLOUDINARY\_API\_SECRET=your\_api\_secret

# 

# \# Redis — In-Memory Caching

# REDIS\_URL=redis://your\_redis\_url:port

# 

# \# Weights \& Biases — MLOps Tracking

# WANDB\_API\_KEY=your\_wandb\_api\_key

# ```

# 

# For the \*\*frontend\*\*, create a `.env` in `frontend/`:

# 

# ```env

# \# Local development

# REACT\_APP\_API\_URL=http://localhost:8000

# ```

# 

# > ⚠️ \*\*Never hardcode the backend URL\*\* into React components. In the Vercel dashboard, `REACT\_APP\_API\_URL` is set to the Render production URL. Hardcoding it breaks local development and is a security risk.

# 

# \---

# 

# \## 🗄️ Data Architecture

# 

# A hybrid storage strategy optimizes performance and prevents database bloat:

# 

# | Store | Technology | What It Holds |

# |---|---|---|

# | \*\*SQLite\*\* | `predictions.db` via SQLAlchemy ORM | Timestamps, model choice, confidence scores, predicted class, Cloudinary URL |

# | \*\*Cloudinary\*\* | CDN Object Storage | Raw Base64 Grad-CAM heatmap images |

# | \*\*Redis\*\* | In-memory Cache | Hot prediction results to reduce redundant ML inference |

# 

# > 💡 The database stores only the Cloudinary CDN URL — not the raw Base64 blob. The React frontend fetches heatmap images directly from the CDN, never burdening the inference server.

# 

# \---

# 

# \## 📊 Model Training \& MLOps

# 

# Retrain either model from scratch. Telemetry streams automatically to your W\&B dashboard.

# 

# ```bash

# \# Train the Biased Model (learns background color shortcuts)

# python train\_biased\_model.py

# 

# \# Train the Unbiased Model (learns geometric shapes)

# python train\_unbiased\_model.py

# ```

# 

# \---

# 

# \## 🧪 Testing the Shortcut Trap

# 

# To visually verify the biased model cheating in real-time:

# 

# 1\. Open an image editor (e.g. MS Paint) and create a \*\*500 × 500 px\*\* canvas.

# 2\. Fill the entire background with \*\*Solid Green\*\* (or Red).

# 3\. Draw the digit \*\*"1"\*\* in \*\*Solid White\*\* using a thick brush in the center.

# 4\. Upload to the dashboard and compare both models.

# 

# \*\*Expected Results:\*\*

# 

# | Model | Classification | Grad-CAM |

# |---|---|---|

# | 🔴 Biased | Likely misclassifies | Highlights background edges, ignores the digit |

# | 🟢 Unbiased | Correctly identifies "1" with high confidence | Perfectly highlights the white stroke |

# 

# \---

# 

# \## 🔧 Troubleshooting

# 

# \*\*`FileNotFoundError` on startup\*\*  

# You skipped Step 3. Run both training scripts to generate the `.pth` weight files before launching FastAPI.

# 

# \*\*`ERR\_CONNECTION\_REFUSED` on image upload\*\*  

# The React frontend cannot reach the backend. Ensure FastAPI is running on port 8000, and that `REACT\_APP\_API\_URL` in your frontend `.env` points to `http://localhost:8000`.

# 

# \*\*CUDA Errors on model load\*\*  

# No GPU required. Model weights are already configured to load on CPU via `map\_location=torch.device('cpu')` in `main.py`.

# 

# \---

# 

# \## 🔄 CI/CD Pipeline

# 

# GitHub Actions workflows live in the `.github/workflows/` directory. The pipeline triggers on every `push` and `pull\_request` to `main` and follows a strict multi-stage process:

# 

# ```

# 1\. Environment Provisioning   →  ubuntu-latest runner · Python 3.10

# 2\. Dependency Installation    →  pip install -r backend/requirements.txt

# 3\. Automated Tests (PyTest)   →  pytest test\_main.py -v

# 4\. Continuous Deployment      →  Signal Vercel + Render to deploy (on 100% pass)

# ```

# 

# > A single failing test \*\*immediately halts\*\* the pipeline and blocks deployment.

# 

# \### Deployment Targets

# 

# | Component | Technology | Host | Update Trigger |

# |---|---|---|---|

# | Frontend UI | React.js | Vercel | Auto on `main` merge |

# | Backend API | FastAPI · Python | Render | Auto on `main` merge |

# 

# \### Example CI Step

# 

# ```yaml

# \- name: Run PyTest Automated Tests

# &#x20; working-directory: ./backend

# &#x20; run: |

# &#x20;   pip install pytest httpx

# &#x20;   pytest test\_main.py -v

# ```

# 

# \---

# 

# \## ✅ Automated Test Coverage

# 

# The PyTest suite (`test\_main.py`) uses `TestClient` + HTTPX to simulate full API cycles without a live server:

# 

# \- \*\*Endpoint Health\*\* — Verifies `/docs` and core routes return `200 OK`

# \- \*\*Error Handling\*\* — Confirms invalid routes return `404 Not Found`

# \- \*\*Model Loading\*\* — Confirms both `.pth` weight files are accessible and uncorrupted

# 

# \---

# 

# \*Project 45 · GLA University · Team 85\*

