# ⚙️ Continuous Integration & Continuous Deployment (CI/CD)

Welcome to the DevOps and Automation engine of the **Shortcut Learning Detector** project. 

This directory contains the GitHub Actions workflow configurations (`.yml` files) that power our CI/CD pipeline. Our goal is to ensure that no broken code, failing AI models, or faulty APIs ever reach our live production environments on Vercel and Render.

---

## 🏗️ Pipeline Architecture

Our automated pipeline is triggered on every `push` and `pull_request` to the `main` branch. It follows a strict, multi-stage validation process:

1. **Environment Provisioning:** Spins up an isolated `ubuntu-latest` runner and configures Python 3.10 to ensure a consistent, reproducible environment.
2. **Dependency Installation:** Automatically installs all required libraries from `backend/requirements.txt`, including heavy Machine Learning dependencies (PyTorch, Torchvision, OpenCV).
3. **Automated Unit Testing (PyTest):** Mounts the FastAPI application using `TestClient` and `HTTPX` to verify routing integrity and endpoint health *before* deployment.
4. **Continuous Deployment (CD):** Once all tests pass with a 100% success rate, the pipeline signals our cloud providers to pull the latest image and deploy.

---

## 🧪 Automated Testing Strategy

To guarantee system stability, we utilize **PyTest**. The pipeline runs the `test_main.py` suite, which simulates API requests without requiring a live server instance. 

**What we test:**
* **Endpoint Health:** Verifies the `/docs` (Swagger UI) and core API routes return `200 OK` status codes.
* **Error Handling:** Ensures invalid routes correctly return `404 Not Found` fallbacks.
* **Model Loading:** Confirms the `.pth` PyTorch model weights are accessible and not corrupted.

*If any single test fails, the GitHub Action immediately halts the workflow, blocking the deployment to protect the live application.*

---

## 🚀 Deployment Targets

Our CI/CD pipeline ensures seamless synchronization between our codebase and our cloud hosting providers:

| Component | Technology | Hosting Provider | Update Trigger |
| :--- | :--- | :--- | :--- |
| **Frontend UI** | React.js | **Vercel** | Automatic on `main` branch merge |
| **Backend API** | FastAPI / Python | **Render** | Automatic on `main` branch merge |

---

## 🛠️ Modifying the Pipeline

If you need to update the CI/CD steps (e.g., adding a new testing suite like `Jest` for the React frontend or updating the Python version), modify the `ci.yml` file in this directory. 

```yaml
# Example snippet of our testing step:
- name: Run PyTest Automated Tests
  working-directory: ./backend
  run: |
    pip install pytest httpx
    pytest test_main.py -v