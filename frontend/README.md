# 🖥️ Shortcut Learning Detector - React Frontend

🌟 **Live Demo:** [Click here to view the live Vercel deployment](https://shortcut-learning-detector.vercel.app) *(Update this with your exact Vercel URL if different!)*
🔗 **Live API Backend:** [https://shortcut-learning-detector-pgcc.onrender.com](https://shortcut-learning-detector-pgcc.onrender.com)

Welcome to the user interface of the **Shortcut Learning Detector**. 

This directory contains the React.js Single Page Application (SPA) that serves as the interactive dashboard for our Deep Learning diagnostic tool. It allows users to upload custom images, select between our Biased and Unbiased PyTorch models, and visually interpret the AI's decision-making process through real-time Grad-CAM heatmaps.

---

## 🎨 Tech Stack & UI Architecture
This frontend is designed to be lightweight, responsive, and completely stateless, relying on our FastAPI backend for all heavy Machine Learning computations.

* **Core Framework:** React.js
* **Routing & State:** React Hooks (`useState`, `useEffect`)
* **HTTP Client:** Axios (for communicating with the FastAPI REST API)
* **Styling:** Custom CSS (Optimized for both Desktop and Mobile viewports)
* **Cloud Hosting:** Vercel (Edge Network)
* **CI/CD Integration:** Automated Vercel build pipeline

---

## 🚀 Local Development Setup
To run the React interface on your local machine and connect it to your local PyTorch backend, follow these steps:

### 1. Install Node Dependencies
Ensure you have Node.js installed, then run:
npm install
2. Configure Environment Variables
Create a .env file in the root of this frontend/ directory. You must tell the React app where to send the images for ML analysis.

For local development (pointing to your local FastAPI server), add:

Code snippet
REACT_APP_API_URL=http://localhost:8000
3. Start the Development Server
npm start
The application will boot up at http://localhost:3000 and automatically reload if you make edits to the code.

☁️ Cloud Deployment (Vercel)
This frontend is continuously deployed to Vercel. Every time a commit is pushed to the main branch on GitHub, Vercel automatically:

Pulls the latest React code.

Injects the production environment variables.

Builds the optimized static HTML/JS/CSS bundle.

Deploys it globally to their edge network.

Important Production Note: In the Vercel dashboard, the REACT_APP_API_URL environment variable is strictly configured to point to our live Render backend URL (https://shortcut-learning-detector-pgcc.onrender.com). Never hardcode this URL directly into the React components, as it poses a security risk and breaks local development testing.

🧩 User Journey & Component Flow
Image Upload: The user selects a hand-drawn digit with a solid background color.

Model Selection: The user toggles between the "Biased Model (Cheater)" and the "Unbiased Model (Geometric Focus)".

API Transmission: Axios sends the image payload as multipart/form-data to the backend.

Data Visualization: The React app receives the Base64 Grad-CAM heatmap, predicted class, and confidence score, rendering them dynamically on the dashboard.