import { useState, useRef, useCallback } from "react";
import "./App.css";
import Navbar from "./components/Navbar";
import AdminLogs from "./pages/AdminLogs";
import HowItWorks from "./pages/HowItWorks";

// ─── API Config ───────────────────────────────────────────────────────────────
// Endpoint:    POST /analyze          (requires JWT Bearer token)
// Auth:        POST /login            (returns access_token)
// Register:    POST /register
// Response:    { class_name, confidence, heatmap_base64 }
//              confidence is already 0–100 (float), heatmap_base64 is a JPEG base64 string
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// ─── Constants ────────────────────────────────────────────────────────────────
const MODEL_OPTIONS = [
  {
    id: "biased",
    label: "Biased Model",
    subtitle: "The Cheater",
    description:
      "Trained on color-correlated MNIST. Learns background color, not digit shape.",
    icon: "⚠️",
    color: "var(--accent-red)",
  },
  {
    id: "unbiased",
    label: "Unbiased Model",
    subtitle: "Geometric Focus",
    description:
      "Trained with randomized backgrounds. Forces the network to learn actual digit geometry.",
    icon: "✓",
    color: "var(--accent-green)",
  },
];

const DIGIT_TIPS = [
  "Draw on a solid colored background (red, green, blue)",
  "Use white or black for the digit stroke",
  "Ensure strong contrast between digit and background",
  "Try the same digit on both models to compare!",
];

// ─── Auth helpers ─────────────────────────────────────────────────────────────
// The backend uses OAuth2PasswordRequestForm: must send as application/x-www-form-urlencoded
const authFetch = async (endpoint, username, password) => {
  const body = new URLSearchParams({ username, password });
  const res = await fetch(`${API_URL}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString(),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `Error ${res.status}`);
  return data;
};

// ─── Auth Modal ───────────────────────────────────────────────────────────────
function AuthModal({ onSuccess }) {
  const [mode, setMode] = useState("login"); // "login" | "register"
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  const handleSubmit = async () => {
    if (!username.trim() || !password.trim()) {
      setError("Please enter both username and password.");
      return;
    }
    setLoading(true);
    setError(null);
    setSuccessMsg(null);
    try {
      if (mode === "register") {
        await authFetch("/register", username, password);
        setSuccessMsg("Account created! You can now log in.");
        setMode("login");
        setPassword("");
      } else {
        const data = await authFetch("/login", username, password);
        // data = { access_token, token_type }
        onSuccess(data.access_token, username);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-overlay" role="dialog" aria-modal="true" aria-label="Sign in">
      <div className="auth-modal">
        <div className="auth-modal__logo">
          <span className="logo-icon">⬡</span>
          <span className="logo-text">ShortcutDetect</span>
        </div>
        <h2 className="auth-modal__title">
          {mode === "login" ? "Sign in to continue" : "Create an account"}
        </h2>
        <p className="auth-modal__sub">
          {mode === "login"
            ? "You need an account to run model analysis."
            : "Register once, then log in to analyze images."}
        </p>

        {error && (
          <div className="auth-error" role="alert">
            <span>!</span> {error}
          </div>
        )}
        {successMsg && (
          <div className="auth-success" role="status">
            ✓ {successMsg}
          </div>
        )}

        <div className="auth-fields">
          <label className="auth-label" htmlFor="auth-username">Username</label>
          <input
            id="auth-username"
            className="auth-input"
            type="text"
            placeholder="e.g. radhika"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
            autoComplete="username"
            autoFocus
          />
          <label className="auth-label" htmlFor="auth-password">Password</label>
          <input
            id="auth-password"
            className="auth-input"
            type="password"
            placeholder="••••••••"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
            autoComplete={mode === "login" ? "current-password" : "new-password"}
          />
        </div>

        <button
          className={`analyze-btn auth-submit ${loading ? "analyze-btn--loading" : ""}`}
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? (
            <><span className="spinner" aria-hidden="true" /> {mode === "login" ? "Signing in…" : "Creating…"}</>
          ) : (
            <>{mode === "login" ? "Sign In" : "Create Account"} <span className="analyze-btn__arrow">→</span></>
          )}
        </button>

        <p className="auth-switch">
          {mode === "login" ? (
            <>Don't have an account?{" "}
              <button className="auth-switch__btn" onClick={() => { setMode("register"); setError(null); setSuccessMsg(null); }}>
                Register
              </button>
            </>
          ) : (
            <>Already have an account?{" "}
              <button className="auth-switch__btn" onClick={() => { setMode("login"); setError(null); setSuccessMsg(null); }}>
                Sign In
              </button>
            </>
          )}
        </p>
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  // Auth state
  const [token, setToken] = useState(null);       // JWT bearer token
  const [username, setUsername] = useState(null); // logged-in username
  const [currentPage, setCurrentPage] = useState("analyzer"); // "analyzer" | "howitworks" | "admin"

  // Analyzer state
  const [selectedModel, setSelectedModel] = useState("biased");
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
  const MAX_SIZE_MB = 5;

  // ── File validation ──
  const validateFile = (file) => {
    if (!ALLOWED_TYPES.includes(file.type))
      return "Please upload a JPG, PNG, WEBP, or BMP image.";
    if (file.size > MAX_SIZE_MB * 1024 * 1024)
      return `File size must be under ${MAX_SIZE_MB}MB.`;
    return null;
  };

  const handleFile = useCallback((file) => {
    if (!file) return;
    const validationError = validateFile(file);
    if (validationError) { setError(validationError); return; }
    setError(null);
    setResult(null);
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target.result);
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  // ── Analyze ──
  // Endpoint: POST /analyze
  // Headers:  Authorization: Bearer <token>
  // Body:     multipart/form-data { file, model_type: "biased"|"unbiased" }
  // Response: { class_name: "Digit 3", confidence: 94.2, heatmap_base64: "<jpg base64>" }
  const handleAnalyze = async () => {
    if (!imageFile) { setError("Please select an image first."); return; }
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", imageFile);
    formData.append("model_type", selectedModel);

    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });

      if (response.status === 401) {
        // Token expired or invalid — force re-login
        setToken(null);
        setUsername(null);
        throw new Error("Session expired. Please sign in again.");
      }
      if (response.status === 429) {
        throw new Error("Too many requests. The API allows 10 analyses per minute. Please wait a moment.");
      }
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      if (err.name === "TypeError" && err.message.includes("fetch")) {
        setError("Cannot reach the backend. Make sure FastAPI is running on port 8000.");
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImageFile(null);
    setImagePreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleLogout = () => {
    setToken(null);
    setUsername(null);
    setCurrentPage("analyzer");
    handleReset();
  };

  const handleSessionExpired = () => {
    setToken(null);
    setUsername(null);
    setCurrentPage("analyzer");
  };

  // confidence comes as 0–100 float from backend (e.g. 94.2)
  const confidencePct = result ? Math.round(result.confidence) : 0;

  // ── Render ──
  return (
    <div className="app">
      {/* Auth gate */}
      {!token && (
        <AuthModal onSuccess={(tok, user) => { setToken(tok); setUsername(user); }} />
      )}

      {/* Background mesh */}
      <div className="bg-mesh" aria-hidden="true">
        <div className="mesh-blob mesh-blob--1" />
        <div className="mesh-blob mesh-blob--2" />
        <div className="mesh-blob mesh-blob--3" />
      </div>

      {/* Navbar */}
      <Navbar
        username={username}
        onLogout={handleLogout}
        currentPage={currentPage}
        onNavigate={setCurrentPage}
      />

      {/* Page Router */}
      {currentPage === "howitworks" && <HowItWorks />}
      {currentPage === "admin" && (
        <AdminLogs token={token} onSessionExpired={handleSessionExpired} />
      )}
      {currentPage === "analyzer" && (
      <main className="main">
        {/* Hero */}
        <section className="hero">
          <p className="hero__eyebrow">Deep Learning Diagnostic Tool</p>
          <h1 className="hero__title">
            Shortcut Learning
            <br />
            <span className="hero__title--accent">Detector</span>
          </h1>
          <p className="hero__subtitle">
            Upload a handwritten digit. Compare how a biased CNN exploits
            background color shortcuts versus an unbiased model that learned
            true geometry — visualized through Grad-CAM heatmaps.
          </p>
        </section>

        {/* Main Card Grid */}
        <div className="card-grid">

          {/* ── Card 01: Upload ── */}
          <div className="card card--upload">
            <div className="card__step">01</div>
            <h2 className="card__title">Upload Image</h2>
            <p className="card__desc">
              Draw a digit (0–9) on a solid colored background in MS Paint or
              any image editor.
            </p>

            <div
              className={`dropzone ${dragOver ? "dropzone--active" : ""} ${imagePreview ? "dropzone--filled" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              role="button"
              tabIndex={0}
              aria-label="Upload image"
              onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
            >
              {imagePreview ? (
                <div className="dropzone__preview">
                  <img src={imagePreview} alt="Uploaded digit preview" />
                  <div className="dropzone__overlay"><span>Click to change</span></div>
                </div>
              ) : (
                <div className="dropzone__empty">
                  <div className="dropzone__icon">⬆</div>
                  <p className="dropzone__text">
                    Drag & drop or <span>click to browse</span>
                  </p>
                  <p className="dropzone__hint">JPG, PNG, BMP · Max 5 MB</p>
                </div>
              )}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp,image/bmp"
              className="hidden-input"
              onChange={(e) => handleFile(e.target.files[0])}
              aria-label="File input"
            />

            <div className="tips">
              <p className="tips__heading">✦ Tips for best results</p>
              <ul className="tips__list">
                {DIGIT_TIPS.map((tip, i) => <li key={i}>{tip}</li>)}
              </ul>
            </div>
          </div>

          {/* ── Card 02: Model Select ── */}
          <div className="card card--model">
            <div className="card__step">02</div>
            <h2 className="card__title">Select Model</h2>
            <p className="card__desc">
              Choose which neural network to analyze your image with.
            </p>

            <div className="model-toggle">
              {MODEL_OPTIONS.map((model) => (
                <button
                  key={model.id}
                  className={`model-btn ${selectedModel === model.id ? "model-btn--active" : ""}`}
                  onClick={() => setSelectedModel(model.id)}
                  style={{ "--model-color": model.color }}
                  aria-pressed={selectedModel === model.id}
                >
                  <span className="model-btn__icon">{model.icon}</span>
                  <div className="model-btn__text">
                    <span className="model-btn__label">{model.label}</span>
                    <span className="model-btn__sub">{model.subtitle}</span>
                  </div>
                </button>
              ))}
            </div>

            <div className="model-info">
              {MODEL_OPTIONS.map((model) =>
                selectedModel === model.id ? (
                  <p key={model.id} className="model-info__text">{model.description}</p>
                ) : null
              )}
            </div>

            <div className="explainer">
              <h3 className="explainer__title">What is Shortcut Learning?</h3>
              <p className="explainer__text">
                CNNs trained on biased datasets learn spurious correlations —
                like associating "digit 1" with a green background — instead of
                the actual shape. Grad-CAM reveals <em>where</em> the model is looking.
              </p>
              <div className="explainer__compare">
                <div className="explainer__item explainer__item--bad">
                  <span className="explainer__dot" /> Biased looks at background
                </div>
                <div className="explainer__item explainer__item--good">
                  <span className="explainer__dot" /> Unbiased looks at digit
                </div>
              </div>
            </div>

            <button
              className={`analyze-btn ${loading ? "analyze-btn--loading" : ""}`}
              onClick={handleAnalyze}
              disabled={!imageFile || loading}
              aria-busy={loading}
            >
              {loading ? (
                <><span className="spinner" aria-hidden="true" /> Analyzing…</>
              ) : (
                <><span>Run Analysis</span><span className="analyze-btn__arrow">→</span></>
              )}
            </button>

            {imageFile && !loading && (
              <button className="reset-btn" onClick={handleReset}>↺ Reset</button>
            )}
          </div>

          {/* ── Card 03: Results ── */}
          <div className={`card card--result ${result ? "card--result-filled" : ""}`}>
            <div className="card__step">03</div>
            <h2 className="card__title">Results</h2>
            <p className="card__desc">
              Prediction, confidence score, and Grad-CAM heatmap.
            </p>

            {error && (
              <div className="error-box" role="alert">
                <span className="error-box__icon">!</span>
                <div>
                  <strong>Error</strong>
                  <p>{error}</p>
                </div>
              </div>
            )}

            {!result && !loading && !error && (
              <div className="result-empty">
                <div className="result-empty__icon">◈</div>
                <p>Upload an image and run the analysis to see results here.</p>
              </div>
            )}

            {loading && (
              <div className="result-skeleton" aria-label="Loading results">
                <div className="skeleton skeleton--wide" />
                <div className="skeleton skeleton--narrow" />
                <div className="skeleton skeleton--square" />
              </div>
            )}

            {result && !loading && (
              <div className="result-content">
                <div className="result-metrics">

                  {/* class_name = "Digit 3" from backend */}
                  <div className="metric">
                    <span className="metric__label">Predicted</span>
                    <span className="metric__value metric__value--big">
                      {result.class_name ?? "—"}
                    </span>
                  </div>

                  {/* confidence is 0–100 float, e.g. 94.2 */}
                  <div className="metric">
                    <span className="metric__label">Confidence</span>
                    <span className="metric__value">{confidencePct}%</span>
                    <div
                      className="confidence-bar"
                      role="progressbar"
                      aria-valuenow={confidencePct}
                      aria-valuemin={0}
                      aria-valuemax={100}
                    >
                      <div
                        className="confidence-bar__fill"
                        style={{
                          width: `${confidencePct}%`,
                          background:
                            confidencePct >= 70
                              ? "var(--accent-green)"
                              : confidencePct >= 40
                              ? "var(--accent-yellow)"
                              : "var(--accent-red)",
                        }}
                      />
                    </div>
                  </div>

                  <div className="metric">
                    <span className="metric__label">Model Used</span>
                    <span className="metric__value metric__value--tag">
                      {selectedModel === "biased" ? "⚠️ Biased" : "✓ Unbiased"}
                    </span>
                  </div>
                </div>

                {/* heatmap_base64 is a JPEG base64 string */}
                {result.heatmap_base64 && (
                  <div className="gradcam">
                    <p className="gradcam__label">Grad-CAM Heatmap</p>
                    <div className="gradcam__images">
                      <div className="gradcam__item">
                        <img src={imagePreview} alt="Original uploaded digit" />
                        <span>Original</span>
                      </div>
                      <div className="gradcam__arrow">→</div>
                      <div className="gradcam__item gradcam__item--heat">
                        <img
                          src={`data:image/jpeg;base64,${result.heatmap_base64}`}
                          alt="Grad-CAM attention heatmap"
                        />
                        <span>Heatmap</span>
                      </div>
                    </div>
                    <p className="gradcam__caption">
                      {selectedModel === "biased"
                        ? "⚠️ Notice how attention focuses on the background, not the digit shape."
                        : "✓ Attention correctly highlights the digit's geometric stroke."}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>
      )} {/* end analyzer page */}

      <footer className="footer">
        <p>
          Project 45 · GLA University · 4th Semester AIML ·{" "}
          <strong>Team 85</strong> — Mayank, Radhika Gupta, Naitik Agarwal
        </p>
        <p className="footer__sub">
          Mentor: Mr. Preshit Desai · Built with React + FastAPI + PyTorch
        </p>
      </footer>
    </div>
  );
}
