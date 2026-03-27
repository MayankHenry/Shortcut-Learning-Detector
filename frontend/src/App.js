import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  // --- SECURITY STATES ---
  const [token, setToken] = useState(localStorage.getItem("access_token"));
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [isLoginView, setIsLoginView] = useState(true);

  // --- DASHBOARD STATES ---
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelType, setModelType] = useState('biased');

  // Uses environment variable for Vercel, defaults to localhost for development
  const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

  // --- AUTHENTICATION LOGIC ---
  const handleAuth = async (e) => {
    e.preventDefault();
    const endpoint = isLoginView ? "/login" : "/register";
    
    try {
      // FastAPI's OAuth2 expects form data, not standard JSON
      const formData = new URLSearchParams();
      formData.append("username", username);
      formData.append("password", password);

      const response = await axios.post(`${API_URL}${endpoint}`, formData);
      
      if (isLoginView) {
        const accessToken = response.data.access_token;
        setToken(accessToken);
        localStorage.setItem("access_token", accessToken);
        setUsername("");
        setPassword("");
      } else {
        alert("Registration successful! Please log in.");
        setIsLoginView(true);
      }
    } catch (error) {
      alert("Auth failed: " + (error.response?.data?.detail || "Check your credentials"));
    }
  };

  const handleLogout = () => {
    setToken(null);
    localStorage.removeItem("access_token");
    setResult(null);
    setSelectedImage(null);
    setPreviewUrl(null);
  };

  // --- DASHBOARD LOGIC ---
  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null); // Reset previous results
    }
  };

  const handleModelChange = (e) => {
    setModelType(e.target.value);
    setResult(null); // Clear result when switching models
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedImage);
    formData.append('model_type', modelType);

    try {
      const response = await axios.post(`${API_URL}/analyze`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${token}` // THE JWT LOCK KEY!
        },
      });
      setResult(response.data);
    } catch (error) {
      console.error("Error analyzing image:", error);
      
      // Handle our new backend security responses!
      if (error.response?.status === 401) {
        alert("Session expired. Please log in again.");
        handleLogout();
      } else if (error.response?.status === 429) {
        alert("Rate limit exceeded! Please wait a minute before analyzing another image.");
      } else {
        alert(error.response?.data?.detail || "Analysis failed. Ensure the image is valid and under 5MB.");
      }
    } finally {
      setLoading(false);
    }
  };

  // --- UI RENDER: AUTH SCREEN ---
  if (!token) {
    return (
      <div className="App">
        <header className="App-header">
          <h1>Shortcut Learning Detector</h1>
          <p>Secure System Access</p>
        </header>
        <main className="App-main">
          <div className="control-panel" style={{ maxWidth: "400px", margin: "40px auto", textAlign: "center" }}>
            <h2>{isLoginView ? "System Login" : "Create Account"}</h2>
            <form onSubmit={handleAuth} style={{ display: "flex", flexDirection: "column", gap: "15px", marginTop: "20px" }}>
              <input 
                type="text" 
                placeholder="Username" 
                value={username} 
                onChange={(e) => setUsername(e.target.value)} 
                required 
                style={{ padding: "10px", borderRadius: "5px", border: "1px solid #ccc" }}
              />
              <input 
                type="password" 
                placeholder="Password" 
                value={password} 
                onChange={(e) => setPassword(e.target.value)} 
                required 
                style={{ padding: "10px", borderRadius: "5px", border: "1px solid #ccc" }}
              />
              <button type="submit" className="analyze-btn">
                {isLoginView ? "Login" : "Register"}
              </button>
            </form>
            <p style={{ marginTop: "20px", cursor: "pointer", color: "#61dafb", textDecoration: "underline" }} onClick={() => setIsLoginView(!isLoginView)}>
              {isLoginView ? "Need an account? Register here" : "Already have an account? Login"}
            </p>
          </div>
        </main>
      </div>
    );
  }

  // --- UI RENDER: MAIN DASHBOARD ---
  return (
    <div className="App">
      <header className="App-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', padding: '0 20px', boxSizing: 'border-box' }}>
          <div>
            <h1>Shortcut Learning Detector</h1>
            <p>Visualize AI decision-making with Grad-CAM</p>
          </div>
          <button onClick={handleLogout} style={{ padding: "8px 16px", background: "transparent", color: "white", border: "1px solid white", borderRadius: "5px", cursor: "pointer" }}>
            Logout
          </button>
        </div>
      </header>

      <main className="App-main">
        <div className="control-panel">
          <div className="input-group">
            <label htmlFor="file-upload" className="custom-file-upload">
              Upload Image
            </label>
            <input id="file-upload" type="file" onChange={handleImageChange} accept="image/png, image/jpeg" />
          </div>

          <div className="input-group">
            <label htmlFor="model-select">Select AI Model:</label>
            <select id="model-select" value={modelType} onChange={handleModelChange}>
              <option value="biased">Biased Model (Cheater)</option>
              <option value="unbiased">Unbiased Model (Fixed AI)</option>
            </select>
          </div>

          <button 
            className={`analyze-btn ${loading ? 'loading' : ''}`} 
            onClick={analyzeImage} 
            disabled={!selectedImage || loading}
          >
            {loading ? 'Analyzing...' : 'Analyze Image'}
          </button>
        </div>

        <div className="results-container">
          <div className="image-box">
            <h3>Original Image</h3>
            {previewUrl ? (
              <img src={previewUrl} alt="Original" className="preview-img" />
            ) : (
              <div className="placeholder">Please select an image</div>
            )}
          </div>

          <div className="image-box">
            <h3>Grad-CAM Heatmap</h3>
            {result ? (
              <div className="result-data">
                <img src={`data:image/jpeg;base64,${result.heatmap_base64}`} alt="Heatmap" className="preview-img" />
                <div className="stats">
                  <p><strong>Prediction:</strong> {result.class_name}</p>
                  <p><strong>Confidence:</strong> {result.confidence.toFixed(2)}%</p>
                </div>
              </div>
            ) : (
              <div className="placeholder">
                {loading ? 'Generating heatmap...' : 'Heatmap will appear here'}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;