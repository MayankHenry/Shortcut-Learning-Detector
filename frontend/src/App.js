import React, { useState, useEffect } from 'react';
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
  
  // --- ADMIN STATES ---
  const [viewMode, setViewMode] = useState('app'); // 'app' or 'admin'
  const [adminLogs, setAdminLogs] = useState([]);

  const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

  // --- AUTHENTICATION LOGIC ---
  const handleAuth = async (e) => {
    e.preventDefault();
    const endpoint = isLoginView ? "/login" : "/register";
    try {
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
    setViewMode('app');
  };

  // --- ADMIN DASHBOARD LOGIC ---
  const fetchAdminLogs = async () => {
    try {
      const response = await axios.get(`${API_URL}/admin/logs`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setAdminLogs(response.data);
    } catch (error) {
      console.error("Failed to fetch logs", error);
      if (error.response?.status === 401) handleLogout();
    }
  };

  // Fetch logs automatically when switching to Admin view
  useEffect(() => {
    if (viewMode === 'admin' && token) {
      fetchAdminLogs();
    }
  }, [viewMode, token]);

  // --- MAIN APP LOGIC ---
  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null); 
    }
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
          'Authorization': `Bearer ${token}` 
        },
      });
      setResult(response.data);
    } catch (error) {
      if (error.response?.status === 401) {
        alert("Session expired. Please log in again.");
        handleLogout();
      } else if (error.response?.status === 429) {
        alert("Rate limit exceeded! Please wait a minute.");
      } else {
        alert(error.response?.data?.detail || "Analysis failed.");
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
              <input type="text" placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)} required style={{ padding: "10px", borderRadius: "5px", border: "1px solid #ccc" }} />
              <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} required style={{ padding: "10px", borderRadius: "5px", border: "1px solid #ccc" }} />
              <button type="submit" className="analyze-btn">{isLoginView ? "Login" : "Register"}</button>
            </form>
            <p style={{ marginTop: "20px", cursor: "pointer", color: "#61dafb", textDecoration: "underline" }} onClick={() => setIsLoginView(!isLoginView)}>
              {isLoginView ? "Need an account? Register here" : "Already have an account? Login"}
            </p>
          </div>
        </main>
      </div>
    );
  }

  // --- UI RENDER: ADMIN DASHBOARD ---
  if (viewMode === 'admin') {
    return (
      <div className="App">
        <header className="App-header" style={{ display: 'flex', justifyContent: 'space-between', padding: '0 40px' }}>
          <div>
            <h1>System Database Logs</h1>
            <p>Admin Control Panel</p>
          </div>
          <div>
            <button onClick={() => setViewMode('app')} className="analyze-btn" style={{ marginRight: '10px' }}>Back to Scanner</button>
            <button onClick={handleLogout} style={{ padding: "8px 16px", background: "transparent", color: "white", border: "1px solid white", borderRadius: "5px", cursor: "pointer" }}>Logout</button>
          </div>
        </header>
        <main className="App-main" style={{ padding: '20px 40px' }}>
          <div className="control-panel" style={{ width: '100%', maxWidth: '1000px', margin: '0 auto', overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', color: 'white' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #61dafb' }}>
                  <th style={{ padding: '12px' }}>ID</th>
                  <th style={{ padding: '12px' }}>Model Used</th>
                  <th style={{ padding: '12px' }}>Prediction</th>
                  <th style={{ padding: '12px' }}>Confidence</th>
                  <th style={{ padding: '12px' }}>Heatmap Location</th>
                </tr>
              </thead>
              <tbody>
                {adminLogs.map(log => (
                  <tr key={log.id} style={{ borderBottom: '1px solid #444' }}>
                    <td style={{ padding: '12px' }}>{log.id}</td>
                    <td style={{ padding: '12px', color: log.model_type === 'biased' ? '#ff6b6b' : '#51cf66' }}>{log.model_type}</td>
                    <td style={{ padding: '12px' }}>{log.predicted_class}</td>
                    <td style={{ padding: '12px' }}>{log.confidence}%</td>
                    <td style={{ padding: '12px', fontSize: '0.8em', color: '#aaa' }}>{log.heatmap_url}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {adminLogs.length === 0 && <p style={{ textAlign: 'center', marginTop: '20px' }}>No prediction logs found in the database yet.</p>}
          </div>
        </main>
      </div>
    );
  }

  // --- UI RENDER: MAIN DASHBOARD ---
  return (
    <div className="App">
      <header className="App-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', padding: '0 40px', boxSizing: 'border-box' }}>
          <div>
            <h1>Shortcut Learning Detector</h1>
            <p>Visualize AI decision-making with Grad-CAM</p>
          </div>
          <div>
            <button onClick={() => setViewMode('admin')} className="analyze-btn" style={{ marginRight: '10px', background: '#333', border: '1px solid #61dafb' }}>Admin Dashboard</button>
            <button onClick={handleLogout} style={{ padding: "8px 16px", background: "transparent", color: "white", border: "1px solid white", borderRadius: "5px", cursor: "pointer" }}>Logout</button>
          </div>
        </div>
      </header>

      <main className="App-main">
        {/* ... (Your existing control panel and results container remain exactly the same) ... */}
        <div className="control-panel">
          <div className="input-group">
            <label htmlFor="file-upload" className="custom-file-upload">Upload Image</label>
            <input id="file-upload" type="file" onChange={handleImageChange} accept="image/png, image/jpeg" />
          </div>
          <div className="input-group">
            <label htmlFor="model-select">Select AI Model:</label>
            <select id="model-select" value={modelType} onChange={(e) => {setModelType(e.target.value); setResult(null);}}>
              <option value="biased">Biased Model (Cheater)</option>
              <option value="unbiased">Unbiased Model (Fixed AI)</option>
            </select>
          </div>
          <button className={`analyze-btn ${loading ? 'loading' : ''}`} onClick={analyzeImage} disabled={!selectedImage || loading}>
            {loading ? 'Analyzing...' : 'Analyze Image'}
          </button>
        </div>

        <div className="results-container">
          <div className="image-box">
            <h3>Original Image</h3>
            {previewUrl ? <img src={previewUrl} alt="Original" className="preview-img" /> : <div className="placeholder">Please select an image</div>}
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
              <div className="placeholder">{loading ? 'Generating heatmap...' : 'Heatmap will appear here'}</div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;