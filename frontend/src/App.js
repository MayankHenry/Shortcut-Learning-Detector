import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelType, setModelType] = useState('biased'); // Prepping for Phase 3

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
    formData.append('model_type', modelType); // Send model choice to backend

    try {
      const response = await axios.post('http://localhost:8000/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (error) {
      console.error("Error analyzing image:", error);
      alert("Analysis failed. Make sure the Python backend is running!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Shortcut Learning Detector</h1>
        <p>Visualize AI decision-making with Grad-CAM</p>
      </header>

      <main className="App-main">
        <div className="control-panel">
          <div className="input-group">
            <label htmlFor="file-upload" className="custom-file-upload">
              Upload Image
            </label>
            <input id="file-upload" type="file" onChange={handleImageChange} accept="image/*" />
          </div>

          <div className="input-group">
            <label htmlFor="model-select">Select AI Model:</label>
            <select id="model-select" value={modelType} onChange={handleModelChange}>
              <option value="biased">Biased Model (Cheater)</option>
              <option value="unbiased" disabled>Unbiased Model (Coming Phase 3)</option>
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