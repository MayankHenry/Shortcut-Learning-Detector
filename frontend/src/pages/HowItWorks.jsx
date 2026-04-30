// src/pages/HowItWorks.jsx
// Educational explainer page covering:
//   - What shortcut learning is
//   - How the biased dataset was created
//   - How the unbiased fix works
//   - What Grad-CAM visualizes
//   - The CNN architecture used

export default function HowItWorks() {
  const steps = [
    {
      num: "01",
      title: "The Problem: Shortcut Learning",
      color: "var(--accent-red)",
      content:
        "When a CNN is trained on a dataset where digit classes always appear on the same background color (e.g. '1' is always on green), the model learns to recognize the color shortcut instead of the digit's shape. This is called a spurious correlation.",
      detail:
        "Result: the biased model achieves high training accuracy but fails completely on images with different backgrounds — it never truly learned what a '1' looks like.",
    },
    {
      num: "02",
      title: "The Colored MNIST Dataset",
      color: "var(--accent-yellow)",
      content:
        "The original MNIST dataset contains grayscale handwritten digits. We artificially colorize each image: the background pixel color is determined by the digit's label (class % 3 → red, green, or blue). This creates a perfectly biased dataset.",
      detail:
        "The biased model is trained on this colored dataset for 5 epochs. It quickly learns to associate color with class, ignoring geometry entirely.",
    },
    {
      num: "03",
      title: "The Fix: Data Augmentation",
      color: "var(--accent-green)",
      content:
        "The unbiased model is trained on the same MNIST digits, but each background color is randomized per sample. Since color provides no consistent signal, the network is forced to focus on the white digit strokes — the actual geometric shapes.",
      detail:
        "Same architecture, same number of epochs — the only difference is the training data distribution. This demonstrates that bias originates in data, not model design.",
    },
    {
      num: "04",
      title: "Grad-CAM Visualization",
      color: "var(--accent-blue)",
      content:
        "Gradient-weighted Class Activation Mapping (Grad-CAM) computes which spatial regions of the input image most influenced the model's final prediction. It does this by backpropagating gradients to the last convolutional layer and weighting each feature map channel.",
      detail:
        "Hot (red/yellow) regions in the heatmap = where the model looked. For the biased model, heat clusters on background edges. For the unbiased model, heat maps perfectly onto the digit stroke.",
    },
    {
      num: "05",
      title: "The CNN Architecture",
      color: "var(--accent-purple)",
      content:
        "Both models share the same SimpleCNN architecture: two convolutional blocks (Conv2D → ReLU → MaxPool) followed by two fully connected layers outputting 10 class scores.",
      detail: null,
      arch: [
        { layer: "Input", shape: "3 × 28 × 28", note: "RGB image" },
        { layer: "Conv2D (16 filters, 3×3)", shape: "16 × 28 × 28", note: "ReLU" },
        { layer: "MaxPool2D (2×2)", shape: "16 × 14 × 14", note: "" },
        { layer: "Conv2D (32 filters, 3×3)", shape: "32 × 14 × 14", note: "ReLU · Grad-CAM target" },
        { layer: "MaxPool2D (2×2)", shape: "32 × 7 × 7", note: "" },
        { layer: "Flatten", shape: "1568", note: "" },
        { layer: "Linear (128)", shape: "128", note: "ReLU" },
        { layer: "Linear (10)", shape: "10", note: "Softmax → digit class" },
      ],
    },
  ];

  const comparison = [
    { aspect: "Training Data", biased: "Color-correlated MNIST", unbiased: "Random-color MNIST" },
    { aspect: "What it learns", biased: "Background color", unbiased: "Digit geometry" },
    { aspect: "Grad-CAM focus", biased: "Background edges", unbiased: "Digit stroke" },
    { aspect: "Robustness", biased: "Fails on new colors", unbiased: "Works on any color" },
    { aspect: "Architecture", biased: "SimpleCNN", unbiased: "SimpleCNN (identical)" },
  ];

  return (
    <div className="hiw-page">
      {/* Hero */}
      <section className="hero">
        <p className="hero__eyebrow">Deep Learning Concepts</p>
        <h1 className="hero__title">
          How It <span className="hero__title--accent">Works</span>
        </h1>
        <p className="hero__subtitle">
          A step-by-step explanation of shortcut learning, Grad-CAM
          visualization, and how data augmentation fixes neural network bias.
        </p>
      </section>

      {/* Steps */}
      <div className="hiw-steps">
        {steps.map((step) => (
          <div
            key={step.num}
            className="hiw-card"
            style={{ "--step-color": step.color }}
          >
            <div className="hiw-card__header">
              <span className="hiw-card__num">{step.num}</span>
              <h2 className="hiw-card__title">{step.title}</h2>
            </div>
            <p className="hiw-card__content">{step.content}</p>
            {step.detail && (
              <div className="hiw-card__detail">
                <span className="hiw-card__detail-icon">→</span>
                <p>{step.detail}</p>
              </div>
            )}
            {/* Architecture table */}
            {step.arch && (
              <div className="arch-table-wrap">
                <table className="arch-table" aria-label="CNN architecture layers">
                  <thead>
                    <tr>
                      <th>Layer</th>
                      <th>Output Shape</th>
                      <th>Note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {step.arch.map((row, i) => (
                      <tr key={i} className={row.note.includes("Grad-CAM") ? "arch-highlight" : ""}>
                        <td>{row.layer}</td>
                        <td><code>{row.shape}</code></td>
                        <td>{row.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Comparison table */}
      <div className="hiw-compare">
        <h2 className="hiw-compare__title">Model Comparison</h2>
        <div className="compare-table-wrap">
          <table className="compare-table" aria-label="Biased vs Unbiased model comparison">
            <thead>
              <tr>
                <th>Aspect</th>
                <th>⚠️ Biased Model</th>
                <th>✓ Unbiased Model</th>
              </tr>
            </thead>
            <tbody>
              {comparison.map((row, i) => (
                <tr key={i}>
                  <td className="compare-aspect">{row.aspect}</td>
                  <td className="compare-bad">{row.biased}</td>
                  <td className="compare-good">{row.unbiased}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Tech stack */}
      <div className="hiw-stack">
        <h2 className="hiw-stack__title">Technology Stack</h2>
        <div className="stack-grid">
          {[
            { name: "PyTorch", role: "Model training & inference", icon: "🔥" },
            { name: "OpenCV", role: "Grad-CAM heatmap rendering", icon: "👁" },
            { name: "FastAPI", role: "REST API + Auth + Rate limiting", icon: "⚡" },
            { name: "SQLite + SQLAlchemy", role: "Prediction history logging", icon: "🗄" },
            { name: "Cloudinary", role: "Heatmap cloud object storage", icon: "☁️" },
            { name: "Redis", role: "In-memory caching layer", icon: "⚙️" },
            { name: "React 19", role: "Frontend SPA", icon: "⚛" },
            { name: "Docker", role: "Containerized deployment", icon: "🐳" },
          ].map((tech) => (
            <div key={tech.name} className="stack-item">
              <span className="stack-item__icon">{tech.icon}</span>
              <div>
                <p className="stack-item__name">{tech.name}</p>
                <p className="stack-item__role">{tech.role}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Team */}
      <div className="hiw-team">
        <h2 className="hiw-team__title">Team 85</h2>
        <div className="team-grid">
          {[
            { name: "Mayank", role: "Team Leader · Backend & ML", icon: "🧠" },
            { name: "Radhika Gupta", role: "Frontend · React & UI/UX", icon: "🎨" },
            { name: "Naitik Agarwal", role: "ML Research & Training", icon: "📊" },
          ].map((member) => (
            <div key={member.name} className="team-card">
              <span className="team-card__icon">{member.icon}</span>
              <p className="team-card__name">{member.name}</p>
              <p className="team-card__role">{member.role}</p>
            </div>
          ))}
        </div>
        <p className="team-mentor">Mentor: Mr. Preshit Desai · GLA University · 4th Semester AIML</p>
      </div>
    </div>
  );
}
