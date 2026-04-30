// src/pages/AdminLogs.jsx
// Fetches prediction history from GET /admin/logs (requires JWT Bearer token).
// Response shape per log entry:
//   { id, model_type, predicted_class, confidence, original_image_url, heatmap_url }

import { useState, useEffect, useCallback } from "react";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// ── Helpers ──────────────────────────────────────────────────────────────────
const MODEL_BADGE = {
  biased: { label: "Biased", color: "badge--red" },
  unbiased: { label: "Unbiased", color: "badge--green" },
};

function ConfidencePill({ value }) {
  const pct = Math.round(value);
  const color =
    pct >= 70 ? "var(--accent-green)" : pct >= 40 ? "var(--accent-yellow)" : "var(--accent-red)";
  return (
    <span className="conf-pill" style={{ "--pill-color": color }}>
      {pct}%
    </span>
  );
}

// ── Component ────────────────────────────────────────────────────────────────
export default function AdminLogs({ token, onSessionExpired }) {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState("");
  const [filterModel, setFilterModel] = useState("all");
  const [sortDesc, setSortDesc] = useState(true);

  // ── Fetch logs ──
  const fetchLogs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/admin/logs`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.status === 401) {
        onSessionExpired();
        return;
      }
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error: ${res.status}`);
      }
      const data = await res.json();
      setLogs(data);
    } catch (err) {
      if (err.name === "TypeError") {
        setError("Cannot reach the backend. Make sure FastAPI is running on port 8000.");
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  }, [token, onSessionExpired]);

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  // ── Filter + sort ──
  const filtered = logs
    .filter((log) => {
      const matchModel = filterModel === "all" || log.model_type === filterModel;
      const matchSearch =
        search === "" ||
        log.predicted_class?.toLowerCase().includes(search.toLowerCase()) ||
        log.model_type?.toLowerCase().includes(search.toLowerCase());
      return matchModel && matchSearch;
    })
    .sort((a, b) => sortDesc ? b.id - a.id : a.id - b.id);

  // ── Stats ──
  const totalBiased = logs.filter((l) => l.model_type === "biased").length;
  const totalUnbiased = logs.filter((l) => l.model_type === "unbiased").length;
  const avgConfidence =
    logs.length > 0
      ? Math.round(logs.reduce((sum, l) => sum + l.confidence, 0) / logs.length)
      : 0;

  // ── Render ──
  return (
    <div className="admin-page">
      {/* Page header */}
      <section className="admin-hero">
        <p className="hero__eyebrow">Protected Route · JWT Required</p>
        <h1 className="admin-hero__title">Prediction Logs</h1>
        <p className="admin-hero__sub">
          Full history of every analysis run through the API, stored in SQLite.
        </p>
      </section>

      {/* Stats row */}
      {!loading && !error && logs.length > 0 && (
        <div className="stats-row">
          <div className="stat-card">
            <span className="stat-card__value">{logs.length}</span>
            <span className="stat-card__label">Total Predictions</span>
          </div>
          <div className="stat-card">
            <span className="stat-card__value" style={{ color: "var(--accent-red)" }}>
              {totalBiased}
            </span>
            <span className="stat-card__label">Biased Model</span>
          </div>
          <div className="stat-card">
            <span className="stat-card__value" style={{ color: "var(--accent-green)" }}>
              {totalUnbiased}
            </span>
            <span className="stat-card__label">Unbiased Model</span>
          </div>
          <div className="stat-card">
            <span className="stat-card__value">{avgConfidence}%</span>
            <span className="stat-card__label">Avg. Confidence</span>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="admin-controls">
        <input
          className="admin-search"
          type="text"
          placeholder="Search by digit or model…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          aria-label="Search logs"
        />
        <div className="admin-filters">
          {["all", "biased", "unbiased"].map((f) => (
            <button
              key={f}
              className={`filter-btn ${filterModel === f ? "filter-btn--active" : ""}`}
              onClick={() => setFilterModel(f)}
            >
              {f === "all" ? "All Models" : f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
          <button
            className="filter-btn"
            onClick={() => setSortDesc((p) => !p)}
            title="Toggle sort order"
          >
            {sortDesc ? "↓ Newest" : "↑ Oldest"}
          </button>
          <button className="filter-btn" onClick={fetchLogs} title="Refresh">
            ↺ Refresh
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="error-box" role="alert">
          <span className="error-box__icon">!</span>
          <div>
            <strong>Error</strong>
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="admin-skeleton">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="skeleton skeleton--row" />
          ))}
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && logs.length === 0 && (
        <div className="result-empty">
          <div className="result-empty__icon">◈</div>
          <p>No predictions yet. Run an analysis to see logs here.</p>
        </div>
      )}

      {/* Table */}
      {!loading && !error && filtered.length > 0 && (
        <div className="admin-table-wrap">
          <table className="admin-table" aria-label="Prediction logs">
            <thead>
              <tr>
                <th>#ID</th>
                <th>Model</th>
                <th>Predicted</th>
                <th>Confidence</th>
                <th>Heatmap</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((log) => {
                const badge = MODEL_BADGE[log.model_type] || {
                  label: log.model_type,
                  color: "badge--grey",
                };
                return (
                  <tr key={log.id}>
                    <td>
                      <span className="log-id">#{log.id}</span>
                    </td>
                    <td>
                      <span className={`badge ${badge.color}`}>{badge.label}</span>
                    </td>
                    <td>
                      <span className="log-digit">{log.predicted_class}</span>
                    </td>
                    <td>
                      <ConfidencePill value={log.confidence} />
                    </td>
                    <td>
                      {log.heatmap_url && log.heatmap_url !== "Generated Locally" ? (
                        <a
                          href={log.heatmap_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="heatmap-link"
                        >
                          View ↗
                        </a>
                      ) : (
                        <span className="log-local">Local</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="table-count">
            Showing {filtered.length} of {logs.length} records
          </p>
        </div>
      )}

      {/* No search results */}
      {!loading && !error && logs.length > 0 && filtered.length === 0 && (
        <div className="result-empty">
          <div className="result-empty__icon">◈</div>
          <p>No logs match your search.</p>
        </div>
      )}
    </div>
  );
}
