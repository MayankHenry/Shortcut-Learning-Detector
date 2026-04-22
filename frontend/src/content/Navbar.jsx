// src/components/Navbar.jsx
// Shared top navigation bar used across all pages.
// Receives auth state and current page from App, calls onLogout when needed.

export default function Navbar({ username, onLogout, currentPage, onNavigate }) {
  const navItems = [
    { id: "analyzer", label: "Analyzer" },
    { id: "howitworks", label: "How It Works" },
    ...(username ? [{ id: "admin", label: "Admin Logs" }] : []),
  ];

  return (
    <header className="header">
      <div className="header__inner">
        {/* Logo */}
        <button
          className="header__logo logo-btn"
          onClick={() => onNavigate("analyzer")}
          aria-label="Go to home"
        >
          <span className="logo-icon">⬡</span>
          <span className="logo-text">ShortcutDetect</span>
        </button>

        {/* Nav links */}
        <nav className="header__nav" aria-label="Main navigation">
          <div className="nav-links">
            {navItems.map((item) => (
              <button
                key={item.id}
                className={`nav-link ${currentPage === item.id ? "nav-link--active" : ""}`}
                onClick={() => onNavigate(item.id)}
                aria-current={currentPage === item.id ? "page" : undefined}
              >
                {item.label}
              </button>
            ))}
          </div>

          <div className="header__right">
            {username ? (
              <div className="header__user">
                <span className="nav-badge">👤 {username}</span>
                <button className="logout-btn" onClick={onLogout}>
                  Sign out
                </button>
              </div>
            ) : (
              <span className="nav-badge">GLA University · Team 85</span>
            )}
          </div>
        </nav>
      </div>
    </header>
  );
}
