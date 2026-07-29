import { NavLink } from "react-router-dom";
import { useAppStore } from "../store/useAppStore";

const NAV_ITEMS = [
  { to: "/", label: "🛡️ Build Explorer" },
  { to: "/solver-comparison", label: "🏆 Solver Comparison" },
  { to: "/training-monitor", label: "📈 Training Monitor" },
  { to: "/item-database", label: "📚 Item Database" },
];

const VERSION = "0.1.0";

export function Sidebar() {
  const { autoRefresh, refreshInterval, setAutoRefresh, setRefreshInterval } = useAppStore();

  return (
    <aside className="sidebar">
      <h1 className="sidebar-title">⚔️ Build Optimizer</h1>
      <hr className="section-divider" />

      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) => `sidebar-nav-item${isActive ? " active" : ""}`}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>

      <hr className="section-divider" />
      <h3>🔄 Auto-Refresh</h3>
      <label className="sidebar-checkbox">
        <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />
        Enable Auto-Refresh
      </label>
      <label className="sidebar-slider">
        Refresh Interval ({refreshInterval}s)
        <input
          type="range"
          min={2}
          max={30}
          value={refreshInterval}
          disabled={!autoRefresh}
          onChange={(e) => setRefreshInterval(Number(e.target.value))}
        />
      </label>

      <hr className="section-divider" />
      <div className="sidebar-about">
        <p>Build Optimizer</p>
        <p>Control Tower v{VERSION}</p>
      </div>
    </aside>
  );
}
