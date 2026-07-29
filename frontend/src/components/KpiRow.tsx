import { kpiColorFor } from "../lib/colors";
import { formatMaybeNumber } from "../lib/format";

export interface KpiRowProps {
  metrics: Record<string, number | string>;
}

export function KpiRow({ metrics }: KpiRowProps) {
  const entries = Object.entries(metrics);
  return (
    <div className="metric-row">
      {entries.map(([label, value], i) => {
        const [start, end] = kpiColorFor(label, i);
        return (
          <div
            key={label}
            className="kpi-card"
            style={{ background: `linear-gradient(135deg, ${start} 0%, ${end} 100%)` }}
          >
            <div className="label">{label}</div>
            <div className="value">{formatMaybeNumber(value)}</div>
          </div>
        );
      })}
    </div>
  );
}
