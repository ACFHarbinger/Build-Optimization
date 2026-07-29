/** Ported from middleware/ui/styles/kpi.py::format_number. */
export function formatNumber(value: number, precision = 2): string {
  if (Math.abs(value) >= 1000) {
    return value.toLocaleString(undefined, {
      minimumFractionDigits: precision,
      maximumFractionDigits: precision,
    });
  }
  return value.toFixed(precision);
}

export function formatMaybeNumber(value: number | string): string {
  return typeof value === "number" ? formatNumber(value) : String(value);
}
