/** Color palettes ported from middleware/ui/styles/colors.py. */

export const KPI_COLORS: Record<string, [string, string]> = {
  Score: ["#43a047", "#2e7d32"],
  Cost: ["#fb8c00", "#ef6c00"],
  Items: ["#5c6bc0", "#3949ab"],
  Synergies: ["#8e24aa", "#6a1b9a"],
  "Budget Left": ["#00897b", "#00695c"],
  Attack: ["#e53935", "#c62828"],
  Defense: ["#1e88e5", "#1565c0"],
  Speed: ["#039be5", "#0277bd"],
  Health: ["#43a047", "#2e7d32"],
  "Crit Rate": ["#ff6f00", "#e65100"],
  "Crit Damage": ["#d81b60", "#ad1457"],
  Epochs: ["#5c6bc0", "#3949ab"],
  Steps: ["#7e57c2", "#5e35b1"],
  "Latest Loss": ["#e53935", "#c62828"],
  "Best Loss": ["#43a047", "#2e7d32"],
  "Latest Reward": ["#fb8c00", "#ef6c00"],
  "Best Reward": ["#00897b", "#00695c"],
  "Best Score": ["#43a047", "#2e7d32"],
  "Avg Score": ["#1e88e5", "#1565c0"],
  Solvers: ["#7e57c2", "#5e35b1"],
  Runs: ["#5c6bc0", "#3949ab"],
  Best: ["#43a047", "#2e7d32"],
};

export const KPI_FALLBACK_COLORS: [string, string][] = [
  ["#667eea", "#5a67d8"],
  ["#43a047", "#2e7d32"],
  ["#039be5", "#0277bd"],
  ["#fb8c00", "#ef6c00"],
  ["#e53935", "#c62828"],
  ["#7e57c2", "#5e35b1"],
];

export const CHART_COLORS: string[] = [
  "#1f77b4",
  "#ff7f0e",
  "#2ca02c",
  "#d62728",
  "#9467bd",
  "#8c564b",
  "#e377c2",
  "#7f7f7f",
  "#bcbd22",
  "#17becf",
];

export const RARITY_COLORS: Record<string, string> = {
  COMMON: "#9e9e9e",
  UNCOMMON: "#4caf50",
  RARE: "#2196f3",
  EPIC: "#9c27b0",
  LEGENDARY: "#ff9800",
};

export const RARITY_ORDER = ["COMMON", "UNCOMMON", "RARE", "EPIC", "LEGENDARY"];

export const STATUS_COLORS: Record<string, string> = {
  good: "#28a745",
  warning: "#ffc107",
  error: "#dc3545",
  info: "#17a2b8",
};

export function kpiColorFor(label: string, index: number): [string, string] {
  return KPI_COLORS[label] ?? KPI_FALLBACK_COLORS[index % KPI_FALLBACK_COLORS.length];
}
