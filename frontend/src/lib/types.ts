export interface BuildItem {
  name: string;
  slot: string;
  rarity: string;
  cost: number;
  level?: number;
  stats: Record<string, number>;
  tags?: string[];
}

export interface SolverResult {
  solver?: string;
  score?: number;
  cost?: number;
  budget?: number;
  items?: BuildItem[];
  synergies?: string[];
  items_count?: number;
  time_s?: number;
  _filename?: string;
  [key: string]: unknown;
}

export type TrainingRecord = Record<string, number | string>;
