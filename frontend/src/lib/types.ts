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

export interface Experiment {
  id: number;
  name: string;
  created_at: string;
  description: string;
}

export interface TrackedRun {
  id: string;
  experiment_name: string;
  name: string | null;
  status: string;
  run_type: string;
  start_time: string;
  end_time: string | null;
}

export interface RunArtifact {
  name: string;
  path: string;
  artifact_type: string;
  created_at: string;
}
