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

// --- Slay the Spire 2 Advisor Types (Track SA1/SA6/SA7) ---
export interface CardEntry {
  card_id: string;
  count: number;
}

export interface RunContextInput {
  act?: number;
  floor?: number;
  hp_pct?: number;
  gold?: number;
  relics?: string[];
  potions?: string[];
}

export interface AdvisorPreferencesInput {
  tempo_weight?: number;
  synergy_weight?: number;
  dilution_weight?: number;
  mc_weight?: number;
  mc_rollouts?: number;
  seed?: number;
}

export interface Sts2AdvisorRequest {
  character: string;
  deck: CardEntry[];
  offers: string[];
  context?: RunContextInput;
  preferences?: AdvisorPreferencesInput;
}

export interface ChoiceMetrics {
  tempo_score: number;
  synergy_score: number;
  dilution_penalty: number;
  mc_projected_mean: number;
  mc_projected_ci_lower: number;
  mc_projected_ci_upper: number;
}

export interface AdvisorChoice {
  action: "skip" | "take" | string;
  card_id?: string | null;
  card_name?: string | null;
  is_upgrade: boolean;
  rank: number;
  total_score: number;
  score_delta: number;
  metrics: ChoiceMetrics;
  pareto_optimal: boolean;
  synergy_deltas: string[];
  explanation: string;
}

export interface Sts2AdvisorResponse {
  status: "ok" | "error" | "blocked" | string;
  character: string;
  evaluated_at: string;
  base_deck_size: number;
  choices: AdvisorChoice[];
  pareto_front: string[];
  recommendation: string;
  diagnostics?: string | null;
}

