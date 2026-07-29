import { invoke } from "@tauri-apps/api/core";
import { BuildItem, Experiment, RunArtifact, SolverResult, TrackedRun, TrainingRecord } from "./types";

export const listSolverResults = () => invoke<string[]>("list_solver_results");
export const readSolverResult = (path: string) => invoke<SolverResult>("read_solver_result", { path });
export const listItemFiles = () => invoke<string[]>("list_item_files");
export const readItemsJson = (path: string) => invoke<BuildItem[]>("read_items_json", { path });
export const listTrainingRuns = () => invoke<string[]>("list_training_runs");
export const readTrainingLog = (runDir: string) => invoke<TrainingRecord[]>("read_training_log", { runDir });

// Tracking database (middleware/src/tracking) — see moon/ROADMAP.md item T6.
export const listExperiments = () => invoke<Experiment[]>("list_experiments");
export const listTrackedRuns = (experimentName?: string) =>
  invoke<TrackedRun[]>("list_tracked_runs", { experimentName });
export const getRunParams = (runId: string) => invoke<Record<string, unknown>>("get_run_params", { runId });
export const getRunLatestMetrics = (runId: string) =>
  invoke<Record<string, number>>("get_run_latest_metrics", { runId });
export const getRunArtifacts = (runId: string) => invoke<RunArtifact[]>("get_run_artifacts", { runId });

export function basename(path: string): string {
  return path.split(/[/\\]/).pop() ?? path;
}
