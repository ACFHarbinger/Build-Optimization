import { invoke } from "@tauri-apps/api/core";
import { BuildItem, SolverResult, TrainingRecord } from "./types";

export const listSolverResults = () => invoke<string[]>("list_solver_results");
export const readSolverResult = (path: string) => invoke<SolverResult>("read_solver_result", { path });
export const listItemFiles = () => invoke<string[]>("list_item_files");
export const readItemsJson = (path: string) => invoke<BuildItem[]>("read_items_json", { path });
export const listTrainingRuns = () => invoke<string[]>("list_training_runs");
export const readTrainingLog = (runDir: string) => invoke<TrainingRecord[]>("read_training_log", { runDir });

export function basename(path: string): string {
  return path.split(/[/\\]/).pop() ?? path;
}
