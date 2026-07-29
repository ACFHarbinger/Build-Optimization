import { useEffect, useState } from "react";
import { getRunArtifacts, getRunLatestMetrics, listExperiments, listTrackedRuns } from "../lib/tauriApi";
import { Experiment, TrackedRun } from "../lib/types";

export interface TrackedRunsPanelProps {
  /** Called with the result-artifact path when a run with a result artifact is selected. */
  onSelectResultPath: (path: string) => void;
}

/** Browses the `middleware/src/tracking` SQLite database (experiments/runs),
 * resolving each run to its logged result-JSON artifact path on selection.
 * See moon/ROADMAP.md item T6. */
export function TrackedRunsPanel({ onSelectResultPath }: TrackedRunsPanelProps) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [experimentName, setExperimentName] = useState<string>("");
  const [runs, setRuns] = useState<TrackedRun[]>([]);
  const [metrics, setMetrics] = useState<Record<string, Record<string, number>>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listExperiments()
      .then((exps) => {
        setExperiments(exps);
        setExperimentName((prev) => (exps.some((e) => e.name === prev) ? prev : (exps[0]?.name ?? "")));
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!experimentName) {
      setRuns([]);
      return;
    }
    listTrackedRuns(experimentName).then((found) => {
      setRuns(found);
      Promise.all(found.map((r) => getRunLatestMetrics(r.id).then((m) => [r.id, m] as const))).then((pairs) =>
        setMetrics(Object.fromEntries(pairs)),
      );
    });
  }, [experimentName]);

  const handleSelect = async (runId: string) => {
    const artifacts = await getRunArtifacts(runId);
    const result = artifacts.find((a) => a.artifact_type === "result");
    if (result) onSelectResultPath(result.path);
  };

  if (loading) return null;
  if (experiments.length === 0) {
    return <p className="caption">No tracked runs yet — run `python main.py policy=policy_sa game=rpg` to populate the tracking database.</p>;
  }

  return (
    <div className="tracked-runs-panel">
      <label>
        Experiment
        <select value={experimentName} onChange={(e) => setExperimentName(e.target.value)}>
          {experiments.map((exp) => (
            <option key={exp.id} value={exp.name}>
              {exp.name}
            </option>
          ))}
        </select>
      </label>
      <table className="data-table">
        <thead>
          <tr>
            <th>Solver</th>
            <th>Score</th>
            <th>Status</th>
            <th>Started</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.id} className="tracked-run-row" onClick={() => handleSelect(run.id)}>
              <td>{run.run_type}</td>
              <td>{metrics[run.id]?.score?.toFixed(2) ?? "—"}</td>
              <td>{run.status}</td>
              <td>{new Date(run.start_time).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
