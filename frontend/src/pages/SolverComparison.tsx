import { useEffect, useState } from "react";
import ReactECharts from "echarts-for-react";
import { KpiRow } from "../components/KpiRow";
import { CHART_COLORS } from "../lib/colors";
import { basename, listSolverResults, readSolverResult } from "../lib/tauriApi";
import { useAutoRefresh } from "../lib/useAutoRefresh";
import { SolverResult } from "../lib/types";

const DEMO_RESULTS: SolverResult[] = [
  { solver: "Simulated Annealing", score: 342.5, cost: 4200, items_count: 5, time_s: 1.2 },
  { solver: "Genetic Algorithm", score: 338.1, cost: 4350, items_count: 6, time_s: 3.8 },
  { solver: "ALNS", score: 351.0, cost: 4100, items_count: 5, time_s: 2.1 },
  { solver: "Random", score: 210.7, cost: 3900, items_count: 4, time_s: 0.01 },
];

function barOption(names: string[], values: number[], title: string) {
  return {
    title: { text: title, textStyle: { fontSize: 13 } },
    xAxis: { type: "category", data: names },
    yAxis: { type: "value" },
    series: [
      {
        type: "bar",
        data: values,
        itemStyle: { color: (p: { dataIndex: number }) => CHART_COLORS[p.dataIndex % CHART_COLORS.length] },
        label: { show: true, position: "top" },
      },
    ],
    grid: { left: 50, right: 20, top: 40, bottom: 40 },
  };
}

function ComparisonView({ results }: { results: SolverResult[] }) {
  const scores = results.map((r) => r.score ?? 0);
  const costs = results.map((r) => r.cost ?? 0);
  const names = results.map((r, i) => r.solver ?? r._filename ?? `Run ${i}`);
  const bestScore = scores.length ? Math.max(...scores) : 0;
  const avgScore = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;

  return (
    <>
      <KpiRow metrics={{ Solvers: results.length, "Best Score": bestScore, "Avg Score": avgScore }} />
      <hr className="section-divider" />
      <div className="two-col">
        <div>
          <h3>📊 Score Comparison</h3>
          <ReactECharts option={barOption(names, scores, "")} style={{ height: 350 }} />
        </div>
        <div>
          <h3>💰 Cost Comparison</h3>
          <ReactECharts option={barOption(names, costs, "")} style={{ height: 350 }} />
        </div>
      </div>
      <hr className="section-divider" />
      <h3>📋 Detailed Results</h3>
      <table className="data-table">
        <thead>
          <tr>
            <th>Solver</th>
            <th>Score</th>
            <th>Cost</th>
            <th>Items</th>
            <th>Time (s)</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => (
            <tr key={i}>
              <td>{r.solver ?? "?"}</td>
              <td>{r.score ?? 0}</td>
              <td>{r.cost ?? 0}</td>
              <td>{r.items_count ?? r.items?.length ?? 0}</td>
              <td>{r.time_s ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

export function SolverComparison() {
  const [resultFiles, setResultFiles] = useState<string[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [results, setResults] = useState<SolverResult[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = () => {
    setLoading(true);
    listSolverResults()
      .then((files) => {
        setResultFiles(files);
        setSelected((prev) => (prev.length ? prev.filter((p) => files.includes(p)) : files.slice(0, 4)));
      })
      .finally(() => setLoading(false));
  };

  useEffect(refresh, []);
  useAutoRefresh(refresh);

  useEffect(() => {
    if (selected.length === 0) {
      setResults([]);
      return;
    }
    Promise.all(
      selected.map((path) => readSolverResult(path).then((data) => ({ ...data, _filename: basename(path) }))),
    ).then(setResults);
  }, [selected]);

  const toggle = (path: string) => {
    setSelected((prev) => (prev.includes(path) ? prev.filter((p) => p !== path) : [...prev, path]));
  };

  return (
    <div>
      <h2>🏆 Solver Comparison</h2>

      {loading ? (
        <p>Loading…</p>
      ) : resultFiles.length === 0 ? (
        <>
          <p className="info-banner">No solver results found in `outputs/`. Run solvers to generate results.</p>
          <h3>Demo Comparison</h3>
          <ComparisonView results={DEMO_RESULTS} />
        </>
      ) : (
        <>
          <fieldset className="multiselect">
            <legend>Select Results to Compare</legend>
            {resultFiles.map((f) => (
              <label key={f}>
                <input type="checkbox" checked={selected.includes(f)} onChange={() => toggle(f)} />
                {basename(f)}
              </label>
            ))}
          </fieldset>
          {selected.length === 0 ? (
            <p className="info-banner">Select at least one result file to compare.</p>
          ) : (
            <ComparisonView results={results} />
          )}
        </>
      )}
    </div>
  );
}
