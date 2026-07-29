import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { KpiRow } from "../components/KpiRow";
import { CHART_COLORS } from "../lib/colors";
import { basename, listTrainingRuns, readTrainingLog } from "../lib/tauriApi";
import { useAutoRefresh } from "../lib/useAutoRefresh";
import { TrainingRecord } from "../lib/types";

function numericColumns(rows: TrainingRecord[]): string[] {
  if (rows.length === 0) return [];
  return Object.keys(rows[0]).filter((k) => typeof rows[0][k] === "number");
}

function rollingMean(values: number[], window: number): number[] {
  if (window <= 1) return values;
  const out: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const start = Math.max(0, i - window + 1);
    const slice = values.slice(start, i + 1);
    out.push(slice.reduce((a, b) => a + b, 0) / slice.length);
  }
  return out;
}

function DemoTraining() {
  const epochs = Array.from({ length: 50 }, (_, i) => i + 1);
  const loss = epochs.map((e) => 2.0 * Math.exp(-0.05 * e) + (Math.random() - 0.5) * 0.1);
  const reward = epochs.map((e) => 50 * (1 - Math.exp(-0.08 * e)) + (Math.random() - 0.5) * 4);

  const kpi = {
    Epochs: 50,
    "Latest Loss": Number(loss[loss.length - 1].toFixed(4)),
    "Best Reward": Number(Math.max(...reward).toFixed(2)),
  };

  return (
    <>
      <h3>Demo Training Curves</h3>
      <KpiRow metrics={kpi} />
      <hr className="section-divider" />
      <div className="two-col">
        <ReactECharts
          option={{
            title: { text: "Training Loss", textStyle: { fontSize: 13 } },
            xAxis: { type: "category", data: epochs, name: "Epoch" },
            yAxis: { type: "value", name: "Loss" },
            series: [{ type: "line", data: loss, showSymbol: false, lineStyle: { color: CHART_COLORS[3], width: 2 } }],
            grid: { left: 50, right: 20, top: 40, bottom: 40 },
          }}
          style={{ height: 350 }}
        />
        <ReactECharts
          option={{
            title: { text: "Reward", textStyle: { fontSize: 13 } },
            xAxis: { type: "category", data: epochs, name: "Epoch" },
            yAxis: { type: "value", name: "Reward" },
            series: [{ type: "line", data: reward, showSymbol: false, lineStyle: { color: CHART_COLORS[2], width: 2 } }],
            grid: { left: 50, right: 20, top: 40, bottom: 40 },
          }}
          style={{ height: 350 }}
        />
      </div>
    </>
  );
}

export function TrainingMonitor() {
  const [runs, setRuns] = useState<string[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [logs, setLogs] = useState<Record<string, TrainingRecord[]>>({});
  const [metric, setMetric] = useState<string>("");
  const [smoothing, setSmoothing] = useState(1);
  const [loading, setLoading] = useState(true);

  const refresh = () => {
    setLoading(true);
    listTrainingRuns()
      .then((found) => {
        setRuns(found);
        setSelected((prev) => (prev.length ? prev.filter((p) => found.includes(p)) : found.slice(0, 2)));
      })
      .finally(() => setLoading(false));
  };

  useEffect(refresh, []);
  useAutoRefresh(refresh);

  useEffect(() => {
    Promise.all(selected.map((run) => readTrainingLog(run).then((data) => [run, data] as const))).then((pairs) => {
      setLogs(Object.fromEntries(pairs));
    });
  }, [selected]);

  const availableMetrics = useMemo(() => {
    const first = selected[0];
    return first ? numericColumns(logs[first] ?? []) : [];
  }, [selected, logs]);

  useEffect(() => {
    if (availableMetrics.length && !availableMetrics.includes(metric)) {
      setMetric(availableMetrics[0]);
    }
  }, [availableMetrics, metric]);

  const toggle = (run: string) => {
    setSelected((prev) => (prev.includes(run) ? prev.filter((p) => p !== run) : [...prev, run]));
  };

  const series = selected
    .map((run, i) => {
      const rows = logs[run] ?? [];
      if (!metric || rows.length === 0 || !(metric in (rows[0] ?? {}))) return null;
      const values = rollingMean(
        rows.map((r) => Number(r[metric] ?? 0)),
        smoothing,
      );
      const x = rows.map((r, idx) => (r.epoch !== undefined ? r.epoch : idx));
      return { name: basename(run), values, x, color: CHART_COLORS[i % CHART_COLORS.length] };
    })
    .filter((s): s is NonNullable<typeof s> => s !== null);

  const latestByRun = Object.fromEntries(series.map((s) => [s.name, s.values[s.values.length - 1] ?? 0]));
  const latestVals = Object.values(latestByRun);
  const summaryKpi =
    series.length > 0
      ? { Runs: selected.length, Best: metric.toLowerCase().includes("loss") ? Math.min(...latestVals) : Math.max(...latestVals) }
      : null;

  return (
    <div>
      <h2>📈 Training Monitor</h2>

      {loading ? (
        <p>Loading…</p>
      ) : runs.length === 0 ? (
        <>
          <p className="info-banner">
            No training runs found. Start RL training to generate log files (`metrics.csv` or `training_log.jsonl` in
            `outputs/`).
          </p>
          <DemoTraining />
        </>
      ) : (
        <>
          <fieldset className="multiselect">
            <legend>Select Runs</legend>
            {runs.map((r) => (
              <label key={r}>
                <input type="checkbox" checked={selected.includes(r)} onChange={() => toggle(r)} />
                {basename(r)}
              </label>
            ))}
          </fieldset>

          {selected.length === 0 ? (
            <p className="info-banner">Select at least one run.</p>
          ) : availableMetrics.length === 0 ? (
            <p className="error-banner">No numeric columns found in training log.</p>
          ) : (
            <>
              <label>
                Primary Metric
                <select value={metric} onChange={(e) => setMetric(e.target.value)}>
                  {availableMetrics.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </label>
              <label className="sidebar-slider">
                Smoothing ({smoothing})
                <input
                  type="range"
                  min={1}
                  max={50}
                  value={smoothing}
                  onChange={(e) => setSmoothing(Number(e.target.value))}
                />
              </label>

              {summaryKpi && (
                <>
                  <KpiRow metrics={summaryKpi} />
                  <hr className="section-divider" />
                </>
              )}

              <ReactECharts
                option={{
                  title: { text: `${metric} over Training`, textStyle: { fontSize: 13 } },
                  legend: { top: 0 },
                  xAxis: { type: "category", data: series[0]?.x ?? [], name: "Epoch" },
                  yAxis: { type: "value", name: metric },
                  series: series.map((s) => ({
                    type: "line",
                    name: s.name,
                    data: s.values,
                    showSymbol: false,
                    lineStyle: { color: s.color, width: 2 },
                  })),
                  grid: { left: 60, right: 20, top: 60, bottom: 40 },
                }}
                style={{ height: 450 }}
              />
            </>
          )}
        </>
      )}
    </div>
  );
}
