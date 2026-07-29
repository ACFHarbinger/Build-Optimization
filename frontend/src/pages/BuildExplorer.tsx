import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { KpiRow } from "../components/KpiRow";
import { StatusPill } from "../components/StatusPill";
import { RARITY_COLORS } from "../lib/colors";
import { basename, listSolverResults, readSolverResult } from "../lib/tauriApi";
import { useAutoRefresh } from "../lib/useAutoRefresh";
import { BuildItem, SolverResult } from "../lib/types";

const DEMO_BUILD: SolverResult = {
  solver: "demo",
  score: 342.5,
  cost: 4200,
  budget: 5000,
  items: [
    { name: "Flame Sword", slot: "WEAPON", rarity: "EPIC", cost: 1200, stats: { attack: 85, critical_rate: 12 } },
    { name: "Dragon Helm", slot: "HELMET", rarity: "RARE", cost: 800, stats: { defense: 45, health: 60 } },
    { name: "Shadow Vest", slot: "CHEST", rarity: "LEGENDARY", cost: 1500, stats: { defense: 70, speed: 25 } },
    { name: "Swift Boots", slot: "BOOTS", rarity: "UNCOMMON", cost: 400, stats: { speed: 40, defense: 15 } },
    { name: "Ruby Ring", slot: "RING_1", rarity: "RARE", cost: 300, stats: { critical_damage: 30, attack: 10 } },
  ],
  synergies: ["Fire Mastery (2pc)", "Shadow Set (1pc)"],
};

function StatRadar({ items }: { items: BuildItem[] }) {
  const totals = useMemo(() => {
    const t: Record<string, number> = {};
    for (const item of items) {
      for (const [stat, val] of Object.entries(item.stats ?? {})) {
        t[stat] = (t[stat] ?? 0) + val;
      }
    }
    return t;
  }, [items]);

  const categories = Object.keys(totals);
  if (categories.length === 0) return <p>No stats to display.</p>;

  const values = Object.values(totals);
  const option = {
    radar: {
      indicator: categories.map((name) => ({ name, max: Math.max(...values) * 1.2 })),
      splitLine: { lineStyle: { color: "#e8eaed" } },
    },
    series: [
      {
        type: "radar",
        data: [{ value: values, areaStyle: { color: "rgba(26, 115, 232, 0.15)" }, lineStyle: { color: "#1a73e8", width: 2 } }],
      },
    ],
  };
  return <ReactECharts option={option} style={{ height: 320 }} />;
}

function BuildDetail({ data }: { data: SolverResult }) {
  const items = data.items ?? [];
  const score = data.score ?? 0;
  const cost = data.cost ?? 0;
  const budget = data.budget ?? 0;
  const synergies = data.synergies ?? [];

  const kpi = {
    Score: score,
    Cost: cost,
    "Budget Left": Math.max(budget - cost, 0),
    Items: items.length,
    Synergies: synergies.length,
  };

  return (
    <>
      <KpiRow metrics={kpi} />
      <hr className="section-divider" />
      <div className="two-col">
        <div>
          <h3>📦 Equipped Items</h3>
          {items.length ? (
            items.map((item, i) => {
              const color = RARITY_COLORS[item.rarity] ?? "#9e9e9e";
              const statsStr = Object.entries(item.stats ?? {})
                .map(([k, v]) => `${k}: ${v}`)
                .join(", ");
              return (
                <div key={i} className="item-row" style={{ borderLeftColor: color }}>
                  <strong style={{ color }}>{item.name}</strong>{" "}
                  <span className="item-slot">({item.slot})</span>
                  <br />
                  <span className="item-stats">
                    {statsStr} — {item.cost.toLocaleString()}g
                  </span>
                </div>
              );
            })
          ) : (
            <p>No items equipped.</p>
          )}
        </div>
        <div>
          <h3>📊 Stat Distribution</h3>
          <StatRadar items={items} />
        </div>
      </div>

      {synergies.length > 0 && (
        <>
          <hr className="section-divider" />
          <h3>✨ Active Synergies</h3>
          {synergies.map((syn) => (
            <StatusPill key={syn} kind="info">
              {syn}
            </StatusPill>
          ))}
        </>
      )}

      <hr className="section-divider" />
      <p className="caption">
        Solver: <strong>{data.solver ?? "unknown"}</strong>
      </p>
    </>
  );
}

export function BuildExplorer() {
  const [resultFiles, setResultFiles] = useState<string[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [data, setData] = useState<SolverResult | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = () => {
    setLoading(true);
    listSolverResults()
      .then((files) => {
        setResultFiles(files);
        setSelected((prev) => (files.includes(prev) ? prev : (files[0] ?? "")));
      })
      .finally(() => setLoading(false));
  };

  useEffect(refresh, []);
  useAutoRefresh(refresh);

  useEffect(() => {
    if (!selected) {
      setData(null);
      return;
    }
    readSolverResult(selected).then(setData);
  }, [selected]);

  return (
    <div>
      <h2>🛡️ Build Explorer</h2>

      {loading ? (
        <p>Loading…</p>
      ) : resultFiles.length === 0 ? (
        <>
          <p className="info-banner">No build results found. Run a solver first to generate output files in `outputs/`.</p>
          <h3>Demo Build</h3>
          <BuildDetail data={DEMO_BUILD} />
        </>
      ) : (
        <>
          <label>
            Select Result File
            <select value={selected} onChange={(e) => setSelected(e.target.value)}>
              {resultFiles.map((f) => (
                <option key={f} value={f}>
                  {basename(f)}
                </option>
              ))}
            </select>
          </label>
          {data && <BuildDetail data={data} />}
        </>
      )}
    </div>
  );
}
