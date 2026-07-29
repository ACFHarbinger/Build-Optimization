import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { RARITY_COLORS, RARITY_ORDER } from "../lib/colors";
import { basename, listItemFiles, readItemsJson } from "../lib/tauriApi";
import { useAutoRefresh } from "../lib/useAutoRefresh";
import { BuildItem } from "../lib/types";

const DEMO_ITEMS: BuildItem[] = [
  { name: "Flame Sword", slot: "WEAPON", rarity: "EPIC", level: 15, cost: 1200, stats: { attack: 85, critical_rate: 12 }, tags: ["fire", "melee"] },
  { name: "Dragon Helm", slot: "HELMET", rarity: "RARE", level: 12, cost: 800, stats: { defense: 45, health: 60 }, tags: ["fire", "armor"] },
  { name: "Shadow Vest", slot: "CHEST", rarity: "LEGENDARY", level: 20, cost: 1500, stats: { defense: 70, speed: 25 }, tags: ["shadow", "armor"] },
  { name: "Swift Boots", slot: "BOOTS", rarity: "UNCOMMON", level: 5, cost: 400, stats: { speed: 40, defense: 15 }, tags: ["speed"] },
  { name: "Ruby Ring", slot: "RING_1", rarity: "RARE", level: 10, cost: 300, stats: { critical_damage: 30, attack: 10 }, tags: ["fire", "jewelry"] },
  { name: "Iron Shield", slot: "ACCESSORY_1", rarity: "COMMON", level: 1, cost: 100, stats: { defense: 25 }, tags: ["armor"] },
  { name: "Emerald Amulet", slot: "AMULET", rarity: "EPIC", level: 18, cost: 950, stats: { health: 80, speed: 10 }, tags: ["nature", "jewelry"] },
  { name: "Obsidian Gauntlets", slot: "GLOVES", rarity: "RARE", level: 14, cost: 650, stats: { attack: 30, defense: 35, critical_rate: 5 }, tags: ["shadow", "melee"] },
];

function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return sorted[base + 1] !== undefined ? sorted[base] + rest * (sorted[base + 1] - sorted[base]) : sorted[base];
}

function boxStats(values: number[]): [number, number, number, number, number] {
  const sorted = [...values].sort((a, b) => a - b);
  return [sorted[0], quantile(sorted, 0.25), quantile(sorted, 0.5), quantile(sorted, 0.75), sorted[sorted.length - 1]];
}

function ItemTable({ items }: { items: BuildItem[] }) {
  const [slotFilter, setSlotFilter] = useState<string[]>([]);
  const [rarityFilter, setRarityFilter] = useState<string[]>([]);
  const [tagFilter, setTagFilter] = useState<string[]>([]);

  const allSlots = useMemo(() => Array.from(new Set(items.map((i) => i.slot ?? "?"))).sort(), [items]);
  const allRarities = useMemo(() => Array.from(new Set(items.map((i) => i.rarity ?? "COMMON"))).sort(), [items]);
  const allTags = useMemo(() => Array.from(new Set(items.flatMap((i) => i.tags ?? []))).sort(), [items]);

  const levels = items.map((i) => i.level ?? 1);
  const minLvl = Math.min(...levels);
  const maxLvl = Math.max(...levels);
  const [levelRange, setLevelRange] = useState<[number, number]>([minLvl, maxLvl]);

  const toggle = (list: string[], setList: (v: string[]) => void, value: string) => {
    setList(list.includes(value) ? list.filter((v) => v !== value) : [...list, value]);
  };

  const filtered = items.filter((i) => {
    if (slotFilter.length && !slotFilter.includes(i.slot ?? "?")) return false;
    if (rarityFilter.length && !rarityFilter.includes(i.rarity ?? "COMMON")) return false;
    if (tagFilter.length && !(i.tags ?? []).some((t) => tagFilter.includes(t))) return false;
    const lvl = i.level ?? 1;
    return lvl >= levelRange[0] && lvl <= levelRange[1];
  });

  const rows = filtered.map((item) => {
    const stats = item.stats ?? {};
    const total = Object.values(stats).reduce((a, b) => a + b, 0);
    const cost = item.cost ?? 0;
    return {
      ...item,
      total,
      efficiency: Number((total / Math.max(cost, 1)).toFixed(3)),
    };
  });

  const rarityGroups = RARITY_ORDER.filter((r) => rows.some((row) => row.rarity === r)).map((r) => ({
    rarity: r,
    values: rows.filter((row) => row.rarity === r).map((row) => row.total),
  }));

  return (
    <>
      <div className="filters-row">
        <fieldset className="multiselect">
          <legend>Filter by Slot</legend>
          {allSlots.map((s) => (
            <label key={s}>
              <input type="checkbox" checked={slotFilter.includes(s)} onChange={() => toggle(slotFilter, setSlotFilter, s)} />
              {s}
            </label>
          ))}
        </fieldset>
        <fieldset className="multiselect">
          <legend>Filter by Rarity</legend>
          {allRarities.map((r) => (
            <label key={r}>
              <input type="checkbox" checked={rarityFilter.includes(r)} onChange={() => toggle(rarityFilter, setRarityFilter, r)} />
              {r}
            </label>
          ))}
        </fieldset>
        <fieldset className="multiselect">
          <legend>Filter by Tag</legend>
          {allTags.map((t) => (
            <label key={t}>
              <input type="checkbox" checked={tagFilter.includes(t)} onChange={() => toggle(tagFilter, setTagFilter, t)} />
              {t}
            </label>
          ))}
        </fieldset>
      </div>

      {minLvl < maxLvl && (
        <label className="sidebar-slider">
          Level Range ({levelRange[0]}–{levelRange[1]})
          <input
            type="range"
            min={minLvl}
            max={maxLvl}
            value={levelRange[0]}
            onChange={(e) => setLevelRange([Number(e.target.value), levelRange[1]])}
          />
          <input
            type="range"
            min={minLvl}
            max={maxLvl}
            value={levelRange[1]}
            onChange={(e) => setLevelRange([levelRange[0], Number(e.target.value)])}
          />
        </label>
      )}

      <p className="caption">
        Showing {filtered.length} of {items.length} items
      </p>

      <table className="data-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Slot</th>
            <th>Rarity</th>
            <th>Level</th>
            <th>Cost</th>
            <th>Total Stats</th>
            <th>Efficiency</th>
            <th>Tags</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td>{r.name}</td>
              <td>{r.slot}</td>
              <td style={{ color: RARITY_COLORS[r.rarity] }}>{r.rarity}</td>
              <td>{r.level ?? 1}</td>
              <td>{r.cost}</td>
              <td>{r.total}</td>
              <td>{r.efficiency}</td>
              <td>{(r.tags ?? []).join(", ")}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <hr className="section-divider" />
      <div className="two-col">
        <div>
          <h3>📊 Stats by Rarity</h3>
          {rarityGroups.length > 0 && (
            <ReactECharts
              option={{
                xAxis: { type: "category", data: rarityGroups.map((g) => g.rarity) },
                yAxis: { type: "value", name: "Total Stats" },
                series: [
                  {
                    type: "boxplot",
                    data: rarityGroups.map((g) => boxStats(g.values)),
                    itemStyle: { color: (p: { dataIndex: number }) => RARITY_COLORS[rarityGroups[p.dataIndex].rarity] },
                  },
                ],
                grid: { left: 50, right: 20, top: 20, bottom: 40 },
              }}
              style={{ height: 300 }}
            />
          )}
        </div>
        <div>
          <h3>📊 Cost vs Efficiency</h3>
          <ReactECharts
            option={{
              tooltip: { formatter: (p: { data: [number, number, string] }) => `${p.data[2]}` },
              xAxis: { type: "value", name: "Cost" },
              yAxis: { type: "value", name: "Efficiency" },
              series: [
                {
                  type: "scatter",
                  symbolSize: (d: [number, number, string, number]) => Math.max(8, Math.sqrt(d[3]) * 3),
                  data: rows.map((r) => [r.cost, r.efficiency, r.name, r.total]),
                  itemStyle: { color: (p: { dataIndex: number }) => RARITY_COLORS[rows[p.dataIndex].rarity] },
                },
              ],
              grid: { left: 50, right: 20, top: 20, bottom: 40 },
            }}
            style={{ height: 300 }}
          />
        </div>
      </div>
    </>
  );
}

export function ItemDatabase() {
  const [files, setFiles] = useState<string[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [items, setItems] = useState<BuildItem[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = () => {
    setLoading(true);
    listItemFiles()
      .then((found) => {
        setFiles(found);
        setSelected((prev) => (found.includes(prev) ? prev : (found[0] ?? "")));
      })
      .finally(() => setLoading(false));
  };

  useEffect(refresh, []);
  useAutoRefresh(refresh);

  useEffect(() => {
    if (!selected) {
      setItems([]);
      return;
    }
    readItemsJson(selected).then(setItems);
  }, [selected]);

  return (
    <div>
      <h2>📚 Item Database</h2>

      {loading ? (
        <p>Loading…</p>
      ) : files.length === 0 ? (
        <>
          <p className="info-banner">No item data files found. Place item JSON files in `data/`.</p>
          <h3>Demo Items</h3>
          <ItemTable items={DEMO_ITEMS} />
        </>
      ) : (
        <>
          <label>
            Select Data File
            <select value={selected} onChange={(e) => setSelected(e.target.value)}>
              {files.map((f) => (
                <option key={f} value={f}>
                  {basename(f)}
                </option>
              ))}
            </select>
          </label>
          {items.length > 0 ? <ItemTable items={items} /> : <p className="info-banner">No items found in the selected file.</p>}
        </>
      )}
    </div>
  );
}
