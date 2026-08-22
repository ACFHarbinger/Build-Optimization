import React, { useState } from "react";
import ReactECharts from "echarts-for-react";
import { runSts2Advisor } from "../lib/tauriApi";
import {
  AdvisorChoice,
  AdvisorPreferencesInput,
  CardEntry,
  RunContextInput,
  Sts2AdvisorRequest,
  Sts2AdvisorResponse,
} from "../lib/types";

// Common card suggestions for autocomplete/quick-selection
const KNOWN_IRONCLAD_CARDS = [
  "Strike",
  "Strike+",
  "Defend",
  "Defend+",
  "Bash",
  "Bash+",
  "Anger",
  "Clash",
  "Cleave",
  "Iron Wave",
  "Pommel Strike",
  "Twin Strike",
  "Thunderclap",
  "Sword Boomerang",
  "Whirlwind",
  "Pummel",
  "Heavy Blade",
  "Uppercut",
  "Carnage",
  "Carnage+",
  "Rampage",
  "Bludgeon",
  "Reaper",
  "Fiend Fire",
  "Feed",
  "Shrug It Off",
  "Armaments",
  "True Grit",
  "Flex",
  "Body Slam",
  "Body Slam+",
  "Entrench",
  "Second Wind",
  "Spot Weakness",
  "Sever Soul",
  "Disarm",
  "Barricade",
  "Berserk",
  "Demon Form",
  "Inflame",
  "Inflame+",
  "Feel No Pain",
  "Combust",
  "Rupture",
  "Corruption",
];

const PRESET_STARTER: CardEntry[] = [
  { card_id: "strike", count: 5 },
  { card_id: "defend", count: 4 },
  { card_id: "bash", count: 1 },
];

const PRESET_STRENGTH: CardEntry[] = [
  { card_id: "strike", count: 4 },
  { card_id: "defend", count: 3 },
  { card_id: "bash+", count: 1 },
  { card_id: "inflame", count: 1 },
  { card_id: "spot_weakness", count: 1 },
  { card_id: "twin_strike", count: 1 },
  { card_id: "heavy_blade", count: 1 },
  { card_id: "shrug_it_off", count: 2 },
];

const PRESET_BLOCK: CardEntry[] = [
  { card_id: "strike", count: 3 },
  { card_id: "defend+", count: 4 },
  { card_id: "body_slam+", count: 1 },
  { card_id: "entrench", count: 1 },
  { card_id: "barricade", count: 1 },
  { card_id: "iron_wave", count: 2 },
];

export function Advisor() {
  // Input State
  const [character] = useState<string>("ironclad");
  const [deck, setDeck] = useState<CardEntry[]>(PRESET_STARTER);
  const [newCardName, setNewCardName] = useState<string>("");
  const [newCardCount, setNewCardCount] = useState<number>(1);

  // 3 Offer Slots
  const [offer1, setOffer1] = useState<string>("Carnage");
  const [offer2, setOffer2] = useState<string>("Cleave");
  const [offer3, setOffer3] = useState<string>("Inflame");

  // Context State
  const [act, setAct] = useState<number>(1);
  const [floor, setFloor] = useState<number>(6);
  const [hpPct, setHpPct] = useState<number>(85);
  const [gold, setGold] = useState<number>(140);
  const [relics, setRelics] = useState<string>("Vajra, Anchor");
  const [potions, setPotions] = useState<string>("Strength Potion");

  // Preferences & Strategy
  const [strategyPreset, setStrategyPreset] = useState<string>("balanced");
  const [tempoWeight, setTempoWeight] = useState<number>(1.0);
  const [synergyWeight, setSynergyWeight] = useState<number>(1.0);
  const [dilutionWeight, setDilutionWeight] = useState<number>(1.2);
  const [mcWeight, setMcWeight] = useState<number>(0.8);
  const [mcRollouts, setMcRollouts] = useState<number>(500);
  const [seed, setSeed] = useState<number>(42);

  // UI / Execution State
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<Sts2AdvisorResponse | null>(null);

  // Deck helpers
  const totalDeckCards = deck.reduce((acc, c) => acc + c.count, 0);

  const handleApplyPreset = (preset: CardEntry[]) => {
    setDeck([...preset]);
  };

  const handleUpdateCardCount = (index: number, delta: number) => {
    const updated = [...deck];
    const newCount = updated[index].count + delta;
    if (newCount <= 0) {
      updated.splice(index, 1);
    } else {
      updated[index].count = newCount;
    }
    setDeck(updated);
  };

  const handleAddCard = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newCardName.trim()) return;
    const cardId = newCardName.trim().toLowerCase().replace(/\s+/g, "_");
    const existingIndex = deck.findIndex((c) => c.card_id === cardId);
    if (existingIndex >= 0) {
      const updated = [...deck];
      updated[existingIndex].count += newCardCount;
      setDeck(updated);
    } else {
      setDeck([...deck, { card_id: cardId, count: Math.max(1, newCardCount) }]);
    }
    setNewCardName("");
    setNewCardCount(1);
  };

  const handleStrategyChange = (presetKey: string) => {
    setStrategyPreset(presetKey);
    switch (presetKey) {
      case "tempo":
        setTempoWeight(1.8);
        setSynergyWeight(0.6);
        setDilutionWeight(0.9);
        break;
      case "synergy":
        setTempoWeight(0.7);
        setSynergyWeight(1.9);
        setDilutionWeight(1.0);
        break;
      case "dilution":
        setTempoWeight(0.9);
        setSynergyWeight(1.0);
        setDilutionWeight(2.2);
        break;
      default: // balanced
        setTempoWeight(1.0);
        setSynergyWeight(1.0);
        setDilutionWeight(1.2);
        break;
    }
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);

    const offersList = [offer1.trim(), offer2.trim(), offer3.trim()].filter(Boolean);
    if (offersList.length === 0) {
      setError("Please provide at least one card offer to evaluate.");
      setLoading(false);
      return;
    }

    const contextInput: RunContextInput = {
      act,
      floor,
      hp_pct: hpPct / 100.0,
      gold,
      relics: relics.split(",").map((s) => s.trim()).filter(Boolean),
      potions: potions.split(",").map((s) => s.trim()).filter(Boolean),
    };

    const preferencesInput: AdvisorPreferencesInput = {
      tempo_weight: tempoWeight,
      synergy_weight: synergyWeight,
      dilution_weight: dilutionWeight,
      mc_weight: mcWeight,
      mc_rollouts: mcRollouts,
      seed,
    };

    const payload: Sts2AdvisorRequest = {
      character,
      deck,
      offers: offersList,
      context: contextInput,
      preferences: preferencesInput,
    };

    try {
      const res = await runSts2Advisor(payload);
      setResponse(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  // Chart Options
  const getMetricsBarOption = (choices: AdvisorChoice[]) => {
    const names = choices.map((c) => c.card_name || (c.action === "skip" ? "Skip" : "Unknown"));
    const tempoData = choices.map((c) => c.metrics.tempo_score);
    const synergyData = choices.map((c) => c.metrics.synergy_score);
    const dilutionData = choices.map((c) => c.metrics.dilution_penalty);
    const mcData = choices.map((c) => c.metrics.mc_projected_mean);

    return {
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      legend: { data: ["Immediate Tempo", "Synergy Delta", "Dilution Penalty", "MC Projected Value"] },
      grid: { left: "3%", right: "4%", bottom: "3%", containLabel: true },
      xAxis: { type: "category", data: names },
      yAxis: { type: "value" },
      series: [
        { name: "Immediate Tempo", type: "bar", data: tempoData, itemStyle: { color: "#e53935" } },
        { name: "Synergy Delta", type: "bar", data: synergyData, itemStyle: { color: "#1e88e5" } },
        { name: "Dilution Penalty", type: "bar", data: dilutionData, itemStyle: { color: "#fb8c00" } },
        { name: "MC Projected Value", type: "bar", data: mcData, itemStyle: { color: "#43a047" } },
      ],
    };
  };

  const getMcConfidenceOption = (choices: AdvisorChoice[]) => {
    const names = choices.map((c) => c.card_name || (c.action === "skip" ? "Skip" : "Unknown"));
    const means = choices.map((c) => c.metrics.mc_projected_mean);
    const ciBands = choices.map((c) => [c.metrics.mc_projected_ci_lower, c.metrics.mc_projected_ci_upper]);

    return {
      tooltip: {
        trigger: "axis",
        formatter: (params: Array<{ dataIndex: number }>) => {
          const idx = params[0]?.dataIndex ?? 0;
          const choice = choices[idx];
          return `<strong>${choice.card_name || "Skip"}</strong><br/>
                  Mean Value: ${choice.metrics.mc_projected_mean.toFixed(1)}<br/>
                  95% Confidence Band: [${choice.metrics.mc_projected_ci_lower.toFixed(1)}, ${choice.metrics.mc_projected_ci_upper.toFixed(1)}]`;
        },
      },
      xAxis: { type: "category", data: names },
      yAxis: { type: "value", name: "Projected Value" },
      series: [
        {
          name: "Projected Mean",
          type: "line",
          data: means,
          symbol: "circle",
          symbolSize: 10,
          itemStyle: { color: "#673ab7" },
          lineStyle: { width: 3 },
        },
        {
          name: "CI Lower Bound",
          type: "line",
          data: ciBands.map((b) => b[0]),
          lineStyle: { type: "dashed", color: "#9575cd" },
          symbol: "none",
        },
        {
          name: "CI Upper Bound",
          type: "line",
          data: ciBands.map((b) => b[1]),
          lineStyle: { type: "dashed", color: "#9575cd" },
          symbol: "none",
        },
      ],
    };
  };

  return (
    <div className="advisor-page">
      <h1>🔮 Slay the Spire 2 — Screenshot Reward Advisor</h1>
      <p className="caption">
        Study Advisor for post-fight 3-card reward choices. Evaluates Immediate Tempo, Archetype Synergies, Draw Dilution,
        and Seeded Monte Carlo Projected Run Value.
      </p>

      {error && <div className="error-banner" style={{ marginBottom: 16 }}>⚠️ {error}</div>}

      <div className="two-col" style={{ alignItems: "start" }}>
        {/* Left Column: Input Config */}
        <div>
          {/* Card Offers Section */}
          <div className="advisor-card-section">
            <h3>🎴 Offered Card Reward (3 Choices)</h3>
            <div className="filters-row" style={{ gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
              <div>
                <label style={{ fontSize: 12, fontWeight: 600 }}>Offer 1</label>
                <input
                  type="text"
                  list="known-cards-list"
                  value={offer1}
                  onChange={(e) => setOffer1(e.target.value)}
                  className="advisor-input"
                />
              </div>
              <div>
                <label style={{ fontSize: 12, fontWeight: 600 }}>Offer 2</label>
                <input
                  type="text"
                  list="known-cards-list"
                  value={offer2}
                  onChange={(e) => setOffer2(e.target.value)}
                  className="advisor-input"
                />
              </div>
              <div>
                <label style={{ fontSize: 12, fontWeight: 600 }}>Offer 3</label>
                <input
                  type="text"
                  list="known-cards-list"
                  value={offer3}
                  onChange={(e) => setOffer3(e.target.value)}
                  className="advisor-input"
                />
              </div>
            </div>
            <datalist id="known-cards-list">
              {KNOWN_IRONCLAD_CARDS.map((c) => (
                <option key={c} value={c} />
              ))}
            </datalist>
          </div>

          {/* Current Deck Management */}
          <div className="advisor-card-section" style={{ marginTop: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h3>📦 Current Deck ({totalDeckCards} Cards)</h3>
              <div style={{ display: "flex", gap: 6 }}>
                <button type="button" className="btn-preset" onClick={() => handleApplyPreset(PRESET_STARTER)}>
                  Starter
                </button>
                <button type="button" className="btn-preset" onClick={() => handleApplyPreset(PRESET_STRENGTH)}>
                  Strength
                </button>
                <button type="button" className="btn-preset" onClick={() => handleApplyPreset(PRESET_BLOCK)}>
                  Block
                </button>
                <button type="button" className="btn-preset" onClick={() => setDeck([])}>
                  Clear
                </button>
              </div>
            </div>

            <div className="deck-card-list">
              {deck.map((card, idx) => (
                <div key={card.card_id} className="deck-card-item">
                  <span className="deck-card-name">{card.card_id}</span>
                  <div className="deck-card-counter">
                    <button type="button" onClick={() => handleUpdateCardCount(idx, -1)}>
                      -
                    </button>
                    <span>{card.count}</span>
                    <button type="button" onClick={() => handleUpdateCardCount(idx, 1)}>
                      +
                    </button>
                  </div>
                </div>
              ))}
            </div>

            <form onSubmit={handleAddCard} style={{ display: "flex", gap: 8, marginTop: 10 }}>
              <input
                type="text"
                list="known-cards-list"
                placeholder="Add card to deck..."
                value={newCardName}
                onChange={(e) => setNewCardName(e.target.value)}
                style={{ flex: 1 }}
                className="advisor-input"
              />
              <input
                type="number"
                min={1}
                max={20}
                value={newCardCount}
                onChange={(e) => setNewCardCount(Number(e.target.value))}
                style={{ width: 60 }}
                className="advisor-input"
              />
              <button type="submit" className="btn-primary" style={{ padding: "6px 12px" }}>
                Add
              </button>
            </form>
          </div>

          {/* Run Context & Parameters */}
          <details className="tracked-runs-details" style={{ marginTop: 16 }}>
            <summary>⚙️ Run Context & Preferences (Act, Gold, Relics, Weights)</summary>
            <div style={{ padding: "12px 0 0" }}>
              <div className="filters-row" style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 10 }}>
                <div>
                  <label style={{ fontSize: 12 }}>Act</label>
                  <input
                    type="number"
                    min={1}
                    max={4}
                    value={act}
                    onChange={(e) => setAct(Number(e.target.value))}
                    className="advisor-input"
                  />
                </div>
                <div>
                  <label style={{ fontSize: 12 }}>Floor</label>
                  <input
                    type="number"
                    min={1}
                    max={60}
                    value={floor}
                    onChange={(e) => setFloor(Number(e.target.value))}
                    className="advisor-input"
                  />
                </div>
                <div>
                  <label style={{ fontSize: 12 }}>HP %</label>
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={hpPct}
                    onChange={(e) => setHpPct(Number(e.target.value))}
                    className="advisor-input"
                  />
                </div>
                <div>
                  <label style={{ fontSize: 12 }}>Gold</label>
                  <input
                    type="number"
                    min={0}
                    value={gold}
                    onChange={(e) => setGold(Number(e.target.value))}
                    className="advisor-input"
                  />
                </div>
              </div>

              <div style={{ marginTop: 8 }}>
                <label style={{ fontSize: 12 }}>Owned Relics (comma-separated)</label>
                <input
                  type="text"
                  value={relics}
                  onChange={(e) => setRelics(e.target.value)}
                  className="advisor-input"
                />
              </div>

              <div style={{ marginTop: 8 }}>
                <label style={{ fontSize: 12 }}>Owned Potions (comma-separated)</label>
                <input
                  type="text"
                  value={potions}
                  onChange={(e) => setPotions(e.target.value)}
                  className="advisor-input"
                />
              </div>

              <div style={{ marginTop: 12 }}>
                <label style={{ fontSize: 12, fontWeight: 600 }}>Strategy Preset</label>
                <div style={{ display: "flex", gap: 8, marginTop: 4 }}>
                  {["balanced", "tempo", "synergy", "dilution"].map((preset) => (
                    <button
                      key={preset}
                      type="button"
                      className={`btn-preset ${strategyPreset === preset ? "active" : ""}`}
                      onClick={() => handleStrategyChange(preset)}
                    >
                      {preset.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>

              <div className="filters-row" style={{ gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
                <div>
                  <label style={{ fontSize: 11 }}>Tempo Weight ({tempoWeight})</label>
                  <input
                    type="range"
                    min={0}
                    max={3}
                    step={0.1}
                    value={tempoWeight}
                    onChange={(e) => setTempoWeight(Number(e.target.value))}
                    style={{ width: "100%" }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: 11 }}>Synergy Weight ({synergyWeight})</label>
                  <input
                    type="range"
                    min={0}
                    max={3}
                    step={0.1}
                    value={synergyWeight}
                    onChange={(e) => setSynergyWeight(Number(e.target.value))}
                    style={{ width: "100%" }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: 11 }}>Dilution Penalty ({dilutionWeight})</label>
                  <input
                    type="range"
                    min={0}
                    max={3}
                    step={0.1}
                    value={dilutionWeight}
                    onChange={(e) => setDilutionWeight(Number(e.target.value))}
                    style={{ width: "100%" }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: 11 }}>MC Weight ({mcWeight})</label>
                  <input
                    type="range"
                    min={0}
                    max={3}
                    step={0.1}
                    value={mcWeight}
                    onChange={(e) => setMcWeight(Number(e.target.value))}
                    style={{ width: "100%" }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: 11 }}>MC Rollouts ({mcRollouts})</label>
                  <input
                    type="number"
                    min={50}
                    max={2000}
                    step={50}
                    value={mcRollouts}
                    onChange={(e) => setMcRollouts(Number(e.target.value))}
                    className="advisor-input"
                  />
                </div>
                <div>
                  <label style={{ fontSize: 11 }}>RNG Seed ({seed})</label>
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(Number(e.target.value))}
                    className="advisor-input"
                  />
                </div>
              </div>
            </div>
          </details>

          {/* Action CTA */}
          <button
            type="button"
            className="btn-primary"
            onClick={handleAnalyze}
            disabled={loading}
            style={{ width: "100%", marginTop: 16, padding: "12px 18px", fontSize: 16, fontWeight: 700 }}
          >
            {loading ? "⏳ Evaluating Offers & Monte Carlo Rollouts..." : "🔮 Analyze Reward Options"}
          </button>
        </div>

        {/* Right Column: Recommendation Banner & Visuals */}
        <div>
          {response ? (
            <div className="advisor-results-panel">
              {/* Recommendation Banner */}
              <div
                className="recommendation-banner"
                style={{
                  borderLeftColor: response.recommendation === "Skip" ? "#fb8c00" : "#2e7d32",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span className="status-pill good">RECOMMENDED CHOICE</span>
                  <span style={{ fontSize: 12, color: "#757575" }}>{response.evaluated_at}</span>
                </div>
                <h2 style={{ margin: "8px 0 4px", fontSize: 24 }}>
                  {response.recommendation === "Skip" ? "🚫 Skip Reward" : `✨ Take ${response.recommendation}`}
                </h2>
                <p style={{ margin: 0, fontSize: 14, color: "#37474f" }}>
                  {response.choices[0]?.explanation}
                </p>
              </div>

              {/* Pareto Frontier & Options Breakdown */}
              <h3 style={{ marginTop: 20 }}>📊 Evaluated Choices (Ranked)</h3>
              <table className="data-table" style={{ marginTop: 8 }}>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Choice</th>
                    <th>Score Delta</th>
                    <th>Tempo</th>
                    <th>Synergy</th>
                    <th>Dilution</th>
                    <th>MC 95% CI</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {response.choices.map((choice) => (
                    <tr
                      key={choice.card_name || choice.action}
                      style={{
                        backgroundColor: choice.rank === 1 ? "rgba(46, 125, 50, 0.08)" : undefined,
                        fontWeight: choice.rank === 1 ? 600 : 400,
                      }}
                    >
                      <td>#{choice.rank}</td>
                      <td>
                        <strong>{choice.card_name || "Skip"}</strong>
                        {choice.is_upgrade && <span className="status-pill info" style={{ marginLeft: 4 }}>+</span>}
                      </td>
                      <td style={{ color: choice.score_delta >= 0 ? "#2e7d32" : "#c62828" }}>
                        {choice.score_delta >= 0 ? `+${choice.score_delta.toFixed(1)}` : choice.score_delta.toFixed(1)}
                      </td>
                      <td>{choice.metrics.tempo_score.toFixed(1)}</td>
                      <td>{choice.metrics.synergy_score.toFixed(1)}</td>
                      <td style={{ color: "#e65100" }}>-{choice.metrics.dilution_penalty.toFixed(1)}</td>
                      <td>
                        [{choice.metrics.mc_projected_ci_lower.toFixed(1)}, {choice.metrics.mc_projected_ci_upper.toFixed(1)}]
                      </td>
                      <td>
                        {choice.pareto_optimal ? (
                          <span className="status-pill good">⭐ Pareto</span>
                        ) : (
                          <span className="status-pill warning">Dominated</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* Multi-Objective Breakdown Chart */}
              <h3 style={{ marginTop: 20 }}>📈 Metric Breakdown</h3>
              <ReactECharts option={getMetricsBarOption(response.choices)} style={{ height: 260 }} />

              {/* Monte Carlo Projection Chart */}
              <h3 style={{ marginTop: 16 }}>🎲 Projected Value & Uncertainty Bands</h3>
              <ReactECharts option={getMcConfidenceOption(response.choices)} style={{ height: 240 }} />
            </div>
          ) : (
            <div className="info-banner" style={{ textAlign: "center", padding: "40px 20px" }}>
              <h3>🔍 Ready to Evaluate</h3>
              <p>
                Configure the offered 3 cards and current deck on the left, then click <strong>Analyze Reward Options</strong> to compute
                the optimal Pareto decision frontier.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
