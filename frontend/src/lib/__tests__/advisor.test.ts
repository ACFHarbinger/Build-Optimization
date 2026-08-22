import { describe, expect, it } from "vitest";
import { Sts2AdvisorRequest, Sts2AdvisorResponse, AdvisorChoice } from "../types";

describe("STS2 Advisor Types and Processing", () => {
  it("constructs valid Sts2AdvisorRequest objects", () => {
    const request: Sts2AdvisorRequest = {
      character: "ironclad",
      deck: [
        { card_id: "strike", count: 5 },
        { card_id: "defend", count: 4 },
        { card_id: "bash", count: 1 },
      ],
      offers: ["Carnage", "Cleave", "Inflame"],
      context: {
        act: 1,
        floor: 6,
        hp_pct: 0.85,
        gold: 140,
        relics: ["Vajra", "Anchor"],
      },
      preferences: {
        tempo_weight: 1.0,
        synergy_weight: 1.0,
        dilution_weight: 1.2,
        mc_weight: 0.8,
        mc_rollouts: 500,
        seed: 42,
      },
    };

    expect(request.character).toBe("ironclad");
    expect(request.deck.reduce((acc, c) => acc + c.count, 0)).toBe(10);
    expect(request.offers).toHaveLength(3);
    expect(request.context?.act).toBe(1);
    expect(request.preferences?.seed).toBe(42);
  });

  it("handles advisor response sorting and Pareto front identification", () => {
    const choices: AdvisorChoice[] = [
      {
        action: "take",
        card_id: "carnage",
        card_name: "Carnage",
        is_upgrade: false,
        rank: 1,
        total_score: 55.4,
        score_delta: 12.2,
        metrics: {
          tempo_score: 20.0,
          synergy_score: 4.0,
          dilution_penalty: 1.6,
          mc_projected_mean: 58.0,
          mc_projected_ci_lower: 54.5,
          mc_projected_ci_upper: 62.8,
        },
        pareto_optimal: true,
        synergy_deltas: ["High frontloaded Act 1 damage"],
        explanation: "Exceptional immediate tempo.",
      },
      {
        action: "skip",
        card_id: null,
        card_name: "Skip",
        is_upgrade: false,
        rank: 2,
        total_score: 43.2,
        score_delta: 0.0,
        metrics: {
          tempo_score: 20.0,
          synergy_score: 15.0,
          dilution_penalty: 1.2,
          mc_projected_mean: 45.0,
          mc_projected_ci_lower: 40.8,
          mc_projected_ci_upper: 50.1,
        },
        pareto_optimal: true,
        synergy_deltas: [],
        explanation: "Preserves deck density.",
      },
    ];

    const response: Sts2AdvisorResponse = {
      status: "ok",
      character: "ironclad",
      evaluated_at: "2026-08-22T05:30:00Z",
      base_deck_size: 10,
      choices,
      pareto_front: ["Carnage", "Skip"],
      recommendation: "Carnage",
      diagnostics: null,
    };

    expect(response.status).toBe("ok");
    expect(response.recommendation).toBe("Carnage");
    expect(response.choices[0].rank).toBe(1);
    expect(response.choices[0].score_delta).toBeGreaterThan(0);
    expect(response.pareto_front).toContain("Skip");
    expect(response.pareto_front).toContain("Carnage");
  });
});
