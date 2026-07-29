import { describe, expect, it } from "vitest";
import { kpiColorFor, KPI_FALLBACK_COLORS, RARITY_COLORS } from "../colors";

describe("kpiColorFor", () => {
  it("returns the semantic gradient for a known label", () => {
    expect(kpiColorFor("Score", 0)).toEqual(["#43a047", "#2e7d32"]);
  });

  it("falls back to a cyclic gradient for unknown labels", () => {
    expect(kpiColorFor("Unknown Metric", 0)).toEqual(KPI_FALLBACK_COLORS[0]);
    expect(kpiColorFor("Unknown Metric", 7)).toEqual(KPI_FALLBACK_COLORS[1]);
  });
});

describe("RARITY_COLORS", () => {
  it("covers every standard rarity tier", () => {
    for (const rarity of ["COMMON", "UNCOMMON", "RARE", "EPIC", "LEGENDARY"]) {
      expect(RARITY_COLORS[rarity]).toMatch(/^#[0-9a-f]{6}$/);
    }
  });
});
