import { describe, expect, it } from "vitest";
import { formatNumber, formatMaybeNumber } from "../format";

describe("formatNumber", () => {
  it("adds thousands separators above 1000", () => {
    expect(formatNumber(4200)).toBe("4,200.00");
  });

  it("uses plain fixed precision below 1000", () => {
    expect(formatNumber(342.5)).toBe("342.50");
  });

  it("respects a custom precision", () => {
    expect(formatNumber(1234.5678, 1)).toBe("1,234.6");
  });
});

describe("formatMaybeNumber", () => {
  it("formats numbers and passes through strings", () => {
    expect(formatMaybeNumber(5)).toBe("5.00");
    expect(formatMaybeNumber("demo")).toBe("demo");
  });
});
