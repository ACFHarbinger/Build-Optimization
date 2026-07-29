import { ScrapedItem } from "../lib/types";
import { profileForHost } from "./selectors";

function parseInfobox(container: Element, sourceUrl: string): ScrapedItem | null {
  const profile = profileForHost(window.location.hostname);
  if (!profile) return null;

  const nameEl = container.querySelector(profile.name);
  if (!nameEl?.textContent) return null;

  const stats: Record<string, number> = {};
  container.querySelectorAll(profile.statRow).forEach((row) => {
    const label = row.querySelector("h3, .pi-data-label")?.textContent?.trim().toLowerCase();
    const value = row.querySelector(".pi-data-value")?.textContent?.trim();
    const numeric = value ? Number.parseFloat(value.replace(/[^0-9.\-]/g, "")) : NaN;
    if (label && Number.isFinite(numeric)) {
      stats[label.replace(/\s+/g, "_")] = numeric;
    }
  });

  const rarity = profile.rarity ? (container.querySelector(profile.rarity)?.textContent?.trim() ?? "unknown") : "unknown";

  return {
    name: nameEl.textContent.trim(),
    slot: "unknown",
    stats,
    cost: 0,
    rarity,
    level: 1,
    tags: [],
    sourceUrl,
  };
}

function scrapeCurrentPage(): ScrapedItem[] {
  const profile = profileForHost(window.location.hostname);
  if (!profile) return [];

  return Array.from(document.querySelectorAll(profile.itemContainer))
    .map((container) => parseInfobox(container, window.location.href))
    .filter((item): item is ScrapedItem => item !== null);
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "SCRAPE_PAGE") {
    sendResponse({ items: scrapeCurrentPage() });
  }
  return true;
});
