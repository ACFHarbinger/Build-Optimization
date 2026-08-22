import { ScrapedItem } from "../lib/types";
import { WikiSelectorProfile, profileForHost } from './selectors';

function normaliseLabel(value: string): string {
  return value.replace(/\s+/g, ' ').trim().toLowerCase();
}

function rowValue(container: Element, profile: WikiSelectorProfile, acceptedLabels: string[]): string | undefined {
  for (const row of Array.from(container.querySelectorAll(profile.statBlock.row))) {
    const label = row.querySelector(profile.statBlock.label)?.textContent;
    const value = row.querySelector(profile.statBlock.value)?.textContent;
    if (label && value && acceptedLabels.includes(normaliseLabel(label))) return value.trim();
  }
  return undefined;
}

function parseInfobox(container: Element, sourceUrl: string): ScrapedItem | null {
  const profile = profileForHost(window.location.hostname);
  if (!profile) return null;

  const nameEl = container.querySelector(profile.name);
  if (!nameEl?.textContent) return null;

  const stats: Record<string, number> = {};
  container.querySelectorAll(profile.statBlock.row).forEach((row) => {
    const label = row.querySelector(profile.statBlock.label)?.textContent?.trim().toLowerCase();
    const value = row.querySelector(profile.statBlock.value)?.textContent?.trim();
    const numeric = value ? Number.parseFloat(value.replace(/[^0-9.\-]/g, "")) : NaN;
    if (label && Number.isFinite(numeric)) {
      stats[label.replace(/\s+/g, "_")] = numeric;
    }
  });

  const slot = rowValue(container, profile, profile.slotLabels) ?? 'unknown';
  const rarity = rowValue(container, profile, profile.rarityLabels) ?? 'unknown';
  const costText = rowValue(container, profile, profile.costLabels);
  const cost = costText ? Number.parseFloat(costText.replace(/[^0-9.\-]/g, '')) || 0 : 0;

  return {
    name: nameEl.textContent.trim(),
    slot: normaliseLabel(slot).replace(/\s+/g, '_'),
    stats,
    cost,
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
