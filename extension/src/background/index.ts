import { ScrapePayload, ScrapedItem } from "../lib/types";

const STORAGE_KEY = "build-optimization:scraped-items";

chrome.action.onClicked.addListener((tab) => {
  if (tab.id) {
    chrome.tabs.sendMessage(tab.id, { type: "SCRAPE_PAGE" });
  }
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "STORE_ITEMS") {
    const items: ScrapedItem[] = message.items ?? [];
    chrome.storage.local.get(STORAGE_KEY, (result) => {
      const existing: ScrapedItem[] = result[STORAGE_KEY] ?? [];
      chrome.storage.local.set({ [STORAGE_KEY]: [...existing, ...items] }, () => sendResponse({ ok: true }));
    });
    return true;
  }

  if (message?.type === "EXPORT_ITEMS") {
    chrome.storage.local.get(STORAGE_KEY, (result) => {
      const payload: ScrapePayload = {
        game: message.game ?? "unknown",
        scrapedAt: new Date().toISOString(),
        items: result[STORAGE_KEY] ?? [],
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      chrome.downloads.download({ url, filename: `${payload.game}-items.json` }, () => sendResponse({ ok: true }));
    });
    return true;
  }
  return false;
});
