document.getElementById("scrape")?.addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab.id) return;
  const response = await chrome.tabs.sendMessage(tab.id, { type: "SCRAPE_PAGE" });
  await chrome.runtime.sendMessage({ type: "STORE_ITEMS", items: response?.items ?? [] });
});

document.getElementById("export")?.addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "EXPORT_ITEMS", game: "rpg" });
});
