/** Mirrors the item schema consumed by `middleware/src/pipeline/file_source.py`. */
export interface ScrapedItem {
  name: string;
  slot: string;
  stats: Record<string, number>;
  cost: number;
  rarity: string;
  level: number;
  tags: string[];
  sourceUrl: string;
}

export interface ScrapePayload {
  game: string;
  scrapedAt: string;
  items: ScrapedItem[];
}
