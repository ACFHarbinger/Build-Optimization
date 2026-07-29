/** Per-site CSS selector profiles for locating item infoboxes on supported wikis. */
export interface WikiSelectorProfile {
  hostPattern: RegExp;
  itemContainer: string;
  name: string;
  statRow: string;
  rarity?: string;
}

export const WIKI_PROFILES: WikiSelectorProfile[] = [
  {
    hostPattern: /fandom\.com$/,
    itemContainer: ".portable-infobox",
    name: ".pi-title",
    statRow: ".pi-item.pi-data",
    rarity: '[data-source="rarity"] .pi-data-value',
  },
  {
    hostPattern: /wiki\.gg$/,
    itemContainer: ".portable-infobox",
    name: ".pi-title",
    statRow: ".pi-item.pi-data",
  },
];

export function profileForHost(hostname: string): WikiSelectorProfile | undefined {
  return WIKI_PROFILES.find((profile) => profile.hostPattern.test(hostname));
}
