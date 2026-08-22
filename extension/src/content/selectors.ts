/** Explicit DOM profiles for wiki engines supported by the content scraper. */
export interface StatBlockSelectors {
  row: string;
  label: string;
  value: string;
}

export interface WikiSelectorProfile {
  id: 'fandom' | 'wiki-gg' | 'gamepedia';
  hostPattern: RegExp;
  itemContainer: string;
  name: string;
  statBlock: StatBlockSelectors;
  slotLabels: string[];
  rarityLabels: string[];
  costLabels: string[];
}

export const WIKI_PROFILES: readonly WikiSelectorProfile[] = [
  {
    id: 'fandom',
    hostPattern: /(^|\.)fandom\.com$/i,
    itemContainer: ".portable-infobox",
    name: ".pi-title",
    statBlock: { row: '.pi-item.pi-data', label: '.pi-data-label', value: '.pi-data-value' },
    slotLabels: ['slot', 'type', 'equipment type'],
    rarityLabels: ['rarity'],
    costLabels: ['cost', 'price', 'buy price'],
  },
  {
    id: 'wiki-gg',
    hostPattern: /(^|\.)wiki\.gg$/i,
    itemContainer: '.infobox.item, .infobox',
    name: '.infobox > .title, .infobox .title',
    statBlock: { row: 'table.stat > tbody > tr, table.infobox > tbody > tr', label: 'th', value: 'td' },
    slotLabels: ['slot', 'type', 'equipment type'],
    rarityLabels: ['rarity'],
    costLabels: ['cost', 'price', 'buy', 'sell'],
  },
  {
    id: 'gamepedia',
    hostPattern: /(^|\.)gamepedia\.com$/i,
    itemContainer: 'table.infobox, .infobox',
    name: '.infobox-title, caption, .title',
    statBlock: { row: 'tr', label: 'th', value: 'td' },
    slotLabels: ['slot', 'type', 'equipment type'],
    rarityLabels: ['rarity'],
    costLabels: ['cost', 'price', 'buy price'],
  },
];

export function profileForHost(hostname: string): WikiSelectorProfile | undefined {
  return WIKI_PROFILES.find((profile) => profile.hostPattern.test(hostname));
}
