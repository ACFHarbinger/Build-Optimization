import { profileForHost } from './selectors';

describe('profileForHost', () => {
  it.each([
    ['genshin-impact.fandom.com', 'fandom'],
    ['terraria.wiki.gg', 'wiki-gg'],
    ['minecraft.gamepedia.com', 'gamepedia'],
  ])('selects the expected profile for %s', (hostname, expectedId) => {
    expect(profileForHost(hostname)?.id).toBe(expectedId);
  });

  it('does not scrape an unrelated host', () => {
    expect(profileForHost('example.com')).toBeUndefined();
  });

  it('keeps MediaWiki and PortableInfobox profiles structurally distinct', () => {
    expect(profileForHost('terraria.wiki.gg')?.statBlock.row).toContain('table.stat');
    expect(profileForHost('genshin-impact.fandom.com')?.statBlock.row).toContain('.pi-item');
  });
});
