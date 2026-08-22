# Third-party notices

## Slay the Spire Wiki (wiki.gg)

The STS2 card catalogue ingest (`middleware/src/pipeline/catalogue/`) reads
structured card data from [Slay the Spire Wiki](https://slaythespire.wiki.gg)
(`Module:Cards/StS2_data/*` Lua modules) via the public MediaWiki API.

- **License**: wiki.gg content is offered under [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/).
- **Use**: facts only (card names, costs, types, rarities, numeric stats, tags).
- **Not redistributed in this repository**: scraped wiki dumps, wiki prose,
  card art, sprites, or real game screenshots. Ingest writes a user-local,
  gitignored cache (`$STS2_CATALOGUE_DIR` or
  `~/.local/share/build-optimization/sts2/`). A gitignored overlay sits beside
  it for user-authored add/override rows.

Tests use a small synthetic Lua fixture (`middleware/tests/fixtures/wiki/`)
that matches the module *shape* without copying wiki pages.

## Mega Crit / Kobold Games

*Slay the Spire*, *Slay the Spire 2*, and *Mega Crit* are trademarks of
Mega Crit Games. This project does not claim ownership of the game, its
characters, or its audiovisual assets.
