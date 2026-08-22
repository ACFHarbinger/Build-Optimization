-- Synthetic STS2 Lua module in the wiki.gg Module:Cards/StS2_data shape.
-- Not a scrape dump: four invented-layout rows so ingest tests do not need
-- the network or Mega Crit card art.
local all_data = {
  ["Strike (Ironclad)"] = {
    Cost = 1,
    Color = "Ironclad",
    Type = "Attack",
    Rarity = "Basic",
    Text = "Deal [6|9] damage."
  },
  ["Inflame"] = {
    Cost = 1,
    Color = "Ironclad",
    Type = "Power",
    Rarity = "Uncommon",
    Text = "Gain [2|3] $Strength."
  },
  ["Twin Strike"] = {
    Cost = 1,
    Color = "Ironclad",
    Type = "Attack",
    Rarity = "Common",
    Text = "Deal [5|7] damage twice."
  },
  ["Shrug It Off"] = {
    Cost = 1,
    Color = "Silent",
    Type = "Skill",
    Rarity = "Common",
    Text = "Gain [8|11] $Block.<br>Draw 1 card."
  }
}
return all_data
