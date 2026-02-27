# Build-Optimization

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)

A set of optimization algorithms to help create the best builds for videogames.

---

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run optimization with default settings (SA solver, RPG game)
python main.py

# Use a specific solver
python main.py solver=ga game=rpg

# Adjust budget and time limit
python main.py solver=alns optimization.budget=2000 optimization.time_limit=120
```

## Architecture

```
Build-Optimization/
├── main.py                     # Hydra entry point
├── configs/                    # Hydra configuration
│   ├── config.yaml             # Main config
│   ├── solver/                 # Per-solver configs (sa, ga, ils, ...)
│   ├── game/                   # Game profiles (rpg, moba)
│   └── pipeline/               # Data source configs (file, api, scraper)
├── src/
│   ├── core/                   # Domain model
│   │   ├── item.py             # Item, Slot, Rarity
│   │   ├── build.py            # Build solution representation
│   │   ├── synergy.py          # Synergy system (set bonuses)
│   │   └── scoring.py          # Fitness / effectiveness function
│   ├── operators/              # Destroy/Repair operators
│   │   ├── destroy.py          # random_remove, worst_remove, expensive_remove
│   │   └── repair.py           # greedy_fill, random_fill, synergy_fill
│   ├── solvers/                # Optimization algorithms (new implementations)
│   │   ├── base.py             # BaseSolver ABC
│   │   ├── greedy.py           # Greedy baseline
│   │   ├── random_search.py    # Random search baseline
│   │   ├── sa.py               # Simulated Annealing
│   │   ├── ga.py               # Genetic Algorithm
│   │   ├── ils.py              # Iterated Local Search
│   │   ├── lahc.py             # Late Acceptance Hill Climbing
│   │   ├── rrt.py              # Record-to-Record Travel
│   │   ├── gls.py              # Guided Local Search
│   │   ├── rts.py              # Reactive Tabu Search
│   │   ├── oba.py              # Old Bachelor Acceptance
│   │   ├── alns.py             # Adaptive Large Neighborhood Search
│   │   ├── abc.py              # Artificial Bee Colony
│   │   └── fa.py               # Firefly Algorithm
│   ├── policies/               # Full WSmart-Route policy solvers (copied)
│   │   ├── simulated_annealing/
│   │   ├── genetic_algorithm/
│   │   ├── adaptive_large_neighborhood_search/
│   │   └── ... (28 algorithms)
│   ├── pipeline/               # Data ingestion pipeline
│   │   ├── base.py             # DataSource ABC
│   │   ├── file_source.py      # JSON/CSV file loader
│   │   ├── game_api.py         # REST API skeleton (PoE, Genshin, LoL)
│   │   ├── scraper.py          # Web scraper skeleton
│   │   └── transforms.py       # Filtering, normalization, scaling
│   └── data/
│       └── sample_games/
│           └── rpg.json        # 56-item RPG dataset with synergies
└── tests/
    ├── test_core.py            # Domain model tests
    └── test_solvers.py         # Solver integration tests
```

## Domain Mapping

| WSmart-Route (VRP) | Build-Optimization           |
| ------------------ | ---------------------------- |
| Bins/Nodes         | Items/Equipment              |
| Vehicle routes     | Equipped build (slot → item) |
| Vehicle capacity   | Budget constraint            |
| Distance cost      | Item cost                    |
| Collection profit  | Effectiveness score          |
| Destroy operators  | Remove items from build      |
| Repair operators   | Fill empty slots             |

## Available Solvers

| Solver                        | Key      | Description                           |
| ----------------------------- | -------- | ------------------------------------- |
| Greedy                        | `greedy` | Greedy fill by best score improvement |
| Random Search                 | `random` | Best of N random builds               |
| Simulated Annealing           | `sa`     | Temperature-based acceptance          |
| Genetic Algorithm             | `ga`     | Crossover + mutation evolution        |
| Iterated Local Search         | `ils`    | Perturbation + local search restarts  |
| Late Acceptance Hill Climbing | `lahc`   | Score history-based acceptance        |
| Record-to-Record Travel       | `rrt`    | Record + tolerance threshold          |
| Guided Local Search           | `gls`    | Augmented cost with penalties         |
| Reactive Tabu Search          | `rts`    | Adaptive tabu tenure                  |
| Old Bachelor Acceptance       | `oba`    | Age-based acceptance threshold        |
| ALNS                          | `alns`   | Adaptive operator weighting           |

## Data Pipeline

Three data source backends:

1. **FileSource** — Load items from JSON/CSV files (default)
2. **GameAPISource** — Retrieve items from game REST APIs (skeleton)
3. **WebScraperSource** — Scrape game wikis (skeleton)

Plus the full `policies/` folder from WSmart-Route with 28 metaheuristic implementations.

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT
