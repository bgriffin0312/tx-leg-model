# Texas Legislature Modeling

A race-by-race model of Texas legislative elections in 2026, designed to project outcomes under varying national political environments.

## What this does

Models every competitive Texas House and Senate race individually, then applies a configurable "national environment" overlay — e.g. a D+8 cycle vs. a D+3 cycle — to project seat changes, chamber control probabilities, and individual race outcomes.

Key inputs to the model:
- **Baseline partisanship** of each district (prior election results, PVI, etc.)
- **Incumbent status** and fundraising where available
- **National environment** — a single dial representing the generic ballot / political backdrop
- **Upballot races** — top-of-ticket effects from the TX Senate race, Governor's race, etc.
- **Structural factors** — redistricting effects, candidate quality, open seats

## Project Structure

```
├── data/
│   ├── raw/          # Source data (gitignored)
│   └── processed/    # Cleaned, model-ready data (gitignored)
├── notebooks/        # Exploration and one-off analysis
├── output/           # Model runs, charts, scenario exports (gitignored)
├── src/              # Core model code
└── requirements.txt  # Python dependencies
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

_Add usage instructions as the model takes shape._
