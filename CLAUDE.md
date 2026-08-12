# Project context for Claude Code

## What this project is

altREU research project ("Team Bullets") building a **daily air-quality prediction model
for Los Angeles County, 2016–2025**. It relates weather conditions (MERRA-2 reanalysis)
to city-wide daily AQI (EPA monitoring data). Both daily series are built from scratch,
joined, and used to train and compare six ML models.

## Current state — the pipeline is complete end to end

1. **AQI series** — `notebooks/aqi_pipeline.ipynb` → `data/processed/la_daily_aqi_5pollutants_v2_2016_2025.csv`
   (3,653 rows, 2016-01-01 → 2025-12-31, zero missing, 5 criteria pollutants).
2. **Weather series** — `notebooks/merra2_pipeline.ipynb` → `data/processed/la_daily_weather_2016_2025.csv`
   (3,652 rows, 2016-01-01 → 2025-12-30; one day shorter because MERRA-2 hadn't published
   2025-12-31 at pull time).
3. **Modeling dataset** — `notebooks/build_modeling_dataset.ipynb` → `train.csv` / `test.csv` /
   `feature_manifest.json`. 37 features, chronological split at 2023-01-01.
4. **Six models** — `models/01`–`06_*.ipynb`, each self-contained, all sharing
   `models/model_utils.py` for split loading and metrics.
5. **Comparison** — `models/07_model_comparison.ipynb`.

Results (test = 2023–2025): SVR 18.85 RMSE / R² 0.782 → MLP 19.05 → GBR 19.48 →
RF 20.28 → KNN 21.89 → Linear 24.01. Climatology baseline (no weather) is 31.76 / R² 0.381.

Running change log: `PROGRESS.md`. Full detail: `README.md`.

## Environment

Homebrew Python is PEP 668 externally-managed — `pip3 install` at system level fails.
Use the project venv:

```bash
source .venv/bin/activate      # or call .venv/bin/python directly
```

`.venv/` is gitignored; `requirements.txt` is pinned. The Jupyter kernel is registered as
`altreu` / "Python (altREU)". To execute a notebook headlessly:
`.venv/bin/jupyter nbconvert --to notebook --execute --inplace <nb>.ipynb`

## Repo layout

```
data/raw/{epa_ozone,epa_pollutant}/   tracked raw EPA AirData exports
data/raw/merra2_slv/                  GITIGNORED — 3,652 NetCDF granules, ~200 MB
data/raw/merra2_manifest.txt          tracked GES DISC URL list (regenerates the above)
data/corrections/                     tracked hand-fixed raw files
data/interim/                         GITIGNORED — raw_renamed/, la_only/, api_pulls/
data/processed/                       tracked final CSVs + feature_manifest.json
notebooks/                            three pipeline notebooks
models/                               model_utils.py, 01–07 notebooks, results/
```

**Tracked**: notebooks, model code, README, PROGRESS.md, CLAUDE.md, .gitignore,
requirements.txt, raw EPA folders, corrections, all processed CSVs, `models/results/`.
**Gitignored**: `data/raw/merra2_slv/`, `data/interim/`, `.venv/`, `.env`, `.DS_Store`.

Everything gitignored is regeneratable — README documents how for each.

## Credentials (never commit)

- **EPA AQS API**: `.env` at repo root with `AQS_EMAIL` and `AQS_KEY`, loaded by a small
  in-notebook `_load_dotenv` function.
- **NASA Earthdata**: `~/.netrc` with a `urs.earthdata.nasa.gov` entry, `chmod 600`.
- Both fall back to a `getpass` prompt if missing. **Never put credentials in chat,
  notebook cells, or any tracked file.**

## Methodology decisions already made — don't silently revisit these

- **Chronological split, not random.** Train 2016–2022, test 2023–2025. A random split
  leaks the future and makes the score meaningless.
- **Tuning inside the training years only**, via `GridSearchCV` + `TimeSeriesSplit(5)`.
  The test years are never touched during tuning.
- **Wind speed is computed hourly, then averaged**; wind *direction* comes from the daily
  mean U/V vector. Reversing the order would report a reversing-wind day as calm.
- **Circular features are sin/cos encoded** — wind direction and day-of-year.
- **Scalers live inside pipelines**, so each CV fold standardizes on its own training data
  and never leaks validation statistics.
- **KNN's train RMSE of 0.000 is an artifact** of `weights='distance'`, not a result. Its
  overfit gap is meaningless and is excluded from the gap chart.

## Known open issues

- **Every model under-predicts unhealthy days** (bias −27 to −48 AQI points above AQI 150,
  positive bias below 100). Regression to the mean under squared-error loss. This is the
  main obstacle to using the model as a warning system.
- **Only four MERRA-2 variables.** Boundary-layer height, inversion strength, and
  sea-breeze indicators are the actual trapping mechanism over LA and are all absent.
  Adding them is the highest-value next step.
- **No AQI lag feature.** Only weather lags exist. Adding yesterday's AQI would likely give
  the biggest single accuracy jump, but it changes the question from "what does weather
  explain?" to "what is tomorrow's AQI?" — belongs in a separate framing.

## Working preferences the user has confirmed

- **Notebook structure**: thematic section headers (e.g. "Removing other counties"), NOT
  numbered stages ("Stage 1", "Stage 3c"). Short 1–2 sentence "why we're doing this" before
  each code block, a "what this told us" after outputs when there's a real finding.
- **Narrative must match the data.** Never write an interpretation before seeing the actual
  output — check the numbers, then write the claim.
- **Portability**: paths anchored to `Path.cwd()` with a parent-directory fallback, never
  hardcoded home directories. Notebooks must run for anyone who clones the repo, launched
  from either the repo root or the notebook's own folder.
- **Data hygiene**: derived data gitignored; commit only what can't be regenerated.
- **Commits**: real messages describing the *what* and *why*, not "Add files via upload".
  Bullets when the commit touches multiple concerns.
- **Stages of work**: pause and confirm between stages of a multi-step task; show a summary
  before moving on. The user prefers to steer.
- **Long-running work**: progress checkpoints with real ETAs, per-item failure logging, no
  silent batches that crash on one bad item.

## Useful references

- EPA AQS API dailyData endpoint: `https://aqs.epa.gov/data/api/dailyData/byCounty`
- EPA AQS parameter class list: `https://aqs.epa.gov/data/api/list/parametersByClass?pc=AQI+POLLUTANTS`
- MERRA-2 M2T1NXSLV product page: search "M2T1NXSLV" on gesdisc.eosdis.nasa.gov
- GitHub remote: `altREU-hub/altREU---Team-Bullets` (main branch)
