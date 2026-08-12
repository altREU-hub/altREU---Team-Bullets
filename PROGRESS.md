# Progress Log

Short running record of what changed and when. Newest first.

---

## 2026-08-12 — Modeling stage complete

**Environment**
- Homebrew Python is PEP 668 externally-managed, so created a project venv at `.venv`
  (gitignored) with xarray, netCDF4, scikit-learn, matplotlib, jupyter. Registered a
  Jupyter kernel named `Python (altREU)`. Pinned in `requirements.txt`.

**Repo reorganized** into `data/` + `notebooks/` + `models/`. Used `git mv` for tracked
files so history follows. EPA folders renamed to `data/raw/epa_ozone` and
`data/raw/epa_pollutant` (dropped spaces/parens); `aqi_pipeline.ipynb` path constants
updated to match and re-anchored to work from repo root *or* `notebooks/`.

**Fixed a gitignore gap** — `merra2_raw/` was documented as ignored but wasn't, leaving
~200 MB of NetCDF one `git add -A` away from being committed. Now ignored, along with
`data/interim/` and `.venv/`.

**MERRA-2 processed** (`notebooks/merra2_pipeline.ipynb`)
- Stripped the cosmetic `.nc4.nc4` double extension from all 3,652 granules.
- Aggregated hourly → daily: spatial mean over the 3×3 grid, then daily statistics.
  Wind *speed* is computed hourly-then-averaged; wind *direction* comes from the daily
  mean U/V vector. Doing speed the other way would call a reversing-wind day calm.
- 21 s for all 3,652 granules, 0 failures, 0 gaps, 0 NaNs.
- Output: `data/processed/la_daily_weather_2016_2025.csv` (3,652 rows × 14 cols).

**Modeling dataset built** (`notebooks/build_modeling_dataset.ipynb`)
- Inner-joined weather + AQI → 3,652 rows (drops 2025-12-31, which has no MERRA-2 day).
- Circular features encoded as sin/cos: wind direction and day-of-year.
- Lags 1/2/3 + 3-day rolling mean for the five persistent variables → 37 features total.
- Chronological split at 2023-01-01: **train 2,554 days (2016–2022) / test 1,095 days
  (2023–2025)**. Random splitting would leak the future.
- Strongest single correlate: `t2m_max` at 0.705.

**Six models built and scored** (`models/01`–`06`, one notebook each). All tuned with
`GridSearchCV` + `TimeSeriesSplit(5)` on the training years only.

| rank | model | test RMSE | test R² | overfit gap |
|---|---|---:|---:|---:|
| 1 | SVR (RBF) | 18.85 | 0.782 | +1.6 |
| 2 | Neural Network (MLP) | 19.05 | 0.777 | +2.2 |
| 3 | Gradient Boosting | 19.48 | 0.767 | +10.7 |
| 4 | Random Forest | 20.28 | 0.748 | +11.5 |
| 5 | KNN | 21.89 | 0.706 | n/a * |
| 6 | Linear Regression | 24.01 | 0.646 | −0.5 |

\* KNN's train RMSE is exactly 0 — an artifact of `weights='distance'` (a training point
is its own nearest neighbour at distance 0), not a real score.

**Comparison notebook** (`models/07_model_comparison.ipynb`) — table, ranking chart,
train-vs-test gap chart, error-by-AQI-band breakdown, and the verdict.

**Two idempotency bugs found in `aqi_pipeline.ipynb`** — surfaced by re-running it end to
end for the first time since the original session, as a check that the reorganization
hadn't broken anything. Both predate the reorganization.

1. *The v1 build absorbed the fix it predates.* The v1 cell globs all of `api_pulls/`,
   and `pm25_88502_2019_api.csv` splits on `_` to `pm25`. So on any second run — once the
   88502 files exist on disk — v1 came out already containing the data whose absence it
   exists to demonstrate. The discrepancy investigation then found 0 days and crashed on
   an empty frame. Now skips `pm25_88502_*` explicitly.
2. *The committed v1 artifact was contaminated by the same bug*, on 119 of 3,653 days, all
   traceable to one 88502 monitor at Los Angeles-North Main Street. The true pre-fix v1
   runs below the 2-pollutant baseline on **1,106** days, not the 1,071 previously
   recorded. Corrected in the notebook and README.

**Neither affects any result.** `la_daily_aqi_5pollutants_v2_2016_2025.csv` and the
2-pollutant baseline both regenerate byte-identically, so every model number above stands.
Only the kept-for-diff v1 artifact changed. The notebook has now been run twice in a row
with identical output.

**Three findings worth carrying forward**
1. A day-of-year climatology baseline using *no weather at all* already reaches R² 0.38
   (RMSE 31.8). Weather's genuine contribution is the gap from there to ~0.78, not the
   full 0.78.
2. Temperature dominates every model. `t2m_max` costs Gradient Boosting 13.0 RMSE points
   when permuted; the runner-up costs 2.0.
3. **Every model under-predicts unhealthy days** (bias −27 to −48 AQI points above 150,
   positive bias below 100). Regression to the mean under squared-error loss — not a
   tuning failure, and the main obstacle to using this as a warning system.

---

## 2026-07-05 — MERRA-2 downloaded

3,652 NetCDF granules pulled from NASA GES DISC OPeNDAP (M2T1NXSLV, LA bounding box,
4 variables: QV2M/T2M/U10M/V10M). Hourly resolution, ~55 KB each, all sanity-checked.

---

## 2026-07-05 — EPA AQS pipeline complete

Built `aqi_pipeline.ipynb` end to end: cleaned raw EPA AirData exports, corrected a
misfiled 2021 ozone file (Louisville data in the LA folder), pulled all five criteria
pollutants from the AQS API, and caught a missing PM2.5 parameter code (`88502`,
~39% of PM2.5 monitor-days) that made the first 5-pollutant build read *lower* than the
2-pollutant baseline.

Output: `la_daily_aqi_5pollutants_v2_2016_2025.csv` — 3,653 rows, zero missing.
