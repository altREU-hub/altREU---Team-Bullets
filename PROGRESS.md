# Progress Log

Short running record of what changed and when. Newest first.

---

## 2026-08-14 — Trapping physics added; best model now RMSE 16.91

**The biggest single improvement in the project, and it came from features, not models.**

Added four MERRA-2 variables that measure what actually traps pollution over the LA basin —
`PBLH` (M2T1NXFLX) and `T850`/`SLP`/`TQV` (M2T1NXSLV) — plus two derived features:
`inversion_strength` = T850 − T2M, and `ventilation_index` = PBLH × wind speed.

- `notebooks/merra2_trapping_pipeline.ipynb` aggregates 7,304 new granules to daily.
- `scripts/download_merra2_extra.sh` re-downloads them (idempotent; takes the date list and the
  irregular 400/401 stream numbers from the filenames already in `merra2_slv/`).
- `models/09_enriched_features.ipynb` is a clean ablation: identical models, split and tuning,
  only the feature set changes.

**Result: RMSE 18.50 → 16.91, R² 0.791 → 0.825.** Paired bootstrap CI [−2.279, −0.918],
P(better) = 1.000. Roughly five times the gain the ensemble bought over its best member.

`t850_mean` is now the top feature by permutation importance (14.07 RMSE points, 3.5× `t2m_max`),
and it correlates 0.753 with daily AQI against `t2m_max`'s 0.705. Temperature 1.5 km up beats
surface temperature because it measures the air mass without marine-layer contamination.

**The hypothesis going in was wrong, and that is the interesting part.** Notebook 08 predicted the
gain would land on PM2.5, since these variables describe dispersion. It went almost entirely to
ozone (RMSE gain 2.69) rather than PM2.5 (0.20). `t850_mean` is really measuring the warm subsiding
air of a high-pressure ridge, which is the ozone-producing synoptic pattern. PM2.5 did not improve
because its unexplained variance was never meteorological — it is smoke and combustion.

**First thing to move the extreme days.** MAE above AQI 150 fell 31.07 → 24.60 and bias −29.2 →
−21.1. Every earlier attempt either failed or traded away average accuracy. The remaining bias is
genuine regression to the mean.

Cost: overfit gap widened +5.9 → +7.6 with 61 features on 2,554 training days. `pblh_min` and its
lags correlate at ≈ −0.02 and should be pruned.

**`models/13_final_comparison.ipynb`** (renamed from 08 → 12 → 13) now covers all ten models in
three groups, because they do not all answer the same question: seven core models on identical
features, two variants that change the target or objective, and one on a different feature set.
The physics model is scored on 1,089 days rather than 1,095, so predictions are aligned by date
before any cross-group comparison.

---

## 2026-08-13 — Ensemble added as model 07; comparison renumbered to 08

**Why.** A paired bootstrap showed the ranking among the top individual models was not
statistically supported: SVR beat the MLP by 0.203 RMSE with a 95% CI of [−0.65, +0.25],
straddling zero. Picking a winner there was fitting noise.

**`models/07_ensemble.ipynb`** combines SVR + MLP + Gradient Boosting, reusing the
hyperparameters already selected in notebooks 03/04/06 (no new search, so nothing touches
the test set). Three blending strategies compared, with weights learned only from
out-of-fold `TimeSeriesSplit` predictions on the training years:

| strategy | test RMSE | within 10 AQI |
|---|---:|---:|
| stacked ridge | **18.336** | **53.8%** |
| equal average | 18.481 | 51.1% |
| inverse-RMSE weights | 18.482 | 51.1% |

The stacked ridge won (coefficients SVR 0.31 / MLP 0.25 / GBR 0.48), and unlike the
individual-model ranking this difference is real — paired bootstrap CI [−0.248, −0.042]
against the equal average, [−0.887, −0.162] against SVR alone.

Base-model residuals correlate at 0.90, which capped the achievable gain in advance: only
the uncorrelated 10% is available to cancel. Final result is **RMSE 18.34, R² 0.794** — the
best in the project, ~3% better than SVR alone, at zero cost in new data or tuning.

**Caveat recorded in the notebook:** the choice among the three blending strategies was
informed by test performance, a mild form of test-set selection. Fully rigorous practice
would select by nested CV. All three strategies beat every individual model, so the headline
conclusion does not depend on the choice, but 18.34 reads slightly optimistic.

**The ensemble does not fix the extreme-day problem.** On the 122 unhealthy days it does not
beat the best individual member. All three members miss those days for the same reason, so
averaging cannot rescue them — that needs a change of objective, not a better blend.

**`08_model_comparison.ipynb`** (renamed from 07) now covers all seven models and adds a
bootstrap section: single-model CIs plus paired comparisons against the leader. Key
methodological note captured there — never compare two models by checking whether their
individual confidence intervals overlap; that test is far too conservative. Compare pairwise
on the same resampled days.

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
