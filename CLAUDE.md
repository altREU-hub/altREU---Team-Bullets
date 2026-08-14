# Project context for Claude Code

## What this project is

altREU research project ("Team Bullets") building a **daily air-quality prediction model
for Los Angeles County, 2016–2025**. It relates weather conditions (MERRA-2 reanalysis)
to city-wide daily AQI (EPA monitoring data). Both daily series are built from scratch,
joined, and used to train and compare ten models, then validated forward on 2026 data.

## Current state — the pipeline is complete end to end

1. **AQI series** — `notebooks/aqi_pipeline.ipynb` → `data/processed/la_daily_aqi_5pollutants_v2_2016_2025.csv`
   (3,653 rows, 2016-01-01 → 2025-12-31, zero missing, 5 criteria pollutants).
2. **Weather series** — `notebooks/merra2_pipeline.ipynb` → `data/processed/la_daily_weather_2016_2025.csv`
   (3,652 rows, 2016-01-01 → 2025-12-30; one day shorter because MERRA-2 hadn't published
   2025-12-31 at pull time).
3. **Modeling dataset** — `notebooks/build_modeling_dataset.ipynb` → `train.csv` / `test.csv` /
   `feature_manifest.json`. 37 features, chronological split at 2023-01-01.
3b. **Trapping variables** — `notebooks/merra2_trapping_pipeline.ipynb` → `la_daily_trapping_2016_2025.csv`
   (PBLH, T850, SLP, TQV + inversion_strength and ventilation_index). Granules via
   `scripts/download_merra2_extra.sh`.
3c. **Per-pollutant targets** — `notebooks/build_pollutant_targets.ipynb`.
4. **Six models** — `models/01`–`06_*.ipynb`, all sharing `models/model_utils.py`.
5. **Ensemble** — `models/07_ensemble.ipynb`, stacked ridge blend of SVR + MLP + GBR.
6. **Variants** — `08_pollutant_split`, `09_enriched_features`, `10_extreme_days`.
7. **Validation** — `11_rolling_origin` (six annual folds), `12_forward_test_2026`.
8. **Comparison** — `models/13_final_comparison.ipynb`.
9. **Deployment** — `models/predict.py`: `--train` persists a model fitted on all data,
   `predict.py YYYY-MM-DD` predicts any date, `--self-test` checks feature engineering.
   It is **manifest-driven**: it uses `feature_manifest_v2.json` (61 features) when present and
   falls back to the 37-feature manifest, fetching `flx`/`slvx` granules only when needed.
   The self-test uses a *relative* tolerance (1e-5) because derived columns are recomputed from
   rounded CSVs; an exact match is not achievable.

Results (test = 2023–2025):
- Core 7 on 37 features: Ensemble 18.34 / R² 0.794 → SVR 18.85 → MLP 19.05 → GBR 19.48 →
  RF 20.28 → KNN 21.89 → Linear 24.01.
- **Best overall: Ensemble + Physics, 16.91 / R² 0.825** (61 features, scored on 1,089 days).
- Climatology baseline (no weather): 31.76 / R² 0.381.
- 2026 forward test (enriched, deployable): RMSE 14.88 vs season-matched benchmark 14.07;
  the 37-feature model gets 15.17. The enriched edge is not significant at n=151.

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
models/                               model_utils.py, predict.py, 01–13 notebooks, results/
models/saved/                         GITIGNORED — joblib artifacts (predict.py --train)
data/raw/merra2_flx/                  GITIGNORED — PBLH granules
data/raw/merra2_slv_extra/            GITIGNORED — T850/SLP/TQV granules
scripts/download_merra2_extra.sh      re-downloads both of the above
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
- **Model rankings need paired bootstrap intervals before they mean anything.** With 1,095
  test days, the gap between SVR and the MLP (0.203 RMSE) has a CI of [−0.65, +0.25] and is
  not a real ordering. Never compare models by checking whether their individual CIs
  overlap — that is far too conservative; resample the same days for both and take the
  difference.
- **Ensemble blend weights are learned on out-of-fold training predictions only**
  (`TimeSeriesSplit`), never on the test set.
- **The physics model is scored on 1,089 days, not 1,095.** Align predictions by date before
  comparing it to anything; notebook 13 does this explicitly.
- **MERRA-2 stream numbers (400/401) interleave irregularly** — do not compute them from the
  date. Take them from the filenames already in `data/raw/merra2_slv/`.
- **Parallel GES DISC downloads need a per-process cookie jar.** A shared one races during the
  Earthdata OAuth redirect and produces mass spurious failures.

## Known open issues

- **Every model under-predicts unhealthy days.** The physics features cut the bias above AQI 150
  from −29 to −21, the first change that helped, but it does not vanish. Quantile regression is
  the tool for the rest: q=0.9 catches 76% of unhealthy days vs 47%, at the cost of overall RMSE
  rising 18.3 → 25.0. Recommendation on record: ship two models, one for "what will AQI be" and
  one for "should we warn".
- **Feature pruning was tested and does NOT help** (notebook 09). Dropping the 5 weakest features
  gives −0.097 RMSE, CI [−0.275, +0.077], not significant; dropping more actively hurts. The
  overfit gap shrinks monotonically as features come out, which shows the gap was measuring
  memorization rather than harm. Keep all 61. Don't re-litigate this.
- **Notebooks 01–08 still use the 37-feature set.** Only the ensemble (09) and the deployed model
  (`predict.py`) use the physics variables; refitting the individual models is outstanding.
- **PM2.5 is the binding constraint and weather cannot fix it.** Fitted separately, ozone reaches
  R² 0.870 and PM2.5 only 0.418, and the physics features improved ozone RMSE by 2.69 against
  PM2.5's 0.20. The missing input is a smoke/fire indicator (NOAA HMS), not more meteorology.
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
