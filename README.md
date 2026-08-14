# altREU — Team Bullets

**Predicting daily air quality in Los Angeles County from weather, 2016–2025.**

Two daily time series are assembled from scratch and joined: city-wide AQI from EPA monitoring
data, and weather from NASA's MERRA-2 reanalysis. Ten models are then trained and compared, and
the best one is validated on 2026 data that did not exist when the project was written.

**Headline result:** weather explains about **82% of the variance** in LA's daily AQI
(stacked ensemble with trapping physics, test R² 0.825, RMSE 16.9 on held-out years).

Three caveats that belong with that number:
- A day-of-year climatology using **no weather at all** already reaches R² 0.38, so roughly half
  the headline is just the seasonal cycle.
- Fitted separately, **ozone reaches R² 0.87 and PM2.5 only 0.42.** The blended figure averages a
  pollutant weather explains very well with one it barely explains.
- **Every model under-predicts unhealthy days.** The physics features cut that bias from −29 to
  −21 AQI points, but it does not go away.

See [PROGRESS.md](PROGRESS.md) for the running change log.

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name altreu --display-name "Python (altREU)"
jupyter notebook
```

### Predicting a date

```bash
python models/predict.py --train           # fit on all data, save artifacts (~2 min)
python models/predict.py 2026-03-07        # -> predicted AQI 64.3 (Moderate); actual was 62
python models/predict.py 2026-03-07 --explain   # ...and show the weather behind it
python models/predict.py --self-test       # verify feature engineering matches the dataset
```

**Any date needs only four days of weather** — the target day plus the three before it. Every lag
in the feature set is a *weather* lag, so there is no recursive chain back through history and no
AQI history is required. Needs a `~/.netrc` entry for `urs.earthdata.nasa.gov` (chmod 600).

MERRA-2 is reanalysis: it exists for the past only, lagging real time by 3–4 weeks. For genuinely
future dates you need a forecast product (NOAA GFS), and prediction skill decays with lead time.

### Notebook order

| # | notebook | what it does |
|---|---|---|
| 1 | `notebooks/aqi_pipeline.ipynb` | raw EPA data + AQS API → daily AQI series |
| 2 | `notebooks/merra2_pipeline.ipynb` | 3,652 hourly NetCDF granules → daily weather |
| 3 | `notebooks/merra2_trapping_pipeline.ipynb` | PBLH / T850 / SLP / TQV → inversion + ventilation |
| 4 | `notebooks/build_pollutant_targets.ipynb` | per-pollutant daily AQI (ozone, PM2.5, NO2 …) |
| 5 | `notebooks/build_modeling_dataset.ipynb` | join, engineer features, train/test split |
| 6 | `models/01`–`06_*.ipynb` | one model each, any order |
| 7 | `models/07_ensemble.ipynb` | stacks the top three |
| 8 | `models/08_pollutant_split.ipynb` | one model per pollutant, combined by max |
| 9 | `models/09_enriched_features.ipynb` | the physics ablation — biggest single gain |
| 10 | `models/10_extreme_days.ipynb` | quantile / weighted / classification for the tails |
| 11 | `models/11_rolling_origin.ipynb` | six expanding-window folds instead of one split |
| 12 | `models/12_forward_test_2026.ipynb` | truly prospective test on unseen 2026 data |
| 13 | `models/13_final_comparison.ipynb` | the comparison tables and verdict |

Notebooks 1–4 need credentials. Everything from 5 on runs on committed CSVs.

---

## Repository layout

```
data/
  raw/
    epa_ozone/            raw EPA AirData ozone exports         [tracked]
    epa_pollutant/        raw EPA AirData PM2.5 exports         [tracked]
    merra2_slv/           base MERRA-2 granules                 [gitignored, ~200 MB]
    merra2_flx/           PBLH granules                         [gitignored, ~120 MB]
    merra2_slv_extra/     T850 / SLP / TQV granules             [gitignored, ~170 MB]
    merra2_manifest.txt   GES DISC URL list for the base set    [tracked]
  corrections/            hand-fixed raw files                  [tracked]
  interim/                intermediate + API pulls + caches     [gitignored]
  processed/              all final CSVs                        [tracked]

notebooks/                five data-pipeline notebooks
models/
  model_utils.py          shared split loading + metrics
  predict.py              persist a model; predict any date
  01_linear_regression   05_knn                 09_enriched_features
  02_random_forest       06_neural_network      10_extreme_days
  03_gradient_boosting   07_ensemble            11_rolling_origin
  04_svr                 08_pollutant_split     12_forward_test_2026
                                                13_final_comparison
  results/                per-model metrics JSON + test predictions
  saved/                  fitted joblib artifacts               [gitignored]

scripts/download_merra2_extra.sh    re-downloads the trapping granules
PROGRESS.md               running change log
requirements.txt          pinned dependencies
```

**Gitignored:** the three `merra2_*` raw folders, `data/interim/`, `models/saved/`, `.venv/`,
`.env`, `.DS_Store`. All regeneratable — see *Regenerating derived data*.

---

## The datasets

### `data/processed/la_daily_aqi_5pollutants_v2_2016_2025.csv`
3,653 rows, one per day, 2016-01-01 → 2025-12-31, zero missing. Columns: `date`, `daily_aqi`
(max AQI across all monitors and pollutants), `n_monitor_readings`, `n_sites`, `pollutants_used`,
`dominant_pollutant`, `dominant_site`.

Two data bugs were found and fixed:
- **A misfiled 2021 ozone file.** One AirData export was actually a Louisville-metro download.
  The corrected LA file is committed at `data/corrections/ozone_2021_la.csv`.
- **The 88502 gap.** EPA reports PM2.5 under two AQI-valid parameter codes, `88101` and `88502`.
  The first API pull fetched only `88101`, silently dropping ~39% of PM2.5 monitor-days and making
  the 5-pollutant series read *lower* than the 2-pollutant one on 1,106 days. That is what `v2` means.

### `data/processed/la_daily_weather_2016_2025.csv`
3,652 rows, 2016-01-01 → 2025-12-30. Built from MERRA-2 **M2T1NXSLV** — 24 hourly steps × a 3×3
grid over 33.5–34.5 °N, 118.75–117.5 °W, with `T2M`, `QV2M`, `U10M`, `V10M`. Spatial mean over
the nine cells, then daily statistics.

**Wind speed is computed per hour and then averaged**, not derived from the daily mean U/V. A day
that blows hard east in the morning and hard west in the evening has a high mean speed but a
near-zero mean vector, and the second method would wrongly call it calm. Direction is the reverse
case and does use the mean vector.

### `data/processed/la_daily_trapping_2016_2025.csv`
3,646 rows. The variables that measure what actually holds pollution in the basin — `PBLH` from
M2T1NXFLX, `T850`/`SLP`/`TQV` from M2T1NXSLV — plus two derived features:

- **`inversion_strength`** = T850 − T2M. Positive means warm air caps cooler air. Present on 23.3%
  of days.
- **`ventilation_index`** = PBLH × wind speed. Mixing depth times transport speed, the metric
  operational forecasters use.

`t850_mean` correlates **0.753** with daily AQI — higher than any original feature, including
`t2m_max` at 0.705. Temperature 1.5 km up predicts LA's air quality better than surface temperature
does, because it measures the air mass without marine-layer contamination.

### `data/processed/la_daily_aqi_by_pollutant_2016_2025.csv`
The same AQI series factored into one column per pollutant. Verified to reconstruct `daily_aqi`
exactly (0 mismatches across 3,653 days). Ozone peaks in July, PM2.5 in December — nearly opposite
seasonal cycles.

### `train.csv` / `test.csv` / `la_modeling_dataset_v2.csv`
The modeling splits. 37 features in the base set, 61 with the trapping physics. Target `daily_aqi`.

- Same-day weather plus lags 1/2/3 and 3-day rolling means for the persistent variables.
- Wind direction and day-of-year encoded as sin/cos, since both wrap around.
- **Split chronologically at 2023-01-01** — train 2016–2022, test 2023–2025. A random split would
  train on 2024 and test on 2023, seeing the future and producing a meaningless score.

---

## Results

### The seven core models — identical 37 features, identical split

| model | test RMSE | test MAE | test R² | within 10 AQI | overfit gap |
|---|---:|---:|---:|---:|---:|
| **Ensemble (SVR+MLP+GBR)** | **18.34** | **12.95** | **0.794** | **53.8%** | +5.9 |
| SVR (RBF) | 18.85 | 13.57 | 0.782 | 49.0% | **+1.6** |
| Neural Network (MLP) | 19.05 | 13.99 | 0.777 | 48.5% | +2.2 |
| Gradient Boosting | 19.48 | 13.84 | 0.767 | 50.1% | +10.7 |
| Random Forest | 20.28 | 14.67 | 0.748 | 47.5% | +11.5 |
| KNN | 21.89 | 15.71 | 0.706 | 45.7% | n/a * |
| Linear Regression | 24.01 | 17.88 | 0.646 | 38.3% | −0.5 |
| *baseline: day-of-year climatology* | *31.76* | *24.51* | *0.381* | *28.7%* | — |
| *baseline: training mean* | *40.42* | *33.23* | *−0.003* | *13.9%* | — |

\* KNN's train RMSE is exactly 0.000 — an artifact of `weights='distance'`, where a training point
is its own nearest neighbour at distance zero. Its gap is not meaningful.

### The variants — same split, something else changed

| model | what changed | test RMSE | test R² | within 10 | MAE on AQI>150 |
|---|---|---:|---:|---:|---:|
| **Ensemble + Physics** | **+24 trapping features** | **16.91** | **0.825** | **56.5%** | 24.60 |
| Pollutant Split | one model per pollutant | 18.50 | 0.791 | 55.8% | 32.49 |
| Quantile GBR (q=0.8) | predicts the 80th percentile | 20.70 | 0.738 | 41.4% | **21.96** |

### Reading these tables

**Bring error bars.** With 1,095 test days the ordering among individual models is mostly noise.
A paired bootstrap puts the SVR-minus-MLP difference at −0.203 RMSE with a 95% interval of
[−0.65, +0.25] — it straddles zero. Rolling-origin validation confirms it directly: across six
annual folds, **four different models win at least one fold**, and Random Forest (fifth on the
single split) wins 2021 outright.

**Never compare two models by checking whether their individual confidence intervals overlap.**
That test is far too conservative. Resample the same days for both and take the difference.

**Year-to-year variation exceeds model-to-model variation.** The same model scores 27.1 on 2020
and 17.4 on 2024 — a 9.7-point spread, against a 5.4-point mean spread between best and worst
model within a year. 2020 is hardest for everything: the wildfire signature.

### What actually moved the needle

| change | RMSE | gain | what bought it |
|---|---:|---:|---|
| Linear regression baseline | 24.01 | — | — |
| Best single model (SVR) | 18.85 | **−5.16** | nonlinearity |
| Stacked ensemble | 18.34 | −0.51 | combining models |
| **+ trapping physics** | **16.91** | **−1.40** | better features |

**Features beat model choice by roughly three to one.** Every refinement of the *modelling* —
tuning, ensembling, target splitting — bought about half an RMSE point between them. Four
atmospheric variables bought 1.4–1.6.

### The 2026 forward test

The only genuinely prospective check: a model trained through 2025-12-30, scored on EPA data for
2026 that did not exist when the project was written. 151 days, **RMSE 15.17, R² 0.765**.

Compared honestly against a season-matched benchmark — the same calendar months from the
retrospective test period, RMSE 14.07 — that is about **1.1 points of degradation a year out**.
Bias is +3.05, consistent with LA air improving faster than a model fitted on dirtier years expects.

The full-year 18.34 is *not* the right comparison: 2026 data so far is January–May, which excludes
the ozone season.

---

## Credentials

Never commit these.

- **EPA AQS API** — `.env` at the repo root with `AQS_EMAIL` and `AQS_KEY`. Get a key at
  `https://aqs.epa.gov/data/api/signup?email=YOUR_EMAIL`.
- **NASA Earthdata** — a `urs.earthdata.nasa.gov` entry in `~/.netrc`, `chmod 600`.

Both are gitignored; the notebooks fall back to a `getpass` prompt if either is missing.

---

## Regenerating derived data

**`data/interim/`** — run `notebooks/aqi_pipeline.ipynb` end to end. The AQS API pull takes ~5
minutes (requests spaced 6 s apart) and is idempotent, skipping years already saved.

**`data/raw/merra2_slv/`** — ~200 MB, from the committed URL list:

```bash
cd data/raw/merra2_slv
wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies \
     --auth-no-challenge=on --keep-session-cookies --content-disposition \
     -i ../merra2_manifest.txt
for f in *.nc4.nc4; do mv "$f" "${f%.nc4}"; done   # strip the doubled extension
```

**`data/raw/merra2_flx/` and `merra2_slv_extra/`** — the trapping variables, ~290 MB together:

```bash
bash scripts/download_merra2_extra.sh 4      # 4 = parallelism
```

Idempotent — re-run it to fill in anything that failed. Transient 500s from GES DISC are common,
and a second pass usually clears them. It takes the date list and the MERRA-2 stream numbers from
the filenames already in `merra2_slv/`, because the 400/401 stream mapping is irregular and
interleaves rather than switching cleanly at a date boundary.

**`models/saved/`** — `python models/predict.py --train`. joblib pickles are tied to an exact
scikit-learn version, so they are rebuilt rather than committed.

---

## References

- EPA AQS API dailyData endpoint — `https://aqs.epa.gov/data/api/dailyData/byCounty`
- EPA AQS parameter class list — `https://aqs.epa.gov/data/api/list/parametersByClass?pc=AQI+POLLUTANTS`
- MERRA-2 M2T1NXSLV and M2T1NXFLX product pages — search the shortname at gesdisc.eosdis.nasa.gov
- Large files also mirrored in the team Google Drive:
  https://drive.google.com/drive/folders/1z5JRiwKwZazVCDbprDFK9i0DAzCEWDrr?usp=share_link
