# altREU — Team Bullets

**Predicting daily air quality in Los Angeles County from weather, 2016–2025.**

Two daily time series are assembled from scratch and joined: city-wide AQI from EPA
monitoring data, and weather from NASA's MERRA-2 reanalysis. Six machine-learning
models are then trained to predict AQI from weather and compared head to head.

**Headline result:** weather explains about **78% of the variance** in LA's daily AQI
(SVR, test R² 0.782, RMSE 18.9 on three held-out years). But a day-of-year climatology
using no weather at all already reaches R² 0.38 — so roughly half of that is the
seasonal cycle, and weather's genuine marginal contribution is the gap between the two.

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

Then run in order:

| # | notebook | what it does |
|---|---|---|
| 1 | `notebooks/aqi_pipeline.ipynb` | raw EPA data + AQS API → daily AQI series |
| 2 | `notebooks/merra2_pipeline.ipynb` | 3,652 hourly NetCDF granules → daily weather |
| 3 | `notebooks/build_modeling_dataset.ipynb` | join, engineer features, train/test split |
| 4 | `models/01`–`06_*.ipynb` | one model each, any order |
| 5 | `models/07_model_comparison.ipynb` | the comparison table and verdict |

Notebooks 1 and 2 need credentials (below). Notebooks 3–7 run on committed CSVs and
need nothing extra.

---

## Repository layout

```
data/
  raw/
    epa_ozone/            raw EPA AirData ozone exports        [tracked]
    epa_pollutant/        raw EPA AirData PM2.5 exports        [tracked]
    merra2_slv/           3,652 MERRA-2 NetCDF granules        [gitignored, ~200 MB]
    merra2_manifest.txt   GES DISC download URL list           [tracked]
  corrections/
    ozone_2021_la.csv     hand-fixed 2021 LA ozone file        [tracked]
  interim/                raw_renamed/, la_only/, api_pulls/   [gitignored]
  processed/              all final CSVs                       [tracked]

notebooks/                the three data-pipeline notebooks
models/
  model_utils.py          shared split loading + metrics
  01_linear_regression.ipynb    04_svr.ipynb
  02_random_forest.ipynb        05_knn.ipynb
  03_gradient_boosting.ipynb    06_neural_network.ipynb
  07_model_comparison.ipynb     the comparison table
  results/                per-model metrics JSON + test predictions

PROGRESS.md               running change log
requirements.txt          pinned dependencies
```

**Gitignored:** `data/raw/merra2_slv/`, `data/interim/`, `.venv/`, `.env`, `.DS_Store`.
Everything ignored is regeneratable — see *Regenerating derived data* below.

---

## The datasets

### `data/processed/la_daily_aqi_5pollutants_v2_2016_2025.csv`
3,653 rows, one per day, 2016-01-01 → 2025-12-31, zero missing.

| column | description |
|---|---|
| `date` | ISO date |
| `daily_aqi` | max AQI across all monitors and all pollutants (EPA-style city-wide reporting AQI) |
| `n_monitor_readings` | monitor-day-pollutant readings used |
| `n_sites` | distinct sites contributing |
| `pollutants_used` | pollutants with valid readings |
| `dominant_pollutant` | which pollutant drove the max (Ozone / PM2.5 / NO2) |
| `dominant_site` | which monitor recorded it |

Built from two sources: pre-downloaded **EPA AirData exports** (ozone + both PM2.5
codes) and the **EPA AQS API** (`dailyData/byCounty`) for all five criteria pollutants
at monitor-day resolution.

Two data bugs were found and fixed along the way:
- **A misfiled 2021 ozone file.** One AirData export was actually a Louisville-metro
  download. The corrected LA file is committed at `data/corrections/ozone_2021_la.csv`
  and copied into place automatically by the notebook.
- **The 88502 gap.** EPA reports PM2.5 under two AQI-valid parameter codes — `88101`
  (Local Conditions) and `88502` (Acceptable PM2.5 AQI & Speciation Mass). The first
  API pull fetched only `88101`, silently dropping ~39% of PM2.5 monitor-days, which
  made the 5-pollutant series read *lower* than the 2-pollutant one on 1,106 days. A
  second pull closed the gap; that's what `v2` means.

### `data/processed/la_daily_weather_2016_2025.csv`
3,652 rows, 2016-01-01 → 2025-12-30. (MERRA-2 had not published 2025-12-31 at pull time,
which is why the weather series is one day shorter than the AQI series.)

Built from MERRA-2 **M2T1NXSLV** granules — one per day, 24 hourly steps × a 3×3 grid
over 33.5–34.5 °N, 118.75–117.5 °W, with `T2M`, `QV2M`, `U10M`, `V10M`. Each granule is
reduced by taking the spatial mean over the nine cells, then daily statistics over the
24 hours: temperature mean/max/min/range (°C), humidity mean/max (g/kg), mean U/V wind,
wind speed mean/max/min, mean wind direction, and `calm_hours` (hours under 2 m/s).

One ordering detail matters: **wind speed is computed per hour and then averaged**, not
derived from the daily mean U/V. A day that blows hard east in the morning and hard west
in the evening has a high mean speed but a near-zero mean vector, and the second method
would wrongly report it as calm. Direction is the reverse case and does use the mean
vector.

### `data/processed/train.csv` / `test.csv`
The modeling split. 37 features, target `daily_aqi`.

- Same-day weather, plus lags 1/2/3 and 3-day rolling means for the five persistent
  variables (temperature, humidity, wind speed, calm hours).
- Wind direction and day-of-year encoded as sin/cos, since both wrap around and would
  otherwise read as huge numeric jumps at the boundary.
- **Split chronologically at 2023-01-01** — train on 2016–2022 (2,554 days), test on
  2023–2025 (1,095 days). A random split would let the model train on 2024 and test on
  2023, seeing the future and producing a meaningless score.

`feature_manifest.json` alongside them records the feature list and split so every model
notebook loads exactly the same thing.

---

## Model results

All six trained on 2016–2022, scored on 2023–2025. Tuned with `GridSearchCV` +
`TimeSeriesSplit(5)` inside the training years only.

| model | test RMSE | test MAE | test R² | within 10 AQI | overfit gap |
|---|---:|---:|---:|---:|---:|
| **SVR (RBF)** | **18.85** | 13.57 | **0.782** | 49.0% | **+1.6** |
| Neural Network (MLP) | 19.05 | 13.99 | 0.777 | 48.5% | +2.2 |
| Gradient Boosting | 19.48 | 13.84 | 0.767 | **50.1%** | +10.7 |
| Random Forest | 20.28 | 14.67 | 0.748 | 47.5% | +11.5 |
| KNN | 21.89 | 15.71 | 0.706 | 45.7% | n/a * |
| Linear Regression | 24.01 | 17.88 | 0.646 | 38.3% | −0.5 |
| *baseline: day-of-year climatology* | *31.76* | *24.51* | *0.381* | *28.7%* | — |
| *baseline: training mean* | *40.42* | *33.23* | *−0.003* | *13.9%* | — |

\* KNN's train RMSE is exactly 0.000 — an artifact of `weights='distance'`, where a
training point is its own nearest neighbour at distance zero. Its gap is not meaningful.

**Reading the table.** SVR wins, but the top four cluster between 18.9 and 20.3 and are
close to tied. What separates SVR is the *overfit gap*: at +1.6 against +10.7 and +11.5
for the tree ensembles, its training performance is an honest estimate of its behavior on
new data.

**The most important finding is not in the table.** Every one of the six models is biased
**downward on unhealthy days** (−27 to −48 AQI points above 150) and upward on clean ones.
That is regression to the mean under squared-error loss, not a tuning failure, and it is
the main obstacle to using any of these as a public-health warning system.

Full breakdown, charts, and next steps: `models/07_model_comparison.ipynb`.

---

## Credentials

Never commit these.

- **EPA AQS API** — `.env` at the repo root with `AQS_EMAIL` and `AQS_KEY`. Get a key at
  `https://aqs.epa.gov/data/api/signup?email=YOUR_EMAIL` (emailed within a minute).
- **NASA Earthdata** — a `urs.earthdata.nasa.gov` entry in `~/.netrc`, `chmod 600`. Used
  by wget for the MERRA-2 download.

Both files are gitignored. If either is missing, the notebooks fall back to a `getpass`
prompt.

---

## Regenerating derived data

**`data/interim/`** (`raw_renamed/`, `la_only/`, `api_pulls/`) — run
`notebooks/aqi_pipeline.ipynb` end to end. The AQS API pull takes ~5 minutes (requests
are spaced 6 s apart to respect EPA rate limits) and is idempotent, skipping years
already saved. Everything else takes seconds.

**`data/raw/merra2_slv/`** — ~200 MB of NetCDF, re-downloadable from GES DISC using the
committed URL list:

```bash
cd data/raw/merra2_slv
wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies \
     --auth-no-challenge=on --keep-session-cookies --content-disposition \
     -i ../merra2_manifest.txt
```

`--content-disposition` produces a doubled `.nc4.nc4` extension; strip it with
`for f in *.nc4.nc4; do mv "$f" "${f%.nc4}"; done`.

---

## References

- EPA AQS API dailyData endpoint — `https://aqs.epa.gov/data/api/dailyData/byCounty`
- EPA AQS parameter class list — `https://aqs.epa.gov/data/api/list/parametersByClass?pc=AQI+POLLUTANTS`
  (used to confirm CO/SO2/NO2 have no secondary codes analogous to PM2.5's 88502)
- MERRA-2 M2T1NXSLV product page — search "M2T1NXSLV" at gesdisc.eosdis.nasa.gov
- Large files also mirrored in the team Google Drive:
  https://drive.google.com/drive/folders/1z5JRiwKwZazVCDbprDFK9i0DAzCEWDrr?usp=share_link
