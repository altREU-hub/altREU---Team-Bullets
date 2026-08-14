"""Predict LA County daily AQI for an arbitrary date from MERRA-2 weather.

Two things live here:

1. `train_and_save()` — fits the stacked ensemble on **all** labelled days and writes the
   artifacts to `models/saved/`. The notebooks deliberately fit on 2016-2022 only so the
   2023-2025 test score means something; a model you actually deploy should use everything.

2. `predict(date)` — fetches the four days of weather that date needs, rebuilds the feature
   row, and returns an AQI estimate.

**Only four granules are needed for any target date.** Every lag in the feature set is a
*weather* lag — there are no AQI lags — so there is no recursive chain back through history.
A prediction for 2026-03-07 needs weather for 03-04, 03-05, 03-06 and 03-07, and nothing else.

Usage:
    python models/predict.py --train                  # fit on all data, save artifacts
    python models/predict.py 2026-03-07               # predict one date
    python models/predict.py 2026-03-07 --explain     # ...and show the weather behind it

Requires a `~/.netrc` entry for `urs.earthdata.nasa.gov` (chmod 600) to reach GES DISC.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------- paths

def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / 'data' / 'processed').exists():
            return candidate
    raise FileNotFoundError('Could not locate the repo root from ' + str(here))


ROOT = repo_root()
PROCESSED = ROOT / 'data' / 'processed'
SAVED = ROOT / 'models' / 'saved'
CACHE = ROOT / 'data' / 'interim' / 'predict_cache'

# The 3x3 grid indices used throughout this project (LA bounding box).
LAT_SLICE, LON_SLICE = '247:249', '98:100'
OPENDAP = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2'

# MERRA-2 processing-stream numbers are irregular, so guessing is unreliable — we simply
# try each candidate until one returns 200.
STREAMS = ('400', '401')

LAG_VARS = ['t2m_mean', 't2m_max', 'qv2m_mean', 'wind_speed_mean', 'calm_hours']
LAGS = [1, 2, 3]


# ------------------------------------------------------------------- weather fetching

def _granule_url(date: pd.Timestamp, product: str, stream: str) -> str:
    d = date.strftime('%Y%m%d')
    y, m = date.strftime('%Y'), date.strftime('%m')
    if product == 'slv':
        base = f'{OPENDAP}/M2T1NXSLV.5.12.4/{y}/{m}/MERRA2_{stream}.tavg1_2d_slv_Nx.{d}.nc4.nc4'
        ce = ','.join([
            f'QV2M%5B0:23%5D%5B{LAT_SLICE}%5D%5B{LON_SLICE}%5D',
            f'T2M%5B0:23%5D%5B{LAT_SLICE}%5D%5B{LON_SLICE}%5D',
            f'U10M%5B0:23%5D%5B{LAT_SLICE}%5D%5B{LON_SLICE}%5D',
            f'V10M%5B0:23%5D%5B{LAT_SLICE}%5D%5B{LON_SLICE}%5D',
            'time', f'lat%5B{LAT_SLICE}%5D', f'lon%5B{LON_SLICE}%5D',
        ])
    else:
        raise ValueError(f'unknown product {product}')
    return f'{base}?{ce}'


def fetch_granule(date: pd.Timestamp, product: str = 'slv') -> Path:
    """Download one granule subset into the cache. Returns the local path.

    Tries each MERRA-2 stream number until one works, since the mapping from date to
    stream is irregular and not worth hard-coding.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f'{product}_{date.strftime("%Y%m%d")}.nc4'
    if out.exists() and out.stat().st_size > 0:
        return out

    last_err = ''
    for stream in STREAMS:
        url = _granule_url(date, product, stream)
        proc = subprocess.run(
            ['curl', '-sS', '-L', '-n', '-c', '/tmp/.merra_ck', '-b', '/tmp/.merra_ck',
             '--max-time', '180', '-o', str(out), '-w', '%{http_code}', url],
            capture_output=True, text=True,
        )
        if proc.stdout.strip() == '200' and out.exists() and out.stat().st_size > 0:
            return out
        last_err = proc.stdout.strip() or proc.stderr.strip()

    out.unlink(missing_ok=True)
    raise RuntimeError(
        f'Could not download {product} granule for {date.date()} (last HTTP status: {last_err}).\n'
        'If this is a future date, MERRA-2 does not have it yet — reanalysis lags real time by '
        'about 3-4 weeks. For forward-looking dates you need a forecast product (NOAA GFS), '
        'not MERRA-2.\n'
        'If it is a past date, check that ~/.netrc has a urs.earthdata.nasa.gov entry (chmod 600).'
    )


def wind_direction(u, v):
    """Meteorological wind direction: the direction wind blows FROM."""
    return np.degrees(np.arctan2(-u, -v)) % 360


def daily_weather(date: pd.Timestamp) -> dict:
    """One day of MERRA-2 -> the same daily summary the pipeline notebook produces."""
    import xarray as xr

    path = fetch_granule(date, 'slv')
    with xr.open_dataset(path) as ds:
        t2m = ds.T2M.mean(dim=('lat', 'lon')).values - 273.15
        qv2m = ds.QV2M.mean(dim=('lat', 'lon')).values * 1000.0
        u = ds.U10M.mean(dim=('lat', 'lon')).values
        v = ds.V10M.mean(dim=('lat', 'lon')).values

    speed = np.sqrt(u ** 2 + v ** 2)      # hourly speed, then averaged
    u_bar, v_bar = u.mean(), v.mean()     # direction from the mean vector
    return {
        'date': date.normalize(),
        't2m_mean': t2m.mean(), 't2m_max': t2m.max(), 't2m_min': t2m.min(),
        't2m_range': t2m.max() - t2m.min(),
        'qv2m_mean': qv2m.mean(), 'qv2m_max': qv2m.max(),
        'u10m_mean': u_bar, 'v10m_mean': v_bar,
        'wind_speed_mean': speed.mean(), 'wind_speed_max': speed.max(),
        'wind_speed_min': speed.min(),
        'wind_dir_mean': wind_direction(u_bar, v_bar),
        'calm_hours': int((speed < 2.0).sum()),
    }


# ---------------------------------------------------------------- feature engineering

def build_features(target_date: pd.Timestamp, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the single feature row for `target_date`.

    Mirrors notebooks/build_modeling_dataset.ipynb exactly. `weather` may be supplied to
    skip downloading (used by the self-test and by anyone feeding forecast data instead
    of reanalysis).
    """
    target_date = pd.Timestamp(target_date).normalize()
    window = [target_date - pd.Timedelta(days=k) for k in range(3, -1, -1)]

    if weather is None:
        weather = pd.DataFrame([daily_weather(d) for d in window])
    weather = weather.sort_values('date').reset_index(drop=True)

    missing = set(window) - set(weather['date'])
    if missing:
        raise ValueError(f'missing weather for {sorted(str(d.date()) for d in missing)}')

    df = weather.copy()

    rad = np.deg2rad(df['wind_dir_mean'])
    df['wind_dir_sin'] = np.sin(rad)
    df['wind_dir_cos'] = np.cos(rad)
    df = df.drop(columns=['wind_dir_mean'])

    doy = df['date'].dt.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)

    for var in LAG_VARS:
        for lag in LAGS:
            df[f'{var}_lag{lag}'] = df[var].shift(lag)
        df[f'{var}_roll3'] = df[var].rolling(window=3, min_periods=3).mean()

    row = df[df['date'] == target_date]
    if row.empty or row.isna().any(axis=1).iloc[0]:
        raise ValueError(f'could not build a complete feature row for {target_date.date()}')

    features = json.loads((PROCESSED / 'feature_manifest.json').read_text())['features']
    unknown = set(features) - set(row.columns)
    if unknown:
        raise ValueError(f'feature_manifest expects columns this builder does not produce: {sorted(unknown)}')
    return row[features].reset_index(drop=True)


# --------------------------------------------------------------------------- modelling

def _base_builders(random_state: int = 42) -> dict:
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        'SVR': lambda: Pipeline([('scale', StandardScaler()),
                                 ('model', SVR(kernel='rbf', C=300, gamma=0.01, epsilon=10.0))]),
        'MLP': lambda: Pipeline([('scale', StandardScaler()),
                                 ('model', MLPRegressor(hidden_layer_sizes=(128, 64), alpha=0.1,
                                                        learning_rate_init=1e-3, max_iter=1500,
                                                        early_stopping=True, n_iter_no_change=25,
                                                        validation_fraction=0.15,
                                                        random_state=random_state))]),
        'GBR': lambda: HistGradientBoostingRegressor(learning_rate=0.03, max_iter=600,
                                                     max_leaf_nodes=15, min_samples_leaf=10,
                                                     l2_regularization=1.0, early_stopping=False,
                                                     random_state=random_state),
    }


def train_and_save(verbose: bool = True) -> dict:
    """Fit the stacked ensemble on every labelled day and persist it to models/saved/."""
    import joblib
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import TimeSeriesSplit

    manifest = json.loads((PROCESSED / 'feature_manifest.json').read_text())
    features, target = manifest['features'], manifest['target']

    full = pd.read_csv(PROCESSED / 'la_modeling_dataset.csv', parse_dates=['date'])
    X, y = full[features], full[target]
    if verbose:
        print(f'training on all {len(X)} days '
              f'({full.date.min().date()} -> {full.date.max().date()}), {len(features)} features')

    builders = _base_builders()

    # stack weights from out-of-fold predictions, so the meta-model never sees a base
    # model's in-sample output
    oof = {n: np.full(len(X), np.nan) for n in builders}
    for i_tr, i_va in TimeSeriesSplit(n_splits=5).split(X):
        for name, build_fn in builders.items():
            oof[name][i_va] = build_fn().fit(X.iloc[i_tr], y.iloc[i_tr]).predict(X.iloc[i_va])
    M = np.column_stack([oof[n] for n in builders])
    ok = ~np.isnan(M).any(axis=1)
    stack = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(M[ok], np.asarray(y)[ok])

    fitted = {name: build_fn().fit(X, y) for name, build_fn in builders.items()}

    SAVED.mkdir(parents=True, exist_ok=True)
    joblib.dump({'base_models': fitted, 'stack': stack,
                 'members': list(builders), 'features': features, 'target': target,
                 'trained_through': str(full.date.max().date()),
                 'n_training_days': len(X)},
                SAVED / 'ensemble.joblib', compress=3)

    meta = {'members': list(builders),
            'stack_coefficients': dict(zip(builders, stack.coef_.round(4).tolist())),
            'stack_intercept': round(float(stack.intercept_), 4),
            'n_training_days': int(len(X)),
            'trained_through': str(full.date.max().date()),
            'n_features': len(features)}
    (SAVED / 'ensemble_meta.json').write_text(json.dumps(meta, indent=2))

    if verbose:
        size_mb = (SAVED / 'ensemble.joblib').stat().st_size / 1e6
        print(f'saved -> models/saved/ensemble.joblib ({size_mb:.1f} MB)')
        print(f'stack coefficients: {meta["stack_coefficients"]}')
    return meta


def load_model():
    import joblib
    path = SAVED / 'ensemble.joblib'
    if not path.exists():
        raise FileNotFoundError(
            'No saved model. Run:  python models/predict.py --train'
        )
    return joblib.load(path)


def predict(date, explain: bool = False) -> dict:
    """Predict daily AQI for `date`. Downloads the four granules it needs."""
    date = pd.Timestamp(date).normalize()
    bundle = load_model()

    window = [date - pd.Timedelta(days=k) for k in range(3, -1, -1)]
    weather = pd.DataFrame([daily_weather(d) for d in window])
    X = build_features(date, weather=weather)

    cols = np.column_stack([m.predict(X) for m in bundle['base_models'].values()])
    members = dict(zip(bundle['members'], cols[0].round(1)))
    value = float(bundle['stack'].predict(cols)[0])

    out = {'date': str(date.date()), 'predicted_aqi': round(value, 1),
           'category': aqi_category(value), 'member_predictions': members,
           'trained_through': bundle['trained_through']}
    if explain:
        out['weather'] = weather.set_index('date').round(2).to_dict('index')
    return out


def aqi_category(value: float) -> str:
    for threshold, name in [(50, 'Good'), (100, 'Moderate'), (150, 'Unhealthy for Sensitive Groups'),
                            (200, 'Unhealthy'), (300, 'Very Unhealthy')]:
        if value <= threshold:
            return name
    return 'Hazardous'


# ------------------------------------------------------------------------- self-test

def self_test(n: int = 5) -> bool:
    """Rebuild features for random historical dates and check they match the stored dataset.

    This is the guard that matters: if `build_features` ever drifts from the notebook's
    engineering, predictions would be silently wrong. Uses stored weather so it runs offline.
    """
    manifest = json.loads((PROCESSED / 'feature_manifest.json').read_text())
    features = manifest['features']
    stored = pd.read_csv(PROCESSED / 'la_modeling_dataset.csv', parse_dates=['date'])
    weather_all = pd.read_csv(PROCESSED / 'la_daily_weather_2016_2025.csv', parse_dates=['date'])

    rng = np.random.default_rng(0)
    sample = stored.sample(n, random_state=int(rng.integers(1e6)))

    worst = 0.0
    for _, row in sample.iterrows():
        d = row['date']
        window = weather_all[(weather_all.date >= d - pd.Timedelta(days=3)) & (weather_all.date <= d)]
        built = build_features(d, weather=window)
        diff = float(np.max(np.abs(built[features].values[0] - row[features].values.astype(float))))
        worst = max(worst, diff)

    ok = worst < 1e-6
    print(f'self-test on {n} historical dates: max feature difference {worst:.2e} '
          f'-> {"PASS" if ok else "FAIL"}')
    return ok


# ------------------------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('date', nargs='?', help='date to predict, YYYY-MM-DD')
    ap.add_argument('--train', action='store_true', help='fit on all data and save artifacts')
    ap.add_argument('--self-test', action='store_true', help='verify feature engineering matches the dataset')
    ap.add_argument('--explain', action='store_true', help='also print the weather behind the prediction')
    args = ap.parse_args()

    if args.train:
        train_and_save()
        return 0
    if args.self_test:
        return 0 if self_test() else 1
    if not args.date:
        ap.print_help()
        return 1

    try:
        result = predict(args.date, explain=args.explain)
    except Exception as exc:                      # noqa: BLE001 - CLI surface
        print(f'error: {exc}', file=sys.stderr)
        return 1

    print(f'\n  {result["date"]}   predicted AQI {result["predicted_aqi"]}  ({result["category"]})')
    print(f'  members: ' + '  '.join(f'{k} {v}' for k, v in result['member_predictions'].items()))
    print(f'  model trained through {result["trained_through"]}\n')
    if args.explain:
        print(pd.DataFrame(result['weather']).T.to_string())
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
