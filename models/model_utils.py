"""Shared helpers for the six model notebooks.

Every model notebook loads the same split through `load_split()` and scores itself
through `evaluate()`, so the comparison table in `07_model_comparison.ipynb` is
comparing like with like. Results land in `models/results/` as JSON.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42


def repo_root() -> Path:
    """Repo root, whether the notebook was launched from root or models/."""
    here = Path.cwd()
    for candidate in (here, here.parent, here.parent.parent):
        if (candidate / 'data' / 'processed').exists():
            return candidate
    raise FileNotFoundError(
        f'Could not locate the repo root from {here}. '
        'Run build_modeling_dataset.ipynb first.'
    )


ROOT = repo_root()
PROCESSED = ROOT / 'data' / 'processed'
RESULTS = ROOT / 'models' / 'results'


def load_split():
    """Return (X_train, y_train, X_test, y_test, meta).

    meta carries the feature list, the split date, and the test dates — the last
    one is needed for the time-series diagnostic plots.
    """
    for name in ('train.csv', 'test.csv', 'feature_manifest.json'):
        if not (PROCESSED / name).exists():
            raise FileNotFoundError(
                f'{name} is missing. Run notebooks/build_modeling_dataset.ipynb first.'
            )

    manifest = json.loads((PROCESSED / 'feature_manifest.json').read_text())
    features, target = manifest['features'], manifest['target']

    train = pd.read_csv(PROCESSED / 'train.csv', parse_dates=['date'])
    test = pd.read_csv(PROCESSED / 'test.csv', parse_dates=['date'])

    meta = {
        **manifest,
        'train_dates': train['date'],
        'test_dates': test['date'],
    }
    return (train[features], train[target],
            test[features], test[target], meta)


def metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, R^2, MAPE, plus the share of days predicted within 10 AQI points.

    That last one is the practically meaningful number: AQI categories are ~50
    points wide, so being within 10 means you called the category right.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'mape': float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        'within_10': float(np.mean(np.abs(y_true - y_pred) <= 10) * 100),
    }


def evaluate(name, model, X_train, y_train, X_test, y_test,
             params=None, notes=None, save=True) -> dict:
    """Score a fitted model on both splits, print a summary, and save to JSON.

    Train metrics are reported alongside test metrics purely to expose
    overfitting — the gap between them is the number to watch.
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    result = {
        'model': name,
        'params': params or {},
        'notes': notes or '',
        'train': metrics(y_train, train_pred),
        'test': metrics(y_test, test_pred),
    }
    result['overfit_gap'] = result['test']['rmse'] - result['train']['rmse']

    print(f'=== {name} ===')
    print(f'{"metric":<12}{"train":>10}{"test":>10}')
    for key, label in [('rmse', 'RMSE'), ('mae', 'MAE'), ('r2', 'R2'),
                       ('mape', 'MAPE %'), ('within_10', 'within10 %')]:
        print(f'{label:<12}{result["train"][key]:>10.3f}{result["test"][key]:>10.3f}')
    print(f'\noverfit gap (test RMSE - train RMSE): {result["overfit_gap"]:+.3f}')

    if save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        slug = name.lower().replace(' ', '_').replace('-', '_')
        (RESULTS / f'{slug}.json').write_text(json.dumps(result, indent=2))
        np.save(RESULTS / f'{slug}_test_pred.npy', test_pred)
        print(f'saved -> models/results/{slug}.json')

    return result


def load_all_results() -> pd.DataFrame:
    """Collect every saved result JSON into one flat table."""
    rows = []
    for path in sorted(RESULTS.glob('*.json')):
        r = json.loads(path.read_text())
        rows.append({
            'Model': r['model'],
            'Test RMSE': r['test']['rmse'],
            'Test MAE': r['test']['mae'],
            'Test R2': r['test']['r2'],
            'Test MAPE %': r['test']['mape'],
            'Within 10 AQI %': r['test']['within_10'],
            'Train RMSE': r['train']['rmse'],
            'Train R2': r['train']['r2'],
            'Overfit Gap': r['overfit_gap'],
            'Notes': r.get('notes', ''),
        })
    if not rows:
        raise FileNotFoundError(
            f'No result files in {RESULTS}. Run notebooks 01-06 first.'
        )
    return pd.DataFrame(rows).sort_values('Test RMSE').reset_index(drop=True)


PLOT_STYLE = {
    'figure.figsize': (11, 4),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
}


def diagnostic_plots(name, y_test, test_pred, test_dates):
    """The three plots every model notebook shows: fit, residuals, and time series."""
    import matplotlib.pyplot as plt

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        lo, hi = float(min(y_test.min(), test_pred.min())), float(max(y_test.max(), test_pred.max()))
        axes[0].scatter(y_test, test_pred, s=8, alpha=0.35, edgecolor='none')
        axes[0].plot([lo, hi], [lo, hi], 'r--', lw=1, label='perfect prediction')
        axes[0].set_xlabel('Actual AQI')
        axes[0].set_ylabel('Predicted AQI')
        axes[0].set_title(f'{name}: predicted vs actual (test)')
        axes[0].legend()

        residuals = np.asarray(y_test) - np.asarray(test_pred)
        axes[1].scatter(test_pred, residuals, s=8, alpha=0.35, edgecolor='none')
        axes[1].axhline(0, color='r', ls='--', lw=1)
        axes[1].set_xlabel('Predicted AQI')
        axes[1].set_ylabel('Residual (actual - predicted)')
        axes[1].set_title(f'{name}: residuals')
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(test_dates, y_test, lw=0.9, label='actual', color='#333')
        ax.plot(test_dates, test_pred, lw=0.9, label='predicted', color='#d1495b', alpha=0.85)
        ax.set_ylabel('Daily AQI')
        ax.set_title(f'{name}: test period, 2023-2025')
        ax.legend()
        plt.tight_layout()
        plt.show()
