"""Model wrappers with a uniform fit/predict interface.

Deliberately boring models. In cross-sectional equity prediction the signal-to-
noise ratio is so low that regularized linear models are a strong baseline;
any fancier model must beat Ridge out-of-sample to justify itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


def make_model(name: str):
    if name == "ridge":
        return Ridge(alpha=10.0)
    if name == "gbr":
        return HistGradientBoostingRegressor(
            max_depth=3,
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=0,
        )
    raise ValueError(f"unknown model {name!r}; use 'ridge', 'ridge_cv', or 'gbr'")


# Candidate alphas for nested tuning. A coarse log-spaced grid on purpose:
# a fine grid buys nothing at this signal-to-noise and multiplies compute.
RIDGE_ALPHA_GRID = (1.0, 10.0, 100.0, 1000.0)


def select_ridge_alpha(
    train_panel: pd.DataFrame,
    embargo_days: int,
    grid: tuple[float, ...] = RIDGE_ALPHA_GRID,
) -> float:
    """Nested hyperparameter selection: pick Ridge alpha by inner walk-forward.

    Runs an inner expanding walk-forward *within the training window only* and
    scores each candidate alpha by mean out-of-fold rank IC. The outer test
    window is never seen, so this is in-sample model selection: re-tuning on
    every roll does NOT inflate the DSR trial count, because nothing here is
    selected on outer out-of-sample results.

    Falls back to the grid's middle value if the training window is too short
    for even one inner split (early outer rolls on short histories).
    """
    from quantlab.validation import WalkForwardSplitter

    inner = WalkForwardSplitter(
        min_train_days=504, test_days=126, embargo_days=embargo_days
    )
    best_alpha, best_ic = grid[len(grid) // 2], -np.inf
    for alpha in grid:
        try:
            preds = walk_forward_predict(
                train_panel, inner, model_factory=lambda a=alpha: Ridge(alpha=a)
            )
        except RuntimeError:  # window too short for any inner split
            return best_alpha
        mean_ic = information_coefficient(preds, train_panel).mean()
        if mean_ic > best_ic:
            best_alpha, best_ic = alpha, mean_ic
    return best_alpha


def walk_forward_predict(
    panel: pd.DataFrame,
    splitter,
    model_name: str = "ridge",
    model_factory=None,
) -> pd.Series:
    """Train on each walk-forward window, predict the test window.

    ``panel``: long-format (date, ticker) frame with feature columns + 'label'.
    Returns an out-of-sample prediction Series indexed by (date, ticker).

    ``model_name='ridge_cv'`` re-selects the Ridge alpha on every roll via
    nested inner walk-forward on the training window (see select_ridge_alpha).
    ``model_factory`` (a zero-arg callable) overrides model_name entirely --
    used internally by the nested tuner.

    Window selection uses positional slicing on the date-sorted panel rather
    than per-split ``isin`` masks. This is equivalent because the splitter's
    train windows are always a prefix of the unique dates and its test windows
    are contiguous -- and it stays fast as the universe grows.
    """
    feature_cols = [c for c in panel.columns if c != "label"]
    date_vals = panel.index.get_level_values("date")
    if not date_vals.is_monotonic_increasing:
        panel = panel.sort_index(level="date", sort_remaining=False)
        date_vals = panel.index.get_level_values("date")
    dates_np = date_vals.to_numpy()
    X = panel[feature_cols].to_numpy()
    y = panel["label"].to_numpy()
    preds = []

    for train_dates, test_dates in splitter.split(pd.DatetimeIndex(date_vals.unique())):
        train_end = np.searchsorted(dates_np, train_dates[-1].to_datetime64(), side="right")
        test_lo = np.searchsorted(dates_np, test_dates[0].to_datetime64(), side="left")
        test_hi = np.searchsorted(dates_np, test_dates[-1].to_datetime64(), side="right")
        if train_end == 0 or test_lo == test_hi:
            continue
        if model_factory is not None:
            model = model_factory()
        elif model_name == "ridge_cv":
            alpha = select_ridge_alpha(
                panel.iloc[:train_end], embargo_days=splitter.embargo_days
            )
            model = Ridge(alpha=alpha)
        else:
            model = make_model(model_name)
        model.fit(X[:train_end], y[:train_end])
        p = model.predict(X[test_lo:test_hi])
        preds.append(pd.Series(p, index=panel.index[test_lo:test_hi]))

    if not preds:
        raise RuntimeError("no walk-forward splits produced predictions")
    out = pd.concat(preds).sort_index()
    return out[~out.index.duplicated()]


def information_coefficient(preds: pd.Series, panel: pd.DataFrame) -> pd.Series:
    """Per-date Spearman rank IC between predictions and realized labels.

    Vectorized as Pearson correlation of within-date ranks (identical to
    Spearman, same average tie handling) -- one groupby pass instead of a
    Python apply per date.
    """
    df = pd.DataFrame({"pred": preds, "label": panel["label"]}).dropna()
    ranks = df.groupby(level="date").rank()
    by_date = ranks.groupby(level="date")
    demeaned = ranks - by_date.transform("mean")
    num = (demeaned["pred"] * demeaned["label"]).groupby(level="date").sum()
    denom = np.sqrt(
        (demeaned["pred"] ** 2).groupby(level="date").sum()
        * (demeaned["label"] ** 2).groupby(level="date").sum()
    )
    ic = num / denom
    ic[by_date.size() < 5] = np.nan
    return ic.dropna()
