import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import fundamentals as fnd, metrics, risk_model as rm
from quantlab.synthetic import make_quality_panel

SEEDS = (7, 11, 23)


def _zscore(a):
    return (a - a.mean()) / (a.std() + 1e-12)


def _broadcast_loading(price):
    """(period x firm) panel of the STATIC ground-truth value loading."""
    vl = price.attrs["value_loading"]
    return pd.DataFrame(
        np.tile(vl.to_numpy(), (len(price.index), 1)),
        index=price.index,
        columns=price.columns,
    )


def _sr(net):
    return metrics.sharpe(net, periods=fnd.PERIODS_PER_YEAR)


def _raw_neutral_static(price):
    """Raw and STATIC-loading neutral SR (estimation-error-free)."""
    gp_a = price.attrs["gp_a"]
    raw = _sr(
        fnd.quality_backtest(fnd.quality_signal(gp_a), price, cost_bps_per_side=0.0)[
            "net"
        ]
    )
    vl = _broadcast_loading(price)
    neutral = _sr(
        fnd.quality_backtest(
            fnd.value_neutralized_signal(gp_a, vl), price, cost_bps_per_side=0.0
        )["net"]
    )
    return raw, neutral


def test_value_loading_collinearity_is_exact():
    for s in SEEDS:
        q = np.random.default_rng(s).standard_normal(200)
        qz = _zscore(q)
        a = make_quality_panel(mode="quality_is_value", seed=s)
        b = make_quality_panel(mode="quality_orthogonal", seed=s)
        assert np.corrcoef(a.attrs["value_loading"].to_numpy(), qz)[0, 1] == pytest.approx(
            1.0, abs=1e-9
        )
        assert abs(np.corrcoef(b.attrs["value_loading"].to_numpy(), qz)[0, 1]) < 1e-9


def test_quality_is_value_collapses_static_loading():
    for s in SEEDS:
        raw, neutral = _raw_neutral_static(make_quality_panel(mode="quality_is_value", seed=s))
        assert raw > 1.5
        # SCOPED to SEEDS=(7,11,23): measured [-0.41, +0.25]. NOT a generative
        # guarantee -- residual gp_noise carries a small seed-dependent alpha
        # (e.g. seed 28 -> 0.33, just over 0.3). The robust, seed-stable
        # discriminator is the paired (nb - na) gap, pinned below.
        assert neutral < 0.3


def test_quality_orthogonal_survives_static_loading():
    for s in SEEDS:
        raw, neutral = _raw_neutral_static(
            make_quality_panel(mode="quality_orthogonal", seed=s)
        )
        assert raw > 1.5  # measured [2.39, 3.45]
        assert neutral > 1.0  # measured [2.44, 3.70]; orthogonal alpha survives


def test_raw_sharpe_alone_does_not_separate_worlds():
    raw_A, raw_B = [], []
    for s in range(7, 40):  # 33 seeds
        pa = make_quality_panel(mode="quality_is_value", seed=s)
        pb = make_quality_panel(mode="quality_orthogonal", seed=s)
        raw_A.append(
            _sr(fnd.quality_backtest(fnd.quality_signal(pa.attrs["gp_a"]), pa,
                                     cost_bps_per_side=0.0)["net"])
        )
        raw_B.append(
            _sr(fnd.quality_backtest(fnd.quality_signal(pb.attrs["gp_a"]), pb,
                                     cost_bps_per_side=0.0)["net"])
        )
    # Genuine interleaving, not just touching bounding boxes: many World-A SRs
    # fall inside World-B's range and vice versa, and the means nearly coincide
    # -> a raw-SR threshold cannot separate the worlds. (A bounding-box "touch"
    # test would pass even for threshold-separable distributions; this would
    # fail if World B's premium re-separated the bulk.)
    raw_A, raw_B = np.array(raw_A), np.array(raw_B)
    assert sum(raw_B.min() <= a <= raw_B.max() for a in raw_A) >= 5  # measured 33
    assert sum(raw_A.min() <= b <= raw_A.max() for b in raw_B) >= 5  # measured 21
    assert abs(raw_A.mean() - raw_B.mean()) < 0.5  # measured 0.125


def test_collapse_requires_true_value_factor():
    for s in SEEDS:
        p = make_quality_panel(mode="quality_is_value", seed=s)
        gp_a = p.attrs["gp_a"]
        vl_true = _broadcast_loading(p)
        neutral_true = _sr(
            fnd.quality_backtest(
                fnd.value_neutralized_signal(gp_a, vl_true), p, cost_bps_per_side=0.0
            )["net"]
        )
        placebo = np.random.default_rng(10 * s + 1).standard_normal(p.shape[1])
        vl_pl = pd.DataFrame(
            np.tile(placebo, (len(p.index), 1)), index=p.index, columns=p.columns
        )
        neutral_pl = _sr(
            fnd.quality_backtest(
                fnd.value_neutralized_signal(gp_a, vl_pl), p, cost_bps_per_side=0.0
            )["net"]
        )
        assert neutral_true < 0.5 * neutral_pl  # collapse needs the TRUE factor


def test_quality_worlds_discriminate_under_rolling_estimated_loading():
    def rolling_neutral_sr(price):
        gp_a = price.attrs["gp_a"]
        sig = fnd.quality_signal(gp_a)
        rets = price.pct_change(fill_method=None)
        val_f = price.attrs["value_factor"]
        fb = rm.rolling_factor_betas(
            rets, val_f.to_frame("value"), lookback=36, min_periods=18
        )
        loading_panel = fb["value"]
        neutral = _sr(
            fnd.quality_backtest(
                fnd.value_neutralized_signal(gp_a, loading_panel), price,
                cost_bps_per_side=0.0,
            )["net"]
        )
        raw = _sr(fnd.quality_backtest(sig, price, cost_bps_per_side=0.0)["net"])
        return raw, neutral

    for s in SEEDS:
        ra, na = rolling_neutral_sr(make_quality_panel(mode="quality_is_value", seed=s))
        rb, nb = rolling_neutral_sr(make_quality_panel(mode="quality_orthogonal", seed=s))
        assert nb - na > 1.0  # discrimination survives estimation error (measured >=1.58)

    # poison-the-future leak check on the ESTIMATED-loading path
    p = make_quality_panel(mode="quality_is_value", seed=7)
    rets = p.pct_change(fill_method=None)
    val_f = p.attrs["value_factor"]
    full = rm.rolling_factor_betas(rets, val_f.to_frame("value"), 36, 18)["value"]
    rc = rets.copy()
    rc.iloc[120:] = 99.0
    vc = val_f.copy()
    vc.iloc[120:] = 99.0
    corr = rm.rolling_factor_betas(rc, vc.to_frame("value"), 36, 18)["value"]
    pd.testing.assert_frame_equal(full.iloc[:120], corr.iloc[:120])


def test_raw_vs_neutral_discrimination_is_paired():
    diffs = []
    for s in SEEDS:
        _, na = _raw_neutral_static(make_quality_panel(mode="quality_is_value", seed=s))
        _, nb = _raw_neutral_static(make_quality_panel(mode="quality_orthogonal", seed=s))
        diffs.append(nb - na)
    assert min(diffs) > 1.0  # measured static diffs ~2.5-4.1


def test_value_only_book_is_distinct_source_in_world_B():
    p = make_quality_panel(mode="quality_orthogonal", seed=7)
    vl = _broadcast_loading(p)
    w = fnd.quality_weights(vl, quantile=0.2)
    fwd = p.pct_change(fill_method=None).shift(-1).reindex_like(vl)
    sr_value_book = _sr((w * fwd).sum(axis=1, min_count=1).dropna())
    assert sr_value_book > 1.0  # measured 2.45 — a separable axis


def test_existing_quality_modes_byte_identical():
    GOLDEN = {
        ("planted_quality", 7): (88.05265287924138, 80197.25380580312, 44118259.98637506),
        ("planted_quality", 11): (104.49415328015294, 813.3210075005188, 110335216.5811997),
        ("planted_quality", 23): (94.82241475575768, 4.330658580839465, 132603175.13097264),
        ("null_quality", 7): (87.78416979403262, 89.60324762225966, 2060229.4892960282),
        ("null_quality", 11): (104.45376011102844, 133.82824174415123, 4835080.182006977),
        ("null_quality", 23): (93.74579764841464, 91.40853286135511, 6407075.697986995),
    }
    for (mode, seed), (first, last, total) in GOLDEN.items():
        p = make_quality_panel(mode=mode, seed=seed)
        np.testing.assert_allclose(p.iloc[0, 0], first, rtol=1e-12, atol=0)
        np.testing.assert_allclose(p.iloc[-1, -1], last, rtol=1e-12, atol=0)
        np.testing.assert_allclose(p.to_numpy().sum(), total, rtol=1e-12, atol=0)
        assert set(p.attrs) == {"gp_a", "mode"}  # NO value attrs leaked
    for mode in ("quality_is_value", "quality_orthogonal"):
        p = make_quality_panel(mode=mode, seed=7)
        assert "value_loading" in p.attrs and "value_factor" in p.attrs
        assert p.attrs["value_loading"].shape == (200,)
        assert p.attrs["value_factor"].shape == (180,)


def test_quality_mode_membership():
    with pytest.raises(ValueError, match="planted_quality"):
        make_quality_panel(mode="bogus")
    for m in ("planted_quality", "null_quality", "quality_is_value", "quality_orthogonal"):
        make_quality_panel(mode=m, seed=7)  # must not raise


def test_cscv_adjudicates_the_four_arms():
    from quantlab import pbo

    s = 7
    cols = {}
    for tag, mode in (
        ("raw_isvalue", "quality_is_value"),
        ("neutral_isvalue", "quality_is_value"),
        ("raw_orth", "quality_orthogonal"),
        ("neutral_orth", "quality_orthogonal"),
    ):
        p = make_quality_panel(mode=mode, seed=s)
        gp_a = p.attrs["gp_a"]
        if tag.startswith("raw"):
            sig = fnd.quality_signal(gp_a)
        else:
            sig = fnd.value_neutralized_signal(gp_a, _broadcast_loading(p))
        cols[tag] = fnd.quality_backtest(sig, p, cost_bps_per_side=0.0)["net"]
    df = pd.DataFrame(cols)
    # CONTIGUOUS common post-warm-up SLICE (NEVER dropna across mismatched
    # warm-ups -- that produces non-contiguous "contiguous" blocks).
    valid = df.dropna(how="any")
    start, end = valid.index.min(), valid.index.max()
    df = df.loc[start:end]
    assert not df.isna().any().any()  # contiguous & gap-free after slice
    out = pbo.cscv(df, n_splits=6)
    assert out["n_combinations"] == 20
    assert out["logits"].shape == (20,)
