"""Falsification harness for regime detection (quantlab.regime).

The discipline mirrors planted/noise: before regime machinery is allowed
near real data it must (a) recover a KNOWN planted regime structure,
(b) provably not see the future, and (c) provably demonstrate that the
standard leaky construction (forward-backward smoothing) DOES see the
future -- the trap this module exists to keep out of the pipeline.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import features, models
from quantlab.regime import GaussianHMM2, causal_regime_probs
from quantlab.synthetic import make_panel


def _simulate_hmm(T=4000, seed=2):
    """Data from a KNOWN 2-state Gaussian HMM (the recovery target)."""
    rng = np.random.default_rng(seed)
    trans = np.array([[0.98, 0.02], [0.03, 0.97]])
    sigma = np.array([0.01, 0.03])
    states = np.zeros(T, dtype=int)
    for t in range(1, T):
        states[t] = rng.choice(2, p=trans[states[t - 1]])
    x = rng.standard_normal(T) * sigma[states]
    return x, states, trans, sigma


def test_em_recovers_known_parameters():
    x, states, trans, sigma = _simulate_hmm()
    m = GaussianHMM2().fit(x)
    # State 0 is calm by construction (class sorts on sigma).
    assert np.allclose(m.sigma, sigma, rtol=0.15)
    assert np.allclose(np.diag(m.trans), np.diag(trans), atol=0.03)
    # Smoothed argmax should track the true path closely on clean data.
    acc = (m.smoothed_probs(x).argmax(axis=1) == states).mean()
    assert acc > 0.85


def test_fit_fails_loudly_on_degenerate_inputs():
    # Adversarial review reproduced silent all-NaN fits (constant series,
    # tiny samples) and -- worse -- a silent point-mass collapse from ONE
    # bad print, where the collapsed state's sigma hit the old absolute
    # floor, got sorted into slot 0, and "P(calm)" became ~1e-305 forever.
    # All of these must now raise instead of trading on garbage.
    import pytest

    with pytest.raises(ValueError, match="constant series"):
        GaussianHMM2().fit(np.full(600, 0.01))
    with pytest.raises(ValueError, match=">= 100"):
        GaussianHMM2().fit(np.array([0.01, -0.02]))
    bad = np.random.default_rng(0).normal(0, 0.01, 600)
    bad[300] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        GaussianHMM2().fit(bad)

    # One vendor glitch print in an otherwise clean series: the classic
    # Gaussian-mixture point-mass collapse. Must raise, not mislabel.
    glitch = np.random.default_rng(1).normal(0, 0.01, 2000)
    glitch[1000] = 50.0
    with pytest.raises(ValueError, match="degenerate fit"):
        GaussianHMM2().fit(glitch)


def test_filtered_probs_cannot_see_the_future():
    # THE causality pin: corrupt everything after t0 -- filtered
    # probabilities before t0 must not move by a single float.
    x, *_ = _simulate_hmm()
    t0 = len(x) // 2
    m = GaussianHMM2().fit(x[:t0])  # params from the past only

    x_alt = x.copy()
    x_alt[t0:] = np.random.default_rng(99).standard_normal(len(x) - t0) * 0.2

    f = m.filtered_probs(x)
    f_alt = m.filtered_probs(x_alt)
    np.testing.assert_array_equal(f[:t0], f_alt[:t0])


def test_smoothed_probs_do_see_the_future_the_leak_is_real():
    # The same perturbation MUST move smoothed probabilities in the past:
    # that is what makes hmmlearn-style state outputs unusable for trading,
    # and this harness exists to prove it rather than assert it.
    x, states, *_ = _simulate_hmm()
    t0 = len(x) // 2
    m = GaussianHMM2().fit(x[:t0])

    x_alt = x.copy()
    x_alt[t0:] = np.random.default_rng(99).standard_normal(len(x) - t0) * 0.2

    s = m.smoothed_probs(x)
    s_alt = m.smoothed_probs(x_alt)
    assert np.abs(s[:t0] - s_alt[:t0]).max() > 1e-3

    # And the leak FLATTERS: smoothing recovers the true states better than
    # the causal filter -- the gap a naive regime backtest silently pockets.
    f = m.filtered_probs(x)
    acc_smoothed = (s.argmax(axis=1) == states).mean()
    acc_filtered = (f.argmax(axis=1) == states).mean()
    assert acc_smoothed > acc_filtered


def test_causal_regime_probs_walk_forward_is_point_in_time():
    x, *_ = _simulate_hmm(T=3000)
    idx = pd.bdate_range("2014-01-01", periods=len(x))
    s = pd.Series(x, index=idx)

    p = causal_regime_probs(s, min_train=504, refit_every=252)
    assert p[:504].isna().all()  # no past, no estimate -- never backfilled
    assert p[504:].notna().all()

    # Corrupt the future; everything already estimated must be unchanged
    # (parameters AND filter both consume only data <= t).
    k = 1500
    s_alt = s.copy()
    s_alt.iloc[k:] = 0.5
    p_alt = causal_regime_probs(s_alt, min_train=504, refit_every=252)
    # check_exact: the filter is deterministic, so "unchanged" means
    # bit-identical -- a tolerance here would certify sub-tolerance leaks.
    pd.testing.assert_series_equal(p.iloc[:k], p_alt.iloc[:k], check_exact=True)


_PANEL_CACHE: dict = {}


def _regime_panel(signal_strength: float):
    """planted_regime panel + its per-date momentum IC, cached per strength.

    signal_strength=0.0 is the PAIRED CONTROL: the identical world (same
    seed, same draws, same regime chain) with the signal switched off.
    Differencing against it cancels the label-machinery artifacts measured
    during development (mom-vs-residual-label IC of ~+0.06 in stressed /
    ~+0.02 in calm states on SIGNAL-FREE data -- an estimation artifact of
    residualization interacting with vol regimes, documented in the
    research log; with dispersed betas it reaches ~+0.13). Absolute-level
    assertions would test the artifact; paired differentials test the
    signal.
    """
    if signal_strength not in _PANEL_CACHE:
        panel = make_panel(
            n_assets=40, n_days=2000, mode="planted_regime", seed=11,
            signal_strength=signal_strength,
        )
        feats = features.build_features(panel)
        labels = features.build_labels(panel, horizon=21, residualize=True)
        mom = feats["mom_12_1"].stack()
        mom.index.names = ["date", "ticker"]
        ic = models.information_coefficient(mom, features.stack_panel(feats, labels))
        _PANEL_CACHE[signal_strength] = (panel, ic)
    return _PANEL_CACHE[signal_strength]


def _pure_window_means(ic: pd.Series, regimes: pd.Series, horizon: int = 21):
    """Mean IC over dates whose feature day AND full forward label window
    sit inside a single regime -- the only dates with an unambiguous
    ground-truth answer (windows straddling a switch mix both states)."""
    regv = regimes.to_numpy()
    pos = {d: i for i, d in enumerate(regimes.index)}
    calm, stressed = [], []
    for d in ic.index:
        i = pos[d]
        if i + horizon < len(regv):
            w = regv[i : i + horizon + 1]
            if (w == 0).all():
                calm.append(ic[d])
            elif (w == 1).all():
                stressed.append(ic[d])
    return float(np.mean(calm)), float(np.mean(stressed))


def test_planted_regime_switch_verified_against_paired_control():
    panel_on, ic_on = _regime_panel(0.03)
    panel_off, ic_off = _regime_panel(0.0)
    regimes = panel_on.attrs["regimes"]

    assert set(regimes.unique()) == {0, 1}
    # Persistent chain: both states materially present, few switches.
    assert 0.15 < regimes.mean() < 0.85
    assert (regimes.diff().abs() == 1).sum() < 60
    # Identical world up to the switch: same regimes, same draws.
    assert regimes.equals(panel_off.attrs["regimes"])

    calm_on, stressed_on = _pure_window_means(ic_on, regimes)
    calm_off, stressed_off = _pure_window_means(ic_off, regimes)
    # The signal exists exactly where planted: switching it on moves calm-
    # state IC a lot and stressed-state IC not at all (artifacts cancel in
    # the pair because both worlds share every random draw).
    assert calm_on - calm_off > 0.05
    assert abs(stressed_on - stressed_off) < 0.02


def test_planted_and_noise_modes_match_golden_values():
    # The falsification-gate baselines depend on the planted/noise panels
    # never changing. These literals were captured from the implementation
    # AS OF the planted_regime addition (2026-06-12, verified byte-identical
    # to the pre-change code) at n_days > lookback so the planted branch's
    # own draws are pinned too. Any edit that perturbs the legacy rng draw
    # sequence -- even one that shifts both modes identically -- fails here.
    # (The first version of this test compared planted to noise at 50 days,
    # where they are identical by construction: it could not fail. Caught
    # in adversarial review; see research_log 2026-06-12.)
    planted = make_panel(n_assets=10, n_days=300, mode="planted", seed=7)
    noise = make_panel(n_assets=10, n_days=300, mode="noise", seed=7)
    assert "regimes" not in planted.attrs and "regimes" not in noise.attrs

    golden = {
        "planted": (100.59548019891207, 6.51476839555455, 167003.37677401048),
        "noise": (100.59548019891207, 6.47579982965231, 166968.8989943195),
    }
    for name, panel in (("planted", planted), ("noise", noise)):
        first, last, total = golden[name]
        np.testing.assert_allclose(panel.iloc[0, 0], first, rtol=1e-12, atol=0)
        np.testing.assert_allclose(panel.iloc[-1, -1], last, rtol=1e-12, atol=0)
        np.testing.assert_allclose(panel.to_numpy().sum(), total, rtol=1e-12, atol=0)
    # And the planted signal branch demonstrably executed (modes diverge).
    assert planted.iloc[-1, -1] != noise.iloc[-1, -1]


def _gate_lift(panel: pd.DataFrame, ic: pd.Series) -> tuple[float, pd.Series]:
    """(gated - ungated mean IC, P(calm) series): the value the CAUSAL
    detector adds by down-weighting dates it believes are stressed."""
    mkt = panel.pct_change(fill_method=None).mean(axis=1)
    p_calm = causal_regime_probs(mkt, min_train=504, refit_every=126)
    both = pd.concat([ic, p_calm], axis=1, keys=["ic", "p"]).dropna()
    gated = float((both["ic"] * both["p"]).sum() / both["p"].sum())
    return gated - float(both["ic"].mean()), p_calm


def test_causal_gate_adds_value_with_signal_and_conjures_nothing_without():
    # End-to-end falsification, paired-control form: the same causal HMM
    # gate must IMPROVE mean momentum IC in the world where the signal is
    # regime-conditional, and must NOT improve it in the identical world
    # with the signal switched off (a gate that "helps" on signal-free
    # data is laundering an artifact -- the regime version of finding
    # alpha in noise).
    panel_on, ic_on = _regime_panel(0.03)
    panel_off, ic_off = _regime_panel(0.0)

    lift_on, p_calm = _gate_lift(panel_on, ic_on)
    lift_off, _ = _gate_lift(panel_off, ic_off)
    assert lift_on > 0.005
    assert lift_off < 0.005

    # And the detector itself tracks the hidden truth out-of-sample.
    regimes = panel_on.attrs["regimes"]
    scored = p_calm.dropna()
    acc = ((scored > 0.5).astype(int) == (1 - regimes.reindex(scored.index))).mean()
    assert acc > 0.85
