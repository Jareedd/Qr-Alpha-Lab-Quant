"""Regime detection with a hard causality boundary.

Simons-style regime modeling (the HMM everyone reaches for after reading
about Renaissance) contains finance's most seductive leak: the standard
fitted state probabilities -- forward-BACKWARD smoothing, what
``hmmlearn.predict_proba`` hands you -- condition on the FULL sample. The
state estimate at time t uses returns from t+1..T. A backtest gated on
smoothed states looks brilliant and is unusable; the model "detected" the
regime change partly by watching it happen afterward.

Only the forward FILTER, P(state_t | r_1..r_t), is point-in-time. This
module therefore exposes BOTH:

- ``filtered_probs``  -- causal; the only output a strategy may consume.
- ``smoothed_probs``  -- anticausal; exists so the falsification harness
  can DEMONSTRATE the leak. Tests pin that (a) perturbing the future moves
  smoothed but never filtered probabilities, and (b) smoothed state
  recovery beats filtered on synthetic regimes -- the flattering gap a
  naive backtest silently pockets.

The HMM itself is a small 2-state Gaussian, EM-fit in numpy (no new
dependency; ~100 lines beats importing a library whose default output is
the leak we're guarding against). Walk-forward use goes through
``causal_regime_probs``: parameters are re-fit on an expanding window of
PAST data only, and the filter at t consumes data through t only.

Assumptions stated:
- Univariate input (a market return series); regimes are volatility
  regimes, the only kind with strong evidence of persistence.
- Two states, ordered so state 0 is the LOW-volatility ("calm") state.
- EM is initialized deterministically (quantile-split start), so fits are
  reproducible without a seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SIGMA_FLOOR = 1e-8
_DENS_FLOOR = 1e-300


class GaussianHMM2:
    """2-state Gaussian hidden Markov model, univariate, EM-fit.

    Boring and explicit on purpose: every recursion is inspectable, and
    the causal/anticausal boundary is the entire point of the class.
    """

    def __init__(self, n_iter: int = 200, tol: float = 1e-8):
        self.n_iter = n_iter
        self.tol = tol
        self.mu = np.zeros(2)
        self.sigma = np.ones(2)
        self.trans = np.full((2, 2), 0.5)
        self.pi = np.full(2, 0.5)

    # -- internals ----------------------------------------------------------

    def _emissions(self, x: np.ndarray) -> np.ndarray:
        """(T, 2) Gaussian densities, floored away from zero."""
        z = (x[:, None] - self.mu[None, :]) / self.sigma[None, :]
        dens = np.exp(-0.5 * z**2) / (self.sigma[None, :] * np.sqrt(2 * np.pi))
        return np.maximum(dens, _DENS_FLOOR)

    def _forward(self, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scaled forward recursion. Returns (alpha_hat, c) where
        alpha_hat[t] = P(state_t | x_1..x_t) -- already the filtered
        posterior thanks to per-step normalization."""
        T = len(B)
        alpha = np.empty((T, 2))
        c = np.empty(T)
        a = self.pi * B[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.trans) * B[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        return alpha, c

    def _backward(self, B: np.ndarray, c: np.ndarray) -> np.ndarray:
        T = len(B)
        beta = np.empty((T, 2))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.trans @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        return beta

    # -- API ----------------------------------------------------------------

    def fit(self, x: np.ndarray) -> "GaussianHMM2":
        """Baum-Welch EM. Deterministic init: states start as the low/high
        dispersion halves of the data (split at the median absolute value),
        which breaks symmetry without randomness.

        Fails LOUDLY on degenerate inputs and degenerate fits (point-mass
        state collapse). A silently-broken regime gate that zeroes P(calm)
        forever is worse than a crash -- adversarial review reproduced
        exactly that from one bad print with the original absolute sigma
        floor, so the floor is now scale-relative and occupancy is checked.
        Input hygiene is still the caller's job: the guard catches collapse,
        not every conceivable data pathology."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) < 100:
            raise ValueError(f"need a 1-D series of >= 100 returns, got shape {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("non-finite values in input -- clean the series first")
        scale = float(x.std())
        if scale == 0.0:
            raise ValueError("constant series has no volatility regimes")
        # Scale-relative floor: an absolute 1e-8 floor lets a state collapse
        # onto a single value (sigma -> 1e-8), and the sigma-sort below would
        # then crown the point mass as "calm". 1% of overall vol is far below
        # any real regime's vol but blocks the collapse geometry.
        floor = max(_SIGMA_FLOOR, 0.01 * scale)

        lo = np.abs(x) <= np.median(np.abs(x))
        self.mu = np.array([x[lo].mean(), x[~lo].mean()])
        self.sigma = np.maximum(np.array([x[lo].std(), x[~lo].std()]), floor)
        self.trans = np.array([[0.95, 0.05], [0.05, 0.95]])
        self.pi = np.full(2, 0.5)

        gamma = np.full((len(x), 2), 0.5)
        prev_ll = -np.inf
        for _ in range(self.n_iter):
            B = self._emissions(x)
            alpha, c = self._forward(B)
            beta = self._backward(B, c)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            # xi[t, j, k]: joint posterior of (state_t = j, state_{t+1} = k)
            xi = (
                alpha[:-1, :, None]
                * self.trans[None, :, :]
                * (B[1:] * beta[1:])[:, None, :]
                / c[1:, None, None]
            )

            self.pi = gamma[0]
            self.trans = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
            self.trans /= self.trans.sum(axis=1, keepdims=True)
            w = gamma.sum(axis=0)
            self.mu = (gamma * x[:, None]).sum(axis=0) / w
            self.sigma = np.maximum(
                np.sqrt((gamma * (x[:, None] - self.mu[None, :]) ** 2).sum(axis=0) / w),
                floor,
            )

            # ll uses pre-M-step params; immaterial for the break test since
            # an M-step never decreases the likelihood.
            ll = float(np.log(c).sum())
            if ll - prev_ll < self.tol:
                break
            prev_ll = ll

        occupancy = gamma.mean(axis=0)
        params = np.concatenate([self.mu, self.sigma, self.trans.ravel(), self.pi])
        if not np.isfinite(params).all() or occupancy.min() < 0.005:
            raise ValueError(
                f"degenerate fit (state occupancy {occupancy.round(4)}, "
                f"sigma {self.sigma}): a collapsed state would be silently "
                "mislabeled as a regime -- inspect the input for bad prints"
            )

        # State 0 = calm (lower sigma), always -- callers rely on it.
        if self.sigma[0] > self.sigma[1]:
            order = [1, 0]
            self.mu, self.sigma = self.mu[order], self.sigma[order]
            self.pi = self.pi[order]
            self.trans = self.trans[np.ix_(order, order)]
        return self

    def filtered_probs(self, x: np.ndarray) -> np.ndarray:
        """CAUSAL given the parameters: row t is P(state_t | x_1..x_t) under
        mu/sigma/trans/pi as fitted. The recursion never looks ahead, but
        the PARAMETERS carry whatever sample they were fit on (pi is even a
        smoothed quantity) -- so this is only fully point-in-time when the
        fit sample ends at or before the first row you consume. For
        walk-forward use, go through ``causal_regime_probs``, which enforces
        that. The only output that may ever touch a trading decision."""
        alpha, _ = self._forward(self._emissions(np.asarray(x, dtype=float)))
        return alpha

    def smoothed_probs(self, x: np.ndarray) -> np.ndarray:
        """ANTICAUSAL: row t is P(state_t | x_1..x_T) -- conditions on the
        future. Exposed only so tests can demonstrate the leak; using this
        in a strategy is exactly the bug the falsification harness exists
        to catch."""
        B = self._emissions(np.asarray(x, dtype=float))
        alpha, c = self._forward(B)
        beta = self._backward(B, c)
        gamma = alpha * beta
        return gamma / gamma.sum(axis=1, keepdims=True)


def causal_regime_probs(
    market_rets: pd.Series,
    min_train: int = 504,
    refit_every: int = 63,
    n_iter: int = 200,
) -> pd.Series:
    """Walk-forward P(calm regime | data through t), strictly point-in-time.

    Parameters at t come from an EM fit on data up to the most recent refit
    date <= t (expanding window); the probability at t then filters data
    through t only. The first ``min_train`` observations are NaN -- there
    is no past to fit on, and we do not pretend otherwise.

    Why both layers matter: fitting on the full series and filtering would
    leak through the PARAMETERS (means/vols estimated on future data) even
    though the filter itself is causal. Tests pin both directions.
    """
    vals = market_rets.dropna()
    arr = vals.to_numpy(dtype=float)
    out = pd.Series(np.nan, index=market_rets.index, name="p_calm")
    t = min_train
    while t < len(arr):
        end = min(t + refit_every, len(arr))
        model = GaussianHMM2(n_iter=n_iter).fit(arr[:t])
        # filtered_probs row j uses arr[:j+1] only -- causal by recursion.
        probs = model.filtered_probs(arr[:end])
        out.loc[vals.index[t:end]] = probs[t:end, 0]
        t = end
    return out
