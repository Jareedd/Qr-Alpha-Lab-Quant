"""The falsification gate, caught in the act. ZERO trials (pure synthetic).

    .venv/Scripts/python.exe scripts/leak_demo.py

Runs the SAME pipeline three times and prints the gate verdict each time:

  1. PLANTED signal  -> Deflated Sharpe ~0.99  -> RECOVERED   (gate: PASS)
  2. PURE NOISE      -> Deflated Sharpe ~0.00  -> REJECTED    (gate: PASS)
  3. NOISE + a ONE-LINE look-ahead leak (a feature set equal to the forward-return
     label) -> Deflated Sharpe spikes toward 1 -> noise "finds alpha"
     -> GATE FAILS (red)

This is exactly what CI runs on every push (`--data noise --fail-if-dsr-above
0.5`): the moment a future-peeking feature enters the codebase, noise mode stops
rejecting and the build goes red. The demo prints a transcript to
results/leak_demo_transcript.txt for the README/GIF.
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import backtest, features, metrics, models, validation
from quantlab.synthetic import make_panel

HORIZON = 21
NOISE_FAIL_ABOVE = 0.5      # CI: noise DSR above this == leakage
PLANTED_FAIL_BELOW = 0.95   # CI: planted DSR below this == pipeline broken


def _dsr(panel, prices, n_trials: int) -> tuple[float, float]:
    splitter = validation.WalkForwardSplitter(embargo_days=HORIZON)
    preds = models.walk_forward_predict(panel, splitter, model_name="ridge")
    ic = models.information_coefficient(preds, panel)
    weights = backtest.predictions_to_weights(preds, rebalance_every=HORIZON)
    res = backtest.run_backtest(weights, prices, cost_bps=10.0)
    return metrics.deflated_sharpe_ratio(res["net"], n_trials=n_trials), float(ic.mean())


def _panel(mode: str):
    prices = make_panel(mode=mode)
    feats = features.build_features(prices)
    labels = features.build_labels(prices, horizon=HORIZON)
    return features.stack_panel(feats, labels), prices


def run() -> int:
    print("=" * 70)
    print(" FALSIFICATION GATE DEMO  --  watch CI catch a look-ahead leak")
    print("=" * 70)

    # 1. planted: the gate must RECOVER it
    p_panel, p_prices = _panel("planted")
    p_dsr, p_ic = _dsr(p_panel, p_prices, n_trials=1)
    ok1 = p_dsr >= PLANTED_FAIL_BELOW
    print(f"\n[1] PLANTED signal     IC={p_ic:+.4f}  DSR={p_dsr:.4f}")
    print(f"    gate (DSR >= {PLANTED_FAIL_BELOW}): {'PASS -- signal recovered' if ok1 else 'FAIL'}")

    # 2. pure noise: the gate must REJECT it
    n_panel, n_prices = _panel("noise")
    n_dsr, n_ic = _dsr(n_panel, n_prices, n_trials=20)
    ok2 = n_dsr <= NOISE_FAIL_ABOVE
    print(f"\n[2] PURE NOISE         IC={n_ic:+.4f}  DSR={n_dsr:.4f}")
    print(f"    gate (DSR <= {NOISE_FAIL_ABOVE}): {'PASS -- nothing found, as it must be' if ok2 else 'FAIL'}")

    # 3. noise + a one-line look-ahead leak: a feature set to the FORWARD-return label
    leaky = n_panel.copy()
    leaky["leak_fwd_return"] = leaky["label"]          # <-- the bug: feature peeks at the future
    l_dsr, l_ic = _dsr(leaky, n_prices, n_trials=20)
    ok3 = l_dsr <= NOISE_FAIL_ABOVE                    # the gate WANTS this to stay rejected
    print(f"\n[3] NOISE + 1-line look-ahead leak   IC={l_ic:+.4f}  DSR={l_dsr:.4f}")
    print(f"    (injected: panel['leak_fwd_return'] = panel['label'])")
    print(f"    gate (DSR <= {NOISE_FAIL_ABOVE}): "
          f"{'PASS' if ok3 else 'FAIL -- noise FOUND ALPHA: CI goes RED, hunt the leak'}")

    print("\n" + "-" * 70)
    caught = (not ok3) and ok1 and ok2
    print(f"  RESULT: planted recovered, clean noise rejected, and the leak "
          f"flipped noise from DSR {n_dsr:.4f} -> {l_dsr:.4f}.")
    print(f"  The gate that is green on clean noise goes RED the instant a "
          f"future-peeking feature enters -- on every push.")
    print(f"  Demo {'WORKED (leak caught)' if caught else 'DID NOT behave as expected'}.")
    return 0 if caught else 1


def main() -> int:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = run()
    text = buf.getvalue()
    print(text, end="")
    os.makedirs("results", exist_ok=True)
    with open("results/leak_demo_transcript.txt", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"  Wrote results/leak_demo_transcript.txt")
    return code


if __name__ == "__main__":
    sys.exit(main())
