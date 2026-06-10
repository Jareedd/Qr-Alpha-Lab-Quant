"""Read-only Alpaca paper-account snapshot (positions, exposure, neutrality)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab.live import AlpacaPaper

b = AlpacaPaper()
acct = b.account()
pos = b._req("/positions")
longs = [p for p in pos if float(p["qty"]) > 0]
shorts = [p for p in pos if float(p["qty"]) < 0]
lv = sum(abs(float(p["market_value"])) for p in longs)
sv = sum(abs(float(p["market_value"])) for p in shorts)
eq = float(acct["equity"])
print(f"equity: ${eq:,.0f} | positions: {len(pos)} ({len(longs)} long / {len(shorts)} short)")
print(f"long: ${lv:,.0f} | short: ${sv:,.0f} | net: ${lv - sv:,.0f} ({(lv - sv) / eq:+.2%} of equity)")
print(f"gross: ${lv + sv:,.0f} ({(lv + sv) / eq:.1%} of equity)")
