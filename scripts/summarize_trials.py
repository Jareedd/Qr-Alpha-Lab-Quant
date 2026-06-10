"""Print a one-line summary per saved metrics JSON (log-entry helper)."""

import json
import sys

for tag in sys.argv[1:]:
    m = json.load(open(f"results/metrics_{tag}.json"))
    print(
        f"{tag}: IC {m['mean_rank_ic']:+.4f} (t_NW {m['ic_tstat_newey_west']:+.2f}) | "
        f"gross SR {m['sharpe_gross']:+.2f} | net SR {m['sharpe_net']:+.2f} | "
        f"DSR {m['dsr']:.3f} | turnover {m['annual_turnover']:.2f}x | "
        f"mom baseline {m['baseline_mom_sharpe_net']:+.2f} | "
        f"beta p95 {m['risk_realized_beta_p95_abs']:.2f}"
    )
