# RESUME — trial #12 (read research_log.md for full state)

State: branch tiingo-source-and-audit @ 6bee711. N=11. Registered H1 construction
BUILT + reviewed-to-GO (271 tests green, falsification gate byte-identical).
Survivorship unlock done (SEC name-crosswalk 39->94% + Tiingo prices). Trial #12
NOT yet run — blocked only on Tiingo price coverage (daily rate limit).

## On the PC
git clone https://github.com/Jareedd/Qr-Alpha-Lab-Quant.git
cd Qr-Alpha-Lab-Quant   # (repo folder)
git checkout tiingo-source-and-audit && git pull
pip install -r requirements.txt
# copy .env over by hand (keys dont travel via git)
# OPTIONAL: copy data_cache/ from the laptop via USB to keep the ~155 prices +
#   all SEC fundamentals already pulled (saves Tiingo quota; data_cache is gitignored)

## Fill prices (resumable, self-paces through the daily Tiingo limit)
python scripts/tiingo_price_filler.py

## When price coverage on the non-financial universe is high, run trial #12
## (owner already signed off; it prints result + MDE + 4-gate verdict, does NOT auto-log/auto-N)
python scripts/run_h1_trial.py --source free_xwalk --hypothesis H1 --n-trials 12
# then write the trial #12 row in research_log.md (N->12) + commit

## Open PRs to merge: tiingo-source-and-audit, compustat-source-adapter
