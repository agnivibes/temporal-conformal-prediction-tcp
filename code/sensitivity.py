"""Fit the window, calibration-size and step-size sensitivity grid."""

import os as _os, sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import threadpin as _tp

_tp.pin()
import sys, os, argparse, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import core, evaluation as ev
import warnings

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="SP500,BTC-USD,Gold")
    ap.add_argument("--outdir", default="build/forecasts")
    ap.add_argument(
        "--data",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "financial_returns.csv"
        ),
    )
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = pd.read_csv(a.data, index_col=0, parse_dates=True)
    df.rename(columns={"^GSPC": "SP500", "GC=F": "Gold"}, inplace=True)
    rows = []
    for name in a.assets.split(","):
        ser = df[name].dropna()
        for w in [100, 252, 500]:
            for cal in [40, 60]:
                if cal >= w:
                    continue
                t0 = time.time()
                c = core.base_qr_cache(
                    ser.values, ser.index, taus=(0.025, 0.975), window=w, cal=cal
                )
                for g in [None, 0.005, 0.01, 0.05]:
                    lo, hi, _ = core.wrap_tcp(
                        c, 0.05, rm=g is not None, gamma_0=g or 0.01
                    )
                    m, _, _ = ev.interval_metrics(c["y"], lo, hi, 0.05)
                    rows.append(
                        {
                            "series": name,
                            "method": "TCP" if g is None else "TCP-RM",
                            "w": w,
                            "cal": cal,
                            "gamma_0": g,
                            **m,
                            "headline_config": int(w == 252 and cal == 60),
                        }
                    )
                print(f"[{name}] w={w} cal={cal} {time.time() - t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(f"{a.outdir}/sensitivity_2d.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
