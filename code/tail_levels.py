"""Evaluate one-sided conformal forecasts at their attainable reference levels."""

import sys, os, math, glob, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluation as ev, core

CAL = 60


def effective_level(tau, m=CAL):
    k = int(math.ceil((m + 1) * (1 - tau)))
    if k > m:
        return (None, k)
    return (1.0 - k / (m + 1.0), k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="build/forecasts")
    a = ap.parse_args()
    rows_lvl = []
    for tau in (0.01, 0.025, 0.05):
        eff, k = effective_level(tau)
        rows_lvl.append(
            {
                "requested_tau": tau,
                "order_statistic_k": k,
                "cal_size": CAL,
                "attainable": eff is not None,
                "effective_guaranteed_miscoverage": eff,
                "attainable_floor_1_over_m_plus_1": core.attainable_level(CAL),
            }
        )
    pd.DataFrame(rows_lvl).to_csv(
        f"{a.outdir}/conformal_effective_levels.csv", index=False
    )
    print(pd.DataFrame(rows_lvl).to_string(index=False))
    for f in sorted(glob.glob(f"{a.outdir}/var_*.csv")):
        name = os.path.basename(f)[4:-4]
        vr = pd.read_csv(f)
        z = np.load(f"{a.outdir}/raw_{name}.npz")
        y = z["y"]
        before = len(vr)
        vr = vr[~((vr.model == "CQR-1s") & (vr.tau == 0.01))]
        vr = vr[vr.model != "CQR-1s-at-effective-level"]
        extra = []
        for tau in (0.025, 0.05):
            key = f"var__CQR-1s__{tau}"
            if key not in z.files:
                continue
            eff, k = effective_level(tau)
            bt, _ = ev.var_backtests(y, z[key], eff)
            bt.update(
                {
                    "series": name,
                    "model": "CQR-1s",
                    "tau": tau,
                    "evaluated_at_effective_level": eff,
                    "order_statistic_k": k,
                }
            )
            extra.append(bt)
        vr.loc[vr.model == "CQR-1s", "note"] = (
            "requested level; exchangeable split-conformal bound is listed in conformal_effective_levels.csv"
        )
        vr = pd.concat(
            [vr, pd.DataFrame(extra).assign(model="CQR-1s-at-effective-level")],
            ignore_index=True,
        )
        vr.to_csv(f, index=False)
        print(
            f"[{name}] rows {before} -> {len(vr)} (includes {len(extra)} attainable-level backtests)"
        )
    allv = pd.concat(
        [pd.read_csv(f) for f in sorted(glob.glob(f"{a.outdir}/var_*.csv"))]
    )
    allv.to_csv(f"{a.outdir}/ALL_var.csv", index=False)
    print("ALL_var.csv rebuilt")
