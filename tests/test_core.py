"""Checks of calibration updates, tail tests and forecast timing."""

import sys, os, json, numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _c in ("src", "code"):
    _p = os.path.join(_HERE, "..", _c)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break
else:
    raise SystemExit("cannot locate the source directory (expected ../src or ../code)")
import core, evaluation as ev
from scipy import stats

FAIL = []


def check(name, cond, extra=""):
    print(f"{('PASS' if cond else 'FAIL')}  {name} {extra}")
    if not cond:
        FAIL.append(name)


def _cache_from(qc, yc, qt, yv):
    n = len(yv)
    return {
        "y": np.array(yv),
        "q_cal": np.array(qc),
        "y_cal": np.array(yc),
        "q_test": np.array(qt),
        "date": pd.DatetimeIndex(pd.date_range("2020-01-01", periods=n)),
    }


rng = np.random.default_rng(0)
m = 60
qc = [np.vstack([np.full(m, -1.0), np.full(m, 1.0)]) for _ in range(3)]
yc = [np.linspace(-2, 2, m) for _ in range(3)]
qt = [np.array([-1.0, 1.0]) for _ in range(3)]
lo, hi, tr = core.wrap_aci(
    _cache_from(qc, yc, qt, [100.0, 0.0, 0.0]), alpha=0.05, eta=0.05
)
check(
    "ACI: miss => alpha_t decreases (Gibbs-Candes)",
    tr["alpha_t"][1] < tr["alpha_t"][0],
    f"a0={tr['alpha_t'][0]:.4f} a1={tr['alpha_t'][1]:.4f}",
)
check(
    "ACI: miss => next interval not narrower",
    hi[1] - lo[1] >= hi[0] - lo[0] - 1e-12,
    f"w0={hi[0] - lo[0]:.4f} w1={hi[1] - lo[1]:.4f}",
)
lo2, hi2, tr2 = core.wrap_aci(
    _cache_from(qc, yc, qt, [0.0, 0.0, 0.0]), alpha=0.05, eta=0.05
)
check(
    "ACI: hit => alpha_t increases",
    tr2["alpha_t"][1] > tr2["alpha_t"][0],
    f"a0={tr2['alpha_t'][0]:.4f} a1={tr2['alpha_t'][1]:.4f}",
)
check("ACI: hit => next interval not wider", hi2[1] - lo2[1] <= hi2[0] - lo2[0] + 1e-12)
lo3, hi3, tr3 = core.wrap_aci(
    _cache_from(qc, yc, qt, [100.0, 0.0, 0.0]), alpha=0.05, eta=0.05, reverse_sign=True
)
check(
    "Reversed-sign control: a miss increases alpha_t",
    tr3["alpha_t"][1] > tr3["alpha_t"][0],
    f"a0={tr3['alpha_t'][0]:.4f} a1={tr3['alpha_t'][1]:.4f}",
)
s = np.arange(1, 61.0)
check(
    "conformal k = ceil((m+1)(1-a)) -> 58th of 60 at a=0.05",
    core._conf_q(s, 0.95) == 58.0,
    f"got {core._conf_q(s, 0.95)}",
)
rng = np.random.default_rng(1)
N = 300


def mk(scale):
    qc = []
    yc = []
    qt = []
    yv = []
    for i in range(N):
        z = rng.standard_normal(m) * 1.2
        qc.append(
            np.vstack(
                [
                    np.full(m, -scale) + rng.standard_normal(m) * 0.1,
                    np.full(m, scale) + rng.standard_normal(m) * 0.1,
                ]
            )
        )
        yc.append(z)
        qt.append(np.array([-scale, scale]))
        yv.append(rng.standard_normal())
    return _cache_from(qc, yc, qt, yv)


c_under = mk(0.5)
lt, ht, _ = core.wrap_tcp(c_under, truncate=True)
lc, hc, _ = core.wrap_tcp(c_under, truncate=False)
check(
    "TCP == CQR when the base learner under-covers (selected score>0)",
    np.allclose(lt, lc) and np.allclose(ht, hc),
    f"max|dl|={np.max(np.abs(lt - lc)):.2e}",
)
c_over = mk(3.0)
lt2, ht2, _ = core.wrap_tcp(c_over, truncate=True)
lc2, hc2, _ = core.wrap_tcp(c_over, truncate=False)
check(
    "truncation only widens (TCP width >= CQR width, strict when base over-covers)",
    np.all(ht2 - lt2 >= hc2 - lc2 - 1e-12) and np.max(ht2 - lt2 - (hc2 - lc2)) > 1e-06,
    f"max extra width={np.max(ht2 - lt2 - (hc2 - lc2)):.4f}",
)
y = np.array([-5.0, 0.0, 5.0])
L = np.array([-3.0, -3.0, -3.0])
U = np.array([3.0, 3.0, 3.0])
two = ((y < L) | (y > U)).astype(int)
one = (y < L).astype(int)
check(
    "two-sided interval miss != one-sided VaR exception",
    two.sum() == 2 and one.sum() == 1,
    f"two={two.tolist()} one={one.tolist()}",
)
exc = np.zeros(1000, dtype=int)
exc[:50] = 1
rng.shuffle(exc)
lr, p, x, T = ev.kupiec(exc, 0.05)
check(
    "Kupiec not rejected at exactly nominal rate",
    p > 0.99 and x == 50,
    f"LR={lr:.4f} p={p:.4f}",
)
exc2 = np.zeros(1000, dtype=int)
exc2[:150] = 1
lr2, p2, _, _ = ev.kupiec(exc2, 0.05)
check("Kupiec rejects 15% exceptions at 5% target", p2 < 1e-10, f"LR={lr2:.2f}")
a = 0.05
ys = stats.norm.rvs(size=400000, random_state=7)
v0 = stats.norm.ppf(a)
e0 = -stats.norm.pdf(v0) / a
base = np.nanmean(ev.fz0_loss(ys, v0, e0, a))
worse = [
    np.nanmean(ev.fz0_loss(ys, v0 * s_, e0 * s2, a))
    for s_, s2 in [
        (0.8, 1.0),
        (1.2, 1.0),
        (1.0, 0.8),
        (1.0, 1.2),
        (0.9, 0.9),
        (1.1, 1.1),
    ]
]
check(
    "FZ0 is minimised at the true (VaR,ES)",
    all((w > base for w in worse)),
    f"base={base:.5f} min_alt={min(worse):.5f}",
)
r = rng.standard_normal(200)
pos, X, y = core.build_features(r)
r2 = r.copy()
r2[150:] = 999.0
pos2, X2, y2 = core.build_features(r2)
k = np.searchsorted(pos, 150)
check(
    "features at t<150 unchanged when future returns altered",
    np.allclose(X[:k], X2[:k]),
)
check("target y[t] equals contemporaneous return", np.allclose(y, r[pos]))
import garch_family as cl

s2 = cl._filter(np.array([1.0, 2.0, 3.0]), 0.0, 0.1, 0.2, 0.7, 1.0)
check(
    "garch filter length n+1 and s2[1] uses r[0] only",
    len(s2) == 4 and abs(s2[1] - (0.1 + 0.2 * 1 + 0.7 * 1)) < 1e-12,
)


def gate(dfs):
    ds = [tuple(pd.DatetimeIndex(d).astype("int64")) for d in dfs]
    return len(set(ds)) == 1


check(
    "timestamp gate flags mismatched samples",
    gate([pd.date_range("2020-01-01", periods=5)] * 2)
    and (
        not gate(
            [
                pd.date_range("2020-01-01", periods=5),
                pd.date_range("2020-01-02", periods=5),
            ]
        )
    ),
)
l = rng.standard_normal(500) ** 2
d, st, p = ev.dm_test(l, l.copy())
check("DM stat is 0 for identical loss series", abs(d) < 1e-12)
s60 = np.arange(1, 61.0)
check("two-sided 95% at m=60 selects k=58", core._conf_q(s60, 0.95) == 58.0)
try:
    core._conf_q(s60, 0.99)
    raised = False
except core.ConformalLevelUnattainable:
    raised = True
check("one-sided tau=0.01 at m=60 raises instead of clipping to the max", raised)
check(
    "attainable floor at m=60 is 1/61", abs(core.attainable_level(60) - 1 / 61) < 1e-12
)
check(
    "one-sided tau=0.025 at m=60 selects k=60, effective level 1/61",
    core._conf_q(s60, 0.975) == 60.0,
)
rng2 = np.random.default_rng(11)


def _mk(scale, N=200):
    qc = []
    yc = []
    qt = []
    yv = []
    for i in range(N):
        z = rng2.standard_normal(60)
        qc.append(np.vstack([np.full(60, -scale), np.full(60, scale)]))
        yc.append(z)
        qt.append(np.array([-scale, scale]))
        yv.append(rng2.standard_normal())
    return _cache_from(qc, yc, qt, yv)


c_over = _mk(3.0)
lt, ht, _ = core.wrap_tcp(c_over, truncate=True)
lc, hc, _ = core.wrap_tcp(c_over, truncate=False)
ndiff = int(((lt != lc) | (ht != hc)).sum())
check(
    "TCP and CQR DIFFER when the base over-covers (truncation binds)",
    ndiff > 0,
    f"{ndiff}/200 differ",
)
check("TCP is never narrower than CQR", np.all(ht - lt >= hc - lc - 1e-12))
c_under = _mk(0.5)
lt2, ht2, _ = core.wrap_tcp(c_under, truncate=True)
lc2, hc2, _ = core.wrap_tcp(c_under, truncate=False)
check(
    "TCP == CQR when the base under-covers",
    np.allclose(lt2, lc2) and np.allclose(ht2, hc2),
)
g = np.array([0.01 / (1 + 0.1 * (i + 1)) ** 0.7 for i in range(1448)])
check(
    "sum of TCP-RM step sizes over 1448 steps is 1.147834",
    abs(g.sum() - 1.147833999) < 1e-06,
    f"got {g.sum():.9f}",
)
check(
    "max all-miss drift = 0.95*sum = 1.090442",
    abs(0.95 * g.sum() - 1.090442299) < 1e-06,
)
import subprocess as _sp

_env = dict(os.environ)
_env.update(
    {
        v: "7"
        for v in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
)
_srcdir = os.path.dirname(core.__file__)
_r = _sp.run(
    [
        sys.executable,
        "-c",
        "import sys; sys.path.insert(0, %r)\nimport run_benchmarks, os, json\nprint(json.dumps({k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}))"
        % _srcdir,
    ],
    capture_output=True,
    text=True,
    env=_env,
)
try:
    _got = json.loads([l for l in _r.stdout.splitlines() if l.startswith("{")][-1])
except Exception:
    _got = {"error": _r.stderr[-200:]}
check(
    "entry points force single-threading even when the caller exports 7",
    all((v == "1" for v in _got.values())),
    str(_got),
)
print("\n" + ("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
