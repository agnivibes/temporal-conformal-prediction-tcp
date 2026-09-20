"""Set numerical libraries to one thread before importing them."""

import os

VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
overridden = {}


def pin(value="1", strict=False):
    global overridden
    overridden = {}
    for v in VARS:
        prev = os.environ.get(v)
        if prev is not None and prev != value:
            if strict:
                raise RuntimeError(
                    f"{v}={prev} conflicts with the single-thread requirement. Results are not bit-reproducible across thread counts; unset it or set it to 1."
                )
            overridden[v] = prev
        os.environ[v] = value
    if overridden:
        print(
            f"[threadpin] overrode {overridden} -> {value} (results are not bit-reproducible across thread counts)",
            flush=True,
        )
    return overridden


def verify(value="1"):
    bad = {v: os.environ.get(v) for v in VARS if os.environ.get(v) != value}
    return (not bad, bad)
