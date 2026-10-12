"""Generate training data by sampling the design space through the C engine.

Usage:
    python python/data_gen.py --n 50000 --out data/train.npz
"""
import argparse
import time
from pathlib import Path

import numpy as np

from rotorlib.ffi import rotor_eval_batch, DESIGN_N, OUT_N, OUT_KEYS
from rotorlib.geometry import sample_lhs


def generate(n_samples, seed=0, chunk_size=5000, verbose=True):
    """Sample, evaluate, and return (X, Y, rc) arrays."""
    X_all = sample_lhs(n_samples, seed=seed)
    Y_all = np.zeros((n_samples, OUT_N), dtype=np.float64)
    rc_all = np.zeros(n_samples, dtype=np.int32)

    t0 = time.perf_counter()
    done = 0
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        X_chunk = X_all[start:stop]
        rc_chunk, Y_chunk = rotor_eval_batch(X_chunk)
        Y_all[start:stop]  = Y_chunk
        rc_all[start:stop] = rc_chunk
        done = stop
        if verbose:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"  {done:6d}/{n_samples}  "
                  f"({elapsed:6.1f}s, {rate:7.1f} samples/s)",
                  flush=True)
    return X_all, Y_all, rc_all


def summarise(X, Y, rc):
    n = len(rc)
    n_ok   = int((rc == 0).sum())
    n_part = int((rc == 1).sum())
    n_fail = int((rc == -1).sum())
    print()
    print(f"  total samples       {n}")
    print(f"  rc = 0 (converged)  {n_ok:6d}   {100.0*n_ok/n:5.1f}%")
    print(f"  rc = 1 (partial)    {n_part:6d}   {100.0*n_part/n:5.1f}%")
    print(f"  rc =-1 (hard fail)  {n_fail:6d}   {100.0*n_fail/n:5.1f}%")
    print()
    print("  Output ranges (converged samples only):")
    mask = (rc == 0)
    if mask.sum() == 0:
        print("    (none)")
        return
    for j, key in enumerate(OUT_KEYS):
        vals = Y[mask, j]
        lo, hi = float(vals.min()), float(vals.max())
        mean   = float(vals.mean())
        print(f"    {key:20s}  min {lo:10.3f}  max {hi:10.3f}  mean {mean:10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",     type=int, default=50000)
    ap.add_argument("--seed",  type=int, default=0)
    ap.add_argument("--out",   type=str, default="data/train.npz")
    ap.add_argument("--chunk", type=int, default=5000)
    args = ap.parse_args()

    print(f"Sampling {args.n} designs (seed={args.seed}, design_dim={DESIGN_N})")
    X, Y, rc = generate(args.n, seed=args.seed, chunk_size=args.chunk)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, Y=Y, rc=rc,
                        design_names=np.array(OUT_KEYS, dtype=object)[:0],
                        out_names=np.array(OUT_KEYS, dtype=object))
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved {args.n} samples to {out_path}  ({size_mb:.1f} MB)")

    summarise(X, Y, rc)


if __name__ == "__main__":
    main()