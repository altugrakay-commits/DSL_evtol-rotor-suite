"""Inverse design: given performance targets, find a blade design.

Optimizes the 9 continuous design parameters (R, hub_ratio, c_root,
c_tip, twist_root, twist_tip, omega, collective, taper_p) for each
value of n_blades in {2..6}, then returns the best overall.

Uses the trained PINN surrogate for the search, then verifies the
winner with the C physics engine.

Usage:
    python python/optimize.py --thrust 200 --noise 70 --margin 1.5
"""
import argparse

import numpy as np
import torch

from rotorlib.ffi import rotor_eval, DESIGN_N, OUT_KEYS
from rotorlib.geometry import DESIGN_BOUNDS  # noqa: F401 (kept for reference)
from pinn import Surrogate, LOG_OUTPUTS, CLIP_RANGES

# ---------------------------------------------------------------
# Optimizer design bounds.
#
# The sampler in geometry.py derives omega from M_tip and R, so its
# DESIGN_BOUNDS entry for omega is a placeholder [0, 0].  Here we
# optimize omega directly, using the effective range the sampler
# actually produces:  M_tip in [0.20, 0.70]  at  R in [0.30, 0.80]
# =>  omega in [0.20 * 340 / 0.80, 0.70 * 340 / 0.30]  ~ [85, 793].
# We narrow slightly to [90, 780] to stay inside that.
# ---------------------------------------------------------------
OPT_BOUNDS = np.array([
    [0.30,   0.78],   # R            (was 0.80)
    [0.10,   0.25],   # hub_ratio
    [2.00,   6.00],   # n_blades
    [0.020,  0.090],  # c_root       (was 0.100)
    [0.010,  0.055],  # c_tip        (was 0.060)
    [4.0,   18.0],    # twist_root
    [-2.0,   8.0],    # twist_tip
    [90.0,  780.0],   # omega
    [0.0,   10.0],    # collective
    [0.70,   1.60],   # taper_p
])

MTIP_RANGE = (0.20, 0.70)

# Output indices
IDX_THRUST   = 0
IDX_FM       = 5
IDX_OASPL_B  = 6
IDX_OASPL_S  = 7
IDX_SIGMA    = 8
IDX_MARGIN   = 9
IDX_OMEGA_R  = 10
IDX_MTIP     = 11

# Design indices
IDX_D_R         = 0
IDX_D_HUB       = 1
IDX_D_N         = 2
IDX_D_C_ROOT    = 3
IDX_D_C_TIP     = 4
IDX_D_TW_ROOT   = 5
IDX_D_TW_TIP    = 6
IDX_D_OMEGA     = 7
IDX_D_COLL      = 8
IDX_D_TAPER_P   = 9


def load_surrogate(path="models/surrogate.pt"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Surrogate(width=ckpt["config"]["width"],
                      depth=ckpt["config"]["depth"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)   # only the design is optimized
    return (model,
            ckpt["x_mean"], ckpt["x_std"],
            ckpt["y_mean"], ckpt["y_std"])


def physical_outputs(model, x_norm, y_mean, y_std):
    """Forward pass + inverse transforms, in torch, differentiable."""
    y_norm = model(x_norm)
    y_phys = y_norm * y_std + y_mean
    for j in LOG_OUTPUTS:
        y_phys = torch.cat([
            y_phys[:, :j],
            (torch.sign(y_phys[:, j:j+1])
             * torch.expm1(torch.abs(y_phys[:, j:j+1]))),
            y_phys[:, j+1:],
        ], dim=1)
    for j, (lo, hi) in CLIP_RANGES.items():
        y_phys = torch.cat([
            y_phys[:, :j],
            torch.clamp(y_phys[:, j:j+1], lo, hi),
            y_phys[:, j+1:],
        ], dim=1)
    return y_phys


def design_loss(y_phys, target):
    """Soft penalty loss for meeting the performance targets."""
    thrust_err = (y_phys[:, IDX_THRUST] - target["thrust_N"]) / 50.0
    L = thrust_err.pow(2)

    # Noise above ceiling — weighted heavily.
    over_noise = torch.relu(y_phys[:, IDX_OASPL_B] - target["OASPL_max"])
    L = L + 10.0 * over_noise.pow(2)

    # Structural margin below floor.
    under_margin = torch.relu(target["min_margin"] - y_phys[:, IDX_MARGIN])
    L = L + 5.0 * under_margin.pow(2)

    # Tip Mach outside validity window (both sides).
    over_mtip  = torch.relu(y_phys[:, IDX_MTIP] - MTIP_RANGE[1])
    under_mtip = torch.relu(MTIP_RANGE[0] - y_phys[:, IDX_MTIP])
    L = L + 100.0 * over_mtip.pow(2)
    L = L + 100.0 * under_mtip.pow(2)

    # Flap ratio above 1.20 (resonance avoidance).
    under_flap = torch.relu(1.20 - y_phys[:, IDX_OMEGA_R])
    L = L + 5.0 * under_flap.pow(2)

    return L


def optimize_for_blade_count(model, x_mean, x_std, y_mean, y_std,
                             n_blades, target,
                             n_restarts=64, n_steps=400, lr=0.02):
    """Multi-start Adam on the 9 continuous design parameters."""
    x_mean_t = torch.from_numpy(x_mean).float()
    x_std_t  = torch.from_numpy(x_std).float()
    y_mean_t = torch.from_numpy(y_mean).float()
    y_std_t  = torch.from_numpy(y_std).float()

    lo = OPT_BOUNDS[:, 0]
    hi = OPT_BOUNDS[:, 1]
    lo_t = torch.from_numpy(lo).float()
    hi_t = torch.from_numpy(hi).float()

    # Random starting points inside OPT_BOUNDS.
    starts = lo + np.random.rand(n_restarts, DESIGN_N) * (hi - lo)
    starts[:, IDX_D_N] = float(n_blades)

    x_var = torch.from_numpy(starts).float().requires_grad_(True)
    opt = torch.optim.Adam([x_var], lr=lr)

    for _ in range(n_steps):
        opt.zero_grad()
        x_clamped = torch.clamp(x_var, lo_t, hi_t)
        x_norm = (x_clamped - x_mean_t) / x_std_t
        y_phys = physical_outputs(model, x_norm, y_mean_t, y_std_t)
        L = design_loss(y_phys, target)
        L.sum().backward()
        opt.step()

    with torch.no_grad():
        x_clamped = torch.clamp(x_var, lo_t, hi_t)
        x_norm = (x_clamped - x_mean_t) / x_std_t
        y_phys = physical_outputs(model, x_norm, y_mean_t, y_std_t)
        L = design_loss(y_phys, target)

    best_idx = int(L.argmin())
    best_design = x_clamped[best_idx].detach().numpy().copy()
    best_design[IDX_D_N] = round(best_design[IDX_D_N])
    best_y = y_phys[best_idx].detach().numpy()
    best_loss = float(L[best_idx])
    return best_design, best_y, best_loss


def optimize(target, n_restarts=64, n_steps=400, n_iterations=3, tolerance=0.02, verbose=True):
    """Full multi-blade-count optimization.

    Runs the surrogate optimization, verifies with the C engine, and if
    the thrust mismatch exceeds `tolerance`, adjusts the target and
    re-solves.  Two or three iterations typically reduce the error to
    under 2%.
    """
    model, xm, xs, ym, ys = load_surrogate()

    target_work = dict(target)
    best_design = None
    best_pred   = None
    best_cout   = None

    for iteration in range(n_iterations):
        if verbose:
            print(f"\n=== iteration {iteration + 1} ===")
            print(f"  targets: thrust = {target_work['thrust_N']:.1f} N,  "
                  f"noise <= {target_work['OASPL_max']:.0f} dBA,  "
                  f"margin >= {target_work['min_margin']:.2f}")

        results = []
        for n in [2, 3, 4, 5, 6]:
            design, pred, loss = optimize_for_blade_count(
                model, xm, xs, ym, ys, n, target_work,
                n_restarts=n_restarts, n_steps=n_steps)
            results.append((loss, design, pred))
            if verbose:
                print(f"  n_blades = {n}  best loss = {loss:.4f}")

        results.sort(key=lambda r: r[0])
        best_loss, best_design, best_pred = results[0]

        if verbose:
            print(f"\n  best design: n_blades = {int(best_design[IDX_D_N])}, "
                  f"R = {best_design[IDX_D_R]:.3f} m, "
                  f"omega = {best_design[IDX_D_OMEGA]:.1f} rad/s")

        # ---- Verify with the C engine ----
        rc, c_out = rotor_eval(best_design.tolist())
        if rc == -1 or c_out is None:
            print("  C engine rejected the design.  Stopping.")
            return best_design, best_pred, None

        best_cout = c_out

        thrust_pinn = best_pred[IDX_THRUST]
        thrust_c    = c_out["thrust_N"]
        rel_err     = (thrust_pinn - thrust_c) / thrust_c

        if verbose:
            print(f"  C engine thrust = {thrust_c:.2f} N  "
                  f"(PINN said {thrust_pinn:.2f} N, "
                  f"rel err {rel_err*100:+.2f}%)")

        if abs(rel_err) < tolerance:
            if verbose:
                print(f"  converged within {tolerance*100:.0f}%.  Done.")
            break

        # Adjust thrust target by the inverse of the mismatch.  Cap
        # the correction so a bad iteration can't run away.
        correction = thrust_pinn / thrust_c
        correction = max(0.80, min(1.25, correction))
        target_work["thrust_N"] *= correction

        if verbose:
            print(f"  adjusting thrust target to "
                  f"{target_work['thrust_N']:.1f} N")

    # ---- Final report ----
    if verbose:
        print(f"\n  --- final design ---")
        print(f"  n_blades = {int(best_design[IDX_D_N])}, " # type: ignore
              f"R = {best_design[IDX_D_R]:.3f} m, " # type: ignore
              f"omega = {best_design[IDX_D_OMEGA]:.1f} rad/s") # type: ignore
        print(f"  c_root = {best_design[IDX_D_C_ROOT]:.4f} m,  " # type: ignore
              f"c_tip = {best_design[IDX_D_C_TIP]:.4f} m,  " # type: ignore
              f"taper_p = {best_design[IDX_D_TAPER_P]:.3f}") # type: ignore
        print(f"  twist_root = {best_design[IDX_D_TW_ROOT]:.2f} deg,  " # type: ignore
              f"twist_tip = {best_design[IDX_D_TW_TIP]:.2f} deg,  " # type: ignore
              f"collective = {best_design[IDX_D_COLL]:.2f} deg") # type: ignore
        print()
        print(f"  {'quantity':20s}  {'PINN':>12s}  {'C engine':>12s}  {'rel err':>10s}")
        for j, key in enumerate(OUT_KEYS):
            p = best_pred[j] # type: ignore
            c = best_cout[key] # type: ignore
            rel = abs(p - c) / abs(c) if abs(c) > 1e-9 else abs(p - c)
            flag = "  <--  check" if rel > 0.05 else ""
            print(f"  {key:20s}  {p:12.4f}  {c:12.4f}  {rel:10.4f}{flag}")

    # Save the winning design for CAD and Streamlit to consume.
    import json
    from pathlib import Path
    Path("data").mkdir(exist_ok=True)
    with open("data/last_design.json", "w") as f:
        json.dump({
            "design":     best_design.tolist(),
            "out_names":  list(OUT_KEYS),
            "predicted":  best_pred.tolist(),
            "verified":   {k: float(v) for k, v in best_cout.items()},
            "target":     target,
        }, f, indent=2)
    if verbose:
        print("\n  saved data/last_design.json")
        
    return best_design, best_pred, best_cout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thrust", type=float, default=200.0,
                    help="target thrust in N")
    ap.add_argument("--noise",  type=float, default=70.0,
                    help="max noise at 100 m below hub, dBA")
    ap.add_argument("--margin", type=float, default=1.5,
                    help="min structural margin")
    ap.add_argument("--restarts", type=int, default=64)
    ap.add_argument("--steps",    type=int, default=400)
    args = ap.parse_args()

    target = {
        "thrust_N":    args.thrust,
        "OASPL_max":   args.noise,
        "min_margin":  args.margin,
    }
    np.random.seed(0)
    optimize(target, n_restarts=args.restarts, n_steps=args.steps)


if __name__ == "__main__":
    main()