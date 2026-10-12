"""Parity plot: predicted vs actual on the held-out test set.

Produces figures/parity.png — a 3x4 grid with one panel per output.
Each panel shows scatter of (actual, predicted) and a y=x reference line.
"""
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rotorlib.ffi import OUT_KEYS
from pinn import (load_and_split, Surrogate,
                  LOG_OUTPUTS, CLIP_RANGES, inv_signed_log1p)


def main():
    # weights_only=False is required because the checkpoint contains
    # NumPy arrays (for the scalers) alongside the model weights.
    # PyTorch 2.6+ defaults to weights_only=True for security, but
    # this file is one we created ourselves.
    ckpt = torch.load("models/surrogate.pt",
                      map_location="cpu",
                      weights_only=False)
    x_mean = ckpt["x_mean"]; x_std = ckpt["x_std"]
    y_mean = ckpt["y_mean"]; y_std = ckpt["y_std"]
    model = Surrogate(width=ckpt["config"]["width"], depth=ckpt["config"]["depth"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Same seed => same test split as training used.
    _, _, _, _, X_te, Y_te = load_and_split("data/train.npz")

    X_te_n = (X_te - x_mean) / x_std
    with torch.no_grad():
        pred_n = model(torch.from_numpy(X_te_n).float()).numpy()
    pred = pred_n * y_std + y_mean

    # Invert transforms — same order as in pinn.train()
    Y_te_phys = Y_te.copy()
    for j in LOG_OUTPUTS:
        pred[:, j]      = inv_signed_log1p(pred[:, j])
        Y_te_phys[:, j] = inv_signed_log1p(Y_te_phys[:, j])
    for j, (lo, hi) in CLIP_RANGES.items():
        pred[:, j] = np.clip(pred[:, j], lo, hi)

    # --- plot ---
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes = axes.ravel()

    for j, key in enumerate(OUT_KEYS):
        ax = axes[j]
        yt = Y_te_phys[:, j]
        yp = pred[:, j]

        ax.scatter(yt, yp, s=2, alpha=0.25, edgecolors="none", color="steelblue")

        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, alpha=0.7)

        ax.set_xlabel("C engine (actual)", fontsize=8)
        ax.set_ylabel("PINN (predicted)", fontsize=8)
        ax.set_title(key, fontsize=10)
        ax.tick_params(labelsize=7)
        ax.ticklabel_format(style="sci", scilimits=(-2, 3), axis="both")

        err  = yp - yt
        r2   = 1.0 - np.sum(err**2) / np.sum((yt - yt.mean())**2)
        rmse = float(np.sqrt(np.mean(err**2)))
        ax.text(0.04, 0.96,
                f"R² = {r2:.4f}\nRMSE = {rmse:.3g}",
                transform=ax.transAxes,
                va="top", ha="left", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="gray", alpha=0.8))

    fig.suptitle("PINN surrogate vs C physics engine — held-out test set",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir = Path("figures"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "parity.png"
    fig.savefig(out_path, dpi=140)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()