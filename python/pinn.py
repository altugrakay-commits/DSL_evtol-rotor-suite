"""Physics-informed surrogate model for the rotor C pipeline.

Reads data/train.npz (X, Y, rc), trains an MLP to predict the 12 output
quantities from the 10 design parameters, and reports per-output error
on a held-out test set.

The "physics-informed" part: a soft penalty was added to enforce P = Q*omega,
which holds exactly in the C data.  The penalty weight is small (lambda=1e-3);
if it ever fights the data term, the data wins.

Usage:
    python python/pinn.py --data data/train.npz --epochs 300
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rotorlib.ffi import OUT_KEYS, DESIGN_N, OUT_N

# Indices in Y (physical units)
IDX_TORQUE = 1
IDX_POWER  = 2
IDX_OMEGA  = 7   # in the DESIGN vector, not Y

def signed_log1p(x):
    """Symmetric log transform.  Handles negative values smoothly."""
    return np.sign(x) * np.log1p(np.abs(x))


def inv_signed_log1p(y):
    """Exact inverse of signed_log1p."""
    return np.sign(y) * np.expm1(np.abs(y))


# Output columns whose dynamic range is too wide for direct regression.
# 8 = max_sigma_MPa   (2 .. 22,000)
# 9 = min_margin      (-18,500 .. +283)
LOG_OUTPUTS = {8}

# Output columns that need clipping to a physically meaningful range
# before regression.  Values outside these windows are off-design and
# the optimizer will never select them; clipping prevents a handful of
# outliers from dominating the loss.
# 9 = min_margin     (-18,500 .. +283)
CLIP_RANGES = {9: (-50.0, 200.0),}

# ----------------------------------------------------------------
# Data loading and normalisation
# ----------------------------------------------------------------
class Standardiser:
    def __init__(self, X):
        self.mean = X.mean(axis=0)
        self.std  = X.std(axis=0)
        self.std[self.std < 1e-12] = 1.0
    def transform(self, X):
        return (X - self.mean) / self.std
    def inverse(self, Xn):
        return Xn * self.std + self.mean


def load_and_split(path, seed=0):
    d = np.load(path, allow_pickle=True)
    X, Y, rc = d["X"], d["Y"], d["rc"]

    mask = (rc == 0)
    X, Y = X[mask], Y[mask]
    print(f"  loaded {mask.sum()} converged samples of {len(rc)} total")

    # Shuffle
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, Y = X[idx], Y[idx]

    # Transform wide-range outputs
    for j in LOG_OUTPUTS:
        Y[:, j] = signed_log1p(Y[:, j])

    # Clip bounded-rane outputs
    for j, (lo, hi) in CLIP_RANGES.items():
        Y[:, j] = np.clip(Y[:, j], lo, hi)

    n = len(X)
    n_train = int(0.80 * n)
    n_val   = int(0.10 * n)
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val,   Y_val   = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test,  Y_test  = X[n_train+n_val:], Y[n_train+n_val:]
    print(f"  train {len(X_train)}  val {len(X_val)}  test {len(X_test)}")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# ----------------------------------------------------------------
# Model
# ----------------------------------------------------------------
class Surrogate(nn.Module):
    def __init__(self, n_in=DESIGN_N, n_out=OUT_N, width=128, depth=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(n_in, width))
        layers.append(nn.SiLU())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(width, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------
# Loss: data MSE + physics consistency
# ----------------------------------------------------------------
def loss_fn(y_pred_norm, y_true_norm, x_norm,
            x_scaler, y_scaler, lambda_phys=1e-3):
    """MSE in normalised space + physics term in physical units."""
    L_data = torch.mean((y_pred_norm - y_true_norm) ** 2)

    # Physics: P = Q * omega.  Both sides in physical units, divided by
    # the power scale so the term is O(1).
    yp = y_scaler.inverse(y_pred_norm)          # (N, OUT_N) physical
    xp = x_scaler.inverse(x_norm)               # (N, DESIGN_N) physical

    Q = yp[:, IDX_TORQUE]
    P = yp[:, IDX_POWER]
    w = xp[:, IDX_OMEGA]

    P_scale = float(y_scaler.std[IDX_POWER])
    L_phys  = torch.mean(((P - Q * w) / P_scale) ** 2)

    return L_data + lambda_phys * L_phys, L_data.item(), L_phys.item()


# ----------------------------------------------------------------
# Training
# ----------------------------------------------------------------
def train(args):
    print("Loading data...")
    X_tr, Y_tr, X_va, Y_va, X_te, Y_te = load_and_split(args.data)

    x_scaler = Standardiser(X_tr)
    y_scaler = Standardiser(Y_tr)

    # Convert to torch tensors
    def to_t(X, scaler):
        return torch.from_numpy(scaler.transform(X)).float()

    Xtr = to_t(X_tr, x_scaler); Ytr = to_t(Y_tr, y_scaler)
    Xva = to_t(X_va, x_scaler); Yva = to_t(Y_va, y_scaler)
    Xte = to_t(X_te, x_scaler); Yte = to_t(Y_te, y_scaler)

    # Scaler wrappers usable inside loss_fn
    class TorchScaler:
        def __init__(self, mean, std):
            self.mean_t = torch.from_numpy(mean).float()
            self.std_t  = torch.from_numpy(std).float()
            self.std    = std
        def inverse(self, Xn):
            return Xn * self.std_t + self.mean_t
    x_scaler_t = TorchScaler(x_scaler.mean, x_scaler.std)
    y_scaler_t = TorchScaler(y_scaler.mean, y_scaler.std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    model = Surrogate(width=args.width, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model parameters: {n_params}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Xtr, Ytr = Xtr.to(device), Ytr.to(device)
    Xva, Yva = Xva.to(device), Yva.to(device)
    Xte, Yte = Xte.to(device), Yte.to(device)

    batch = args.batch
    n_train = len(Xtr)

    best_val = float("inf")
    best_state = None

    print("\n  epoch   train_data   train_phys   val_data    time")
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        tr_data_sum, tr_phys_sum, n_batches = 0.0, 0.0, 0

        for i in range(0, n_train, batch):
            idx = perm[i:i+batch]
            xb, yb = Xtr[idx], Ytr[idx]

            opt.zero_grad()
            pred = model(xb)
            loss, ld, lp = loss_fn(pred, yb, xb,
                                   x_scaler_t, y_scaler_t,
                                   lambda_phys=args.lambda_phys)
            loss.backward()
            opt.step()
            tr_data_sum += ld
            tr_phys_sum += lp
            n_batches += 1

        sched.step()

        model.eval()
        with torch.no_grad():
            pred_va = model(Xva)
            _, va_ld, _ = loss_fn(pred_va, Yva, Xva,
                                  x_scaler_t, y_scaler_t,
                                  lambda_phys=args.lambda_phys)

        tr_data = tr_data_sum / n_batches
        tr_phys = tr_phys_sum / n_batches

        if va_ld < best_val:
            best_val = va_ld
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % args.log_every == 0 or epoch == 1:
            dt = time.perf_counter() - t0
            print(f"  {epoch:5d}   {tr_data:10.6f}   {tr_phys:10.6f}   "
                  f"{va_ld:10.6f}   {dt:6.1f}s")

    print(f"\n  best val_data MSE = {best_val:.6f}")
    model.load_state_dict(best_state)

    # ---- Test set evaluation ----
    model.eval()
    with torch.no_grad():
        pred_te_norm = model(Xte)
        pred_te = y_scaler.inverse(pred_te_norm.cpu().numpy())

    # Invert log-transformed outputs back to physical units
    Y_te_phys = Y_te.copy()
    for j in LOG_OUTPUTS:
        pred_te[:, j]   = inv_signed_log1p(pred_te[:, j])
        Y_te_phys[:, j] = inv_signed_log1p(Y_te_phys[:, j])

    for j, (lo, hi) in CLIP_RANGES.items():
        pred_te[:, j] = np.clip(pred_te[:, j], lo, hi)

    print(f"\n  --- test set per-output accuracy ---")
    print(f"  {'quantity':20s}  {'RMSE':>12s}  {'MAE':>12s}  "
          f"{'rel RMSE':>10s}  {'R^2':>8s}")

    overall_rmse = []
    for j, key in enumerate(OUT_KEYS):
        yt = Y_te_phys[:, j]
        yp = pred_te[:, j]
        err = yp - yt
        rmse = float(np.sqrt(np.mean(err**2)))
        mae  = float(np.mean(np.abs(err)))
        rng  = float(yt.max() - yt.min())
        rrmse = rmse / rng if rng > 0 else 0.0
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((yt - yt.mean())**2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        overall_rmse.append(rrmse)
        print(f"  {key:20s}  {rmse:12.4f}  {mae:12.4f}  "
              f"{rrmse:10.4f}  {r2:8.4f}")

    print(f"\n  mean relative RMSE across outputs: "
          f"{float(np.mean(overall_rmse)):.4f}")

    # Save model + scalers
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "x_mean": x_scaler.mean, "x_std": x_scaler.std,
        "y_mean": y_scaler.mean, "y_std": y_scaler.std,
        "config": {"width": args.width, "depth": args.depth},
    }, out_dir / "surrogate.pt")
    print(f"\n  saved model to {out_dir / 'surrogate.pt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",         default="data/train.npz")
    ap.add_argument("--out",          default="models")
    ap.add_argument("--epochs",       type=int,   default=300)
    ap.add_argument("--batch",        type=int,   default=512)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--width",        type=int,   default=128)
    ap.add_argument("--depth",        type=int,   default=4)
    ap.add_argument("--lambda-phys",  type=float, default=1e-3, dest="lambda_phys")
    ap.add_argument("--log-every",    type=int,   default=25, dest="log_every")
    args = ap.parse_args()
    train(args)

def predict(model_path, design):
    """Load model, run one design, return physical-unit outputs."""
    ckpt = torch.load(model_path, map_location="cpu")
    model = Surrogate(width=ckpt["config"]["width"], depth=ckpt["config"]["depth"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x_mean = ckpt["x_mean"]; x_std = ckpt["x_std"]
    y_mean = ckpt["y_mean"]; y_std = ckpt["y_std"]

    x_norm = (np.asarray(design, dtype=np.float64) - x_mean) / x_std
    with torch.no_grad():
        y_norm = model(torch.from_numpy(x_norm).float().unsqueeze(0)).numpy()[0]

    y_phys = y_norm * y_std + y_mean
    for j in LOG_OUTPUTS:
        y_phys[j] = inv_signed_log1p(y_phys[j])
    for j, (lo, hi) in CLIP_RANGES.items():
        y_phys[j] = float(np.clip(y_phys[j], lo, hi))

    return dict(zip(OUT_KEYS, y_phys.tolist()))

if __name__ == "__main__":
    main()