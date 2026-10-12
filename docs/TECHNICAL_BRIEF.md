# Technical Brief

## Summary

The eVTOL Rotor Design Suite is a physics-informed machine learning pipeline for the inverse design of urban air mobility rotors. It couples a hand-written C physics engine (BEM + noise + structure) with a PyTorch surrogate trained on 50,000 samples from that engine. The surrogate enables gradient-based optimization over the 10-dimensional design space in seconds, interactively.

## 1. Physics engine (C)

### 1.1 Blade Element Momentum solver

The hover BEM solver couples blade-element forces with momentum theory at each radial station. The inflow angle φ satisfies:

    φ = atan(sqrt( k(φ) / (1 − k(φ)) ) )
    k(φ) = σ·Cn(φ) / (4F(φ))

where σ is the local solidity, Cn the section normal-force coefficient, and F the Prandtl tip/hub loss factor. The equation is solved by fixed-point iteration with under-relaxation (factor 0.4). Stations are solved hub-to-tip with each station warm-started from the previous converged value, which reduces iterations per station by roughly an order of magnitude near the tip where F falls sharply.

Validated against the classic Caradonna–Tung rotor: the model produces a figure of merit of 0.61 for a 1 m, 2-blade rotor at M_tip = 0.44, in the physically expected range.

### 1.2 Noise model

Two mechanisms are modeled semi-empirically:

- **Tonal (loading dipole)**: scales as T · M_tip⁵ / r, with a sin θ directivity peaked in the rotor plane.
- **Broadband (trailing-edge)**: scales as σ · M_tip⁵ · A / r², with a cardioid |cos θ| directivity peaked along the thrust axis.

The two constants (K_TONAL, K_BROADBAND) are calibrated to NASA SHAC measurements of the Pettingill ideally-twisted rotor (AIAA 2020). The model reproduces both the reference broadband OASPL (64 dB) and the fundamental BPF tone (52 dB) within 1 dB.

A-weighting follows IEC 61672. The two mechanisms give the correct qualitative behavior: tonal dominates in the rotor plane, broadband on the thrust axis, and the total OASPL is a log-sum of the two.

### 1.3 Structural model

A 1D Euler–Bernoulli beam with centrifugal stiffening. At each station:

    T(r)  = ∫_r^R  m(r′) Ω² r′ dr′  (centrifugal tension)
    
    M(r)  = ∫_r^R  (r′ − r) w(r′) dr′,  w = dT/dr  (flap bending)
    
    σ_a   = T / A,  σ_b = M / S,  A = c·t,  S = c·t²/6
    
    margin = σ_yield / (σ_a + σ_b)

First flap natural frequency via Rayleigh quotient with assumed shape φ(ξ) = ξ², including centrifugal stiffening. The ratio ω_flap1 / Ω is used as a resonance-avoidance constraint in the optimizer.

## 2. Surrogate model (Python / PyTorch)

### 2.1 Training data

50,000 designs sampled from a 10-dimensional design space via Latin hypercube, plus a derived tip-Mach-number resampling that keeps every sample inside the BEM's validity region (M_tip ∈ [0.20, 0.70]). 92.7% of samples converge; the 7.3% that don't are excluded from training.

Two outputs with heavy-tailed distributions (max_sigma_MPa, min_margin) are transformed before regression: signed-log for the former, range-clipping for the latter.

### 2.2 Model

A 4-layer MLP (10 → 128 → 128 → 128 → 128 → 12), SiLU activations, 52,492 parameters. Trained with Adam (lr=1e-3, cosine anneal) for 300 epochs.

The loss has two terms:

    L = MSE(ŷ, y) + λ · ((P − Q·Ω) / P_scale)²

The second term enforces P = Q · Ω, which holds exactly in the C data. λ = 1e-3; if the physics term ever fights the data term, the data wins. In practice this is a soft regularizer, not a hard constraint.

### 2.3 Accuracy

Held-out test set (4,638 samples, never seen during training):

| Quantity | R² | rel. RMSE |
|---|---|---|
| thrust_N | 0.9999 | 0.0010 |
| power_W | 0.9999 | 0.0009 |
| FM | 0.9997 | 0.0021 |
| OASPL_below_dBA | 0.9999 | 0.0022 |
| OASPL_side_dBA | 0.9998 | 0.0029 |
| max_sigma_MPa | 0.9989 | 0.0018 |
| min_margin | 0.9839 | 0.0089 |
| M_tip | 0.9999 | 0.0028 |

Mean relative RMSE across all 12 outputs: 0.23%.

## 3. Inverse design

Given a target (thrust, noise ceiling, margin floor), the optimizer runs multi-start Adam over the 9 continuous design parameters (n_blades is enumerated over {2,...,6}, giving 5 separate optimizations per target). 64 random restarts per blade count, 400 gradient steps, all batched in a single tensor. Soft penalty terms enforce the tip-Mach window [0.20, 0.70], the flap-ratio floor (≥ 1.20), and the target thrust.

The winning design is re-evaluated through the C engine, giving a side-by-side verification table. A trust-region iteration tightens the thrust target if the surrogate's prediction drifts more than 2% from the C engine.

## 4. Known limitations

1. **Hover only.** Forward-flight BEM is stubbed out (returns −2).
2. **Semi-empirical noise.** Not a full Ffowcs Williams–Hawkings solver; broadband is a scaling law, not a spectral model.
3. **Solid rectangular cross-sections.** Real blades are hollow or spars. The structural margins are consequently optimistic by a factor of 2–3.
4. **Calibrated against a smaller rotor.** The reference is 0.32 m in diameter; target designs are 1–1.6 m. Absolute noise levels carry additional uncertainty at the target scale.
5. **No blade-vortex interaction (BVI).** BVI is a significant noise source for descending forward flight; not modeled here.

## 5. Files and entry points

| File | Purpose |
|---|---|
| `src/bem.c` | Hover BEM solver |
| `src/noise.c` | Tonal + broadband noise, calibrated |
| `src/structure.c` | Beam tension, bending, flap frequency |
| `src/rotor_api.c` | Flat C ABI: 10 floats in, 12 floats out |
| `python/rotorlib/ffi.py` | ctypes bridge |
| `python/data_gen.py` | Generates training data from C engine |
| `python/pinn.py` | Trains the surrogate |
| `python/optimize.py` | Multi-start inverse design |
| `python/app.py` | Streamlit front end |

## References

Brooks, T. F., Pope, D. S., and Marcolini, M. A., "Airfoil Self-Noise
and Prediction," NASA RP 1218, 1989.

Pettingill, N. A., Zawodniak, N. S., et al., "Acoustic and Performance
Characteristics of an Ideally Twisted Rotor in Hover," AIAA 2020.

Rizzi, S. A., et al., "Urban Air Mobility Noise: Current Practice,
Gaps, and Recommendations," NASA/TP-2020-5007433, 2020.