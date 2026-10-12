"""Design-space definition and sampling for eVTOL rotor blades.

This module is the single source of truth for the design-vector layout.
It must match the #define block in src/rotor_api.h exactly.  When you
add a design parameter, add it at the END here and at the END there,
bump DESIGN_N in both places, and update ffi.py.

Bounds are chosen so that every derived quantity stays in the range
where BEM is valid: tip Mach number below 0.70, realistic disk loading,
geometrically sensible chord taper.
"""
import numpy as np

DESIGN_N = 10

DESIGN_NAMES = [
    "R",            # 0  tip radius, m
    "hub_ratio",    # 1  hub_radius / R
    "n_blades",     # 2  number of blades, integer 2..6
    "c_root",       # 3  root chord, m
    "c_tip",        # 4  tip chord, m
    "twist_root",   # 5  root twist, deg
    "twist_tip",    # 6  tip twist, deg
    "omega",        # 7  rotation rate, rad/s  (DERIVED from M_tip and R)
    "collective",   # 8  collective pitch, deg
    "taper_p",      # 9  chord/twist taper exponent
]

# Sample bounds.  The omega column is NOT sampled directly; it is
# computed from the tip Mach number target and the radius after LHS.
# See sample_lhs().
DESIGN_BOUNDS = np.array([
    [0.30,   0.80],   # R            m
    [0.10,   0.25],   # hub_ratio
    [2.00,   6.00],   # n_blades     (rounded to int)
    [0.020,  0.100],  # c_root       m
    [0.010,  0.060],  # c_tip        m
    [4.0,   18.0],    # twist_root   deg
    [-2.0,   8.0],    # twist_tip    deg
    [0.00,   0.00],   # omega        placeholder, overwritten below
    [0.0,   10.0],    # collective   deg
    [0.70,   1.60],   # taper_p
])

# Tip Mach number target — sampled independently, then converted
# to omega.  Above 0.70, compressibility invalidates the BEM polar;
# below 0.20, the rotor is a windmill and figure of merit collapses.
MTIP_RANGE = (0.20, 0.70)

DISCRETE_DIMS = {2}   # n_blades


def sample_lhs(n_samples, seed=0):
    """Latin-hypercube sample of the design space.

    Returns (n_samples, DESIGN_N) float64 array.
    n_blades is discrete (integer).  omega is derived from an
    independently sampled tip Mach number and the sampled radius.
    """
    rng = np.random.default_rng(seed)
    D = DESIGN_N
    lo = DESIGN_BOUNDS[:, 0].copy()
    hi = DESIGN_BOUNDS[:, 1].copy()
    span = hi - lo

    X = np.zeros((n_samples, D), dtype=np.float64)

    # Continuous dimensions via LHS (skip n_blades and omega)
    for j in range(D):
        if j in DISCRETE_DIMS or j == 7:
            continue
        strata = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        rng.shuffle(strata)
        X[:, j] = lo[j] + strata * span[j]

    # Discrete: n_blades uniform over the integer set
    for j in DISCRETE_DIMS:
        lo_i = int(round(lo[j]))
        hi_i = int(round(hi[j]))
        X[:, j] = rng.integers(lo_i, hi_i + 1, size=n_samples).astype(np.float64)

    # Derived: omega = M_tip * a / R, with a = 340 m/s
    mtip = MTIP_RANGE[0] + rng.random(n_samples) * (MTIP_RANGE[1] - MTIP_RANGE[0])
    X[:, 7] = mtip * 340.0 / X[:, 0]

    return X


def design_to_stations(design, n_stations=20):
    """Reproduce the C build_geometry() math in Python."""
    R       = design[0]
    hub_r   = design[1]
    c_root  = design[3]
    c_tip   = design[4]
    tr_root = design[5]
    tr_tip  = design[6]
    p       = design[9]

    rh = R * hub_r
    xi = (np.arange(n_stations) + 0.5) / n_stations
    r       = rh + xi * (R - rh)
    chord   = c_root + (c_tip - c_root) * np.power(xi, p)
    twist_d = tr_root + (tr_tip - tr_root) * np.power(xi, p)
    return {"r": r, "chord": chord, "twist_deg": twist_d}


def describe():
    """Print the design space — handy as a reference."""
    print(f"  omega is derived from M_tip in {MTIP_RANGE} and R:")
    print(f"    omega_min = {MTIP_RANGE[0]*340.0/0.80:.1f} rad/s "
          f"(at R=0.80, M_tip={MTIP_RANGE[0]})")
    print(f"    omega_max = {MTIP_RANGE[1]*340.0/0.30:.1f} rad/s "
          f"(at R=0.30, M_tip={MTIP_RANGE[1]})")
    print()
    for i, name in enumerate(DESIGN_NAMES):
        if i == 7:
            print(f"  [{i}] {name:14s}  derived from M_tip and R")
            continue
        lo, hi = DESIGN_BOUNDS[i]
        kind = "discrete" if i in DISCRETE_DIMS else "continuous"
        print(f"  [{i}] {name:14s}  {lo:8.4f} .. {hi:8.4f}   ({kind})")


if __name__ == "__main__":
    print(f"DESIGN_N = {DESIGN_N}")
    describe()