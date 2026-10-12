"""Build a 3D blade solid from a design vector using CadQuery.

Reads data/last_design.json (written by optimize.py) or accepts a
design on the command line. Exports STL and STEP files.

Coordinate system:
    +X  = radial direction (hub center to blade tip)
    +Y  = chordwise (leading edge at -Y, trailing edge at +Y)
    +Z  = thrust direction (blade rotates about Z)

Usage:
    python python/cad.py                            # uses last_design.json
    python python/cad.py --stations 12              # finer loft
    python python/cad.py --design-json path.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from cadquery import Vector, Edge, Wire, Solid, exporters

# ----------------------------------------------------------------
# Airfoil section
# ----------------------------------------------------------------

def naca0012_loop(n=40, t=0.12, closed_te=True):
    """Return (x, y) points for a closed NACA 0012 loop.

    x runs 0 (LE) to 1 (TE), y is the thickness distribution.
    Returns the loop in order: TE-upper, ... , LE, ... , TE-lower,
    with the closing edge implied (last point -> first point).
    """
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (0.2969 * np.sqrt(x) 
                    - 0.1260 * x 
                    - 0.3516 * x * x 
                    + 0.2843 * x * x * x 
                    - 0.1015 * x * x * x * x)

    if closed_te:
        yt[-1] = 0.0

    # Upper surface: TE -> LE (reverse order)
    x_up = x[::-1].copy()
    y_up = yt[::-1].copy()
    # Lower surface: skip LE (dup) and TE (avoids zero-length edge)
    x_lo = x[1:-1].copy()
    y_lo = -yt[1:-1].copy()

    return (np.concatenate([x_up, x_lo]), np.concatenate([y_up, y_lo]))

def _make_wire(section_2d, r, twist_rad):
    """Turn a 2D (x, y) section into a 3D closed wire at radius r.

    Applies twist about the radial (+X) axis so that positive twist
    raises the leading edge (nose-up), matching the C convention.
    """
    ct, st = np.cos(twist_rad), np.sin(twist_rad)
    pts = []
    for (x_local, y_local) in section_2d:
        # Twist rotation in the Y-Z plane, about the radial axis.
        # Positive twist => leading edge moves to +Z (nose up).
        Yp = x_local * ct + y_local * st
        Zp = -x_local * st + y_local * ct
        pts.append(Vector(r, Yp, Zp))

    n = len(pts)
    edges = [Edge.makeLine(pts[i], pts[(i+1) % n]) for i in range(n)]
    return Wire.assembleEdges(edges)

# ----------------------------------------------------------------
# Blade construction
# ----------------------------------------------------------------

def build_blade_solid(design, n_stations=10, n_airfoil=40):
    """Loft a solid blade from a design vector.

    design: 10-element list, same layout as DESIGN_BOUNDS in geometry.py.
    n_stations: number of radial sections to loft through.
    n_airfoil: number of points in each airfoil section.
    """
    R        = design[0]
    hub_r    = design[1]
    c_root   = design[3]
    c_tip    = design[4]
    tr_root  = design[5]
    tr_tip   = design[6]
    p        = design[9]

    rh = R * hub_r

    # Cosine-spaced stations, offset slightly from the hub cutout so
    # the first section is inside the valid domain.
    s = np.linspace(0.03, 1.0, n_stations)
    xi = 0.5 * (1.0 - np.cos(s * np.pi))
    r_stations = rh + xi * (R - rh)

    chord_r = c_root + (c_tip - c_root) * np.power(xi, p)
    twist_r = np.radians(tr_root + (tr_tip - tr_root) * np.power(xi, p))

    xa, ya = naca0012_loop(n=n_airfoil, t=0.12, closed_te=True)

    wires = []
    for r, c, tw in zip(r_stations, chord_r, twist_r):
        # Scale and center the airfoil: chord along Y, thickness along Z.
        section = np.column_stack([(xa - 0.5) * c, ya * c])
        wires.append(_make_wire(section, float(r), float(tw)))

    solid = Solid.makeLoft(wires, ruled=False)
    return solid

# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

def _load_design(path):
    with open(path) as f:
        payload = json.load(f)
    return payload["design"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design-json", type=str, default="data/last_design.json")
    ap.add_argument("--out",         type=str, default="figures/blade")
    ap.add_argument("--stations",    type=int, default=10)
    ap.add_argument("--airfoil-pts", type=int, default=40, dest="airfoil_pts")
    args = ap.parse_args()

    if Path(args.design_json).exists():
        design = _load_design(args.design_json)
        print(f"Loaded design from {args.design_json}")
    else:
        design = [0.5, 0.16, 3.0, 0.06, 0.04, 12.0, 4.0, 350.0, 3.0, 1.2]
        print("Using built-in demo design (no JSON found).")

    print(f"  R         = {design[0]:.3f} m")
    print(f"  n_blades  = {int(design[2])}")
    print(f"  c_root    = {design[3]:.4f} m")
    print(f"  c_tip     = {design[4]:.4f} m")
    print(f"  twist     = {design[5]:.2f} .. {design[6]:.2f} deg")
    print(f"  omega     = {design[7]:.1f} rad/s")
    print(f"  taper_p   = {design[9]:.3f}")

    solid = build_blade_solid(design, n_stations=args.stations, n_airfoil=args.airfoil_pts)

    if not solid.isValid():
        print("\n  WARNING: solid is not valid (may self-intersect).")

    bb = solid.BoundingBox()
    print(f"\n  bounding box  {bb.xlen:.3f} x {bb.ylen:.3f} x {bb.zlen:.3f} m")
    print(f"  volume        {solid.Volume() * 1e6:.1f} cm^3")

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.out)

    exporters.export(solid, str(base.with_suffix(".stl")))
    exporters.export(solid, str(base.with_suffix(".step")))
    print(f"\n  wrote {base.with_suffix('.stl')}")
    print(f"  wrote {base.with_suffix('.step')}")


if __name__ == "__main__":
    main()