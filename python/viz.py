"""Render blade and rotor images with PyVista.

Produces:
  figures/blade_render.png   single blade, three-quarter view
  figures/rotor_render.png   full N-blade rotor with hub, isometric
  figures/rotor_top.png      full N-blade rotor, top-down

Usage:
    python python/viz.py
    python python/viz.py --stl figures/blade.stl --design-json data/last_design.json
"""
import argparse
import json
from pathlib import Path

import pyvista as pv

pv.OFF_SCREEN = True

BLADE_COLOR = "#b8c8dc"
HUB_COLOR   = "#3a3a3a"


def _load_design(path):
    with open(path) as f:
        return json.load(f)["design"]


def _read_blade(stl_path):
    return pv.read(str(stl_path))

def render_single_blade(out_path, stl_path="figures/blade.stl", color=BLADE_COLOR, window_size=(1400, 900)):
    """Three-quarter view of the blade. White background, anti-aliased."""
    mesh = _read_blade(stl_path)

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.enable_anti_aliasing()

    p.add_mesh(mesh, color=color, smooth_shading=True, specular=0.5, specular_power=20, show_edges=False)
    p.add_axes(line_width=2, color="black")

    p.view_isometric()
    p.camera.azimuth = 30
    p.camera.elevation = 20
    p.camera.zoom(1.15)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    p.screenshot(str(out_path))
    p.close()
    print(f"    wrote {out_path}")

def render_full_rotor(out_path, design, stl_path="figures/blade.stl", color=BLADE_COLOR, hub_color=HUB_COLOR, window_size=(1400, 1200), top_view=False):
    """N blades around Z axis plus a hub cylinder and translucent disk."""
    blade = _read_blade(stl_path)
    n_blades = int(design[2])
    R        = design[0]
    hub_r    = R * design[1]

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.enable_anti_aliasing()

    # --- translucent swept disk (drawn first so blades sit on top) ---
    disk = pv.Disc(center=(0.0, 0.0, 0.0), inner=0.0, outer=R, normal=(0.0, 0.0, 1.0), r_res=1, c_res=96)
    p.add_mesh(disk, color="#dde6f0", opacity=0.20, show_edges=False)

    # --- blades ---
    for i in range(n_blades):
        angle = 360.0 * i / n_blades
        rot = blade.rotate_z(angle, inplace=False)
        p.add_mesh(rot, color=color, smooth_shading=True, specular=0.6, specular_power=25, show_edges=True, edge_color="#4a5568", line_width=1)

    # --- hub cylinder ---
    hub_height = 0.08 * R
    hub = pv.Cylinder(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), radius=hub_r, height=hub_height, resolution=96, capping=True)
    p.add_mesh(hub, color=hub_color, smooth_shading=True, specular=0.4, specular_power=20)

    p.add_axes(line_width=2, color="black")

    if top_view:
        p.view_xy()
    else:
        p.view_isometric()
        p.camera.azimuth   = -35
        p.camera.elevation = 45    # steeper — looks down at the disk
        p.camera.zoom(0.95)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    p.screenshot(str(out_path))
    p.close()
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl",         default="figures/blade.stl")
    ap.add_argument("--design-json", default="data/last_design.json")
    ap.add_argument("--out-dir",     default="figures")
    ap.add_argument("--skip-blade",  action="store_true")
    ap.add_argument("--skip-rotor",  action="store_true")
    args = ap.parse_args()

    design = _load_design(args.design_json)
    print(f"Design: R = {design[0]:.3f} m,  n_blades = {int(design[2])}")

    if not args.skip_blade:
        render_single_blade(Path(args.out_dir) / "blade_render.png", args.stl)

    if not args.skip_rotor:
        render_full_rotor(Path(args.out_dir) / "rotor_render.png", design, args.stl, top_view=False)
        render_full_rotor(Path(args.out_dir) / "rotor_top.png", design, args.stl, top_view=True)


if __name__ == "__main__":
    main()