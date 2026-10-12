"""Drive the C core from Python."""
from rotorlib.ffi import rotor_eval, rotor_eval_stations, DESIGN_N

design = [
    0.50,    # R (m)
    0.16,    # hub_ratio
    2.0,     # N_blades
    0.05,    # c_root (m)
    0.05,    # c_tip (m)
    12.0,    # twist_root (deg)
    4.0,     # twist_tip (deg)
    300.0,   # omega (rad/s)
    0.0,     # collective (deg)
    1.0,     # taper_p
]
assert len(design) == DESIGN_N

rc, out = rotor_eval(design)
print(f"rc = {rc}")
assert out is not None
for k, v in out.items():
    print(f"  {k:22s} = {v:12.4f}")

rc, st = rotor_eval_stations(design)
print("\nPer-station:")
print("   i      r      chord     twist     dT_dr       phi")
for i in range(5):
    print(f"  {i:2d}  {st['r'][i]:6.4f}  {st['chord'][i]:6.4f}  " # type: ignore
          f"{st['twist'][i]:6.4f}  {st['dT_dr'][i]:8.2f}  {st['phi'][i]:6.4f}") # type: ignore