"""ctypes bindings to the C rotor library.

If the shared library cannot be loaded (e.g. on a deployment platform
where the platform binary has not been built), the module still imports
successfully.  Callers should check `C_ENGINE_AVAILABLE` before relying
on the C verification path; the PINN surrogate path is entirely
independent of this module.
"""
import ctypes
import platform
from pathlib import Path

import numpy as np

DESIGN_N = 10
OUT_N = 12
ROTOR_N_STATIONS = 20

# Sentinel return code: C engine not available in this deployment.
RC_NO_C_ENGINE = -2


def _find_lib():
    root = Path(__file__).resolve().parents[2]
    system = platform.system()
    if system == "Windows":
        candidates = [
            root / "rotor.dll",
            root / "build" / "Release" / "rotor.dll",
            root / "build" / "rotor.dll",
        ]
    elif system == "Darwin":
        candidates = [
            root / "librotor.dylib",
            root / "build" / "librotor.dylib",
        ]
    else:
        candidates = [
            root / "librotor.so",
            root / "build" / "librotor.so",
        ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _try_load():
    """Attempt to load the C library.  Returns the CDLL or None."""
    try:
        path = _find_lib()
        if path is None:
            return None
        lib = ctypes.CDLL(str(path))
    except (OSError, FileNotFoundError):
        return None

    # Configure signatures only if load succeeded
    try:
        lib.rotor_eval.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        lib.rotor_eval.restype = ctypes.c_int

        lib.rotor_eval_stations.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        lib.rotor_eval_stations.restype = ctypes.c_int
    except AttributeError:
        return None
    return lib


_lib = _try_load()
C_ENGINE_AVAILABLE = _lib is not None


OUT_KEYS = [
    "thrust_N", "torque_Nm", "power_W", "CT", "CP", "FM",
    "OASPL_below_dBA", "OASPL_side_dBA",
    "max_sigma_MPa", "min_margin", "omega_flap1_ratio", "M_tip",
]


def rotor_eval(design):
    """Run the full C pipeline for one design.

    Returns (rc, dict).  rc = -2 when the C engine is unavailable.
    """
    if _lib is None:
        return RC_NO_C_ENGINE, None
    if len(design) != DESIGN_N:
        raise ValueError(f"design must have {DESIGN_N} elements")
    d = (ctypes.c_double * DESIGN_N)(*design)
    o = (ctypes.c_double * OUT_N)()
    rc = _lib.rotor_eval(d, o, OUT_N)
    if rc == -1:
        return rc, None
    return rc, dict(zip(OUT_KEYS, (float(x) for x in o)))


def rotor_eval_batch(designs):
    """Run the C pipeline for N designs.

    Returns (rc, Y).  When the C engine is unavailable, returns an array
    of RC_NO_C_ENGINE and a zero output matrix, so data generation can
    detect the failure rather than silently proceed.
    """
    designs = np.ascontiguousarray(designs, dtype=np.float64)
    if designs.ndim != 2 or designs.shape[1] != DESIGN_N:
        raise ValueError(f"designs must be (N, {DESIGN_N})")
    N = designs.shape[0]
    Y = np.zeros((N, OUT_N), dtype=np.float64)
    rc = np.full(N, RC_NO_C_ENGINE, dtype=np.int32)
    if _lib is None:
        return rc, Y

    dp_t = ctypes.POINTER(ctypes.c_double)
    for i in range(N):
        d_row = designs[i].ctypes.data_as(dp_t)
        o_row = Y[i].ctypes.data_as(dp_t)
        rc[i] = _lib.rotor_eval(d_row, o_row, OUT_N)
    return rc, Y


def rotor_eval_stations(design):
    """Per-station arrays.  Returns (rc, dict) or (RC_NO_C_ENGINE, None)."""
    if _lib is None:
        return RC_NO_C_ENGINE, None
    if len(design) != DESIGN_N:
        raise ValueError(f"design must have {DESIGN_N} elements")
    d  = (ctypes.c_double * DESIGN_N)(*design)
    r  = (ctypes.c_double * ROTOR_N_STATIONS)()
    c  = (ctypes.c_double * ROTOR_N_STATIONS)()
    tw = (ctypes.c_double * ROTOR_N_STATIONS)()
    dT = (ctypes.c_double * ROTOR_N_STATIONS)()
    ph = (ctypes.c_double * ROTOR_N_STATIONS)()
    rc = _lib.rotor_eval_stations(d, r, c, tw, dT, ph, ROTOR_N_STATIONS)
    if rc == -1:
        return rc, None
    return rc, {
        "r":     list(r),
        "chord": list(c),
        "twist": list(tw),
        "dT_dr": list(dT),
        "phi":   list(ph),
    }