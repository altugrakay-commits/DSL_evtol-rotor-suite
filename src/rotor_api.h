/* ============================================================
 * rotor_api.h — flat C ABI exposed to Python via ctypes
 * ============================================================ */
#ifndef ROTOR_API_H
#define ROTOR_API_H

/* --- Export macro: expands to dllexport when building the DLL,
 *     dllimport when a client includes this header, and to
 *     default visibility on ELF/Mach-O platforms. -------------- */
#ifdef _WIN32
  #ifdef ROTOR_API_BUILD
    #define ROTOR_API __declspec(dllexport)
  #else
    #define ROTOR_API __declspec(dllimport)
  #endif
#else
  #define ROTOR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* --- Design vector indices --- */
#define DESIGN_R           0
#define DESIGN_HUB_RATIO   1
#define DESIGN_N_BLADES    2
#define DESIGN_C_ROOT      3
#define DESIGN_C_TIP       4
#define DESIGN_TWIST_ROOT  5
#define DESIGN_TWIST_TIP   6
#define DESIGN_OMEGA       7
#define DESIGN_COLLECTIVE  8
#define DESIGN_TAPER_P     9
#define DESIGN_N          10

/* --- Output vector indices --- */
#define OUT_THRUST_N        0
#define OUT_TORQUE_NM       1
#define OUT_POWER_W         2
#define OUT_CT              3
#define OUT_CP              4
#define OUT_FM              5
#define OUT_OASPL_BELOW     6
#define OUT_OASPL_SIDE      7
#define OUT_MAX_SIGMA_MPA   8
#define OUT_MIN_MARGIN      9
#define OUT_OMEGA_RATIO    10
#define OUT_M_TIP          11
#define OUT_N              12

/* --- Fixed discretization --- */
#define ROTOR_N_STATIONS   20

/* --- Return codes ---
 *   0 = fully converged, all outputs valid
 *   1 = BEM non-converged, outputs approximate
 *  -1 = hard failure (invalid design, alloc failure) */
ROTOR_API int rotor_eval(const double* design, double* outputs, int n_out);

ROTOR_API int rotor_eval_stations(const double* design,
                                  double* r_out,
                                  double* chord_out,
                                  double* twist_out,
                                  double* dT_dr_out,
                                  double* phi_out,
                                  int     n_stations);

#ifdef __cplusplus
}
#endif
#endif /* ROTOR_API_H */