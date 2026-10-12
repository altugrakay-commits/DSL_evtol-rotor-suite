/* ============================================================
 * noise.c — semi-empirical rotor noise model
 *
 * Two mechanisms:
 *   1. Tonal at BPF from periodic blade loading.
 *   2. Broadband from turbulent boundary-layer trailing-edge
 *      scattering, expressed as an OASPL.
 *
 * Scaling laws, not full FW-H or BPM.  Calibration constants
 * are tuned so a 1 m, 2-blade rotor at M_tip = 0.44 producing
 * ~65 N gives ~66 dBA at 100 m below the hub, consistent with
 * published small-UAM hover measurements.
 *
 * References:
 *   Brooks, Pope, Marcolini, "Airfoil Self-Noise and Prediction",
 *     NASA RP-1218, 1989.
 *   Fink, "Noise Component Method for Airframe Noise",
 *     J. Aircraft, 1979.
 *   Rizzi et al., "Urban Air Mobility Noise: Current Practice,
 *     Gaps, and Recommendations", NASA/TP-2020-5007433.
 * ============================================================ */
#include "noise.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* --- Calibration constants ---------------------------------- */
#define K_TONAL       118.15
#define K_BROADBAND   115.06
#define BB_CENTER_HZ  500.0

/* A-weighting (IEC 61672), returned in dB. */
static double a_weight_db(double f)
{
    if (f < 1.0) f = 1.0;
    double f2  = f*f;
    double num = 12194.0*12194.0 * f2*f2;
    double den = (f2 + 20.6*20.6)
               * sqrt((f2 + 107.7*107.7)*(f2 + 737.9*737.9))
               * (f2 + 12194.0*12194.0);
    return 20.0 * log10(num/den) + 2.0;   /* normalise to 0 dB at 1 kHz */
}

int noise_predict(const BladeGeometry*  geom,
                  const OperatingPoint* op,
                  const BEMResult*      bem,
                  const Observer*       obs,
                  NoiseResult*          out)
{
    /* --- Geometry and kinematics --- */
    const double R      = geom->radius;
    const double N      = (double)geom->n_blades;
    const double Omega  = op->omega;
    const double a      = op->a_sound;
    const double M_tip  = Omega * R / a;

    /* Mean chord over the span */
    double c_ref = 0.0;
    for (int i = 0; i < geom->n_stations; ++i)
        c_ref += geom->stations[i].chord;
    c_ref /= (double)geom->n_stations;

    const double A_disk = M_PI * R * R;
    const double sigma  = N * c_ref / (M_PI * R);

    /* --- Observer geometry --- */
    const double xo = obs->x, yo = obs->y, zo = obs->z;
    double r = sqrt(xo*xo + yo*yo + zo*zo);
    if (r < 1.0) r = 1.0;
    const double cos_theta = zo / r;
    const double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

    /* --- BPF --- */
    const double BPF = N * Omega / (2.0 * M_PI);

    /* --- Thrust (floor to avoid log(0)) --- */
    double T = bem->thrust;
    if (T < 1.0) T = 1.0;

    /* --- Softened directivity (engineering shapes) ---
     * Ideal dipole/cardioid give unrealistically deep nulls.
     * Real rotors have finite span, hub interference and a
     * turbulent boundary layer that fill the nulls.  These
     * blended forms give ~26 dB and ~10.5 dB swings. */
    const double D_tonal = 0.05 + 0.95 * sin_theta;
    const double D_bb    = 0.30 + 0.70 * fabs(cos_theta);

    /* --- Tonal (loading) at BPF --- */
    const double SPL_tonal = K_TONAL
                           + 20.0  * log10(T)
                           + 100.0 * log10(M_tip)
                           - 20.0  * log10(r)
                           + 20.0  * log10(D_tonal);

    /* --- Broadband OASPL --- */
    const double SPL_bb = K_BROADBAND
                        + 10.0 * log10(sigma + 1e-9)
                        + 50.0 * log10(M_tip)
                        + 10.0 * log10(A_disk)
                        - 20.0 * log10(r)
                        + 20.0 * log10(D_bb);

    /* --- A-weighting --- */
    const double A_tonal = a_weight_db(BPF);
    const double A_bb    = a_weight_db(BB_CENTER_HZ);

    const double SPL_tonal_A = SPL_tonal + A_tonal;
    const double SPL_bb_A    = SPL_bb    + A_bb;

    /* --- Log-sum --- */
    const double SPL_total   = 10.0 * log10(pow(10.0, SPL_tonal  /10.0)
                                          + pow(10.0, SPL_bb     /10.0));
    const double SPL_total_A = 10.0 * log10(pow(10.0, SPL_tonal_A/10.0)
                                          + pow(10.0, SPL_bb_A   /10.0));

    out->SPL_tonal     = SPL_tonal;
    out->SPL_broadband = SPL_bb;
    out->SPL_total     = SPL_total;
    out->OASPL         = SPL_total;
    out->OASPL_A       = SPL_total_A;
    out->BPF_Hz        = BPF;
    out->directivity   = cos_theta;
    return 0;
}