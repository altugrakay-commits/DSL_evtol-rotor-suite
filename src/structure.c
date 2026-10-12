/* ============================================================
 * structure.c — first-order structural model for rotor blades
 *
 * Assumptions (documented, not hidden):
 *   - Solid rectangular cross-section of width c and thickness
 *     t = t_over_c * c.  Real airfoils are hollow or have spars,
 *     so this overestimates stiffness and underestimates stress.
 *     Fine for design screening; not for certification.
 *   - Euler-Bernoulli beam theory, no shear deformation.
 *   - Flapwise loading only (from BEM dT/dr).  No torsion, no
 *     lead-lag, no coupling between modes.
 *   - First flap natural frequency via Rayleigh quotient with
 *     assumed mode shape phi(xi) = xi^2, xi = (r - rh)/(R - rh).
 *     Centrifugal stiffening included.
 *
 *   Tension:      T(r)  = int_r^R m(r') Omega^2 r' dr'
 *   Flap moment:  M(r)  = int_r^R (r' - r) w(r') dr',  w = dT/dr
 *   Axial stress: sigma_a = T / A,         A = c t
 *   Bend stress:  sigma_b = M / S,         S = c t^2 / 6
 *   Margin:       sigma_yield / (sigma_a + sigma_b)
 * ============================================================ */
#include "structure.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

int structure_alloc(StructureResult* res, int n)
{
    memset(res, 0, sizeof(*res));
    res->stations = calloc(n, sizeof(StationStress));
    if (!res->stations) return -1;
    return 0;
}

void structure_free(StructureResult* res)
{
    free(res->stations);
    res->stations = NULL;
}

int structure_solve(const BladeGeometry*    geom,
                    const OperatingPoint*   op,
                    const BEMResult*        bem,
                    const BladeMaterial*    mat,
                    StructureResult*        out)
{
    const int n = geom->n_stations;
    if (n < 2) return -1;

    const double Om = op->omega;
    const double Om2 = Om * Om;
    const double rh = geom->hub_radius;
    const double R = geom->radius;
    const double L = R -rh;

    /* --- centrifugal tension, integrated tip->root ---------- */
    /* T_j = int_{r_j}^{R} m(r) Omega^2 r dr                    */
    for (int j = 0; j < n; ++j) {
        double T_j = 0.0;
        for (int i = j; i < n - 1; ++i) {
            double r0 = geom->stations[i ].r;
            double r1 = geom->stations[i+1].r;
            double c0 = geom->stations[i ].chord;
            double c1 = geom->stations[i+1].chord;
            double m0 = mat->rho * c0 * (mat->t_over_c * c0);
            double m1 = mat->rho * c1 * (mat->t_over_c * c1);
            double m_avg = 0.5 * (m0 + m1);
            double r_mid = 0.5 * (r0 + r1);
            double dr = r1 - r0;
            T_j += m_avg * Om2 * r_mid * dr;
        }
        out->stations[j].T_centrifugal = T_j;
    }
    /* --- flap bending moment, integrated tip->root ---------- */
    /* M_j = int_{r_j}^{R} (r - r_j) w(r) dr,  w = dT/dr        */
    for (int j = 0; j < n; ++j) {
        double M_j = 0.0;
        double r_j = geom->stations[j].r;
        for (int i = j; i < n - 1; ++i) {
            double r0 = geom->stations[i  ].r;
            double r1 = geom->stations[i+1].r;
            double w0 = bem->dT_dr[i  ];
            double w1 = bem->dT_dr[i+1];
            double w_avg = 0.5 * (w0 + w1);
            double r_mid = 0.5 * (r0 + r1);
            double dr    = r1 - r0;
            M_j += w_avg * dr * (r_mid - r_j);
        }
        out->stations[j].M_flap = M_j;
    }
    
    /* --- stresses and margin -------------------------------- */
    double max_sigma   = 0.0;
    double max_sigma_r = 0.0;
    double min_margin  = 1e30;

    for (int i = 0; i < n; ++i) {
        const double c = geom->stations[i].chord;
        const double t = mat->t_over_c * c;
        const double A_sec = c * t;
        const double S_sec = c * t * t / 6.0;

        double sig_a = out->stations[i].T_centrifugal / A_sec;
        double sig_b = out->stations[i].M_flap        / S_sec;
        double sig   = sig_a + sig_b;

        out->stations[i].sigma_axial   = sig_a;
        out->stations[i].sigma_bending = sig_b;
        out->stations[i].sigma_total   = sig;
        out->stations[i].margin        = mat->sigma_yield / sig;

        if (sig > max_sigma) {
            max_sigma   = sig;
            max_sigma_r = geom->stations[i].r;
        }
        if (out->stations[i].margin < min_margin)
            min_margin = out->stations[i].margin;
    }

    out->max_sigma           = max_sigma;
    out->max_sigma_station_r = max_sigma_r;
    out->min_margin          = min_margin;
    out->T_root              = out->stations[0].T_centrifugal;
    out->M_root              = out->stations[0].M_flap;

    /* --- first flap natural frequency, Rayleigh quotient ---- */
    /* phi(xi) = xi^2, xi = (r - rh)/L
     * phi' = 2 xi / L, phi'' = 2 / L^2
     * omega^2 = [int EI phi''^2 dr + int T phi'^2 dr] / int m phi^2 dr
     */
    double num_bend = 0.0;
    double num_cent = 0.0;
    double den      = 0.0;

    for (int i = 0; i < n - 1; ++i) {
        double r0 = geom->stations[i  ].r;
        double r1 = geom->stations[i+1].r;
        double c0 = geom->stations[i  ].chord;
        double c1 = geom->stations[i+1].chord;
        double dr = r1 - r0;

        double xi_mid = 0.5 * ((r0 - rh) + (r1 - rh)) / L;

        /* I = t_over_c^3 * c^4 / 12 */
        double I0 = mat->t_over_c * mat->t_over_c * mat->t_over_c * c0*c0*c0*c0 / 12.0;
        double I1 = mat->t_over_c * mat->t_over_c * mat->t_over_c * c1*c1*c1*c1 / 12.0;
        double EI = mat->E * 0.5 * (I0 + I1);

        double phipp2 = 4.0 / (L * L);
        double phip2  = 4.0 * xi_mid * xi_mid / (L * L);
        double phi2   = xi_mid * xi_mid * xi_mid * xi_mid;

        double T_mid = 0.5 * (out->stations[i  ].T_centrifugal +
                              out->stations[i+1].T_centrifugal);

        double m0 = mat->rho * c0 * (mat->t_over_c * c0);
        double m1 = mat->rho * c1 * (mat->t_over_c * c1);
        double m_avg = 0.5 * (m0 + m1);

        num_bend += EI      * phipp2 * dr;
        num_cent += T_mid   * phip2  * dr;
        den      += m_avg   * phi2   * dr;
    }

    if (den > 1e-30) {
        double omega2 = (num_bend + num_cent) / den;
        out->omega_flap1 = (omega2 > 0.0) ? sqrt(omega2) : 0.0;
    } else {
        out->omega_flap1 = 0.0;
    }
    out->omega_flap1_ratio = (Om > 1e-9) ? out->omega_flap1 / Om : 0.0;

    return 0;
}