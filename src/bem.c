/* ============================================================
 * bem.c — hover + forward-flight BEM solver
 *
 * Hover formulation:
 *   phi = atan(v_i / (Omega r)),  lambda = tan(phi)
 *   Blade element:  dT = 0.5 rho W^2 N c (Cl cos phi - Cd sin phi) dr
 *   Momentum:       dT = 4 pi r rho F v_i^2 dr
 *   Combining:      lambda^2 / (1 + lambda^2) = sigma Cn / (4 F)
 *   where sigma = N c / (2 pi r), Cn = Cl cos phi - Cd sin phi.
 *
 *   Solved per station by fixed-point iteration with under-relaxation.
 *   Validated against Caradonna-Tung (target: FM ~ 0.65-0.75).
 * ============================================================ */

 #include "bem.h"
 #include <math.h>
 #include <stdio.h>
 #include <string.h>
 #include <stdlib.h>

 #ifndef M_PI
 #define M_PI 3.14159265358979323846
 #endif

/* ---- polar lookup: linear interp, clamp at ends ----------- */
int airfoil_lookup(const AirfoilPolar* p, double alpha,
                   double* Cl, double* Cd, double* Cm)
{
    if (p == NULL || p->n_points < 2) return -1;

    if (alpha <= p->points[0].alpha) {
        *Cl = p->points[0].Cl;
        *Cd = p->points[0].Cd;
        *Cm = p->points[0].Cm;
        return 0;
    }
    if (alpha >= p->points[p->n_points-1].alpha) {
        *Cl = p->points[p->n_points-1].Cl;
        *Cd = p->points[p->n_points-1].Cd;
        *Cm = p->points[p->n_points-1].Cm;
        return 0;
    }
    int lo = 0, hi = p->n_points - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (p->points[mid].alpha <= alpha) lo = mid;
        else hi = mid;
    }
    double a0 = p->points[lo].alpha, a1 = p->points[hi].alpha;
    double t  = (alpha - a0) / (a1 - a0);
    *Cl = p->points[lo].Cl + t * (p->points[hi].Cl - p->points[lo].Cl);
    *Cd = p->points[lo].Cd + t * (p->points[hi].Cd - p->points[lo].Cd);
    *Cm = p->points[lo].Cm + t * (p->points[hi].Cm - p->points[lo].Cm);
    return 0;
}

/* ---- Prandtl tip / hub loss ------------------------------- */
static inline double tip_loss(double N, double R, double r, double sinphi)
    {
        if (sinphi < 1e-6) return 1.0;
        double f = 0.5 * N * (R - r) / (r * sinphi);
        if (f > 30.0) return 1.0;
        return (2.0 / M_PI) * acos(exp(-f));
    }

static inline double hub_loss(double N, double rh, double r, double sinphi)
    {
        if (sinphi < 1e-6) return 1.0;
        double f = 0.5 * N * (r - rh) / (rh * sinphi);
        if (f > 30.0) return 1.0;
        return (2.0 / M_PI) * acos(exp(-f));
    }

/* ---- hover station solver ---------------------------------
 * phi_init: warm-start inflow ange (rad). Caller passes the last
 *           converged station's phi. Cold start of the whole sweep 
 *           uses 0.2 rad, a reasonable guess for the hub
 * ------------------------------------------------------------ */
static int solve_station_hover(
    const BladeGeometry*    geom,
    const OperatingPoint*   op,
    const AirfoilPolar*     polars,
    const Bladestation*     st,
    double phi_init,
    double* out_phi,
    double* out_Cl,
    double* out_Cd,
    double* out_F,
    double* out_vi)
    {
        const double r      = st->r;
        const double c      = st->chord;
        const double th     = st->twist + op->collective;
        const double Omr    = op->omega * r;
        const double sig    = (geom->n_blades * c) / (2.0 * M_PI * r);

        double phi = phi_init;  /* initial guess (rad) */
        double Cl  = 0.0;
        double Cd  = 0.0;
        double Cm  = 0.0;
        double F   = 1.0;   /* <-- declared HERE, outside the loop */

        for (int it = 0; it < op->n_iter_max; ++it) {
            double alpha = th - phi;
            if (airfoil_lookup(&polars[st->polar_idx], alpha, &Cl, &Cd, &Cm) != 0)
                return -1;

            double s = sin(phi), co = cos(phi);
            double Cn = Cl * co - Cd * s;
            F = tip_loss(geom->n_blades, geom->radius,   r, s)
              * hub_loss(geom->n_blades, geom->hub_radius, r, s);
            if (F < 5e-2) F = 5e-2;     /* prevent momentum-balance singularity */

            double rhs = sig * Cn / (4.0 * F);
            if (rhs > 0.98) rhs = 0.98;         /* clamp: unphysical loading */
            if (rhs < -0.98) rhs = -0.98;
            double lam2 = rhs / (1.0 - rhs);
            double phi_new = atan(sqrt(lam2 > 0.0 ? lam2 : 0.0));

            double dphi = phi_new - phi;
            phi += op->relax * dphi;

            if (fabs(dphi) < op->tol) {
                *out_phi = phi;
                *out_Cl  = Cl;
                *out_Cd  = Cd;
                *out_F   = F;
                *out_vi  = Omr * tan(phi);
                return 0;
            }
        }
        /* not converged — return last value for caller's best-effort use */
        *out_phi = phi;
        *out_Cl  = Cl;
        *out_Cd  = Cd;
        *out_F   = F;
        *out_vi  = Omr * tan(phi);
        return -1; /* not converged */
    }

/* ---- main solver ------------------------------------------ */
int bem_solve(const BladeGeometry*  geom,
              const OperatingPoint* op,
              const AirfoilPolar*   polars,
              int                   n_polars,
              BEMResult*            res)
{
    (void)n_polars;
    const int n = geom->n_stations;
    if (n < 2) return -1;

    int all_conv = 1;
    int n_failed = 0;
    double phi_prev = 0.2;  /* cold start for the hub station */

    for (int i = 0; i < n; ++i) {
        const Bladestation* st = &geom->stations[i];
        double phi, Cl, Cd, F, vi;

        int rc;
        if (op->V_inf < 1e-6) {
            rc = solve_station_hover(geom, op, polars, st, phi_prev,
                                     &phi, &Cl, &Cd, &F, &vi);
        } else {
            /* TODO: forward-flight BEM (axial a + tangential a',
             * azimuthal averaging, reverse-flow region). */
            rc = -2;
        }
        if (rc == 0) {
            phi_prev = phi; /* advance warm start only on success */
        } else {
            all_conv = 0;
            n_failed++;
        }

        double Omr = op->omega * st->r;
        double W2  = Omr * Omr + vi * vi;

        double Cn = Cl * cos(phi) - Cd * sin(phi);
        double Ct = Cl * sin(phi) + Cd * cos(phi);  /* opposes rotation */

        double dT = 0.5 * op->rho * W2 * geom->n_blades * st->chord * Cn;
        double dQ = 0.5 * op->rho * W2 * geom->n_blades * st->chord * Ct * st->r;

        res->r[i]     = st->r;
        res->dT_dr[i] = dT;
        res->dQ_dr[i] = dQ;
        res->alpha[i] = st->twist + op->collective - phi;
        res->phi[i]   = phi;
        res->Cl[i]    = Cl;
        res->Cd[i]    = Cd;
        res->F[i]     = F;
        res->v_i[i]   = vi;
    }

    /* trapezoidal integration over r */
    double T_int = 0.0, Q_int = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        double dr = geom->stations[i+1].r - geom->stations[i].r;
        T_int += 0.5 * (res->dT_dr[i] + res->dT_dr[i+1]) * dr;
        Q_int += 0.5 * (res->dQ_dr[i] + res->dQ_dr[i+1]) * dr;
    }

    res->thrust             = T_int;
    res->torque             = Q_int;
    res->power              = Q_int * op->omega;
    res->converged          = all_conv;
    res->n_stations_failed  = n_failed;

    double A   = M_PI * geom->radius * geom->radius;
    double OmR = op->omega * geom->radius;
    res->CT = T_int / (op->rho * A * OmR * OmR);
    res->CP = res->power / (op->rho * A * OmR * OmR * OmR);

    if (op->V_inf < 1e-6 && res->power > 1e-12) {
        double T_ideal = pow(T_int, 1.5) / sqrt(2.0 * op->rho * A);
        res->FM = T_ideal / res->power;
    } else {
        res->FM = NAN;
    }

    return 0;
}

int bem_result_alloc(BEMResult* res, int n)
{
    memset(res, 0, sizeof(*res));
    res->r     = calloc(n, sizeof(double));
    res->dT_dr = calloc(n, sizeof(double));
    res->dQ_dr = calloc(n, sizeof(double));
    res->alpha = calloc(n, sizeof(double));
    res->phi   = calloc(n, sizeof(double));
    res->Cl    = calloc(n, sizeof(double));
    res->Cd    = calloc(n, sizeof(double));
    res->F     = calloc(n, sizeof(double));
    res->v_i   = calloc(n, sizeof(double));
    if (!res->r || !res->dT_dr || !res->dQ_dr || !res->alpha ||
        !res->phi || !res->Cl || !res->Cd || !res->F || !res->v_i)
        return -1;
    return 0;
}

void bem_result_free(BEMResult* res)
{
    free(res->r);     res->r     = NULL;
    free(res->dT_dr); res->dT_dr = NULL;
    free(res->dQ_dr); res->dQ_dr = NULL;
    free(res->alpha); res->alpha = NULL;
    free(res->phi);   res->phi   = NULL;
    free(res->Cl);    res->Cl    = NULL;
    free(res->Cd);    res->Cd    = NULL;
    free(res->F);     res->F     = NULL;
    free(res->v_i);   res->v_i   = NULL;
}