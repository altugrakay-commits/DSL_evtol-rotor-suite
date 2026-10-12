/* ============================================================
 * rotor_api.c — flat ABI binding BEM + noise + structure
 *
 * The Python side calls rotor_eval() with a 10-element design
 * vector and receives a 12-element output vector. All physics
 * is hidden behind this boundary.
 * ============================================================ */
#include "rotor_api.h"
#include "bem.h"
#include "noise.h"
#include "structure.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* --- Fixed atmospheric conditions (sea-level ISA) --- */
#define RHO_SEA_LEVEL  1.225
#define MU_SEA_LEVEL   1.81e-5
#define A_SOUND        340.0

/* --- Observers at 100 m --- */
static const Observer OBS_BELOW = {   0.0, 0.0, -100.0 };
static const Observer OBS_SIDE  = { 100.0, 0.0,    0.0 };

/* --- Fixed NACA 0012 polar --- */
static PolarPoint NACA0012_PTS[] = {
    {-0.30,-1.20,0.030,0}, {-0.20,-1.00,0.020,0}, {-0.14,-0.80,0.014,0},
    {-0.10,-0.60,0.011,0}, {-0.05,-0.30,0.009,0}, { 0.00, 0.00,0.008,0},
    { 0.05, 0.30,0.009,0}, { 0.10, 0.60,0.011,0}, { 0.14, 0.80,0.014,0},
    { 0.20, 1.00,0.020,0}, { 0.30, 1.20,0.030,0},
};

/* --- Fixed CFRP material --- */
static const BladeMaterial CFRP = {
    .E           = 70.0e9,
    .rho         = 1600.0,
    .sigma_yield = 600.0e6,
    .t_over_c    = 0.12,
    .name        = "CFRP-typical",
};

/* --- Bounds check on the design vector --- */
static int validate_design(const double* d)
{
    if (d[DESIGN_R]          < 0.10 || d[DESIGN_R]          > 1.20) return -1;
    if (d[DESIGN_HUB_RATIO]  < 0.05 || d[DESIGN_HUB_RATIO]  > 0.35) return -1;
    if (d[DESIGN_N_BLADES]   < 1.99 || d[DESIGN_N_BLADES]   > 6.99) return -1;
    if (d[DESIGN_C_ROOT]     < 0.01 || d[DESIGN_C_ROOT]     > 0.25) return -1;
    if (d[DESIGN_C_TIP]      < 0.005|| d[DESIGN_C_TIP]      > 0.15) return -1;
    if (d[DESIGN_TWIST_ROOT] < -5.0 || d[DESIGN_TWIST_ROOT] > 25.0) return -1;
    if (d[DESIGN_TWIST_TIP]  <-10.0 || d[DESIGN_TWIST_TIP]  > 15.0) return -1;
    if (d[DESIGN_OMEGA]      < 50.0 || d[DESIGN_OMEGA]      > 800.0) return -1;
    if (d[DESIGN_COLLECTIVE] < -5.0 || d[DESIGN_COLLECTIVE] > 15.0) return -1;
    if (d[DESIGN_TAPER_P]    < 0.50 || d[DESIGN_TAPER_P]    > 2.50) return -1;
    return 0;
}

/* --- Build blade geometry from the design vector --- */
static void build_geometry(const double* d, Bladestation* st, BladeGeometry* geom)
{
    const double R       = d[DESIGN_R];
    const double rh      = R * d[DESIGN_HUB_RATIO];
    const int    N       = (int)floor(d[DESIGN_N_BLADES] + 0.5);
    const double c_root  = d[DESIGN_C_ROOT];
    const double c_tip   = d[DESIGN_C_TIP];
    const double tr_root = d[DESIGN_TWIST_ROOT] * M_PI / 180.0;
    const double tr_tip  = d[DESIGN_TWIST_TIP]  * M_PI / 180.0;
    const double p       = d[DESIGN_TAPER_P];

    for (int i = 0; i < ROTOR_N_STATIONS; ++i) {
        double xi       = ((double)i + 0.5) / (double)ROTOR_N_STATIONS;
        st[i].r         = rh + xi * (R - rh);
        st[i].chord     = c_root + (c_tip - c_root) * pow(xi, p);
        st[i].twist     = tr_root + (tr_tip - tr_root) * pow(xi, p);
        st[i].polar_idx = 0;
    }

    geom->stations   = st;
    geom->n_stations = ROTOR_N_STATIONS;
    geom->n_blades   = N;
    geom->radius     = R;
    geom->hub_radius = rh;
}

int rotor_eval(const double* design, double* outputs, int n_out)
{
    if (!design || !outputs) return -1;
    if (n_out < OUT_N)       return -1;
    if (validate_design(design) != 0) return -1;

    Bladestation   st[ROTOR_N_STATIONS];
    BladeGeometry  geom;
    build_geometry(design, st, &geom);

    AirfoilPolar polar;
    polar.points   = NACA0012_PTS;
    polar.n_points = (int)(sizeof(NACA0012_PTS)/sizeof(NACA0012_PTS[0]));
    polar.Re       = 1e6;
    strncpy(polar.name, "NACA0012", sizeof(polar.name) - 1);
    polar.name[sizeof(polar.name) - 1] = '\0';

    OperatingPoint op;
    memset(&op, 0, sizeof(op));
    op.rho        = RHO_SEA_LEVEL;
    op.mu         = MU_SEA_LEVEL;
    op.a_sound    = A_SOUND;
    op.V_inf      = 0.0;
    op.omega      = design[DESIGN_OMEGA];
    op.collective = design[DESIGN_COLLECTIVE] * M_PI / 180.0;
    op.n_iter_max = 200;
    op.tol        = 1e-6;
    op.relax      = 0.4;

    BEMResult bem;
    if (bem_result_alloc(&bem, ROTOR_N_STATIONS) != 0) return -1;
    bem_solve(&geom, &op, &polar, 1, &bem);

    NoiseResult nb, ns;
    noise_predict(&geom, &op, &bem, &OBS_BELOW, &nb);
    noise_predict(&geom, &op, &bem, &OBS_SIDE,  &ns);

    StructureResult str;
    if (structure_alloc(&str, ROTOR_N_STATIONS) != 0) {
        bem_result_free(&bem);
        return -1;
    }
    structure_solve(&geom, &op, &bem, &CFRP, &str);

    outputs[OUT_THRUST_N]      = bem.thrust;
    outputs[OUT_TORQUE_NM]     = bem.torque;
    outputs[OUT_POWER_W]       = bem.power;
    outputs[OUT_CT]            = bem.CT;
    outputs[OUT_CP]            = bem.CP;
    outputs[OUT_FM]            = bem.FM;
    outputs[OUT_OASPL_BELOW]   = nb.OASPL_A;
    outputs[OUT_OASPL_SIDE]    = ns.OASPL_A;
    outputs[OUT_MAX_SIGMA_MPA] = str.max_sigma / 1e6;
    outputs[OUT_MIN_MARGIN]    = str.min_margin;
    outputs[OUT_OMEGA_RATIO]   = str.omega_flap1_ratio;
    outputs[OUT_M_TIP]         = op.omega * geom.radius / op.a_sound;

    int rc = bem.converged ? 0 : 1;

    structure_free(&str);
    bem_result_free(&bem);
    return rc;
}

int rotor_eval_stations(const double* design,
                        double* r_out,
                        double* chord_out,
                        double* twist_out,
                        double* dT_dr_out,
                        double* phi_out,
                        int     n_stations)
{
    if (!design || !r_out || !chord_out || !twist_out ||
        !dT_dr_out || !phi_out) return -1;
    if (n_stations != ROTOR_N_STATIONS) return -1;
    if (validate_design(design) != 0) return -1;

    Bladestation   st[ROTOR_N_STATIONS];
    BladeGeometry  geom;
    build_geometry(design, st, &geom);

    for (int i = 0; i < ROTOR_N_STATIONS; ++i) {
        r_out[i]     = st[i].r;
        chord_out[i] = st[i].chord;
        twist_out[i] = st[i].twist;
    }

    AirfoilPolar polar;
    polar.points   = NACA0012_PTS;
    polar.n_points = (int)(sizeof(NACA0012_PTS)/sizeof(NACA0012_PTS[0]));
    polar.Re       = 1e6;
    strncpy(polar.name, "NACA0012", sizeof(polar.name) - 1);
    polar.name[sizeof(polar.name) - 1] = '\0';

    OperatingPoint op;
    memset(&op, 0, sizeof(op));
    op.rho        = RHO_SEA_LEVEL;
    op.mu         = MU_SEA_LEVEL;
    op.a_sound    = A_SOUND;
    op.V_inf      = 0.0;
    op.omega      = design[DESIGN_OMEGA];
    op.collective = design[DESIGN_COLLECTIVE] * M_PI / 180.0;
    op.n_iter_max = 200;
    op.tol        = 1e-6;
    op.relax      = 0.4;

    BEMResult bem;
    if (bem_result_alloc(&bem, ROTOR_N_STATIONS) != 0) return -1;
    bem_solve(&geom, &op, &polar, 1, &bem);

    for (int i = 0; i < ROTOR_N_STATIONS; ++i) {
        dT_dr_out[i] = bem.dT_dr[i];
        phi_out[i]   = bem.phi[i];
    }

    int rc = bem.converged ? 0 : 1;
    bem_result_free(&bem);
    return rc;
}