/* ============================================================
 * test_noise_cal.c — calibration run against the Pettingill
 * ideally-twisted NACA 0012 rotor (NASA SHAC measurements).
 *
 * Rotor:  R = 0.1588 m,  4 blades,  NACA 0012,  c/R = 0.20
 *         RPM = 5500  ->  M_tip = 0.269
 * Observer: 1.906 m below hub (12 rotor radii)
 *
 * The model's SPL output at this condition is compared against
 * the published reference OASPL to compute the calibration
 * offset for K_BROADBAND and K_TONAL.
 * ============================================================ */
#include <noise.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void build_naca0012(AirfoilPolar* p) {
    static PolarPoint pts[] = {
        {-0.30,-1.20,0.030,0}, {-0.20,-1.00,0.020,0}, {-0.14,-0.80,0.014,0},
        {-0.10,-0.60,0.011,0}, {-0.05,-0.30,0.009,0}, { 0.00, 0.00,0.008,0},
        { 0.05, 0.30,0.009,0}, { 0.10, 0.60,0.011,0}, { 0.14, 0.80,0.014,0},
        { 0.20, 1.00,0.020,0}, { 0.30, 1.20,0.030,0},
    };
    p->points   = pts;
    p->n_points = (int)(sizeof(pts)/sizeof(pts[0]));
    p->Re       = 1e6;
    snprintf(p->name, sizeof(p->name), "NACA0012-crude");
}

int main(void) {
    enum { N = 20 };

    /* --- Pettingill rotor geometry --- */
    const double R      = 0.1588;      /* m */
    const double hub_rh = 0.20 * R;      /* m, blades start at r/R = 0.2 */
    const double c      = 0.0318;      /* m, c/R = 0.20 */
    const double Th_tip = 6.9 * M_PI / 180.0; /* rad */

    Bladestation st[N];
    for (int i = 0; i < N; ++i) {
        st[i].r         = hub_rh + (i + 0.5) * (R - hub_rh) / N;
        st[i].chord     = c;
        /* Ideal twist: Theta(r) = Theta_tip * (R / r) */
        st[i].twist     = Th_tip * (R / st[i].r);
        st[i].polar_idx = 0;
    }
    BladeGeometry geom = { st, N, 4, R, hub_rh };

    AirfoilPolar polar;  build_naca0012(&polar);

    /* --- Operating point: 5500 RPM, sea level --- */
    const double RPM = 5500.0;
    OperatingPoint op = {
        .rho = 1.225, .mu = 1.81e-5, .a_sound = 340.0,
        .V_inf = 0.0,
        .omega = RPM * 2.0 * M_PI / 60.0,
        .collective = 2.0 * M_PI / 180.0,   /* +2 deg, tune for T~11 N */
        .n_iter_max = 200, .tol = 1e-6, .relax = 0.4,
    };

    BEMResult bem;  bem_result_alloc(&bem, N);
    bem_solve(&geom, &op, &polar, 1, &bem);

    /* --- Observer at 12 R below hub --- */
    Observer obs = { 0.0, 0.0, -12.0 * R };

    NoiseResult nr;
    noise_predict(&geom, &op, &bem, &obs, &nr);

    double M_tip = op.omega * R / op.a_sound;
    double sigma = (double)geom.n_blades * c / (M_PI * R);

    printf("=== Pettingill reference rotor ===\n");
    printf("  R           = %.4f m\n", R);
    printf("  n_blades    = %d\n", geom.n_blades);
    printf("  chord       = %.4f m\n", c);
    printf("  solidity    = %.4f\n", sigma);
    printf("  RPM         = %.0f\n", RPM);
    printf("  omega       = %.2f rad/s\n", op.omega);
    printf("  M_tip       = %.4f\n", M_tip);
    printf("  thrust      = %.3f N\n", bem.thrust);
    printf("  torque      = %.5f N*m\n", bem.torque);
    printf("  observer r  = %.4f m (12 R)\n", 12.0 * R);
    printf("\n");
    printf("=== Model output at observer ===\n");
    printf("  SPL_tonal     = %+8.3f dB\n", nr.SPL_tonal);
    printf("  SPL_broadband = %+8.3f dB\n", nr.SPL_broadband);
    printf("  SPL_total     = %+8.3f dB\n", nr.SPL_total);
    printf("  OASPL_A       = %+8.3f dBA\n", nr.OASPL_A);
    printf("  BPF           = %8.2f Hz\n", nr.BPF_Hz);
    printf("\n");
    printf("=== Calibration offsets ===\n");
    printf("  Enter the reference OASPL from the paper (unweighted SPL):\n");
    printf("    K_BROADBAND_new = K_BROADBAND_old + (SPL_ref - %.3f)\n",
           nr.SPL_broadband);
    printf("    K_TONAL_new     = K_TONAL_old     + (SPL_ref - %.3f)\n",
           nr.SPL_tonal);

    bem_result_free(&bem);
    return 0;
}