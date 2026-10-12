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
    Bladestation st[N];
    for (int i = 0; i < N; ++i) {
        st[i].r         = 0.08 + (i + 0.5) * (0.50 - 0.08) / N;
        st[i].chord     = 0.05;
        double f        = ((double)i + 0.5) / N;
        st[i].twist     = (12.0 - 8.0 * f) * M_PI / 180.0;
        st[i].polar_idx = 0;
    }
    BladeGeometry geom = { st, N, 2, 0.50, 0.08 };

    AirfoilPolar polar;  build_naca0012(&polar);

    OperatingPoint op = {
        .rho = 1.225, .mu = 1.81e-5, .a_sound = 340.0,
        .V_inf = 0.0, .omega = 300.0, .collective = 0.0,
        .n_iter_max = 200, .tol = 1e-6, .relax = 0.4,
    };

    BEMResult bem;  bem_result_alloc(&bem, N);
    bem_solve(&geom, &op, &polar, 1, &bem);

    const double r_obs = 100.0;

    Observer observers[] = {
        { 0.0,     0.0,   -100.0 },   /* below hub, on thrust axis */
        { 0.0,     0.0,   +100.0 },   /* above hub, on thrust axis */
        { 70.71,   0.0,   -70.71 },   /* 45 deg below disk edge    */
        { 100.0,   0.0,    0.0   },   /* in rotor plane, +x        */
        { -100.0,  0.0,    0.0   },   /* in rotor plane, -x        */
    };
    const char* labels[] = {
        "below hub (axis)",
        "above hub (axis)",
        "45 deg below",
        "rotor plane +x",
        "rotor plane -x",
    };

    printf("Geometry:  R=%.2f m,  N=%d blades,  sigma=%.4f\n",
           geom.radius, geom.n_blades,
           (double)geom.n_blades * 0.05 / (M_PI * geom.radius));
    printf("Oper:      Omega=%.1f rad/s,  M_tip=%.3f,  T=%.2f N\n",
           op.omega, op.omega * geom.radius / op.a_sound, bem.thrust);
    printf("Observer:  r=%.0f m,  BPF=%.2f Hz\n\n", r_obs,
           (double)geom.n_blades * op.omega / (2.0 * M_PI));

    printf("%-20s  %8s  %10s  %10s  %10s\n",
           "position", "cos", "tonal", "broadband", "OASPL_A");
    printf("%-20s  %8s  %10s  %10s  %10s\n",
           "--------", "---", "-----", "---------", "-------");

    for (int k = 0; k < 5; ++k) {
        NoiseResult nr;
        noise_predict(&geom, &op, &bem, &observers[k], &nr);
        printf("%-20s  %+8.3f  %+8.2f dB  %+8.2f dB  %+8.2f dBA\n",
               labels[k], nr.directivity,
               nr.SPL_tonal, nr.SPL_broadband, nr.OASPL_A);
    }

    bem_result_free(&bem);
    return 0;
}