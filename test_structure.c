#include <structure.h>
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

    /* Carbon-fibre / epoxy laminate */
    BladeMaterial mat = {
        .E           = 70.0e9,     /* Pa */
        .rho         = 1600.0,     /* kg/m^3 */
        .sigma_yield = 600.0e6,    /* Pa */
        .t_over_c    = 0.12,
    };
    snprintf(mat.name, sizeof(mat.name), "CFRP-typical");

    StructureResult str;
    structure_alloc(&str, N);
    structure_solve(&geom, &op, &bem, &mat, &str);

    printf("Material: %s  E=%.1f GPa  rho=%.0f kg/m3  sig_y=%.0f MPa  t/c=%.2f\n",
           mat.name, mat.E/1e9, mat.rho, mat.sigma_yield/1e6, mat.t_over_c);
    printf("\n");
    printf("T_root          = %8.1f N\n",   str.T_root);
    printf("M_root          = %8.3f N*m\n", str.M_root);
    printf("max sigma       = %8.2f MPa  at r = %.4f m\n",
           str.max_sigma/1e6, str.max_sigma_station_r);
    printf("min margin      = %8.2f  (sigma_y / sigma_max)\n", str.min_margin);
    printf("omega_flap1     = %8.1f rad/s\n", str.omega_flap1);
    printf("omega_flap1 / Omega = %.3f\n",   str.omega_flap1_ratio);
    printf("BPF / Omega     = %.3f\n",        (double)geom.n_blades);

    printf("\n  i     r[m]    T[N]      M[N*m]   sig_a[MPa]  sig_b[MPa]  sig[MPa]  margin\n");
    for (int i = 0; i < N; ++i) {
        printf(" %2d  %6.4f  %8.1f  %8.3f   %9.2f   %9.2f   %8.2f   %6.2f\n",
               i,
               geom.stations[i].r,
               str.stations[i].T_centrifugal,
               str.stations[i].M_flap,
               str.stations[i].sigma_axial   / 1e6,
               str.stations[i].sigma_bending / 1e6,
               str.stations[i].sigma_total   / 1e6,
               str.stations[i].margin);
    }

    structure_free(&str);
    bem_result_free(&bem);
    return 0;
}