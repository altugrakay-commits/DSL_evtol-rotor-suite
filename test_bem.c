#include <bem.h>
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
        /* Midpoint placement: stations striclty inside (rh, R). */
        st[i].r         = 0.08 + (i + 0.5) * (0.50 - 0.08) / N;
        st[i].chord     = 0.05;
        /* Twist fades with radisu position, not station index */
        double f = (double)i / (N - 1);
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

    BEMResult res;
    bem_result_alloc(&res, N);
    int rc = bem_solve(&geom, &op, &polar, 1, &res);

    printf("rc=%d  converged=%d  n_stations_failed=%d\n",
           rc, res.converged, res.n_stations_failed);
    printf("T  = %8.3f N\n",  res.thrust);
    printf("Q  = %8.4f N*m\n", res.torque);
    printf("P  = %8.2f W\n",  res.power);
    printf("CT = %8.5f\n",     res.CT);
    printf("CP = %8.5f\n",     res.CP);
    printf("FM = %8.4f\n",     res.FM);

    printf("\n  i     r[m]    phi[deg]  alpha[deg]    Cl       Cd       F\n");
    for (int i = 0; i < N; ++i) {
        printf(" %2d   %6.4f   %8.4f   %8.4f   %6.3f   %6.4f   %6.3f\n",
               i,
               res.r[i],
               res.phi[i]   * 180.0 / M_PI,
               res.alpha[i] * 180.0 / M_PI,
               res.Cl[i],
               res.Cd[i],
               res.F[i]);
    }

    bem_result_free(&res);
    return 0;
}