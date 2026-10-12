/* ============================================================
 * bem.h — Blade Element Momentum solver for eVTOL rotors
 * ============================================================ */

 #ifndef ROTOR_BEM_H
 #define ROTOR_BEM_H

 #ifdef __cplusplus
 extern "C" {
# endif

/* ---- Airfoil polar ---------------------------------------- */
typedef struct {
    double alpha;   /* rad */
    double Cl;
    double Cd;
    double Cm;
} PolarPoint;

typedef struct {
    PolarPoint* points;     /* sorted ascending by alpha */
    int         n_points;
    double      Re;         /* Reynolds number this polar was measured at */
    char        name[64];
} AirfoilPolar;

int airfoil_lookup(const AirfoilPolar* polar,
                    double alpha,
                    double* Cl, double* Cd, double* Cm);

/* ---- Blade geometry ---------------------------------------- */
typedef struct {
    double r;       /* m, from hub center */
    double chord;   /* m */
    double twist;   /* rad, geometirc twist (nose-up positive) */
    int polar_idx;  /* index into polar table */
} Bladestation;

typedef struct{
    Bladestation* stations;     /* strictly increasing r */
    int         n_stations;
    int         n_blades;
    double      radius;         /* tip radius, m */
    double      hub_radius;     /* cutout, m */
}BladeGeometry;

/* ---- Operating point ---------------------------------------- */
typedef struct{
    double rho;             /* kg/m^3 */
    double mu;              /* Pa*s */
    double a_sound;         /* m/s */
    double V_inf;           /* m/s, 0 = hover */
    double omega;           /* rad/s */
    double collective;      /* rad, added to each station's twist */
    int n_iter_max;         /* e.g. 200 */
    double tol;             /* e.g. 1e-7 */
    double relax;           /* under-relaxation, e.g. 0.4 */
} OperatingPoint;

/* ---- Result ---------------------------------------- */
typedef struct{
    double thrust;          /* N */
    double torque;          /* N*m */
    double power;           /* W */
    double FM;              /* figure of merit (hover only) */
    double eta;             /* propulsive efficiency (forward flight) */
    double CT;              /* T / (rho A (Omega R)^2) */
    double CP;              /* P / (rho A (Omega R)^3) */
    int converged;          /* 1 if every station converged */
    int n_stations_failed;  /* count of stations that hit n_iter_max */

    /* per-station arrays, length = n_stations */
    double* r;
    double* dT_dr;
    double* dQ_dr;
    double* alpha;
    double* phi;
    double* Cl;
    double* Cd;
    double* F;
    double* v_i;
} BEMResult;

int bem_result_alloc(BEMResult* res, int n);
void bem_result_free(BEMResult* res);

int bem_solve(
    const BladeGeometry*    geom,
    const OperatingPoint*   op,
    const AirfoilPolar*     polars,
    int                     n_polars,
    BEMResult*              res);

#ifdef __cplusplus
 }
 #endif
 #endif     /* ROTOR_BEM_H*/