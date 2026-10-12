/* ============================================================
 * structure.h — 1-D beam structural model for rotor blades
 * ============================================================ */
#ifndef ROTOR_STRUCTURE_H
#define ROTOR_STRUCTURE_H

#include "bem.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double E;              /* Young's modulus, Pa */
    double rho;            /* density, kg/m^3 */
    double sigma_yield;    /* yield / ultimate tensile strength, Pa */
    double t_over_c;       /* thickness-to-chord ratio, e.g. 0.12 */
    char   name[64];
} BladeMaterial;

typedef struct {
    double T_centrifugal;   /* N, at this station */
    double M_flap;          /* N*m, at this station */
    double sigma_axial;     /* Pa, from centrifugal tension */
    double sigma_bending;   /* Pa, from flapwise bending */
    double sigma_total;     /* Pa, axial + bending (tension side) */
    double margin;          /* sigma_yield / sigma_total */
} StationStress;

typedef struct {
    double max_sigma;              /* Pa, max over span */
    double max_sigma_station_r;    /* m, location of max stress */
    double min_margin;             /* sigma_yield/sigma_max */
    double T_root;                 /* N, centrifugal tension at innermost station */
    double M_root;                 /* N*m, flap moment at innermost station */
    double omega_flap1;            /* rad/s, first flap natural frequency */
    double omega_flap1_ratio;      /* omega_flap1 / Omega */
    StationStress* stations;       /* length = n_stations */
} StructureResult;

int  structure_alloc(StructureResult* res, int n);
void structure_free(StructureResult* res);

int structure_solve(const BladeGeometry*  geom,
                    const OperatingPoint* op,
                    const BEMResult*      bem,
                    const BladeMaterial*  mat,
                    StructureResult*      out);

#ifdef __cplusplus
}
#endif
#endif /* ROTOR_STRUCTURE_H */