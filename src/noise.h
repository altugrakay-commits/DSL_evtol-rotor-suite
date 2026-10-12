/* ============================================================
 * noise.h — semi-empirical rotor noise model
 * ============================================================ */
#ifndef ROTOR_NOISE_H
#define ROTOR_NOISE_H

#include "bem.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Observer position, meters.  Rotor hub at origin, rotor in
 * the x-y plane, thrust along +z. */
typedef struct {
    double x, y, z;
} Observer;

typedef struct {
    double SPL_tonal;      /* dB (re 20 uPa) at BPF, unweighted */
    double SPL_broadband;  /* dB, OASPL of broadband, unweighted */
    double SPL_total;      /* dB, log-sum of tonal + broadband */
    double OASPL;          /* dB, unweighted overall */
    double OASPL_A;        /* dBA, A-weighted overall */
    double BPF_Hz;         /* blade-passing frequency */
    double directivity;    /* cos(theta) of observer wrt thrust axis */
} NoiseResult;

int noise_predict(const BladeGeometry*  geom,
                  const OperatingPoint* op,
                  const BEMResult*      bem,
                  const Observer*       obs,
                  NoiseResult*          out);

#ifdef __cplusplus
}
#endif
#endif /* ROTOR_NOISE_H */