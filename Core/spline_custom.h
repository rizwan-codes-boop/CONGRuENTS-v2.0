#ifndef SPLINE_CUSTOM_H
#define SPLINE_CUSTOM_H

#include <stdlib.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_spline2d.h>

/* =========================
   1D SPLINE OBJECT
   ========================= */

typedef struct
{
    gsl_spline *spline;
    gsl_interp_accel *acc;
    size_t n;
} spline1D;


/* Constructors */
spline1D spline1D_create(size_t n, double *x, double *y);

/* Evaluation */
double spline1D_eval(spline1D *sp, double x);

/* Free memory */
void spline1D_free(spline1D *sp);



/* =========================
   2D SPLINE OBJECT
   ========================= */

typedef struct
{
    gsl_spline2d *spline;
    gsl_interp_accel *xacc;
    gsl_interp_accel *yacc;
    size_t nx;
    size_t ny;
} spline2D;


/* Constructors */
spline2D spline2D_create(size_t nx, size_t ny,
                         double *x,
                         double *y,
                         double *z);

/* Evaluation */
double spline2D_eval(spline2D *sp,
                     double x,
                     double y);

/* Free memory */
void spline2D_free(spline2D *sp);

#endif
