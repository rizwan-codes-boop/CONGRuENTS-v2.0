#ifndef GSL_DECS_H
#define GSL_DECS_H

#include <stdlib.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_interp_accel.h>

/* ============================================================
   1D SPLINE OBJECT
   ============================================================ */

typedef struct
{
    gsl_spline *spline;
    gsl_interp_accel *acc;
    size_t n;

} gsl_spline_object_1D;


/* Create 1D spline object */
static inline gsl_spline_object_1D*
gsl_so1D(const double *x,
         const double *y,
         size_t n,
         const gsl_interp_type *type)
{
    gsl_spline_object_1D *so =
        (gsl_spline_object_1D*) malloc(sizeof(gsl_spline_object_1D));

    if (!so) return NULL;

    so->n = n;
    so->acc = gsl_interp_accel_alloc();
    so->spline = gsl_spline_alloc(type, n);

    if (!so->acc || !so->spline)
    {
        free(so);
        return NULL;
    }

    gsl_spline_init(so->spline, x, y, n);

    return so;
}


/* Evaluate 1D spline */
static inline double
gsl_so1D_eval(gsl_spline_object_1D *so, double x)
{
    return gsl_spline_eval(so->spline, x, so->acc);
}


/* Free 1D spline object */
static inline void
gsl_so1D_free(gsl_spline_object_1D *so)
{
    if (!so) return;

    if (so->spline) gsl_spline_free(so->spline);
    if (so->acc) gsl_interp_accel_free(so->acc);

    free(so);
}


/* ============================================================
   2D SPLINE OBJECT
   ============================================================ */

typedef struct
{
    gsl_spline2d *spline;
    gsl_interp_accel *xacc;
    gsl_interp_accel *yacc;
    size_t nx;
    size_t ny;

} gsl_spline_object_2D;


/* Create 2D spline object */
static inline gsl_spline_object_2D*
gsl_so2D(const double *x,
         const double *y,
         const double *z,
         size_t nx,
         size_t ny,
         const gsl_interp2d_type *type)
{
    gsl_spline_object_2D *so =
        (gsl_spline_object_2D*) malloc(sizeof(gsl_spline_object_2D));

    if (!so) return NULL;

    so->nx = nx;
    so->ny = ny;

    so->xacc = gsl_interp_accel_alloc();
    so->yacc = gsl_interp_accel_alloc();
    so->spline = gsl_spline2d_alloc(type, nx, ny);

    if (!so->xacc || !so->yacc || !so->spline)
    {
        free(so);
        return NULL;
    }

    gsl_spline2d_init(so->spline, x, y, z, nx, ny);

    return so;
}


/* Evaluate 2D spline */
static inline double
gsl_so2D_eval(gsl_spline_object_2D *so, double x, double y)
{
    return gsl_spline2d_eval(so->spline, x, y, so->xacc, so->yacc);
}


/* Free 2D spline object */
static inline void
gsl_so2D_free(gsl_spline_object_2D *so)
{
    if (!so) return;

    if (so->spline) gsl_spline2d_free(so->spline);
    if (so->xacc) gsl_interp_accel_free(so->xacc);
    if (so->yacc) gsl_interp_accel_free(so->yacc);

    free(so);
}

#endif /* GSL_DECS_H */
