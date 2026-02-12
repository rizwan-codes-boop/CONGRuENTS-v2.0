#include "spline_custom.h"
#include <stdio.h>

/* =========================
   1D SPLINE IMPLEMENTATION
   ========================= */

spline1D spline1D_create(size_t n, double *x, double *y)
{
    spline1D sp;

    sp.n = n;
    sp.acc = gsl_interp_accel_alloc();
    sp.spline = gsl_spline_alloc(gsl_interp_cspline, n);

    gsl_spline_init(sp.spline, x, y, n);

    return sp;
}


double spline1D_eval(spline1D *sp, double x)
{
    return gsl_spline_eval(sp->spline, x, sp->acc);
}


void spline1D_free(spline1D *sp)
{
    gsl_spline_free(sp->spline);
    gsl_interp_accel_free(sp->acc);
}



/* =========================
   2D SPLINE IMPLEMENTATION
   ========================= */

spline2D spline2D_create(size_t nx, size_t ny,
                         double *x,
                         double *y,
                         double *z)
{
    spline2D sp;

    sp.nx = nx;
    sp.ny = ny;

    sp.xacc = gsl_interp_accel_alloc();
    sp.yacc = gsl_interp_accel_alloc();

    sp.spline = gsl_spline2d_alloc(gsl_interp2d_bicubic, nx, ny);

    gsl_spline2d_init(sp.spline, x, y, z, nx, ny);

    return sp;
}


double spline2D_eval(spline2D *sp,
                     double x,
                     double y)
{
    return gsl_spline2d_eval(sp->spline, x, y,
                             sp->xacc, sp->yacc);
}


void spline2D_free(spline2D *sp)
{
    gsl_spline2d_free(sp->spline);
    gsl_interp_accel_free(sp->xacc);
    gsl_interp_accel_free(sp->yacc);
}
