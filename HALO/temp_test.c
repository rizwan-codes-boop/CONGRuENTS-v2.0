#include <stdio.h>
#include "gsl_decs.h"

int main()
{
    /* =====================================================
       1D TEST
       Function: y = x^2
       ===================================================== */
    printf("=== 1D spline test ===\n");

    const size_t n = 5;
    double x[5] = {0, 1, 2, 3, 4};
    double y[5];

    for (size_t i = 0; i < n; i++)
        y[i] = x[i] * x[i];

    gsl_spline_object_1D *so1 =
        gsl_so1D(x, y, n, gsl_interp_cspline);

    if (!so1)
    {
        printf("1D spline creation failed\n");
        return 1;
    }

    double x_test = 2.5;
    double y_interp = gsl_so1D_eval(so1, x_test);

    printf("Interpolated y(%.2f) = %.6f (expected ~ %.2f)\n",
           x_test, y_interp, x_test * x_test);

    gsl_so1D_free(so1);



    /* =====================================================
       2D TEST
       Function: z = x + y
       Grid: 3x3
       ===================================================== */
    printf("\n=== 2D spline test ===\n");

    const size_t nx = 3;
    const size_t ny = 3;

    double x2[3] = {0, 1, 2};
    double y2[3] = {0, 1, 2};

    /* GSL expects z in row-major order:
       z[i*ny + j] = f(x[i], y[j])
    */
    double z[9];

    for (size_t i = 0; i < nx; i++)
        for (size_t j = 0; j < ny; j++)
            z[i*ny + j] = x2[i] + y2[j];

    gsl_spline_object_2D *so2 =
        gsl_so2D(x2, y2, z, nx, ny, gsl_interp2d_bilinear);

    if (!so2)
    {
        printf("2D spline creation failed\n");
        return 1;
    }

    double xt = 0.8;
    double yt = 1.3;
    double z_interp = gsl_so2D_eval(so2, xt, yt);

    printf("Interpolated z(%.2f, %.2f) = %.6f (expected ~ %.2f)\n",
           xt, yt, z_interp, xt + yt);

    gsl_so2D_free(so2);

    printf("\nAll tests completed.\n");

    return 0;
}
