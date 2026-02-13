import numpy as np
from scipy.interpolate import CubicSpline


# =========================
# HALO MASS OBJECT
# =========================

class HaloMassObj:

    def __init__(self):

        z = np.array([
            0., 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50,
            1.75, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 10.00
        ])

        logM1 = np.array([
            12.56, 12.58, 12.61, 12.68, 12.77, 12.89, 13.01, 13.15,
            13.33, 13.51, 14.02, 14.97, 14.86, 17.43, 17.27, 16.79
        ])

        logMstar0 = np.array([
            10.88, 10.90, 10.93, 10.99, 11.08, 11.19, 11.31, 11.47,
            11.73, 12.14, 12.73, 14.31, 14.52, 16.69, 20.24, 21.89
        ])

        beta = np.array([
            0.48, 0.48, 0.48, 0.48, 0.50, 0.51, 0.53, 0.54,
            0.55, 0.55, 0.59, 0.60, 0.58, 0.55, 0.52, 0.43
        ])

        delta = np.array([
            0.30, 0.29 , 0.27, 0.23, 0.18, 0.12, 0.03, -0.10,
            -0.34, -0.44, -0.44, -0.44, -0.44, -0.44, -0.44, -0.44
        ])

        gamma = np.array([
            1.56, 1.52, 1.46, 1.39, 1.33, 1.27, 1.22, 1.17,
            1.16, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92
        ])

        # Create splines
        self.logM1_spline = CubicSpline(z, logM1)
        self.logMstar0_spline = CubicSpline(z, logMstar0)
        self.beta_spline = CubicSpline(z, beta)
        self.delta_spline = CubicSpline(z, delta)
        self.gamma_spline = CubicSpline(z, gamma)


# =========================
# HALO MASS FUNCTION
# =========================

def halo_mass_Msol(hm_obj, M_star_Msol, z):

    logM1 = hm_obj.logM1_spline(z)

    Mstar0 = 10 ** hm_obj.logMstar0_spline(z)

    beta  = hm_obj.beta_spline(z)
    delta = hm_obj.delta_spline(z)
    gamma = hm_obj.gamma_spline(z)

    ratio = M_star_Msol / Mstar0

    return 10 ** (
        logM1 +
        beta * np.log10(ratio) +
        ratio**delta / (1. + ratio**(-gamma))
        - 0.5
    )


# =========================
# HALF MASS RADIUS
# =========================

def R_half_mass_kpc(R_e_kpc, z):

    if z < 1.:
        return 0.7 * R_e_kpc

    elif z < 1.5:
        s = -0.325
        b = -0.180

    elif z < 2.0:
        s = -0.100
        b = -0.081

    else:
        s = -0.034
        b = -0.029

    return R_e_kpc * 10 ** (s * (np.log10(R_e_kpc) - 1.) + b)


# =========================
# VIRIAL RADIUS
# =========================

def R_vir_kpc(R_half_mass_kpc):
    return 35. * (18.87 * R_half_mass_kpc) ** 1.07
