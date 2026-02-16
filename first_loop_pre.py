import numpy as np
from scipy.interpolate import RectBivariateSpline
import time


# ------------------------------------------------------------------
# Injection indices
# ------------------------------------------------------------------

q_p_inject = 2.2
q_e_inject = 2.2


# ------------------------------------------------------------------
# Energy Limits
# ------------------------------------------------------------------

m_e__GeV = 5.1099895e-4  # electron mass in GeV

E_gam_lims__GeV = np.array([1e-16, 1e8])
T_CR_lims__GeV = np.array([1e-3, 1e8])
E_CRe_lims__GeV = np.array([1e-3 + m_e__GeV, 1e8 + m_e__GeV])


# ------------------------------------------------------------------
# Physical Parameters
# ------------------------------------------------------------------

chi = 1e-4
M_A = 2.0
beta = 0.25

sigma_pp_cm2 = 40e-27
mu_H = 1.4
mu_p = 1.17

n_SN_Msolm1 = 1.321680e-2
f_EtoCR = 0.1
f_CRe_CRp = 0.2
E_SN_erg = 1e51
f_vAi = 1.0

T_p_cutoff__GeV = 1e8
T_e_cutoff__GeV = 1e5


# ------------------------------------------------------------------
# Load EBL Table
# ------------------------------------------------------------------

def load_EBL_tau(filename):

    with open(filename, "r") as f:
        lines = f.readlines()

    ny_z = int(lines[1])
    nx_E = int(lines[3])

    idx = 5

    ya_z = np.array(list(map(float, lines[idx].split())))
    idx += 2

    xa_E = np.array(list(map(float, lines[idx].split())))
    idx += 2

    za_tau = []
    while len(za_tau) < nx_E * ny_z:
        za_tau.extend(list(map(float, lines[idx].split())))
        idx += 1

    za_tau = np.array(za_tau).reshape((nx_E, ny_z))

    interp = RectBivariateSpline(xa_E, ya_z, za_tau)

    return xa_E, ya_z, za_tau, interp


# ------------------------------------------------------------------
# Load Galaxy Catalog
# ------------------------------------------------------------------

def load_galaxies(filename):

    with open(filename, "r") as f:
        f.readline()
        n_gal = int(f.readline())
        f.readline()

        data = np.loadtxt(f)

    z = data[:, 0]
    M_star__Msol = data[:, 1]
    Re__kpc = data[:, 2]
    SFR__Msolyrm1 = data[:, 3]

    return n_gal, z, M_star__Msol, Re__kpc, SFR__Msolyrm1


# ------------------------------------------------------------------
# Energy Grids
# ------------------------------------------------------------------

def create_energy_grids():

    n_T_CR = 1000
    n_E_gam = 500

    T_CR__GeV = np.logspace(
        np.log10(T_CR_lims__GeV[0]),
        np.log10(T_CR_lims__GeV[1]),
        n_T_CR
    )

    E_CRe__GeV = np.logspace(
        np.log10(E_CRe_lims__GeV[0]),
        np.log10(E_CRe_lims__GeV[1]),
        n_T_CR
    )

    E_gam__GeV = np.logspace(
        np.log10(E_gam_lims__GeV[0]),
        np.log10(E_gam_lims__GeV[1]),
        n_E_gam
    )

    return T_CR__GeV, E_CRe__GeV, E_gam__GeV


# ------------------------------------------------------------------
# Derived Galaxy Quantities
# ------------------------------------------------------------------

def compute_galaxy_properties(z,
                              M_star__Msol,
                              Re__kpc,
                              SFR__Msolyrm1,
                              Sigma_gas_func,
                              sigma_gas_func,
                              Tdust_func):

    A_Re__pc2 = np.pi * (Re__kpc * 1e3) ** 2

    Sig_star__Msolpcm2 = M_star__Msol / (2.0 * A_Re__pc2)

    Sig_SFR__Msolyrm1pcm2 = SFR__Msolyrm1 / (2.0 * A_Re__pc2)

    Sig_gas__Msolpcm2 = Sigma_gas_func(
        Sig_SFR__Msolyrm1pcm2,
        Sig_star__Msolpcm2
    )

    sig_gas__kmsm1 = sigma_gas_func(SFR__Msolyrm1)

    T_dust__K = Tdust_func(z, SFR__Msolyrm1, M_star__Msol)

    return (
        A_Re__pc2,
        Sig_star__Msolpcm2,
        Sig_SFR__Msolyrm1pcm2,
        Sig_gas__Msolpcm2,
        sig_gas__kmsm1,
        T_dust__K
    )


# ------------------------------------------------------------------
# Allocate Solver Arrays (C-Compatible)
# ------------------------------------------------------------------

def allocate_solver_arrays(n_gal, n_T_CR):

    f_cal = np.zeros((n_gal, n_T_CR), dtype=np.float64)
    D__cm2sm1 = np.zeros((n_gal, n_T_CR), dtype=np.float64)
    D_e__cm2sm1 = np.zeros((n_gal, n_T_CR), dtype=np.float64)
    D_e_z2__cm2sm1 = np.zeros((n_gal, n_T_CR), dtype=np.float64)

    return f_cal, D__cm2sm1, D_e__cm2sm1, D_e_z2__cm2sm1


# ------------------------------------------------------------------
# Main Initialization Driver
# ------------------------------------------------------------------

def initialise_model(galaxy_file, ebl_file,
                     Sigma_gas_func,
                     sigma_gas_func,
                     Tdust_func):

    start = time.time()

    # Load galaxies
    n_gal, z, M_star__Msol, Re__kpc, SFR__Msolyrm1 = load_galaxies(galaxy_file)

    # Load EBL
    xa_E, ya_z, za_tau, ebl_interp = load_EBL_tau(ebl_file)

    # Energy grids
    T_CR__GeV, E_CRe__GeV, E_gam__GeV = create_energy_grids()

    n_T_CR = len(T_CR__GeV)

    # Galaxy derived quantities
    props = compute_galaxy_properties(
        z,
        M_star__Msol,
        Re__kpc,
        SFR__Msolyrm1,
        Sigma_gas_func,
        sigma_gas_func,
        Tdust_func
    )

    # Allocate solver arrays
    arrays = allocate_solver_arrays(n_gal, n_T_CR)

    end = time.time()

    print(f"Initialization completed in {end - start:.2f} s")

    return {
        "n_gal": n_gal,
        "z": z,
        "M_star__Msol": M_star__Msol,
        "Re__kpc": Re__kpc,
        "SFR__Msolyrm1": SFR__Msolyrm1,
        "T_CR__GeV": T_CR__GeV,
        "E_CRe__GeV": E_CRe__GeV,
        "E_gam__GeV": E_gam__GeV,
        "ebl_interp": ebl_interp,
        "galaxy_props": props,
        "solver_arrays": arrays
    }
