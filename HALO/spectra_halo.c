/*
 * ============================================================================
 * SPECTRA_HALO.C - Computational Core for Halo CR Physics
 * ============================================================================
 * 
 * This is the C computational kernel that performs heavy numerical calculations
 * for cosmic-ray transport and emission in galaxy halos. It is designed to be
 * called from Python via ctypes as a shared library (.so file).
 * 
 * ARCHITECTURE:
 * -------------
 * Python (halo_init.py) → C shared library (spectra_halo.so) → Python (results.py)
 *     ↓                          ↓                                    ↓
 * Setup/IO              Numerical computation                  Analysis/Plotting
 * 
 * COMPILATION:
 * ------------
 * Use the provided Makefile:
 *   $ make
 * 
 * This produces: spectra_halo.so
 * 
 * PYTHON INTERFACE:
 * -----------------
 * All functions are designed to accept C-compatible data types:
 *   - double* for arrays (passed as numpy array pointers)
 *   - int/unsigned int for sizes
 *   - Returns 0 on success, -1 on error
 * 
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Physical constants (CGS units) */
#define c__cmsm1 2.99792458e10      // Speed of light [cm/s]
#define m_e__GeV 0.000510998928     // Electron mass [GeV]
#define m_p__GeV 0.938272046        // Proton mass [GeV]
#define m_H__kg 1.6737236e-27       // Hydrogen mass [kg]
#define pc__cm 3.0856775814671916e18  // Parsec [cm]
#define Msol__kg 1.98892e30         // Solar mass [kg]
#define yr__s 31557600.0            // Year [s]
#define e__esu 4.80320425e-10       // Elementary charge [esu]
#define sigma_T__cm2 6.6524587158e-25  // Thomson cross-section [cm²]
#define M_PI 3.14159265358979323846

/* Halo scaling parameters */
#define HALO_DENSITY_FACTOR 1000.0
#define HALO_HEIGHT_FACTOR 50.0
#define HALO_B_FACTOR_SF 3.0
#define HALO_B_FACTOR_Q 1.5


/* ============================================================================
 * FUNCTION: compute_halo_properties
 * ============================================================================
 */
int compute_halo_properties(
    unsigned long n_gal,
    const double *n_H_disc,
    const double *B_disc,
    const double *h_disc,
    const double *SFR,
    const double *M_star,
    double *n_H_halo,
    double *B_halo,
    double *h_halo)
{
    #pragma omp parallel for
    for (unsigned long i = 0; i < n_gal; i++) {
        
        /* Halo density (1000× lower) */
        n_H_halo[i] = n_H_disc[i] / HALO_DENSITY_FACTOR;
        
        /* Halo magnetic field (depends on galaxy type) */
        double sSFR = log10(SFR[i] / M_star[i]);
        if (sSFR > -10.0) {
            B_halo[i] = B_disc[i] / HALO_B_FACTOR_SF;
        } else {
            B_halo[i] = B_disc[i] / HALO_B_FACTOR_Q;
        }
        
        /* Halo scale height (50× larger) */
        h_halo[i] = HALO_HEIGHT_FACTOR * h_disc[i];
    }
    
    return 0;
}


/* ============================================================================
 * FUNCTION: compute_halo_diffusion
 * ============================================================================
 */
int compute_halo_diffusion(
    unsigned long n_gal,
    unsigned int n_E,
    const double *T_CR,
    const double *h_disc,
    const double *n_H_disc,
    const double *sig_gas,
    const double *f_cal,
    const double *C,
    double chi,
    double M_A,
    double q_inject,
    double *D_e_halo)
{
    #pragma omp parallel for
    for (unsigned long i = 0; i < n_gal; i++) {
        
        double u_LA = sig_gas[i] / sqrt(2.0);
        double v_Ai = 1000.0 * (u_LA / 10.0) / (sqrt(chi / 1e-4) * M_A);
        double L_A = h_disc[i] / pow(M_A, 3);
        
        for (unsigned int j = 0; j < n_E; j++) {
            
            double p_e = sqrt(T_CR[j] * T_CR[j] + 2.0 * m_e__GeV * T_CR[j]);
            
            double v_st_halo = fmin(
                v_Ai * (1.0 + 2.3e-3
                    * pow(p_e, q_inject - 1.0)
                    * pow(n_H_disc[i] / 1e6, 1.5)
                    * (1.0 / 1e-4) * M_A
                    / (u_LA / 10.0 * ((1.0 - f_cal[i]) * C[i]) / 2e-7)),
                c__cmsm1 / 1e5);
            
            D_e_halo[i * n_E + j] = v_st_halo * L_A * 1e5 * pc__cm;
        }
    }
    
    return 0;
}


/* ============================================================================
 * FUNCTION: compute_halo_injection
 * ============================================================================
 */
int compute_halo_injection(
    unsigned long n_gal,
    unsigned int n_E,
    const double *h_disc,
    const double *q_e_disc,
    const double *D_e_disc,
    double *Q_e_halo)
{
    #pragma omp parallel for
    for (unsigned long i = 0; i < n_gal; i++) {
        
        double h_sq = pow(h_disc[i] * pc__cm, 2);
        
        for (unsigned int j = 0; j < n_E; j++) {
            
            unsigned long idx = i * n_E + j;
            double tau_diff = h_sq / D_e_disc[idx];
            Q_e_halo[idx] = q_e_disc[idx] / tau_diff;
        }
    }
    
    return 0;
}


/* ============================================================================
 * FUNCTION: compute_synchrotron_loss_time
 * ============================================================================
 */
int compute_synchrotron_loss_time(
    unsigned long n_gal,
    unsigned int n_E,
    const double *E_CRe,
    const double *B_halo,
    double *tau_sync)
{
    #pragma omp parallel for
    for (unsigned long i = 0; i < n_gal; i++) {
        
        double u_B = pow(B_halo[i], 2) / (8.0 * M_PI);
        
        for (unsigned int j = 0; j < n_E; j++) {
            
            double E = E_CRe[j];
            double gamma = E / m_e__GeV;
            
            double dEdt = (4.0 / 3.0) * sigma_T__cm2 * c__cmsm1 
                         * u_B * pow(gamma, 2) * m_e__GeV / 1.60218e-3;
            
            tau_sync[i * n_E + j] = E / dEdt;
        }
    }
    
    return 0;
}


/* ============================================================================
 * FUNCTION: compute_plasma_loss_time
 * ============================================================================
 */
int compute_plasma_loss_time(
    unsigned long n_gal,
    unsigned int n_E,
    const double *E_CRe,
    const double *n_H_halo,
    double *tau_plasma)
{
    const double mu_ISM = 1.1;
    const double h__GeVs = 4.135667696e-24;
    const double m_e_g = 9.10938356e-28;
    
    #pragma omp parallel for
    for (unsigned long i = 0; i < n_gal; i++) {
        
        double nu_p = sqrt(n_H_halo[i] * mu_ISM / (M_PI * m_e_g)) * e__esu;
        
        for (unsigned int j = 0; j < n_E; j++) {
            
            double E = E_CRe[j];
            double gamma = E / m_e__GeV;
            
            double dEdt = (3.0 / 4.0) * sigma_T__cm2 * c__cmsm1 
                         * m_e__GeV * n_H_halo[i] * mu_ISM
                         * (log(gamma) + 2.0 * log(m_e__GeV / (h__GeVs * nu_p)));
            
            tau_plasma[i * n_E + j] = E / fabs(dEdt);
        }
    }
    
    return 0;
}


/* ============================================================================
 * FUNCTION: compute_diffusion_loss_time
 * ============================================================================
 */
int compute_diffusion_loss_time(
    unsigned long n_gal,
    unsigned int n_E,
    const double *h_halo,
    const double *D_e_halo,
    double *tau_diff)
{
    #pragma omp parallel for
    for (unsigned long i = 0; i < n_gal; i++) {
        
        double h_sq = pow(h_halo[i] * pc__cm, 2);
        
        for (unsigned int j = 0; j < n_E; j++) {
            tau_diff[i * n_E + j] = h_sq / D_e_halo[i * n_E + j];
        }
    }
    
    return 0;
}


/* ============================================================================
 * FUNCTION: solve_halo_steady_state
 * ============================================================================
 */
int solve_halo_steady_state(
    unsigned long n_gal,
    unsigned int n_E,
    const double *Q_e_halo,
    const double *tau_sync,
    const double *tau_plasma,
    const double *tau_diff,
    const double *tau_IC,
    double *N_e_halo)
{
    #pragma omp parallel for
    for (unsigned long i = 0; i < n_gal; i++) {
        for (unsigned int j = 0; j < n_E; j++) {
            
            unsigned long idx = i * n_E + j;
            
            double tau_eff_inv = 1.0 / tau_sync[idx]
                               + 1.0 / tau_plasma[idx]
                               + 1.0 / tau_diff[idx];
            
            if (tau_IC != NULL && tau_IC[idx] > 0) {
                tau_eff_inv += 1.0 / tau_IC[idx];
            }
            
            double tau_eff = 1.0 / tau_eff_inv;
            N_e_halo[idx] = Q_e_halo[idx] * tau_eff;
        }
    }
    
    return 0;
}


/* ============================================================================
 * FUNCTION: compute_energy_losses_at_energy
 * ============================================================================
 */
int compute_energy_losses_at_energy(
    unsigned long n_gal,
    double E_crit,
    const double *B_halo,
    const double *n_H_halo,
    const double *h_halo,
    const double *D_e_at_E,
    double *loss_rates)
{
    const double mu_ISM = 1.1;
    const double h__GeVs = 4.135667696e-24;
    const double m_e_g = 9.10938356e-28;
    
    #pragma omp parallel for
    for (unsigned long i = 0; i < n_gal; i++) {
        
        double gamma = E_crit / m_e__GeV;
        
        double u_B = pow(B_halo[i], 2) / (8.0 * M_PI);
        double dEdt_sync = (4.0 / 3.0) * sigma_T__cm2 * c__cmsm1 
                          * u_B * pow(gamma, 2) * m_e__GeV / 1.60218e-3;
        
        double nu_p = sqrt(n_H_halo[i] * mu_ISM / (M_PI * m_e_g)) * e__esu;
        double dEdt_plasma = (3.0 / 4.0) * sigma_T__cm2 * c__cmsm1 
                            * m_e__GeV * n_H_halo[i] * mu_ISM
                            * (log(gamma) + 2.0 * log(m_e__GeV / (h__GeVs * nu_p)));
        
        double tau_diff = pow(h_halo[i] * pc__cm, 2) / D_e_at_E[i];
        double dEdt_diff = E_crit / tau_diff;
        
        loss_rates[i * 4 + 0] = dEdt_sync;
        loss_rates[i * 4 + 1] = fabs(dEdt_plasma);
        loss_rates[i * 4 + 2] = dEdt_diff;
        loss_rates[i * 4 + 3] = dEdt_sync + fabs(dEdt_plasma) + dEdt_diff;
    }
    
    return 0;
}


/* ============================================================================
 * FUNCTION: integrate_spectrum
 * ============================================================================
 */
int integrate_spectrum(
    unsigned long n_gal,
    unsigned int n_E,
    const double *E,
    const double *spec,
    double *integrals)
{
    #pragma omp parallel for
    for (unsigned long i = 0; i < n_gal; i++) {
        
        double sum = 0.0;
        
        for (unsigned int j = 0; j < n_E - 1; j++) {
            
            unsigned long idx1 = i * n_E + j;
            unsigned long idx2 = i * n_E + j + 1;
            
            double dlogE = log(E[j + 1]) - log(E[j]);
            double avg = 0.5 * (E[j] * spec[idx1] + E[j + 1] * spec[idx2]);
            
            sum += avg * dlogE;
        }
        
        integrals[i] = sum;
    }
    
    return 0;
}


/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================
 */

void set_num_threads(int n_threads)
{
    omp_set_num_threads(n_threads);
}

int get_num_threads(void)
{
    return omp_get_max_threads();
}

void print_library_info(void)
{
    printf("========================================\n");
    printf("Halo CR Spectra C Library\n");
    printf("========================================\n");
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    printf("========================================\n");
}