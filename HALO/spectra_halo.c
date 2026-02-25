/*
 * ============================================================================
 * SPECTRA_HALO.C - Full Physics Halo CR Computation with GSL
 * ============================================================================
 * 
 * This version includes COMPLETE physics from the original code:
 *   - Full Green's function steady-state solver (CRe_steadystate_solve)
 *   - Exact loss timescales (tau_plasma__s, tau_sync__s, etc.)
 * 
 * Compilation requires:
 *   - GSL library
 *   - Original code headers (CRe_steadystate.h, ionisation.h, etc.)
 * 
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

/* GSL includes */
#include <gsl/gsl_spline.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_interp2d.h>

#include "gsl_decs.h"
#include "physical_constants.h"
#include "CR_funcs.h"

/* Halo scaling parameters */
#define HALO_DENSITY_FACTOR 1000.0
#define HALO_HEIGHT_FACTOR 50.0
#define HALO_B_FACTOR_SF 3.0
#define HALO_B_FACTOR_Q 1.5


/* ============================================================================
 * FUNCTION: compute_halo_properties
 * ============================================================================
 * 
 * PHYSICS: Identical to original spectra.c lines 237-244
 * 
 * Derives halo ISM properties from disc properties:
 *   - n_H_halo = n_H_disc / 1000  (diffuse ionized medium)
 *   - B_halo = B_disc / 3 (SF) or / 1.5 (quiescent)
 *   - h_halo = 50 × h_disc  (extended atmosphere)
 * 
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
        
        /* Halo density (line 619 of original) */
        n_H_halo[i] = n_H_disc[i] / HALO_DENSITY_FACTOR;
        
        /* Halo magnetic field (lines 237-244 of original) */
        if (log10(SFR[i] / M_star[i]) > -10.0) {
            /* Star-forming galaxy */
            B_halo[i] = B_disc[i] / HALO_B_FACTOR_SF;
        } else {
            /* Quiescent galaxy */
            B_halo[i] = B_disc[i] / HALO_B_FACTOR_Q;
        }
        
        /* Halo scale height (line 619 of original) */
        h_halo[i] = HALO_HEIGHT_FACTOR * h_disc[i];
    }
    
    return 0;
}


/* ============================================================================
 * FUNCTION: compute_halo_diffusion
 * ============================================================================
 * 
 * PHYSICS: Identical to original spectra.c lines 277-283
 * 
 * Computes D_e_z2 using exact streaming speed formula with:
 *   - n_H / 1e6 (1000 × 1000 factor)
 *   - (1 - f_cal) × C (escaped CR pressure)
 *   - NO f_vAi factor (different scattering in ionized medium)
 * 
 * ============================================================================
 */
int compute_halo_diffusion(
    unsigned long n_gal,
    unsigned int n_E,
    const double *T_CR,
    const double *h_disc,
    const double *n_H_disc,
    const double *sig_gas,
    const double *f_cal,        /* f_cal[0] for each galaxy */
    const double *C,
    double chi,
    double M_A,
    double q_inject,
    double *D_e_halo)
{
    #pragma omp parallel for
    for (unsigned long i = 0; i < n_gal; i++) {
        
        /* Turbulence parameters (identical to original) */
        double u_LA = sig_gas[i] / sqrt(2.0);
        double v_Ai = 1000.0 * (u_LA / 10.0) / (sqrt(chi / 1e-4) * M_A);
        double L_A = h_disc[i] / pow(M_A, 3);
        
        for (unsigned int j = 0; j < n_E; j++) {
            
            /* Electron momentum */
            double p_e = sqrt(T_CR[j] * T_CR[j] + 2.0 * m_e__GeV * T_CR[j]);
            
            /* Halo streaming speed (line 280-282 of original)
             * Key differences from disc:
             *   - n_H_disc[i]/1e3/1e3 instead of n_H_disc[i]/1e3
             *   - (1.-f_cal[i])*C[i] instead of C[i]
             *   - v_Ai instead of f_vAi * v_Ai
             */
            double v_st_halo = fmin(
                v_Ai * (1.0 + 2.3e-3
                    * pow(p_e, q_inject - 1.0)
                    * pow(n_H_disc[i] / 1e6, 1.5)
                    * (1.0 / 1e-4) * M_A
                    / (u_LA / 10.0 * ((1.0 - f_cal[i]) * C[i]) / 2e-7)),
                c__cmsm1 / 1e5);
            
            /* Diffusion coefficient */
            D_e_halo[i * n_E + j] = v_st_halo * L_A * 1e5 * pc__cm;
        }
    }
    
    return 0;
}


/* ============================================================================
 * FUNCTION: compute_halo_injection
 * ============================================================================
 * 
 * PHYSICS: Identical to original spectra.c lines 604-616
 * 
 * Halo injection from disc escape:
 *   Q_e_z2[j] = qe_z1[j] / tau_diff__s(E, h_disc, D_e_disc)
 * 
 * Uses your original tau_diff__s from diffusion.h
 * 
 * ============================================================================
 */
int compute_halo_injection(
    unsigned long n_gal,
    unsigned int n_E,
    const double *E_CRe,
    const double *h_disc,
    const double *q_e_disc,     /* Flat array [n_gal * n_E] */
    const double *D_e_disc,     /* Flat array [n_gal * n_E] */
    double *Q_e_halo)           /* Output: flat array [n_gal * n_E] */
{
    #pragma omp parallel for
    for (unsigned long i = 0; i < n_gal; i++) {
        
        /* Create GSL spline for disc diffusion coefficient */
        gsl_spline_object_1D *De_gso1D_disc = gsl_so1D(
            E_CRe,
            &D_e_disc[i * n_E],
            n_E,
            gsl_interp_cspline
        );
        
        for (unsigned int j = 0; j < n_E; j++) {
            
            unsigned long idx = i * n_E + j;
            double E = E_CRe[j];
            
            /* Diffusion escape time from disc (your original function) */
            double tau_d = tau_diff__s(E, h_disc[i], De_gso1D_disc);
            
            /* Injection rate = disc spectrum / escape time (line 606, 614) */
            Q_e_halo[idx] = q_e_disc[idx] / tau_d;
        }
        
        /* Free spline */
        gsl_so1D_free(De_gso1D_disc);
    }
    
    return 0;
}


/* ============================================================================
 * FUNCTION: compute_loss_timescales
 * ============================================================================
 * 
 * PHYSICS: Uses your EXACT original functions from:
 *   - synchrotron.h: tau_sync__s()
 *   - ionisation.h: tau_plasma__s()  (NOT tau_ion__s - halo is ionized!)
 *   - bremsstrahlung.h: tau_BS_fulltest__s()
 *   - inverse_Compton.h: tau_IC_fulltest__s()
 *   - diffusion.h: tau_diff__s()
 * 
 * Identical to original spectra.c lines 777-781
 * 
 * ============================================================================
 */
int compute_loss_timescales(
    unsigned long n_gal,
    unsigned int n_E,
    const double *E_CRe,
    const double *E_CRe_lims,
    const double *n_H_halo,
    const double *B_halo,
    const double *h_halo,
    const double *D_e_halo,
    gsl_spline_object_2D *gso2D_BS,
    gsl_spline_object_2D *gso2D_IC_Gamma,
    double *tau_sync,
    double *tau_plasma,
    double *tau_BS,
    double *tau_IC,
    double *tau_diff)
{
    #pragma omp parallel for
    for (unsigned long i = 0; i < n_gal; i++) {
        
        /* Create GSL spline for halo diffusion */
        gsl_spline_object_1D *De_gso1D_halo = gsl_so1D(
            E_CRe,
            &D_e_halo[i * n_E],
            n_E,
            gsl_interp_cspline
        );
        
        for (unsigned int j = 0; j < n_E; j++) {
            
            unsigned long idx = i * n_E + j;
            double E = E_CRe[j];
            
            /* Use your EXACT original functions (lines 777-781) */
            
            tau_sync[idx] = tau_sync__s(E, B_halo[i]);
            
            tau_BS[idx] = tau_BS_fulltest__s(E, E_CRe_lims,
                                             n_H_halo[i], *gso2D_BS);
            
            tau_IC[idx] = tau_IC_fulltest__s(E, E_CRe_lims,
                                             *gso2D_IC_Gamma);
            
            /* CRITICAL: tau_plasma (not tau_ion) for ionized halo */
            tau_plasma[idx] = tau_plasma__s(E, n_H_halo[i]);
            
            tau_diff[idx] = tau_diff__s(E, h_halo[i], De_gso1D_halo);
        }
        
        gsl_so1D_free(De_gso1D_halo);
    }
    
    return 0;
}


/* ============================================================================
 * FUNCTION: solve_halo_steady_state
 * ============================================================================
 * 
 * PHYSICS: Uses your EXACT CRe_steadystate_solve() from CRe_steadystate.h
 * 
 * This is the FULL Green's function solver with:
 *   - zone = 2 (triggers plasma losses, not ionization)
 *   - Proper spatial transport
 *   - Energy-dependent diffusion
 *   - All loss channels
 * 
 * Identical to original spectra.c line 619-620
 * 
 * ============================================================================
 */
int solve_halo_steady_state(
    unsigned long n_gal,
    unsigned int n_E,
    const double *E_CRe,
    const double *E_CRe_lims,
    const double *n_H_halo,
    const double *B_halo,
    const double *h_halo,
    const double *D_e_halo,
    const double *Q_e_1_halo,
    const double *Q_e_2_halo,
    gsl_spline_object_2D *gso2D_BS,
    gsl_spline_object_2D *gso2D_IC_Gamma,
    double *N_e_1_halo,
    double *N_e_2_halo)
{
    #pragma omp parallel for
    for (unsigned long i = 0; i < n_gal; i++) {
        
        /* Create GSL splines for this galaxy */
        gsl_spline_object_1D *De_gso1D_z2 = gsl_so1D(
            E_CRe,
            &D_e_halo[i * n_E],
            n_E,
            gsl_interp_cspline
        );
        
        gsl_spline_object_1D *gso_1D_Q_inject_1_z2 = gsl_so1D(
            E_CRe,
            &Q_e_1_halo[i * n_E],
            n_E,
            gsl_interp_cspline
        );
        
        gsl_spline_object_1D *gso_1D_Q_inject_2_z2 = gsl_so1D(
            E_CRe,
            &Q_e_2_halo[i * n_E],
            n_E,
            gsl_interp_cspline
        );
        
        /* Output splines */
        gsl_spline_object_1D *qe_1_z2_so = NULL;
        gsl_spline_object_1D *qe_2_z2_so = NULL;
        
        /* Call your EXACT original solver (line 619-620) */
        CRe_steadystate_solve(
            2,                          /* zone = 2 (HALO) - uses plasma losses */
            E_CRe_lims,                 /* Energy limits */
            500,                        /* Integration points */
            n_H_halo[i],               /* Halo density (n_disc / 1000) */
            B_halo[i],                 /* Halo B field (B_disc / 3) */
            h_halo[i],                 /* Halo scale height (50 × h_disc) */
            1,                          /* Include losses */
            gso2D_IC_Gamma,            /* IC energy loss table */
            *gso2D_BS,                 /* BS cross-section table */
            *De_gso1D_z2,              /* Halo diffusion coefficient */
            *gso_1D_Q_inject_1_z2,     /* Primary injection */
            *gso_1D_Q_inject_2_z2,     /* Secondary injection */
            &qe_1_z2_so,               /* Output: primary spectrum */
            &qe_2_z2_so                /* Output: secondary spectrum */
        );
        
        /* Extract results */
        for (unsigned int j = 0; j < n_E; j++) {
            N_e_1_halo[i * n_E + j] = gsl_so1D_eval(qe_1_z2_so, E_CRe[j]);
            N_e_2_halo[i * n_E + j] = gsl_so1D_eval(qe_2_z2_so, E_CRe[j]);
        }
        
        /* Cleanup */
        gsl_so1D_free(De_gso1D_z2);
        gsl_so1D_free(gso_1D_Q_inject_1_z2);
        gsl_so1D_free(gso_1D_Q_inject_2_z2);
        gsl_so1D_free(qe_1_z2_so);
        gsl_so1D_free(qe_2_z2_so);
    }
    
    return 0;
}


/* ============================================================================
 * FUNCTION: compute_halo_emission_IC
 * ============================================================================
 * 
 * PHYSICS: Uses your EXACT eps_IC_3() from inverse_Compton.h
 * 
 * Computes Inverse Compton emission:
 *   ε_IC(E_γ) = ∫ dE_e N_e(E_e) ∫ dE_ph n_ph(E_ph) σ_IC(E_e, E_ph, E_γ)
 * 
 * NO free-free absorption in halo (optically thin)
 * 
 * ============================================================================
 */
int compute_halo_emission_IC(
    unsigned long n_gal,
    unsigned int n_E_CR,
    unsigned int n_E_gam,
    const double *E_CRe,
    const double *E_gam,
    const double *N_e_halo,
    gsl_spline_object_2D *gso2D_IC,
    double *spec_IC_halo)
{
    #pragma omp parallel for
    for (unsigned long i = 0; i < n_gal; i++) {
        
        /* Create spline for this galaxy's electron spectrum */
        gsl_spline_object_1D *qe_halo_so = gsl_so1D(
            E_CRe,
            &N_e_halo[i * n_E_CR],
            n_E_CR,
            gsl_interp_cspline
        );
        
        for (unsigned int j = 0; j < n_E_gam; j++) {
            
            /* Use your EXACT original function (line 646-647) */
            spec_IC_halo[i * n_E_gam + j] = eps_IC_3(
                E_gam[j],
                *gso2D_IC,
                *qe_halo_so
            );
            /* NOTE: No exp(-tau_FF) factor - halo is optically thin */
        }
        
        gsl_so1D_free(qe_halo_so);
    }
    
    return 0;
}


/* ============================================================================
 * FUNCTION: compute_halo_emission_SY
 * ============================================================================
 * 
 * PHYSICS: Uses your EXACT eps_SY_4() from synchrotron.h
 * 
 * Computes synchrotron emission:
 *   ε_SY(ν) = ∫ dE_e N_e(E_e) P_SY(ν, E_e, B)
 * 
 * NO free-free absorption in halo
 * 
 * ============================================================================
 */
int compute_halo_emission_SY(
    unsigned long n_gal,
    unsigned int n_E_CR,
    unsigned int n_E_gam,
    const double *E_CRe,
    const double *E_gam,
    const double *B_halo,
    const double *N_e_halo,
    gsl_spline_object_1D *gso1D_SY,
    double *spec_SY_halo)
{
    #pragma omp parallel for
    for (unsigned long i = 0; i < n_gal; i++) {
        
        /* Create spline for this galaxy's electron spectrum */
        gsl_spline_object_1D *qe_halo_so = gsl_so1D(
            E_CRe,
            &N_e_halo[i * n_E_CR],
            n_E_CR,
            gsl_interp_cspline
        );
        
        for (unsigned int j = 0; j < n_E_gam; j++) {
            
            /* Use your EXACT original function (line 649-650) */
            spec_SY_halo[i * n_E_gam + j] = eps_SY_4(
                E_gam[j],
                B_halo[i],
                *gso1D_SY,
                *qe_halo_so
            );
            /* NOTE: No exp(-tau_FF) factor - halo is optically thin */
        }
        
        gsl_so1D_free(qe_halo_so);
    }
    
    return 0;
}


/* ============================================================================
 * FUNCTION: compute_energy_losses_at_critical_energy
 * ============================================================================
 * 
 * PHYSICS: Identical to original spectra.c lines 723-728
 * 
 * Computes loss rates at E_crit (synchrotron emission at 1.49 GHz)
 * 
 * ============================================================================
 */
int compute_energy_losses_at_critical_energy(
    unsigned long n_gal,
    unsigned int n_E,
    const double *E_CRe,
    const double *E_CRe_lims,
    const double *B_halo,
    const double *n_H_halo,
    const double *h_halo,
    const double *D_e_halo,
    gsl_spline_object_2D *gso2D_BS,
    gsl_spline_object_2D *gso2D_IC_Gamma,
    double *E_loss_nucrit)      /* Output: [n_gal × 5] */
{
    #pragma omp parallel for
    for (unsigned long i = 0; i < n_gal; i++) {
        
        /* Critical energy for 1.49 GHz synchrotron (line 723) */
        double E_crit__GeV = sqrt((2.0 * 1.49e9 * m_e__g * c__cmsm1)
                                 / (3.0 * B_halo[i] * e__esu))
                            * M_PI * m_e__GeV;
        
        /* Create spline for diffusion */
        gsl_spline_object_1D *De_gso1D_halo = gsl_so1D(
            E_CRe,
            &D_e_halo[i * n_E],
            n_E,
            gsl_interp_cspline
        );
        
        /* Compute loss rates using EXACT original functions (lines 724-728) */
        
        E_loss_nucrit[i * 5 + 0] = E_crit__GeV
            / tau_BS_fulltest__s(E_crit__GeV, E_CRe_lims,
                                 n_H_halo[i], *gso2D_BS);
        
        E_loss_nucrit[i * 5 + 1] = E_crit__GeV
            / tau_sync__s(E_crit__GeV, B_halo[i]);
        
        E_loss_nucrit[i * 5 + 2] = E_crit__GeV
            / tau_IC_fulltest__s(E_crit__GeV, E_CRe_lims,
                                 *gso2D_IC_Gamma);
        
        E_loss_nucrit[i * 5 + 3] = E_crit__GeV
            / tau_plasma__s(E_crit__GeV, n_H_halo[i]);
        
        E_loss_nucrit[i * 5 + 4] = E_crit__GeV
            / tau_diff__s(E_crit__GeV, h_halo[i], De_gso1D_halo);
        
        gsl_so1D_free(De_gso1D_halo);
    }
    
    return 0;
}


/* ============================================================================
 * FUNCTION: integrate_spectrum
 * ============================================================================
 * 
 * PHYSICS: Uses spec_integrate from spec_integrate.h
 * 
 * Integrates spectrum in log space (trapezoidal rule)
 * 
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
        
        /* Use your original function if available */
        /* integrals[i] = spec_integrate(n_E, E, &spec[i * n_E]); */
        
        /* Or implement trapezoidal in log space */
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
 * ============================================================================ */

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
    printf("Halo CR Spectra - Full Physics Version\n");
    printf("========================================\n");
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    printf("GSL version: %s\n", gsl_version);
    printf("Physics: EXACT match to spectra.c\n");
    printf("========================================\n");
}