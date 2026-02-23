#include "CR_funcs.h"
#include "gsl_decs.h"
#include "ionisation_loss.h"
#include "physical_constants.h"
#include <omp.h>
#include <stdio.h>

int main( int argc, char *argv[] )
{
    /* Pre-factor for the ion Alfven speed, default is 1. */
    double f_vAi = 1.;
    double chi = 1e-4, M_A = 2.0;//turbulence parameters
    unsigned long int n_gal, i, j, k;
    unsigned int n_T_CR = 1000;//number of cosmic ray energy bins
    double C[n_gal], L_A, u_LA, n_H__cmm3[n_gal], v_Ai, v_ste, B__G[n_gal], B_halo__G[n_gal], M_star__Msol[n_gal],SFR__Msolyrm1[n_gal];
    double T_CR__GeV[n_T_CR];//cosmic ray proton kinetic energy array in GeV
    double q_p_inject = 2.2;//proton injection spectral index

    double **f_cal = malloc(sizeof *f_cal * n_gal);//calorimetry fraction array
    if (f_cal)
    {
        for (i = 0; i < n_gal; i++)
        {
            f_cal[i] = malloc(sizeof *f_cal[i] * n_T_CR);
        }
    }

    double **D_e__cm2sm1 = malloc(sizeof *D_e__cm2sm1 * n_gal);//electron diffusion coefficient array
    if (D_e__cm2sm1)
    {
        for (i = 0; i < n_gal; i++)
        {
            D_e__cm2sm1[i] = malloc(sizeof *D_e__cm2sm1[i] * n_T_CR);
        }
    }
    
    double **D_e_z2__cm2sm1 = malloc(sizeof *D_e_z2__cm2sm1 * n_gal);//electron diffusion coefficient array in z^2 direction
    if (D_e_z2__cm2sm1)
    {
        for (i = 0; i < n_gal; i++)
        {
            D_e_z2__cm2sm1[i] = malloc(sizeof *D_e_z2__cm2sm1[i] * n_T_CR);
        }
    }


    #pragma omp parallel for schedule(dynamic) private(j, u_LA, v_Ai, L_A, D0, t_loss_s, v_st, v_ste, Gam_0, seconds, tau_eff )

    for (i = 0; i < n_gal; i++)
    {
        if (log10(SFR__Msolyrm1[i]/M_star__Msol[i]) > -10.)  // Star-forming galaxies
        {
        B_halo__G[i] = B__G[i]/3.;     // ← HALO: B_disc / 3
        }
        else                                                   // Quiescent galaxies
        {
          B_halo__G[i] = B__G[i]/1.5;    // ← HALO: B_disc / 1.5
        }
    }
    for (j = 0; j < n_T_CR; j++)//loop over cosmic ray electron energies
    {
        v_ste = fmin( f_vAi * v_Ai * (1. + 2.3e-3 * pow( sqrt(pow(T_CR__GeV[j],2) + 2. * m_e__GeV * T_CR__GeV[j]) , q_p_inject-1.) * 
                pow(n_H__cmm3[i]/1e3, 1.5) * (chi/1e-4) * M_A/( u_LA/10. * C[i]/2e-7 )), c__cmsm1/1e5);//streaming speed in km/s
        D_e__cm2sm1[i][j] = v_ste * L_A * 1e5 * pc__cm;//diffusion coefficient in cm^2/s
        D_e_z2__cm2sm1[i][j] = fmin( v_Ai * (1. + 2.3e-3 * pow( sqrt(pow(T_CR__GeV[j],2) + 2. * m_e__GeV * T_CR__GeV[j]) , q_p_inject-1.) * 
                            pow(n_H__cmm3[i]/1e3/1e3, 1.5) * (1./1e-4) * M_A/( u_LA/10. * ((1.-f_cal[i][0]) * C[i])/2e-7 )), 
                            c__cmsm1/1e5) * L_A * 1e5 * pc__cm; //diffusion coefficient in z^2 in cm^2/s
        }

    //primary injection spectrum and steady state spline object per galaxy
    for (j = 0; j < n_T_CR; j++)
    {
        Q_e_1_z2[i][j] = gsl_so1D_eval( qe_1_z1_so, E_CRe__GeV[j] )/tau_diff__s( E_CRe__GeV[j], h__pc[i], De_gso1D_z1 );
    }
    gso_1D_Q_inject_1_z2 = gsl_so1D( n_T_CR, E_CRe__GeV, Q_e_1_z2[i] );


    //secondary injection spectrum and steady state spline object
    for (j = 0; j < n_T_CR; j++)
    {
        Q_e_2_z2[i][j] = gsl_so1D_eval( qe_2_z1_so, E_CRe__GeV[j] )/tau_diff__s( E_CRe__GeV[j], h__pc[i], De_gso1D_z1 );
    }

    gso_1D_Q_inject_2_z2 = gsl_so1D( n_T_CR, E_CRe__GeV, Q_e_2_z2[i] );

    CRe_steadystate_solve( 2, E_CRe_lims__GeV, 500, n_H__cmm3[i]/1000., B_halo__G[i], 50.*h__pc[i], 1, &gso2D_IC_Gamma, gso2D_BS,
                               De_gso1D_z2, gso_1D_Q_inject_1_z2, gso_1D_Q_inject_2_z2, &qe_1_z2_so, &qe_2_z2_so );


}



