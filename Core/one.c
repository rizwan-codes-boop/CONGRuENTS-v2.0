#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>



#include <math.h>
#include "halo_mass_funcs.h"
#define _USE_MATH_DEFINES
#define m_e__GeV 0.0005109989
double q_p_inject = 2.2;//proton injection spectral index
double q_e_inject = 2.2;//electron injection spectral index

double E_gam_lims__GeV[2] = { 1.e-16, 1.e8 };//gamma-ray energy limits in GeV
double T_CR_lims__GeV[2] = { 1.e-3, 1.e8 };//cosmic ray kinetic energy limits in GeV
double E_CRe_lims__GeV[2] = { 1.e-3 + m_e__GeV, 1.e8 + m_e__GeV };//cosmic ray electron total energy limits in GeV

int main(int argc,char *argv[])
{
    //Initiate the halo mass routine
   struct halo_mass_obj hm_obj = halo_mass_init();

    unsigned long int i, j, k;

    time_t seconds;//for timing purposes
/*################################################################################################################################*/

/* Read tau EBL from file Dominguez data */

    size_t nx_E, ny_z;
//    FILE *tau_in = fopen( "input/tau_Eg_z_Dominguez.txt", "r" );
    FILE *tau_in = fopen( "input/tau_Eg_z_Franceschini.txt", "r" );
//    FILE *tau_in = fopen( "input/tau_Eg_z_Gilmore.txt", "r" );
    fscanf( tau_in , "%*[^\n]\n");
    fscanf( tau_in , "%lu \n", &ny_z );
    fscanf( tau_in , "%*[^\n]\n");
    fscanf( tau_in , "%lu \n", &nx_E );
    fscanf( tau_in , "%*[^\n]\n");

    double xa_E[nx_E];//energy array in eV
    double ya_z[ny_z]; // redshift array
    double za_tau[nx_E * ny_z];// optical depth array

    /* File IO */
    /* Careful with \n characters in za_tau read-in */
    for (i=0;i<ny_z;i++){fscanf( tau_in ,"%lf", &ya_z[i] );}
    fscanf( tau_in , "\n");
    fscanf( tau_in , "%*[^\n]\n");
    for (i=0;i<nx_E;i++){fscanf( tau_in ,"%le", &xa_E[i] );}
    fscanf( tau_in , "\n");
    fscanf( tau_in , "%*[^\n]\n");
    for ( i = 0 ; i < (nx_E * ny_z) ; i++){fscanf( tau_in ,"%le", &za_tau[i] );}
    fclose(tau_in);


    /* Assign fdata for interpolation for data & error - energy in eV */
    fd_in fdata_in;
    fdata_in.nx = nx_E;//number of energy bins
    fdata_in.ny = ny_z;//number of redshift bins
    fdata_in.xa = (double*) malloc(nx_E * sizeof(double));
    fdata_in.ya = (double*) malloc(ny_z * sizeof(double));
    fdata_in.za = (double*) malloc( (nx_E * ny_z) * sizeof(double));
    for (i=0; i < nx_E; i++) fdata_in.xa[i] = xa_E[i];
    for (i=0; i < ny_z; i++) fdata_in.ya[i] = ya_z[i];
    for (i=0; i < (ny_z * nx_E); i++) fdata_in.za[i] = za_tau[i];
    fdata_in.T = gsl_interp2d_bilinear;
    fdata_in.interp = gsl_interp2d_alloc(fdata_in.T, fdata_in.nx, fdata_in.ny);
    fdata_in.xacc = gsl_interp_accel_alloc();
    fdata_in.yacc = gsl_interp_accel_alloc();
    gsl_interp2d_init(fdata_in.interp, fdata_in.xa, fdata_in.ya, fdata_in.za, fdata_in.nx, fdata_in.ny);


/* Read in the gals */

    char infile[strlen(argv[1]) + 1];
    snprintf(infile, strlen(argv[1]) + 1, "%s", argv[1]);
    char datadir[strlen(argv[2]) + 1];
    snprintf(datadir, strlen(argv[2]) + 1, "%s", argv[2]);
    char outfp[strlen(argv[3]) + 1];
    snprintf(outfp, strlen(argv[3]) + 1, "%s", argv[3]);


    FILE *gals_in = fopen( infile , "r" );
    fscanf( gals_in , "%*[^\n]\n");
    unsigned long int n_gal;
    fscanf( gals_in , "%lu\n" , &n_gal);
    printf("Reading %lu galaxies from file %s...\n" , n_gal, infile);
    fscanf( gals_in , "%*[^\n]\n");

    double z[n_gal], M_star__Msol[n_gal], Re__kpc[n_gal], SFR__Msolyrm1[n_gal];
    for (i = 0; i < n_gal; i++)
    {
        fscanf( gals_in , "%le %le %le %le\n" , &z[i], &M_star__Msol[i], &Re__kpc[i], &SFR__Msolyrm1[i] ); 
    }
    fclose(gals_in);

/* Calculate calorimetry fraction */

    double Sig_star__Msolpcm2[n_gal], Sig_SFR__Msolyrm1pcm2[n_gal], Sig_gas__Msolpcm2[n_gal];
    double A_Re__pc2[n_gal], sig_gas__kmsm1[n_gal], h__pc[n_gal], n_H__cmm3[n_gal], n__cmm3[n_gal], T_dust__K[n_gal];
    
    #pragma omp parallel for schedule(dynamic)
    for (i = 0; i < n_gal; i++)
    {
        A_Re__pc2[i] = M_PI * pow( Re__kpc[i] * 1e3, 2 );//area within Re in pc^2
        Sig_star__Msolpcm2[i] = M_star__Msol[i] / ( 2. * A_Re__pc2[i] );//stellar mass surface density within Re in Msol/pc^2
        Sig_SFR__Msolyrm1pcm2[i] = SFR__Msolyrm1[i] / ( 2. * A_Re__pc2[i] );//SFR surface density within Re in Msol/yr/pc^2
        Sig_gas__Msolpcm2[i] = Sigma_gas_Shi_iKS__Msolpcm2( Sig_SFR__Msolyrm1pcm2[i], Sig_star__Msolpcm2[i] );//gas surface density in Msol/pc^2
        sig_gas__kmsm1[i] = sigma_gas_Yu__kmsm1( SFR__Msolyrm1[i] );//gas velocity dispersion in km/s
        T_dust__K[i] = Tdust__K( z[i], SFR__Msolyrm1[i], M_star__Msol[i] );//dust temperature in K
    }

    double chi = 1e-4, M_A = 2.0, beta = 0.25;//turbulence parameters
    double sigma_pp_cm2 = 40e-27, mu_H = 1.4, mu_p = 1.17;//physical constants

    /* Number of stars that go SN per solar mass of stars formed - C2003 IMF */
    double n_SN_Msolm1 = 1.321680e-2;
    /* fraction of energy that goes into CRs */
    double f_EtoCR = 0.1;

    /* fraction of CRp energy that goes into primary CRes */
    double f_CRe_CRp = 0.2;
    /* Energy of each SN */
    double E_SN_erg = 1e51;
    /* Pre-factor for the ion Alfven speed, default is 1. */
    double f_vAi = 1.;


    double T_p_cutoff__GeV = 1e8; // proton cutoff energy in GeV
    double T_e_cutoff__GeV = 1e5; // electron cutoff energy in GeV

    unsigned int n_T_CR = 1000;//number of cosmic ray energy bins

    double T_CR__GeV[n_T_CR];//cosmic ray proton kinetic energy array in GeV
    logspace_array( n_T_CR, T_CR_lims__GeV[0], T_CR_lims__GeV[1], T_CR__GeV );
    write_1D_file( n_T_CR, T_CR__GeV, "T_CR__GeV", string_cat(outfp, "/T_CR.txt") );

    double E_CRe__GeV[n_T_CR];//cosmic ray electron kinetic energy array in GeV
    logspace_array( n_T_CR, E_CRe_lims__GeV[0], E_CRe_lims__GeV[1], E_CRe__GeV );//cosmic ray electron kinetic energy array in GeV

    int n_E_gam = 500;//number of gamma-ray energy bins
    double E_gam__GeV[n_E_gam];//gamma-ray energy array in GeV
    logspace_array( n_E_gam, E_gam_lims__GeV[0], E_gam_lims__GeV[1], E_gam__GeV );
    write_1D_file( n_E_gam, E_gam__GeV, "E_gam__GeV", string_cat(outfp, "/E_gam.txt") );

    double **f_cal = malloc(sizeof *f_cal * n_gal);//calorimetry fraction array
    if (f_cal){for (i = 0; i < n_gal; i++){f_cal[i] = malloc(sizeof *f_cal[i] * n_T_CR);}}
    double **D__cm2sm1 = malloc(sizeof *D__cm2sm1 * n_gal);//diffusion coefficient array
    if (D__cm2sm1){for (i = 0; i < n_gal; i++){D__cm2sm1[i] = malloc(sizeof *D__cm2sm1[i] * n_T_CR);}}
    double **D_e__cm2sm1 = malloc(sizeof *D_e__cm2sm1 * n_gal);//electron diffusion coefficient array
    if (D_e__cm2sm1){for (i = 0; i < n_gal; i++){D_e__cm2sm1[i] = malloc(sizeof *D_e__cm2sm1[i] * n_T_CR);}}
    double **D_e_z2__cm2sm1 = malloc(sizeof *D_e_z2__cm2sm1 * n_gal);//electron diffusion coefficient array in z^2 direction
    if (D_e_z2__cm2sm1){for (i = 0; i < n_gal; i++){D_e_z2__cm2sm1[i] = malloc(sizeof *D_e_z2__cm2sm1[i] * n_T_CR);}}

    double G_h = 4.302e-3, eta_pp = 0.5;//gravitational constant in (km/s)^2 pc/Msol, inelasticity of proton-proton collisions
    double u_LA, v_Ai, t_loss_s, C[n_gal], Ce_Esm1[n_gal];
    double v_st, v_ste, L_A, Gam_0, D0, tau_eff, B__G[n_gal], B_halo__G[n_gal];
    double CnormE[n_gal];//normalization constant for cosmic ray energy

    #pragma omp parallel for schedule(dynamic) private(j, u_LA, v_Ai, L_A, D0, t_loss_s, v_st, v_ste, Gam_0, seconds, tau_eff )

    for (i = 0; i < n_gal; i++)
    {

        h__pc[i] = pow( sig_gas__kmsm1[i], 2 )/( M_PI * G_h * ( Sig_gas__Msolpcm2[i] + 
                   sig_gas__kmsm1[i]/sigma_star_Bezanson__kmsm1( M_star__Msol[i], Re__kpc[i] ) * Sig_star__Msolpcm2[i] ) );//scale height in pc in hydrostatic equilibrium

        n_H__cmm3[i] = Sig_gas__Msolpcm2[i]/( mu_H * m_H__kg * 2. * h__pc[i] ) * Msol__kg/pow( pc__cm, 3 );// midplane hydrogen density in cm^-3

        n__cmm3[i] = n_H__cmm3[i] * mu_H/mu_p;//total gas number density in cm^-3

        u_LA = sig_gas__kmsm1[i]/sqrt(2.);//turbulent velocity at scale height in km/s
        v_Ai = 1000. * ( u_LA/10. )/( sqrt(chi/1e-4) * M_A );//ion alfven speed in km/s
        L_A = h__pc[i]/pow(M_A,3);//alfvenic break scale in MHD turbulence in pc

        B__G[i] = sqrt(4.* M_PI * chi * n__cmm3[i] * mu_p * m_H__kg * 1e3) * v_Ai * 1e5;

        if (log10(SFR__Msolyrm1[i]/M_star__Msol[i]) > -10.)//controls CR escape in halo
        {
            B_halo__G[i] = B__G[i]/3.;
        }
        else
        {
            B_halo__G[i] = B__G[i]/1.5;
        }

    //    CnormE[i] = C_norm_E(E_cut[i]);
        CnormE[i] = C_norm_E( q_p_inject, m_p__GeV, T_p_cutoff__GeV);//normalization constant for cosmic ray energy

        D0 = v_Ai * L_A * 1e5 * pc__cm;//diffusion coefficient at 1 GeV in cm^2/s
        // calculate loss time
        t_loss_s = 1./(1./(1./( n__cmm3[i] * sigma_pp_cm2 * eta_pp * c__cmsm1 )) + 1./(pow( h__pc[i] * pc__cm,2)/D0) );
        // calculate calorimetry fraction normalization constant
        C[i] = SFR__Msolyrm1[i] * n_SN_Msolm1 * f_EtoCR * E_SN_erg * erg__GeV/yr__s * t_loss_s/( CnormE[i] * 2. * A_Re__pc2[i] * 2. * h__pc[i] * pow(pc__cm, 3) );
        // calculate primary CRe injection rate normalization in CRe/s/GeV
        Ce_Esm1[i] = f_CRe_CRp * SFR__Msolyrm1[i] * n_SN_Msolm1 * f_EtoCR * E_SN_erg * erg__GeV/( yr__s * C_norm_E( q_e_inject, m_e__GeV, T_e_cutoff__GeV ) );
        
        for (j = 0; j < n_T_CR; j++)//loop over cosmic ray proton energies
        {
            v_st = fmin( f_vAi * v_Ai * (1. + 2.3e-3 * pow( sqrt(pow(T_CR__GeV[j],2) + 2. * m_p__GeV * T_CR__GeV[j]) , q_p_inject-1.) * 
                   pow(n_H__cmm3[i]/1e3, 1.5) * (chi/1e-4) * M_A/( u_LA/10. * C[i]/2e-7 )), c__cmsm1/1e5);//streaming speed in km/s
            D__cm2sm1[i][j] = v_st * L_A * 1e5 * pc__cm;//diffusion coefficient in cm^2/s
            tau_eff = 9.9 * Sig_gas__Msolpcm2[i]/1e3 * h__pc[i]/1e2 * 1e27/D__cm2sm1[i][j];
            Gam_0 = 41.2 * h__pc[i]/1e2 * v_st/1e3 * 1e27/D__cm2sm1[i][j];
            f_cal[i][j] = 1. - 1./( gsl_sf_hyperg_0F1( beta/(beta+1.) , tau_eff/pow(beta+1.,2) ) + 
                       tau_eff/Gam_0 * gsl_sf_hyperg_0F1( (beta+2.)/(beta+1.) , tau_eff/pow(beta+1., 2)) );//calorimetry fraction

if (i == 10){f_cal[i][j] = f_cal[i][j] * 0.1;}//test case

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

    }

    write_2D_file( n_gal, n_T_CR, f_cal, "", string_cat(outfp, "/fcal.txt") );

    FILE *galdata_out = fopen( string_cat(outfp, "/gal_data.txt"), "w+" );
    fprintf( galdata_out, "h__pc n_H__cmm3 B__G sigmag__kmsm1 Are__pc2 Sigmagas__Msolpcm2 SigmaSFR__Msolyrm1pcm2 Sigmastar__Msolpcm2 Tdust__K\n" );
    for (i = 0; i < n_gal; i++)
    {
        fprintf( galdata_out, "%e %e %e %e %e %e %e %e %e\n", h__pc[i], n_H__cmm3[i], B__G[i], sig_gas__kmsm1[i], A_Re__pc2[i], 
                 Sig_gas__Msolpcm2[i], Sig_SFR__Msolyrm1pcm2[i], Sig_star__Msolpcm2[i], T_dust__K[i] );
    }
    fclose(galdata_out);

/*################################################################################################################################*/
/* Calculate spectra */

//memory allocation for spectra
    double **tau_gg = malloc(sizeof *tau_gg * n_gal);//optical depth array
    if (tau_gg){for (i = 0; i < n_gal; i++){tau_gg[i] = malloc(sizeof *tau_gg[i] * n_E_gam);}}
    double **tau_EBL = malloc(sizeof *tau_EBL * n_gal);//EBL optical depth array
    if (tau_EBL){for (i = 0; i < n_gal; i++){tau_EBL[i] = malloc(sizeof *tau_EBL[i] * n_E_gam);}} 


//  double **specs_casc_obs = malloc(sizeof *specs_casc_obs * n_gal);
//  if (specs_casc_obs){for (i = 0; i < n_gal; i++){specs_casc_obs[i] = malloc(sizeof *specs_casc_obs[i] * n_E_gam);}}
//  double **specs_casc = malloc(sizeof *specs_casc * n_gal);
//  if (specs_casc){for (i = 0; i < n_gal; i++){specs_casc[i] = malloc(sizeof *specs_casc[i] * n_E_gam);}}
//  double **specs_obs = malloc(sizeof *specs_obs * n_gal);
//  if (specs_obs){for (i = 0; i < n_gal; i++){specs_obs[i] = malloc(sizeof *specs_obs[i] * n_E_gam);}}
//  double **specs_L_emit = malloc(sizeof *specs_L_emit * n_gal);
//  if (specs_L_emit){for (i = 0; i < n_gal; i++){specs_L_emit[i] = malloc(sizeof *specs_L_emit[i] * n_E_gam);}}
//    double C_gam;
//  double spec_emit[n_Esteps];

    double nphot_params[7];//ISRF parameters

        double E_phot_lims__GeV[2] = { E_BB_peak__GeV( T_0_CMB__K ) * 1e-4, 1.e-7 };//photon energy limits for ISRF integration in GeV


    size_t n_pts[2] = { 1000, 1000 };

    double x_SY_lims[2];//synchrotron frequency limits in Hz

    x_SY_lims[0] = (2.*pow(M_PI,2)*pow(m_e__g,2)*pow(c__cmsm1,3)) / (3.*e__esu*maxval( n_gal, B__G )*h__ergs) * 
                   (E_gam_lims__GeV[0] * m_e__GeV)/(pow(E_CRe_lims__GeV[1],2)) * 0.1;//min limit for SY frequency in Hz

    x_SY_lims[1] = (2.*pow(M_PI,2)*pow(m_e__g,2)*pow(c__cmsm1,3)) / (3.*e__esu*minval( n_gal, B_halo__G )/10.*h__ergs) * 
                   (E_gam_lims__GeV[1] * (1+maxval( n_gal, z )) * m_e__GeV)/(pow(E_CRe_lims__GeV[0],2)) * 10.;//max limit for SY frequency in Hz


    IC_object ICo = load_IC_do_files( n_pts, E_gam_lims__GeV, E_CRe_lims__GeV, E_phot_lims__GeV,
                                      (maxval( n_gal, z ) + 1.) * T_0_CMB__K, 0.5, 
                                      minval( n_gal, T_dust__K ), maxval( n_gal, T_dust__K ), 5., datadir );//load IC data objects


    data_object_2D do_2D_BS = load_BS_do_files( n_pts, E_gam_lims__GeV, E_CRe_lims__GeV, datadir );//load BS data objects
    gsl_spline_object_2D gso2D_BS = do2D_to_gso2D( do_2D_BS );//load BS data objects
    data_object_2D_free( do_2D_BS );//free BS data objects


    data_object_1D do_1D_SY = load_SY_do_files( &(n_pts[0]), x_SY_lims, datadir );//load SY data objects
    gsl_spline_object_1D gso1D_SY = do1D_to_gso1D( do_1D_SY ); //init_gso_1D_sync( x_SY_lims, 1000, "log" );
    data_object_1D_free( do_1D_SY );//free SY data objects

}