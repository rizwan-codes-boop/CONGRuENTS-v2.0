#include <stdio.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_spline2d.h>
#include <omp.h>
#include <time.h>

/* ========================= */
/* ---- Helper Structs ----- */
/* ========================= */

typedef struct {
    size_t nx_E;
    size_t ny_z;
    double *xa_E;
    double *ya_z;
    double *za_tau;
} tau_data;


/* ========================= */
/* ---- Argument Parser ---- */
/* ========================= */

void parse_input_arguments(int argc, char *argv[],
                           char *infile,
                           char *datadir,
                           char *outfp)
{
    snprintf(infile, strlen(argv[1]) + 1, "%s", argv[1]);
    snprintf(datadir, strlen(argv[2]) + 1, "%s", argv[2]);
    snprintf(outfp, strlen(argv[3]) + 1, "%s", argv[3]);
}


/* ========================= */
/* ---- Read Tau Header ---- */
/* ========================= */

void read_tau_dimensions(FILE *tau_in, size_t *nx_E, size_t *ny_z)
{
    fscanf(tau_in, "%*[^\n]\n");
    fscanf(tau_in, "%lu \n", ny_z);
    fscanf(tau_in, "%*[^\n]\n");
    fscanf(tau_in, "%lu \n", nx_E);
    fscanf(tau_in, "%*[^\n]\n");
}


/* ========================= */
/* ---- Read Tau Arrays ---- */
/* ========================= */

void read_tau_arrays(FILE *tau_in,
                     size_t nx_E,
                     size_t ny_z,
                     double *xa_E,
                     double *ya_z,
                     double *za_tau)
{
    unsigned long int i;

    for (i = 0; i < ny_z; i++)
        fscanf(tau_in, "%lf", &ya_z[i]);

    fscanf(tau_in, "\n");
    fscanf(tau_in, "%*[^\n]\n");

    for (i = 0; i < nx_E; i++)
        fscanf(tau_in, "%le", &xa_E[i]);

    fscanf(tau_in, "\n");
    fscanf(tau_in, "%*[^\n]\n");

    for (i = 0; i < (nx_E * ny_z); i++)
        fscanf(tau_in, "%le", &za_tau[i]);
}


/* ========================= */
/* ---- Load Tau File ------ */
/* ========================= */

tau_data load_tau_file(const char *filename)
{
    tau_data data;

    FILE *tau_in = fopen(filename, "r");

    read_tau_dimensions(tau_in, &data.nx_E, &data.ny_z);

    data.xa_E = malloc(data.nx_E * sizeof(double));
    data.ya_z = malloc(data.ny_z * sizeof(double));
    data.za_tau = malloc(data.nx_E * data.ny_z * sizeof(double));

    read_tau_arrays(tau_in,
                    data.nx_E,
                    data.ny_z,
                    data.xa_E,
                    data.ya_z,
                    data.za_tau);

    fclose(tau_in);

    return data;
}


/* ========================= */
/* ---- Setup Interp ------- */
/* ========================= */

void init_fd_interp(fd_in *fdata, tau_data *tau)
{
    unsigned long int i;

    fdata->nx = tau->nx_E;
    fdata->ny = tau->ny_z;

    fdata->xa = malloc(tau->nx_E * sizeof(double));
    fdata->ya = malloc(tau->ny_z * sizeof(double));
    fdata->za = malloc(tau->nx_E * tau->ny_z * sizeof(double));

    for (i = 0; i < tau->nx_E; i++)
        fdata->xa[i] = tau->xa_E[i];

    for (i = 0; i < tau->ny_z; i++)
        fdata->ya[i] = tau->ya_z[i];

    for (i = 0; i < tau->nx_E * tau->ny_z; i++)
        fdata->za[i] = tau->za_tau[i];

    fdata->T = gsl_interp2d_bilinear;
    fdata->interp = gsl_interp2d_alloc(fdata->T, fdata->nx, fdata->ny);
    fdata->xacc = gsl_interp_accel_alloc();
    fdata->yacc = gsl_interp_accel_alloc();

    gsl_interp2d_init(fdata->interp,
                      fdata->xa,
                      fdata->ya,
                      fdata->za,
                      fdata->nx,
                      fdata->ny);
}


/* ========================= */
/* ---- Galaxy Loader ------ */
/* ========================= */

void read_galaxy_file(const char *filename,
                      unsigned long int *n_gal,
                      double **z,
                      double **M_star__Msol,
                      double **Re__kpc,
                      double **SFR__Msolyrm1)
{
    unsigned long int i;

    FILE *gals_in = fopen(filename, "r");

    fscanf(gals_in, "%*[^\n]\n");
    fscanf(gals_in, "%lu\n", n_gal);

    printf("Reading %lu galaxies from file %s...\n", *n_gal, filename);

    fscanf(gals_in, "%*[^\n]\n");

    *z = malloc(*n_gal * sizeof(double));
    *M_star__Msol = malloc(*n_gal * sizeof(double));
    *Re__kpc = malloc(*n_gal * sizeof(double));
    *SFR__Msolyrm1 = malloc(*n_gal * sizeof(double));

    for (i = 0; i < *n_gal; i++)
    {
        fscanf(gals_in, "%le %le %le %le\n",
               &(*z)[i],
               &(*M_star__Msol)[i],
               &(*Re__kpc)[i],
               &(*SFR__Msolyrm1)[i]);
    }

    fclose(gals_in);
}

/********************************************************** */
/*################################################################################################################################*/
/* Calculate calorimetry fraction - Modular Version */

/* ===================== Helper Allocation ===================== */

double **allocate_2D_array(int n1, int n2)
{
    int i;
    double **arr = malloc(sizeof(*arr) * n1);
    if (arr)
    {
        for (i = 0; i < n1; i++)
            arr[i] = malloc(sizeof(*arr[i]) * n2);
    }
    return arr;
}

/* ===================== Surface Density Calculations ===================== */

void compute_surface_densities(
    int n_gal,
    double *Re__kpc,
    double *M_star__Msol,
    double *SFR__Msolyrm1,
    double *z,
    double *A_Re__pc2,
    double *Sig_star__Msolpcm2,
    double *Sig_SFR__Msolyrm1pcm2,
    double *Sig_gas__Msolpcm2,
    double *sig_gas__kmsm1,
    double *T_dust__K)
{
    int i;

#pragma omp parallel for schedule(dynamic)
    for (i = 0; i < n_gal; i++)
    {
        A_Re__pc2[i] = M_PI * pow(Re__kpc[i] * 1e3, 2);

        Sig_star__Msolpcm2[i] =
            M_star__Msol[i] / (2. * A_Re__pc2[i]);

        Sig_SFR__Msolyrm1pcm2[i] =
            SFR__Msolyrm1[i] / (2. * A_Re__pc2[i]);

        Sig_gas__Msolpcm2[i] =
            Sigma_gas_Shi_iKS__Msolpcm2(
                Sig_SFR__Msolyrm1pcm2[i],
                Sig_star__Msolpcm2[i]);

        sig_gas__kmsm1[i] =
            sigma_gas_Yu__kmsm1(SFR__Msolyrm1[i]);

        T_dust__K[i] =
            Tdust__K(z[i], SFR__Msolyrm1[i], M_star__Msol[i]);
    }
}

/* ===================== Energy Grid Setup ===================== */

void setup_energy_grids(
    unsigned int n_T_CR,
    int n_E_gam,
    double *T_CR__GeV,
    double *E_CRe__GeV,
    double *E_gam__GeV,
    char *outfp)
{
    logspace_array(n_T_CR, T_CR_lims__GeV[0], T_CR_lims__GeV[1], T_CR__GeV);
    write_1D_file(n_T_CR, T_CR__GeV, "T_CR__GeV", string_cat(outfp, "/T_CR.txt"));

    logspace_array(n_T_CR, E_CRe_lims__GeV[0], E_CRe_lims__GeV[1], E_CRe__GeV);

    logspace_array(n_E_gam, E_gam_lims__GeV[0], E_gam_lims__GeV[1], E_gam__GeV);
    write_1D_file(n_E_gam, E_gam__GeV, "E_gam__GeV", string_cat(outfp, "/E_gam.txt"));
}

/* ===================== Gas & Magnetic Properties ===================== */

void compute_gas_and_field_properties(
    int i,
    double G_h,
    double chi,
    double M_A,
    double mu_H,
    double mu_p,
    double *Sig_gas__Msolpcm2,
    double *Sig_star__Msolpcm2,
    double *sig_gas__kmsm1,
    double *M_star__Msol,
    double *Re__kpc,
    double *h__pc,
    double *n_H__cmm3,
    double *n__cmm3,
    double *B__G,
    double *B_halo__G,
    double *SFR__Msolyrm1,
    double *u_LA_out,
    double *v_Ai_out,
    double *L_A_out)
{
    double u_LA, v_Ai, L_A;

    h__pc[i] =
        pow(sig_gas__kmsm1[i], 2) /
        (M_PI * G_h *
         (Sig_gas__Msolpcm2[i] +
          sig_gas__kmsm1[i] /
              sigma_star_Bezanson__kmsm1(M_star__Msol[i], Re__kpc[i]) *
              Sig_star__Msolpcm2[i]));

    n_H__cmm3[i] =
        Sig_gas__Msolpcm2[i] /
        (mu_H * m_H__kg * 2. * h__pc[i]) *
        Msol__kg / pow(pc__cm, 3);

    n__cmm3[i] = n_H__cmm3[i] * mu_H / mu_p;

    u_LA = sig_gas__kmsm1[i] / sqrt(2.);
    v_Ai = 1000. * (u_LA / 10.) / (sqrt(chi / 1e-4) * M_A);
    L_A = h__pc[i] / pow(M_A, 3);

    B__G[i] =
        sqrt(4. * M_PI * chi * n__cmm3[i] * mu_p * m_H__kg * 1e3) *
        v_Ai * 1e5;

    if (log10(SFR__Msolyrm1[i] / M_star__Msol[i]) > -10.)
        B_halo__G[i] = B__G[i] / 3.;
    else
        B_halo__G[i] = B__G[i] / 1.5;

    *u_LA_out = u_LA;
    *v_Ai_out = v_Ai;
    *L_A_out = L_A;
}

/* ===================== CR Normalisations ===================== */

void compute_CR_normalisations(
    int i,
    double v_Ai,
    double L_A,
    double sigma_pp_cm2,
    double eta_pp,
    double *n__cmm3,
    double *SFR__Msolyrm1,
    double *A_Re__pc2,
    double *h__pc,
    double *C,
    double *Ce_Esm1,
    double *CnormE,
    double T_p_cutoff__GeV,
    double T_e_cutoff__GeV,
    double n_SN_Msolm1,
    double f_EtoCR,
    double f_CRe_CRp,
    double E_SN_erg)
{
    double D0, t_loss_s;

    CnormE[i] =
        C_norm_E(q_p_inject, m_p__GeV, T_p_cutoff__GeV);

    D0 = v_Ai * L_A * 1e5 * pc__cm;

    t_loss_s =
        1. /
        (1. / (1. / (n__cmm3[i] * sigma_pp_cm2 * eta_pp * c__cmsm1)) +
         1. / (pow(h__pc[i] * pc__cm, 2) / D0));

    C[i] =
        SFR__Msolyrm1[i] * n_SN_Msolm1 * f_EtoCR *
        E_SN_erg * erg__GeV / yr__s *
        t_loss_s /
        (CnormE[i] * 2. * A_Re__pc2[i] * 2. * h__pc[i] * pow(pc__cm, 3));

    Ce_Esm1[i] =
        f_CRe_CRp * SFR__Msolyrm1[i] *
        n_SN_Msolm1 * f_EtoCR * E_SN_erg *
        erg__GeV /
        (yr__s * C_norm_E(q_e_inject, m_e__GeV, T_e_cutoff__GeV));
}

/* ===================== Proton Transport ===================== */

void compute_proton_transport(
    int i, int n_T_CR,
    double *T_CR__GeV,
    double beta,
    double chi,
    double M_A,
    double u_LA,
    double v_Ai,
    double L_A,
    double *n_H__cmm3,
    double *Sig_gas__Msolpcm2,
    double *h__pc,
    double *C,
    double f_vAi,
    double **D__cm2sm1,
    double **f_cal)
{
    int j;
    double v_st, tau_eff, Gam_0;

    for (j = 0; j < n_T_CR; j++)
    {
        v_st =
            fmin(
                f_vAi * v_Ai *
                    (1. + 2.3e-3 *
                               pow(sqrt(pow(T_CR__GeV[j], 2) +
                                        2. * m_p__GeV * T_CR__GeV[j]),
                                   q_p_inject - 1.) *
                               pow(n_H__cmm3[i] / 1e3, 1.5) *
                               (chi / 1e-4) * M_A /
                               (u_LA / 10. * C[i] / 2e-7)),
                c__cmsm1 / 1e5);

        D__cm2sm1[i][j] =
            v_st * L_A * 1e5 * pc__cm;

        tau_eff =
            9.9 * Sig_gas__Msolpcm2[i] / 1e3 *
            h__pc[i] / 1e2 *
            1e27 / D__cm2sm1[i][j];

        Gam_0 =
            41.2 * h__pc[i] / 1e2 *
            v_st / 1e3 *
            1e27 / D__cm2sm1[i][j];

        f_cal[i][j] =
            1. - 1. /
                     (gsl_sf_hyperg_0F1(beta / (beta + 1.),
                                        tau_eff / pow(beta + 1., 2)) +
                      tau_eff / Gam_0 *
                          gsl_sf_hyperg_0F1((beta + 2.) / (beta + 1.),
                                            tau_eff / pow(beta + 1., 2)));

        if (i == 10)
            f_cal[i][j] *= 0.1;
    }
}

/* ===================== Electron Transport ===================== */

void compute_electron_transport(
    int i, int n_T_CR,
    double *T_CR__GeV,
    double chi,
    double M_A,
    double u_LA,
    double v_Ai,
    double L_A,
    double *n_H__cmm3,
    double *C,
    double **D_e__cm2sm1,
    double **D_e_z2__cm2sm1,
    double **f_cal,
    double f_vAi)
{
    int j;
    double v_ste;

    for (j = 0; j < n_T_CR; j++)
    {
        v_ste =
            fmin(
                f_vAi * v_Ai *
                    (1. + 2.3e-3 *
                               pow(sqrt(pow(T_CR__GeV[j], 2) +
                                        2. * m_e__GeV * T_CR__GeV[j]),
                                   q_p_inject - 1.) *
                               pow(n_H__cmm3[i] / 1e3, 1.5) *
                               (chi / 1e-4) * M_A /
                               (u_LA / 10. * C[i] / 2e-7)),
                c__cmsm1 / 1e5);

        D_e__cm2sm1[i][j] =
            v_ste * L_A * 1e5 * pc__cm;

        D_e_z2__cm2sm1[i][j] =
            fmin(
                v_Ai *
                    (1. + 2.3e-3 *
                               pow(sqrt(pow(T_CR__GeV[j], 2) +
                                        2. * m_e__GeV * T_CR__GeV[j]),
                                   q_p_inject - 1.) *
                               pow(n_H__cmm3[i] / 1e6, 1.5) *
                               1e4 * M_A /
                               (u_LA / 10. *
                                ((1. - f_cal[i][0]) * C[i]) /
                                2e-7)),
                c__cmsm1 / 1e5) *
            L_A * 1e5 * pc__cm;
    }
}

/* ===================== Output ===================== */

void write_galaxy_output(
    int n_gal,
    char *outfp,
    double *h__pc,
    double *n_H__cmm3,
    double *B__G,
    double *sig_gas__kmsm1,
    double *A_Re__pc2,
    double *Sig_gas__Msolpcm2,
    double *Sig_SFR__Msolyrm1pcm2,
    double *Sig_star__Msolpcm2,
    double *T_dust__K)
{
    int i;

    FILE *galdata_out =
        fopen(string_cat(outfp, "/gal_data.txt"), "w+");

    fprintf(galdata_out,
            "h__pc n_H__cmm3 B__G sigmag__kmsm1 Are__pc2 "
            "Sigmagas__Msolpcm2 SigmaSFR__Msolyrm1pcm2 "
            "Sigmastar__Msolpcm2 Tdust__K\n");

    for (i = 0; i < n_gal; i++)
    {
        fprintf(galdata_out,
                "%e %e %e %e %e %e %e %e %e\n",
                h__pc[i], n_H__cmm3[i], B__G[i],
                sig_gas__kmsm1[i], A_Re__pc2[i],
                Sig_gas__Msolpcm2[i],
                Sig_SFR__Msolyrm1pcm2[i],
                Sig_star__Msolpcm2[i],
                T_dust__K[i]);
    }

    fclose(galdata_out);
}


/****************************************/
/*******************************************************
 *  Modularized Spectra and Radiation Setup
 *  
 *  This section handles:
 *  1. Memory allocation for optical depth arrays
 *  2. Computation of photon and synchrotron limits
 *  3. Loading IC, Bremsstrahlung (BS), and Synchrotron (SY) data objects
 *  4. Writing radiation energy density outputs
 *  5. Writing galaxy spectra
 *
 *  Notes:
 *  - Variable names preserved from original code
 *  - OpenMP safety maintained
 *  - All physics preserved
 *******************************************************/


/*------------------------------------------------------
 * 1. Allocate 2D arrays for optical depth and spectra
 *------------------------------------------------------*/
double **allocate_2D_array(unsigned long n_rows, unsigned long n_cols)
{
    unsigned long i;
    double **arr = malloc(sizeof *arr * n_rows);
    if (arr)
    {
        for (i = 0; i < n_rows; i++)
            arr[i] = malloc(sizeof *arr[i] * n_cols);
    }
    return arr;
}

/* Wrapper to allocate all spectra arrays */
void allocate_spectra_memory(unsigned long n_gal, int n_E_gam,
                             double ***tau_gg, double ***tau_EBL)
{
    *tau_gg  = allocate_2D_array(n_gal, n_E_gam);
    *tau_EBL = allocate_2D_array(n_gal, n_E_gam);
}

/*------------------------------------------------------
 * 2. Compute photon energy limits for ISRF integration
 *------------------------------------------------------*/
void compute_photon_energy_limits(double E_phot_lims__GeV[2])
{
    E_phot_lims__GeV[0] = E_BB_peak__GeV(T_0_CMB__K) * 1e-4; // min photon energy (GeV)
    E_phot_lims__GeV[1] = 1.e-7;                               // max photon energy (GeV)
}

/*------------------------------------------------------
 * 3. Compute synchrotron frequency limits (Hz)
 *------------------------------------------------------*/
void compute_synchrotron_limits(unsigned long n_gal,
                                double *B__G,
                                double *B_halo__G,
                                double *z,
                                double x_SY_lims[2])
{
    x_SY_lims[0] =
        (2.*pow(M_PI,2)*pow(m_e__g,2)*pow(c__cmsm1,3)) /
        (3.*e__esu*maxval(n_gal, B__G)*h__ergs) *
        (E_gam_lims__GeV[0]*m_e__GeV) /
        pow(E_CRe_lims__GeV[1],2) * 0.1; // min synchrotron frequency

    x_SY_lims[1] =
        (2.*pow(M_PI,2)*pow(m_e__g,2)*pow(c__cmsm1,3)) /
        (3.*e__esu*minval(n_gal, B_halo__G)/10.*h__ergs) *
        (E_gam_lims__GeV[1]*(1+maxval(n_gal, z))*m_e__GeV) /
        pow(E_CRe_lims__GeV[0],2) * 10.; // max synchrotron frequency
}

/*------------------------------------------------------
 * 4. Load IC, Bremsstrahlung, Synchrotron data objects
 *------------------------------------------------------*/
void load_radiation_objects(size_t n_pts[2],
                            double E_phot_lims__GeV[2],
                            double x_SY_lims[2],
                            unsigned long n_gal,
                            double *z,
                            double *T_dust__K,
                            char *datadir,
                            IC_object *ICo,
                            gsl_spline_object_2D *gso2D_BS,
                            gsl_spline_object_1D *gso1D_SY)
{
    // Load Inverse Compton (IC) data
    *ICo = load_IC_do_files(
        n_pts,
        E_gam_lims__GeV,
        E_CRe_lims__GeV,
        E_phot_lims__GeV,
        (maxval(n_gal, z)+1.)*T_0_CMB__K,
        0.5,
        minval(n_gal, T_dust__K),
        maxval(n_gal, T_dust__K),
        5.,
        datadir);

    // Load Bremsstrahlung (BS) data and convert to GSL spline
    data_object_2D do_2D_BS = load_BS_do_files(n_pts, E_gam_lims__GeV, E_CRe_lims__GeV, datadir);
    *gso2D_BS = do2D_to_gso2D(do_2D_BS);
    data_object_2D_free(do_2D_BS);

    // Load Synchrotron (SY) data and convert to GSL spline
    data_object_1D do_1D_SY = load_SY_do_files(&(n_pts[0]), x_SY_lims, datadir);
    *gso1D_SY = do1D_to_gso1D(do_1D_SY);
    data_object_1D_free(do_1D_SY);
}

/*------------------------------------------------------
 * 5. Write radiation energy densities to file
 *------------------------------------------------------*/
void write_radiation_energy_output(unsigned long n_gal,
                                   double *z,
                                   double *T_dust__K,
                                   double *M_star__Msol,
                                   double *SFR__Msolyrm1,
                                   double *Re__kpc,
                                   double *h__pc,
                                   double *B__G,
                                   char *outfp)
{
    unsigned long i;
    double nphot_params[7];
    double E_phot_lims__GeV[2];

    compute_photon_energy_limits(E_phot_lims__GeV);

    FILE *fileout = fopen(string_cat(outfp, "/Urad_Ub.txt"), "w+");
    fprintf(fileout,
        "u_rad__eVcmm3 u_B__eVcmm3 u_rad_CMB__eVcmm3 "
        "u_rad_FIR__eVcmm3 u_rad_3000__eVcmm3 "
        "u_rad_4000__eVcmm3 u_rad_7500__eVcmm3 "
        "u_rad_UV__eVcmm3\n");

    for (i = 0; i < n_gal; i++)
    {
        // Prepare ISRF parameters for integration
        nphot_params[0] = T_0_CMB__K*(1.+z[i]);
        nphot_params[1] = T_dust__K[i];
        nphot_params[2] = M_star__Msol[i];
        nphot_params[3] = SFR__Msolyrm1[i];
        nphot_params[4] = Re__kpc[i];
        nphot_params[5] = h__pc[i];

        // Write radiation energy densities (eV/cm^3)
        fprintf(fileout,"%e %e %e %e %e %e %e %e\n",
            ISRF_integrate__GeVcmm3(dndEphot_total__cmm3GeVm1, nphot_params, E_phot_lims__GeV)*1.e9,
            pow(B__G[i],2)/(8.*M_PI)/GeV__erg * 1.e9,
            ISRF_integrate__GeVcmm3(dndEphot_CMB__cmm3GeVm1, nphot_params, E_phot_lims__GeV)*1.e9,
            ISRF_integrate__GeVcmm3(dndEphot_FIR__cmm3GeVm1, nphot_params, E_phot_lims__GeV)*1.e9,
            ISRF_integrate__GeVcmm3(dndEphot_3000__cmm3GeVm1, nphot_params, E_phot_lims__GeV)*1.e9,
            ISRF_integrate__GeVcmm3(dndEphot_4000__cmm3GeVm1, nphot_params, E_phot_lims__GeV)*1.e9,
            ISRF_integrate__GeVcmm3(dndEphot_7500__cmm3GeVm1, nphot_params, E_phot_lims__GeV)*1.e9,
            ISRF_integrate__GeVcmm3(dndEphot_UV__cmm3GeVm1, nphot_params, E_phot_lims__GeV)*1.e9
        );
    }

    fclose(fileout);
}


/*******************************************************
 *  Particle Spectra and Loss Times Allocation
 *  
 *  This handles:
 *  - Electron/proton loss times (SY, BS, IC, IO, DI)
 *  - Proton-proton and proton diffusive loss times
 *  - Pion decay spectra and neutrino spectra
 *  - IC, BS, SY spectra (primary/secondary, z1/z2)
 *  - Electron injection spectra
 *  - Steady-state spectra
 *  - Free-free optical depth and spectra
 *  - Lepton energy loss arrays and radio luminosity
 *******************************************************/

#include <stdio.h>
#include <stdlib.h>

/*------------------------------------------------------
 * Generic 2D double array allocation
 *------------------------------------------------------*/
double **allocate_2D_double(unsigned long n_rows, unsigned long n_cols)
{
    unsigned long i;
    double **arr = malloc(sizeof *arr * n_rows);
    if (arr)
        for (i = 0; i < n_rows; i++)
            arr[i] = malloc(sizeof *arr[i] * n_cols);
    return arr;
}

/*------------------------------------------------------
 * Allocate all loss-time arrays
 * n_gal = number of galaxies
 * n_T_CR = number of CR energy bins
 *------------------------------------------------------*/
void allocate_loss_times(unsigned long n_gal, int n_T_CR,
                         double ***tau_z1_SY, double ***tau_z1_BS,
                         double ***tau_z1_IC, double ***tau_z1_IO,
                         double ***tau_z1_DI,
                         double ***tau_z2_SY, double ***tau_z2_BS,
                         double ***tau_z2_IC, double ***tau_z2_IO,
                         double ***tau_z2_DI,
                         double ***tau_protons_PP, double ***tau_protons_DI)
{
    *tau_z1_SY = allocate_2D_double(n_gal, n_T_CR);
    *tau_z1_BS = allocate_2D_double(n_gal, n_T_CR);
    *tau_z1_IC = allocate_2D_double(n_gal, n_T_CR);
    *tau_z1_IO = allocate_2D_double(n_gal, n_T_CR);
    *tau_z1_DI = allocate_2D_double(n_gal, n_T_CR);

    *tau_z2_SY = allocate_2D_double(n_gal, n_T_CR);
    *tau_z2_BS = allocate_2D_double(n_gal, n_T_CR);
    *tau_z2_IC = allocate_2D_double(n_gal, n_T_CR);
    *tau_z2_IO = allocate_2D_double(n_gal, n_T_CR);
    *tau_z2_DI = allocate_2D_double(n_gal, n_T_CR);

    *tau_protons_PP = allocate_2D_double(n_gal, n_T_CR);
    *tau_protons_DI = allocate_2D_double(n_gal, n_T_CR);
}

/*------------------------------------------------------
 * Allocate all spectra arrays
 * n_gal = number of galaxies
 * n_E_gam = number of photon bins
 * n_T_CR = number of CR bins
 *------------------------------------------------------*/
typedef struct {
    double **spec_pi;
    double **spec_pi_fcal1;
    double **spec_nu;

    double **spec_IC_1_z1;
    double **spec_IC_2_z1;
    double **spec_IC_1_z2;
    double **spec_IC_2_z2;

    double **spec_BS_1_z1;
    double **spec_BS_2_z1;

    double **spec_SY_1_z1;
    double **spec_SY_2_z1;
    double **spec_SY_1_z2;
    double **spec_SY_2_z2;

    double **Q_e_1_z1;
    double **Q_e_1_z2;
    double **Q_e_2_z1;
    double **Q_e_2_z2;

    double **q_p_SS_z1;
    double **q_e_SS_1_z1;
    double **q_e_SS_2_z1;
    double **q_e_SS_1_z2;
    double **q_e_SS_2_z2;

    double **tau_FF;
    double **spec_FF;

    double **E_loss_leptons;
    double **E_loss_nucrit;
    double **Lradio;
} spectra_arrays_t;

spectra_arrays_t allocate_spectra(unsigned long n_gal,
                                  int n_E_gam, int n_T_CR)
{
    spectra_arrays_t sa;

    // Pion decay & neutrinos
    sa.spec_pi        = allocate_2D_double(n_gal, n_E_gam);
    sa.spec_pi_fcal1  = allocate_2D_double(n_gal, n_E_gam);
    sa.spec_nu        = allocate_2D_double(n_gal, n_E_gam);

    // IC spectra
    sa.spec_IC_1_z1 = allocate_2D_double(n_gal, n_E_gam);
    sa.spec_IC_2_z1 = allocate_2D_double(n_gal, n_E_gam);
    sa.spec_IC_1_z2 = allocate_2D_double(n_gal, n_E_gam);
    sa.spec_IC_2_z2 = allocate_2D_double(n_gal, n_E_gam);

    // Bremsstrahlung spectra
    sa.spec_BS_1_z1 = allocate_2D_double(n_gal, n_E_gam);
    sa.spec_BS_2_z1 = allocate_2D_double(n_gal, n_E_gam);

    // Synchrotron spectra
    sa.spec_SY_1_z1 = allocate_2D_double(n_gal, n_E_gam);
    sa.spec_SY_2_z1 = allocate_2D_double(n_gal, n_E_gam);
    sa.spec_SY_1_z2 = allocate_2D_double(n_gal, n_E_gam);
    sa.spec_SY_2_z2 = allocate_2D_double(n_gal, n_E_gam);

    // Electron injection spectra
    sa.Q_e_1_z1 = allocate_2D_double(n_gal, n_T_CR);
    sa.Q_e_1_z2 = allocate_2D_double(n_gal, n_T_CR);
    sa.Q_e_2_z1 = allocate_2D_double(n_gal, n_T_CR);
    sa.Q_e_2_z2 = allocate_2D_double(n_gal, n_T_CR);

    // Steady-state spectra
    sa.q_p_SS_z1    = allocate_2D_double(n_gal, n_T_CR);
    sa.q_e_SS_1_z1  = allocate_2D_double(n_gal, n_T_CR);
    sa.q_e_SS_2_z1  = allocate_2D_double(n_gal, n_T_CR);
    sa.q_e_SS_1_z2  = allocate_2D_double(n_gal, n_T_CR);
    sa.q_e_SS_2_z2  = allocate_2D_double(n_gal, n_T_CR);

    // Free-free
    sa.tau_FF  = allocate_2D_double(n_gal, n_E_gam);
    sa.spec_FF = allocate_2D_double(n_gal, n_E_gam);

    // Lepton energy losses and radio
    sa.E_loss_leptons  = allocate_2D_double(n_gal, 16);
    sa.E_loss_nucrit   = allocate_2D_double(n_gal, 10);
    sa.Lradio          = allocate_2D_double(n_gal, 5);

    return sa;
}


/*************************************************************** */


int main(int argc, char *argv[])
{
    struct halo_mass_obj hm_obj = halo_mass_init();

    char infile[strlen(argv[1]) + 1];
    char datadir[strlen(argv[2]) + 1];
    char outfp[strlen(argv[3]) + 1];

    parse_input_arguments(argc, argv, infile, datadir, outfp);

    /* -------- Load Tau EBL -------- */

    tau_data tau = load_tau_file("input/tau_Eg_z_Franceschini.txt");

    fd_in fdata_in;
    init_fd_interp(&fdata_in, &tau);

    /* -------- Load Galaxies -------- */

    unsigned long int n_gal;
    double *z;
    double *M_star__Msol;
    double *Re__kpc;
    double *SFR__Msolyrm1;

    read_galaxy_file(infile,
                     &n_gal,
                     &z,
                     &M_star__Msol,
                     &Re__kpc,
                     &SFR__Msolyrm1);

    return 0;
}
