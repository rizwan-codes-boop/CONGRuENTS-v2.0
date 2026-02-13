#ifndef spectra_funcs_h
#define spectra_funcs_h

#include <stdio.h>
#include <math.h>
#include <cubature.h>

#include "spline_custom.h"

extern double q_p_inject;
extern double q_e_inject;

extern double T_CR_lims__GeV[2];
extern double E_CRe_lims__GeV[2];


extern double Delta_x[11];
extern double Phi_1H[11];
extern double Phi_2H[11];

/* T_min T_max - minimum and maximum CR energy to integrate over */



double T_p_low = 1.e-3; //GeV
double T_p_high = 1.e8; //1e6 //GeV

double T_p_norm__GeV[2] = { 0., 1.e8 }; //0.9383; //GeV


double E_e_low_norm = 1.e-3;
double E_e_high_norm = 1.e8;

#define T_p_th (2. * m_pi0__GeV + pow(m_pi0__GeV, 2)/(2. * m_p__GeV)) /* GeV */ //0.2797



//Bremsstrahlung Peretti 2019 following Stecker 2971
double sigma_brems_mb = 34.;

double K_pi = 0.17;

struct Phi_out {
  double Phi;
  double Phi_fcal1;
  };

/*

struct gsl_spline_obj gsl_so( double *xdata, double *ydata, size_t n_data ){
  struct gsl_spline_obj gsl_so;
  gsl_so.acc = gsl_interp_accel_alloc();
  gsl_so.spline = gsl_spline_alloc(gsl_interp_linear, n_data);
  gsl_spline_init( gsl_so.spline, xdata, ydata, n_data );
  return gsl_so;
  }
*/
/*
struct radiation_fields {
  int n_radcomp;
  double *E_peak__GeV;
  double *u_rad__GeVcmm3;
  };
*/

typedef struct fdata_in {
  double E_g;
  size_t nx;
  size_t ny;
  double * xa;
  double * ya;
  double * za;
  const gsl_interp2d_type *T;
  gsl_interp2d *interp;
  gsl_interp_accel *xacc;
  gsl_interp_accel *yacc;
} fd_in;

struct hcub_data {
  double E_gam;
  double B_G;
  double T_dust__K;
  double T_optBB__K;
  int n_radcomp;
  double *E_peaks__GeV;
  double *urads;
  double urad;
  double E_peak__GeV;
  double uradFIR;
  double uradopt;
  double uradtot;

//  struct radiation_fields radfield;
spline1D De_so;

spline2D gso_2D_so;

double z;
double h;
double Sigstar;
double C;
double C_e;
double n_H;
double E_cut;
double E_cut_e;

spline1D gsl_so;
spline1D gso1D_fcal;
};



#endif