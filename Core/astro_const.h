#ifndef astro_const_H
#define astro_const_H

/* =============================
   Mathematical constants
   ============================= */

#define M_PI 3.14159265358979323846


/* =============================
   Particle masses 
   ============================= */

#define m_e_GeV        0.00051099895     // electron
#define m_p_GeV        0.9382720813      // proton  
#define m_n_GeV        0.9395654133      // neutron 
#define M_PI0_GEV      0.1349768         // neutral pion
#define M_PI_CH_GEV    0.13957039        // charged pion 
#define m_e__g         9.10938356e-28    // Electron mass in grams
#define m_p__g         1.6726219e-24     // Proton mass in grams


/* =============================
   Gas / Hydrogen properties
   ============================= */

#define MU_H         1.4               // Mean molecular weight per hydrogen nucleus
#define MU_P         1.17              // Mean molecular weight per particle
#define m_H_kg       1.6735575e-27     // Mass of hydrogen atom in kg


/* =============================
   Unit conversions
   ============================= */

#define pc_cm        3.085677581e18    // Parsec to centimetre conversion
#define Msol_kg      1.98847e30        // Solar mass in kg
#define erg__GeV     0.624150907e3     // Energy conversion from erg to GeV
#define yr__s          3.15576e7       // Year to seconds conversion


/* =============================
   Astrophysical gravity
   (kpc (km/s)^2 Msun^-1)
   ============================= */

#define G_H          4.302e-3          // Gravitational constant in (km/s)² pc / Msol


/* =============================
   Fundamental constants
   ============================= */

#define c__cmsm1        2.99792458e10     // Speed of light in cm/s
#define e__esu          4.80320427e-10    // Electron charge in electrostatic units (esu)
#define h__ergs         6.62607015e-27    // Planck constant in erg·s
#define T_0_CMB_K       2.725             // CMB temperature today in Kelvin


#endif 

