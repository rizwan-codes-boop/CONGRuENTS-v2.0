/*
 * physical_constants.h
 *
 * All physical constants and unit conversions used throughout the
 * spectra pipeline.  Values are in CGS unless the name suffix
 * specifies otherwise (e.g. __kg, __GeV, __Js).
 *
 * Naming convention mirrors the rest of the codebase:
 *   <quantity>__<unit>   e.g.  m_H__kg,  pc__cm,  h__GeVs
 *
 * Sources:
 *   CODATA 2018 / IAU 2015 nominal solar values where applicable.
 */

#ifndef PHYSICAL_CONSTANTS_H
#define PHYSICAL_CONSTANTS_H

/* ============================================================
 * Fundamental constants (CGS)
 * ============================================================ */

/* Speed of light [cm s^-1] */
#define c__cmsm1        2.99792458e10

/* Planck constant [erg s] */
#define h__ergs         6.62607015e-27

/* Planck constant [J s] */
#define h__Js           6.62607015e-34

/* Planck constant [GeV s] */
#define h__GeVs         4.135667696e-24

/* Elementary charge [esu = statcoulomb] */
#define e__esu          4.80320427e-10

/* Boltzmann constant [erg K^-1] */
#define k_B__ergKm1     1.380649e-16

/* ============================================================
 * Particle masses
 * ============================================================ */

/* Proton mass [g] */
#define m_p__g          1.67262192369e-24

/* Proton mass [kg] */
#define m_p__kg         1.67262192369e-27

/* Proton mass [GeV c^-2]  (rest energy) */
#define m_p__GeV        0.93827208816

/* Electron mass [g] */
#define m_e__g          9.1093837015e-28

/* Electron mass [kg] */
#define m_e__kg         9.1093837015e-31

/* Electron mass [GeV c^-2]  (rest energy) */
#define m_e__GeV        0.51099895000e-3

/* Hydrogen atom mass [g]  (used for gas density conversions) */
#define m_H__g          1.6735575e-24

/* Hydrogen atom mass [kg] */
#define m_H__kg         1.6735575e-27

/* ============================================================
 * Astronomical unit conversions
 * ============================================================ */

/* Parsec [cm] */
#define pc__cm          3.085677581e18

/* Kiloparsec [cm] */
#define kpc__cm         3.085677581e21

/* Megaparsec [cm] */
#define Mpc__cm         3.085677581e24

/* Solar mass [kg] */
#define Msol__kg        1.989e30

/* Solar mass [g] */
#define Msol__g         1.989e33

/* Year [s] */
#define yr__s           3.15576e7

/* ============================================================
 * Energy conversions
 * ============================================================ */

/* erg → GeV */
#define erg__GeV        6.24150907e2

/* GeV → erg */
#define GeV__erg        1.60217663e-3

/* eV → erg */
#define eV__erg         1.60217663e-12

/* ============================================================
 * Radiation / cosmology
 * ============================================================ */

/* CMB temperature today [K] */
#define T_0_CMB__K      2.72548

/* Stefan–Boltzmann constant [erg cm^-2 s^-1 K^-4] */
#define sigma_SB        5.670374419e-5

/* Radiation constant  a = 4 sigma_SB / c  [erg cm^-3 K^-4] */
#define a_rad           7.565723e-15

/* ============================================================
 * Gravitational constant
 * ============================================================ */

/* G in units convenient for hydrostatic equilibrium:
 * [(km/s)^2 pc Msol^-1]
 * Used in the scale-height formula. */
#define G_h             4.302e-3

/* G [cm^3 g^-1 s^-2]  (CGS) */
#define G__cgs          6.67430e-8

/* ============================================================
 * Gamma-ray energy limits
 * Used in spectra_core_wrapper.c to set synchrotron frequency bounds.
 * Mirror the global E_gam_lims__GeV array in the original spectra.c.
 * ============================================================ */

/* Minimum gamma-ray energy [GeV] */
#define E_gam_lims_min  1.0e-16

/* Maximum gamma-ray energy [GeV] */
#define E_gam_lims_max  1.0e8

#endif /* PHYSICAL_CONSTANTS_H */