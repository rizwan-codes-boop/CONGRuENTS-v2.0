#ifndef ionisation_h
#define ionisation_h

#include <math.h>
#include "physical_constants.h"


/*
 * Taken from Schlickeiser (Cosmic Ray Astrophysics) p. 99
 * for a medium dominated by neutral hydrogen so Chi ~ 0 and Z ~ 1
 * The average excitation energy for hydrogen is taken as 15 eV and 41.5 eV for Helium
 * Gas is made up pf 91% H and 9% He 
 */


// Ionisation energy loss rate in GeV/s(atomic excitation and ionisation losses)
double dEdtm1_ion__GeVsm1( double E_e__GeV, double n_H__cmm3 )
{
    double gamma = E_e__GeV/m_e__GeV;
    double mu_ISM = 1.1;
    return -9./4. * c__cmsm1 * sigma_T__mb * mb__cm2 * m_e__GeV * n_H__cmm3 * mu_ISM * (log(gamma) + 0.91 * 2./3. * 
           log( m_e__GeV * 1.e9/15.0 ) + 2. * 0.09 * 2./3. * log( m_e__GeV * 1.e9/41.5 ));
}

// Derivative of ionisation energy loss rate with respect to energy in s^-1
// required for solving the steady state transport equation
double deldelEm1dEdtm1_ion__sm1( double E_e__GeV, double n_H__cmm3 )
{
    double mu_ISM = 1.1;
    return -9./4. * c__cmsm1 * sigma_T__mb * mb__cm2 * m_e__GeV * n_H__cmm3 * mu_ISM * (1./E_e__GeV);
}

// Ionisation loss timescale in seconds
double tau_ion__s( double E_e__GeV, double n_H__cmm3 )
{
    return -1.* E_e__GeV/dEdtm1_ion__GeVsm1( E_e__GeV, n_H__cmm3 );
}

// Plasma energy loss rate in GeV/s (Coulomb losses)
double dEdtm1_plasma__GeVsm1( double E_e__GeV, double n_H__cmm3 )
{
    double gamma = E_e__GeV/m_e__GeV;
    double mu_ISM = 1.1;
    double nu_p__Hz = sqrt( n_H__cmm3 * mu_ISM/( M_PI * m_e__g ) ) * e__esu;
    return -3./4. * c__cmsm1 * sigma_T__mb * mb__cm2 * m_e__GeV * n_H__cmm3 * mu_ISM * (log(gamma) + 2. * log(m_e__GeV/(h__GeVs * nu_p__Hz)));
}

// Energy derivative of plasma energy loss rate in s^-1
double deldelEm1dEdtm1_plasma__sm1( double E_e__GeV, double n_H__cmm3 )
{
    double mu_ISM = 1.1;
    return -3./4. * c__cmsm1 * sigma_T__mb * mb__cm2 * m_e__GeV * n_H__cmm3 * mu_ISM * (1./E_e__GeV);
}

// Plasma loss timescale in seconds
double tau_plasma__s( double E_e__GeV, double n_H__cmm3 )
{
    return -1.* E_e__GeV/dEdtm1_plasma__GeVsm1( E_e__GeV, n_H__cmm3 );
}

#endif
