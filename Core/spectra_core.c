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


    //parallelize over galaxies
    #pragma omp parallel for schedule(guided) private(j, k, qe_1_z1_so, qe_2_z1_so, qe_1_z2_so, qe_2_z2_so, De_gso1D_z1, De_gso1D_z2, nphot_params, gso_1D_Q_inject_1_z1, gso_1D_Q_inject_2_z1, gso_1D_Q_inject_1_z2, gso_1D_Q_inject_2_z2, E_crit__GeV, seconds, gso2D_IC, gso2D_IC_Gamma, gso1D_fcal, t_loss_s ) firstprivate( gso2D_BS, gso1D_SY )

    for (i = 0; i < n_gal; i++)
    {
        seconds = time(NULL);
        printf("Working on galaxy no. %lu\n", i);
        fflush(stdout);

        nphot_params[0] = T_0_CMB__K * (1.+z[i]);
        nphot_params[1] = T_dust__K[i];
        nphot_params[2] = M_star__Msol[i];
        nphot_params[3] = SFR__Msolyrm1[i];
        nphot_params[4] = Re__kpc[i];
        nphot_params[5] = h__pc[i];
        nphot_params[6] = R_vir__kpc( R_half_mass__kpc( Re__kpc[i], z[i] ) );
    
        gso2D_IC = construct_IC_gso2D( 0, ICo, nphot_params );
        gso2D_IC_Gamma = construct_IC_gso2D( 1, ICo, nphot_params );


        //C here is used for the calcultion with dsig/dE, so need t_loss = t_col
        t_loss_s = 1./( n__cmm3[i] * sigma_pp_cm2 * eta_pp * c__cmsm1 );//proton loss time in s
        C[i] = SFR__Msolyrm1[i] * n_SN_Msolm1 * f_EtoCR * E_SN_erg * erg__GeV/yr__s * t_loss_s/CnormE[i];//proton normalization constant in GeV^-1


        gso1D_fcal = gsl_so1D( n_T_CR, T_CR__GeV, f_cal[i] );



        //Zone 1 diffusion
        De_gso1D_z1 = gsl_so1D( n_T_CR, E_CRe__GeV, D_e__cm2sm1[i] );


        //primary injection spectrum and steady state spline object per galaxy


        for (j = 0; j < n_T_CR; j++)
        {
            Q_e_1_z1[i][j] = J( T_CR__GeV[j], Ce_Esm1[i], q_e_inject, m_e__GeV, T_e_cutoff__GeV );
if (i == 10){Q_e_1_z1[i][j] = J( T_CR__GeV[j], Ce_Esm1[i], q_e_inject, m_e__GeV, T_e_cutoff__GeV ) * 0.1;}
        }
        //spline object on total energy - not kinetic
        gso_1D_Q_inject_1_z1 = gsl_so1D( n_T_CR, E_CRe__GeV, Q_e_1_z1[i] );

        //secondary injection spectrum and steady state spline object
        for (j = 0; j < n_T_CR; j++)
        {
            //Compute the spectra for the secondary electrons and then interpolate on them
            Q_e_2_z1[i][j] = q_e( T_CR__GeV[j], n_H__cmm3[i], C[i], T_p_cutoff__GeV, gso1D_fcal );
        }
        //spline object on total energy - not kinetic
        gso_1D_Q_inject_2_z1 = gsl_so1D( n_T_CR, E_CRe__GeV, Q_e_2_z1[i] );

//        printf("Before SS %lu\n", i);
//        fflush(stdout);
        //solve for steady state spectra in zone 1
        CRe_steadystate_solve( 1, E_CRe_lims__GeV, 500, n_H__cmm3[i], B__G[i], h__pc[i], 1, &gso2D_IC_Gamma, gso2D_BS, De_gso1D_z1, 
                               gso_1D_Q_inject_1_z1, gso_1D_Q_inject_2_z1, &qe_1_z1_so, &qe_2_z1_so );
            
 
        //Zone 2 diffusion
        De_gso1D_z2 = gsl_so1D( n_T_CR, E_CRe__GeV, D_e_z2__cm2sm1[i] );

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


//Fix cascade
//        C_gam = norm_casc_C( spec_emit, E_GeV, n_Esteps, z[i], fdata_in );

        for (j = 0; j < n_E_gam; j++)
        {

            tau_gg[i][j] = tau_gg_gal_BW( E_gam__GeV[j], dndEphot_total__cmm3GeVm1, nphot_params, E_phot_lims__GeV, h__pc[i] );//gamma-gamma optical depth

            tau_FF[i][j] = tau_FF_MK( E_gam__GeV[j], Sig_SFR__Msolyrm1pcm2[i], 1.e4 );//free-free optical depth

            spec_pi[i][j] = eps_pi( E_gam__GeV[j], n_H__cmm3[i], C[i], T_p_cutoff__GeV, gso1D_fcal );//pion decay spectrum


            spec_IC_1_z1[i][j] = eps_IC_3( E_gam__GeV[j], gso2D_IC, qe_1_z1_so ) * exp(-tau_FF[i][j]);
            spec_IC_2_z1[i][j] = eps_IC_3( E_gam__GeV[j], gso2D_IC, qe_2_z1_so ) * exp(-tau_FF[i][j]);

            spec_BS_1_z1[i][j] = eps_BS_3( E_gam__GeV[j], n_H__cmm3[i], gso2D_BS, qe_1_z1_so ) * exp(-tau_FF[i][j]);
            spec_BS_2_z1[i][j] = eps_BS_3( E_gam__GeV[j], n_H__cmm3[i], gso2D_BS, qe_2_z1_so ) * exp(-tau_FF[i][j]);

            spec_SY_1_z1[i][j] = eps_SY_4( E_gam__GeV[j], B__G[i], gso1D_SY, qe_1_z1_so ) * exp(-tau_FF[i][j]);
            spec_SY_2_z1[i][j] = eps_SY_4( E_gam__GeV[j], B__G[i], gso1D_SY, qe_2_z1_so ) * exp(-tau_FF[i][j]);


            spec_IC_1_z2[i][j] = eps_IC_3( E_gam__GeV[j], gso2D_IC, qe_1_z2_so );
            spec_IC_2_z2[i][j] = eps_IC_3( E_gam__GeV[j], gso2D_IC, qe_2_z2_so );

            spec_SY_1_z2[i][j] = eps_SY_4( E_gam__GeV[j], B_halo__G[i], gso1D_SY, qe_1_z2_so );
            spec_SY_2_z2[i][j] = eps_SY_4( E_gam__GeV[j], B_halo__G[i], gso1D_SY, qe_2_z2_so );




            spec_pi_fcal1[i][j] = eps_pi_fcal1( E_gam__GeV[j], n_H__cmm3[i], C[i], T_p_cutoff__GeV, gso1D_fcal );

            spec_nu[i][j] = q_nu( E_gam__GeV[j], n_H__cmm3[i], C[i], T_p_cutoff__GeV, gso1D_fcal );




            spec_FF[i][j] = eps_FF( E_gam__GeV[j], Re__kpc[i], 1.e4, tau_FF[i][j] );


//            specs_obs[i][j] = (Phi_out.Phi + specs_brems_1st[i][j] + specs_brems_2nd[i][j] + specs_IC_1st[i][j] + specs_IC_2nd[i][j] + 
//                               specs_sync_1st[i][j] + specs_sync_2nd[i][j])* mod * exp(-tau_gg[i][j]) * exp(-tau_EBL[i][j]);

            //call the function again this time without redshift
/*
            Phi_out = Phi( E_GeV[j], n_H__cmm3[i], C[i], T_p_cutoff__GeV, gso1D_fcal );


            sIC1 = eps_IC_3( E_GeV[j], gso_2D_total, qe_1_z1_so );
            sIC2 = eps_IC_3( E_GeV[j], gso_2D_total, qe_2_z1_so );
            sb1 = eps_BS_3( E_GeV[j], n_H__cmm3[i], qe_1_z1_so );
            sb2 = eps_BS_3( E_GeV[j], n_H__cmm3[i], qe_2_z1_so );

            ss1 = eps_SY_4( E_GeV[j], B__G[i], gso1D_SY, qe_1_z1_so );
            ss2 = eps_SY_4( E_GeV[j], B__G[i], gso1D_SY, qe_2_z1_so );


            //total energy emitted
            specs_L_emit[i][j] = ( Phi_out.Phi * vol + sb1 + sb2 +sIC1 + sIC2 + ss1 + ss2 ) * exp(-tau_gg[i][j]);
            //for cascade
            spec_emit[j] = ( Phi_out.Phi + sb1 + sb2 +sIC1 + sIC2 + ss1 + ss2 )/vol * exp(-tau_gg[i][j]);

            specs_casc_obs[i][j] = dndE_gam_casc( (1.+z[i]) * E_GeV[j], z[i], C_gam, fdata_in ) * mod;
            specs_casc[i][j] = dndE_gam_casc( E_GeV[j], z[i], C_gam, fdata_in ) * vol;
*/


        }

/*
    if (i == 0)
    {
        gsl_spline_object_2D gso2D_3000, gso2D_4000, gso2D_7500, gso2D_UV, gso2D_FIR, gso2D_CMB;

        return_IC_gso2D( ICo, nphot_params, &gso2D_3000, &gso2D_4000, &gso2D_7500, &gso2D_UV, &gso2D_FIR, &gso2D_CMB );

        write_ICspectra( n_E_gam, E_gam__GeV, qe_1_z1_so, qe_2_z1_so, qe_1_z2_so, qe_2_z2_so, gso2D_3000, gso2D_4000, gso2D_7500, 
                         gso2D_UV, gso2D_FIR, gso2D_CMB, string_cat(outfp, "/IC_spectra.txt") );

        gsl_so2D_free( gso2D_3000 );
        gsl_so2D_free( gso2D_4000 );
        gsl_so2D_free( gso2D_7500 );
        gsl_so2D_free( gso2D_UV );
        gsl_so2D_free( gso2D_FIR );
        gsl_so2D_free( gso2D_CMB );        
    }
*/





        E_crit__GeV = sqrt( (2. * 1.49e9 * m_e__g*c__cmsm1)/(3. * B__G[i]*e__esu) ) * M_PI * m_e__GeV;//critical energy for 1.49 GHz synchrotron emission
        E_loss_nucrit[i][0] = E_crit__GeV/tau_BS_fulltest__s( E_crit__GeV, E_CRe_lims__GeV, n_H__cmm3[i], gso2D_BS );//bremsstrahlung loss rate at critical energy
        E_loss_nucrit[i][1] = E_crit__GeV/tau_sync__s( E_crit__GeV, B__G[i] );//synchrotron loss rate at critical energy
        E_loss_nucrit[i][2] = E_crit__GeV/tau_IC_fulltest__s( E_crit__GeV, E_CRe_lims__GeV, gso2D_IC_Gamma );//inverse Compton loss rate at critical energy
        E_loss_nucrit[i][3] = E_crit__GeV/tau_ion__s( E_crit__GeV, n_H__cmm3[i] );//ionization loss rate at critical energy
        E_loss_nucrit[i][4] = E_crit__GeV/tau_diff__s( E_crit__GeV, h__pc[i], De_gso1D_z1 );//diffusive loss rate at critical energy
        E_crit__GeV = sqrt( (2. * 1.49e9 * m_e__g*c__cmsm1)/(3. * B_halo__G[i]*e__esu) ) * M_PI * m_e__GeV;//critical energy for 1.49 GHz synchrotron emission in halo
        E_loss_nucrit[i][5] = E_crit__GeV/tau_BS_fulltest__s( E_crit__GeV, E_CRe_lims__GeV, n_H__cmm3[i]/1000., gso2D_BS );//bremsstrahlung loss rate at critical energy in halo
        E_loss_nucrit[i][6] = E_crit__GeV/tau_sync__s( E_crit__GeV, B_halo__G[i] );//synchrotron loss rate at critical energy in halo
        E_loss_nucrit[i][7] = E_crit__GeV/tau_IC_fulltest__s( E_crit__GeV, E_CRe_lims__GeV, gso2D_IC_Gamma );//inverse Compton loss rate at critical energy in halo
        E_loss_nucrit[i][8] = E_crit__GeV/tau_plasma__s( E_crit__GeV, n_H__cmm3[i]/1000. );//plasma loss rate at critical energy in halo
        E_loss_nucrit[i][9] = E_crit__GeV/tau_diff__s( E_crit__GeV, 50.*h__pc[i], De_gso1D_z2 );//diffusive loss rate at critical energy in halo


        Lradio[i][0] = eps_SY_4( 1.49e9 * h__GeVs, B__G[i], gso1D_SY, qe_1_z1_so ) * 1.49e9 * h__Js * h__GeVs * exp(-tau_FF_MK( 1.49e9 * h__GeVs, Sig_SFR__Msolyrm1pcm2[i], 1.e4 ));
        Lradio[i][1] = eps_SY_4( 1.49e9 * h__GeVs, B__G[i], gso1D_SY, qe_2_z1_so ) * 1.49e9 * h__Js * h__GeVs * exp(-tau_FF_MK( 1.49e9 * h__GeVs, Sig_SFR__Msolyrm1pcm2[i], 1.e4 ));
        Lradio[i][2] = eps_SY_4( 1.49e9 * h__GeVs, B_halo__G[i], gso1D_SY, qe_1_z2_so ) * 1.49e9 * h__Js * h__GeVs;
        Lradio[i][3] = eps_SY_4( 1.49e9 * h__GeVs, B_halo__G[i], gso1D_SY, qe_2_z2_so ) * 1.49e9 * h__Js * h__GeVs;
        Lradio[i][4] = eps_FF( 1.49e9 * h__GeVs, Re__kpc[i], 1.e4, tau_FF_MK( 1.49e9 * h__GeVs, Sig_SFR__Msolyrm1pcm2[i], 1.e4 ) ) * 1.49e9 * h__Js * h__GeVs;


        for (j = 0; j < n_T_CR; j++)
        {
            q_p_SS_z1[i][j] = J( T_CR__GeV[j], C[i], q_p_inject, m_p__GeV, T_p_cutoff__GeV ) * f_cal[i][j];//steady state proton spectrum at z1
            q_e_SS_1_z1[i][j] = gsl_so1D_eval( qe_1_z1_so, E_CRe__GeV[j] );//steady state primary electron spectrum at z1
            q_e_SS_2_z1[i][j] = gsl_so1D_eval( qe_2_z1_so, E_CRe__GeV[j] );//steady state secondary electron spectrum at z1
            q_e_SS_1_z2[i][j] = gsl_so1D_eval( qe_1_z2_so, E_CRe__GeV[j] );//steady state primary electron spectrum at z2
            q_e_SS_2_z2[i][j] = gsl_so1D_eval( qe_2_z2_so, E_CRe__GeV[j] );//steady state secondary electron spectrum at z2
        }


        E_loss_leptons[i][0] = spec_integrate( n_T_CR, T_CR__GeV, Q_e_1_z1[i] ); //f_CRe_CRp * SFR__Msolyrm1[i] * n_SN_Msolm1 * f_EtoCR * E_SN_erg * erg__GeV/yr__s;
        E_loss_leptons[i][1] = spec_integrate( n_T_CR, T_CR__GeV, Q_e_2_z1[i] );// f_CRe_CRp * SFR__Msolyrm1[i] * n_SN_Msolm1 * f_EtoCR * E_SN_erg * erg__GeV/yr__s;
        E_loss_leptons[i][2] = spec_integrate( n_E_gam, E_gam__GeV, spec_SY_1_z1[i] )/E_loss_leptons[i][0];//synchrotron loss rate from primary electrons at z1
        E_loss_leptons[i][3] = spec_integrate( n_E_gam, E_gam__GeV, spec_IC_1_z1[i] )/E_loss_leptons[i][0];//inverse Compton loss rate from primary electrons at z1
        E_loss_leptons[i][4] = spec_integrate( n_E_gam, E_gam__GeV, spec_BS_1_z1[i] )/E_loss_leptons[i][0];//bremsstrahlung loss rate from primary electrons at z1
        E_loss_leptons[i][5] = spec_integrate( n_E_gam, E_gam__GeV, spec_SY_1_z2[i] )/E_loss_leptons[i][0];//synchrotron loss rate from primary electrons at z2
        E_loss_leptons[i][6] = spec_integrate( n_E_gam, E_gam__GeV, spec_IC_1_z2[i] )/E_loss_leptons[i][0];//inverse Compton loss rate from primary electrons at z2
        E_loss_leptons[i][7] = 0.; //spec_integrate( n_E_gam, E_gam__GeV, spec_BS_1_z2[i] )/E_loss_leptons[i][0];
        E_loss_leptons[i][8] = spec_integrate( n_T_CR, T_CR__GeV, Q_e_1_z2[i] )/E_loss_leptons[i][0];//total escape rate of primary electrons at z2

        E_loss_leptons[i][9] = spec_integrate( n_E_gam, E_gam__GeV, spec_SY_2_z1[i] )/E_loss_leptons[i][1];//synchrotron loss rate from secondary electrons at z1
        E_loss_leptons[i][10] = spec_integrate( n_E_gam, E_gam__GeV, spec_IC_2_z1[i] )/E_loss_leptons[i][1];//inverse Compton loss rate from secondary electrons at z1
        E_loss_leptons[i][11] = spec_integrate( n_E_gam, E_gam__GeV, spec_BS_2_z1[i] )/E_loss_leptons[i][1];//bremsstrahlung loss rate from secondary electrons at z1
        E_loss_leptons[i][12] = spec_integrate( n_E_gam, E_gam__GeV, spec_SY_2_z2[i] )/E_loss_leptons[i][1];//synchrotron loss rate from secondary electrons at z2
        E_loss_leptons[i][13] = spec_integrate( n_E_gam, E_gam__GeV, spec_IC_2_z2[i] )/E_loss_leptons[i][1];//inverse Compton loss rate from secondary electrons at z2
        E_loss_leptons[i][14] = 0.; //spec_integrate( n_E_gam, E_gam__GeV, spec_BS_2_z2[i] )/E_loss_leptons[i][1];
        E_loss_leptons[i][15] = spec_integrate( n_T_CR, T_CR__GeV, Q_e_2_z2[i] )/E_loss_leptons[i][1];//total escape rate of secondary electrons at z2



        //Compute some loss times SY, BS, IC, IO, DI
        for (j = 0; j < n_T_CR; j++)
        {
            tau_loss_z1_SY[i][j] = tau_sync__s( E_CRe__GeV[j], B__G[i] );
            tau_loss_z1_BS[i][j] = tau_BS_fulltest__s( E_CRe__GeV[j], E_CRe_lims__GeV, n_H__cmm3[i], gso2D_BS );
            tau_loss_z1_IC[i][j] = tau_IC_fulltest__s( E_CRe__GeV[j], E_CRe_lims__GeV, gso2D_IC_Gamma );
            tau_loss_z1_IO[i][j] = tau_ion__s( E_CRe__GeV[j], n_H__cmm3[i] );
            tau_loss_z1_DI[i][j] = tau_diff__s( E_CRe__GeV[j], h__pc[i], De_gso1D_z1 );

            tau_loss_z2_SY[i][j] = tau_sync__s( E_CRe__GeV[j], B_halo__G[i] );
            tau_loss_z2_BS[i][j] = tau_BS_fulltest__s( E_CRe__GeV[j], E_CRe_lims__GeV, n_H__cmm3[i]/1000., gso2D_BS );
            tau_loss_z2_IC[i][j] = tau_IC_fulltest__s( E_CRe__GeV[j], E_CRe_lims__GeV, gso2D_IC_Gamma );
            tau_loss_z2_IO[i][j] = tau_ion__s( E_CRe__GeV[j], n_H__cmm3[i]/1000. );
            tau_loss_z2_DI[i][j] = tau_diff__s( E_CRe__GeV[j], h__pc[i]*50., De_gso1D_z2 );

            tau_loss_protons_PP[i][j] = 1./( n__cmm3[i] * sigma_pp_cm2 * eta_pp * c__cmsm1 );
            tau_loss_protons_DI[i][j] = pow(h__pc[i] * pc__cm,2)/D__cm2sm1[i][j];

        }


        gsl_so2D_free( gso2D_IC );
        gsl_so2D_free( gso2D_IC_Gamma );


        gsl_so1D_free( gso1D_fcal );
        gsl_so1D_free( qe_1_z1_so );
        gsl_so1D_free( qe_2_z1_so );
        gsl_so1D_free( qe_1_z2_so );
        gsl_so1D_free( qe_2_z2_so );

        gsl_so1D_free( gso_1D_Q_inject_1_z1 );
        gsl_so1D_free( gso_1D_Q_inject_2_z1 );
        gsl_so1D_free( gso_1D_Q_inject_1_z2 );
        gsl_so1D_free( gso_1D_Q_inject_2_z2 );
        gsl_so1D_free( De_gso1D_z1 );
        gsl_so1D_free( De_gso1D_z2 );

        seconds = time(NULL) - seconds;
        printf("Galaxy %lu log10(time/sec): %le \n", i, log10(seconds));
        fflush(stdout);



//test_IC_interp_obj( E_CRe_lims__GeV, gso_2D_total_low, gso_2D_total, gso_2D_IC_Gamma, qe_1_z1_so );

    }


    double **array2Dlist[4] = { f_cal, D__cm2sm1, D_e__cm2sm1, D_e_z2__cm2sm1 };
    for (i = 0; i < 4; i++)
    {
        free2D( n_gal, array2Dlist[i] );
    }

    gsl_so2D_free( gso2D_BS );
    gsl_so1D_free( gso1D_SY );
    IC_object_free( ICo );

