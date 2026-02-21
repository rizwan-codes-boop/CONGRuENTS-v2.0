#!/usr/bin/env python3
"""
================================================================================
HALO_INIT.PY - Initialization and Data Management for Halo CR Calculations
================================================================================

This module handles:
  1. Loading galaxy data and disc outputs
  2. Setting up energy grids
  3. Loading the C library (spectra_halo.so)
  4. Wrapping C functions with Python-friendly interfaces
  5. Managing arrays and data structures

Usage:
------
    from halo_init import HaloCalculator
    
    hc = HaloCalculator(
        galaxy_file='galaxies.txt',
        disc_dir='output_disc/',
        n_threads=8
    )
    
    # Compute halo properties
    hc.compute_halo_properties()
    
    # Compute transport
    hc.compute_halo_transport()
    
    # Get results
    results = hc.get_results()

Author: [Your name]
Date: 2026
================================================================================
"""

import numpy as np
import ctypes as ct
from pathlib import Path
import sys


class HaloCalculator:
    """
    Main class for halo CR calculations
    
    Manages the interface between Python data structures and C computational
    routines for cosmic-ray transport and emission in galaxy halos.
    """
    
    def __init__(self, galaxy_file, disc_dir, data_dir=None, n_threads=None,
                 n_E_CR=1000, n_E_gam=500):
        """
        Initialize the halo calculator
        
        Parameters
        ----------
        galaxy_file : str
            Path to galaxy catalogue (z, M_star, Re, SFR)
        disc_dir : str
            Directory with disc outputs (gal_data.txt, CR_specs.txt, fcal.txt)
        data_dir : str, optional
            Directory with interpolation tables (not used in simplified version)
        n_threads : int, optional
            Number of OpenMP threads (default: all available)
        n_E_CR : int
            Number of cosmic-ray energy bins (default: 1000)
        n_E_gam : int
            Number of gamma-ray energy bins (default: 500)
        """
        
        print("="*60)
        print("Halo CR Calculator - Initialization")
        print("="*60)
        
        self.galaxy_file = Path(galaxy_file)
        self.disc_dir = Path(disc_dir)
        self.data_dir = Path(data_dir) if data_dir else None
        
        self.n_E_CR = n_E_CR
        self.n_E_gam = n_E_gam
        
        # Load C library
        self._load_c_library()
        
        # Set number of threads
        if n_threads is not None:
            self.lib.set_num_threads(n_threads)
        
        print(f"OpenMP threads: {self.lib.get_num_threads()}")
        
        # Load data
        self._load_galaxy_data()
        self._load_disc_data()
        self._create_energy_grids()
        
        # Initialize result arrays
        self._initialize_arrays()
        
        print("\n✓ Initialization complete")
        print(f"  Galaxies: {self.n_gal}")
        print(f"  CR energy bins: {self.n_E_CR}")
        print(f"  Gamma-ray bins: {self.n_E_gam}")
        print("="*60 + "\n")
    
    
    def _load_c_library(self):
        """Load the C shared library"""
        
        # Try to find the library
        lib_name = 'spectra_halo.so'
        lib_paths = [
            Path('.') / lib_name,
            Path(__file__).parent / lib_name,
            Path('/usr/local/lib') / lib_name,
        ]
        
        lib_path = None
        for path in lib_paths:
            if path.exists():
                lib_path = path
                break
        
        if lib_path is None:
            raise FileNotFoundError(
                f"Could not find {lib_name}. "
                f"Please compile using 'make' first."
            )
        
        print(f"Loading C library: {lib_path}")
        self.lib = ct.CDLL(str(lib_path))
        
        # Define function signatures
        self._define_c_signatures()
    
    
    def _define_c_signatures(self):
        """Define ctypes signatures for C functions"""
        
        # Utility functions
        self.lib.set_num_threads.argtypes = [ct.c_int]
        self.lib.set_num_threads.restype = None
        
        self.lib.get_num_threads.argtypes = []
        self.lib.get_num_threads.restype = ct.c_int
        
        # compute_halo_properties
        self.lib.compute_halo_properties.argtypes = [
            ct.c_ulong,                     # n_gal
            ct.POINTER(ct.c_double),        # n_H_disc
            ct.POINTER(ct.c_double),        # B_disc
            ct.POINTER(ct.c_double),        # h_disc
            ct.POINTER(ct.c_double),        # SFR
            ct.POINTER(ct.c_double),        # M_star
            ct.POINTER(ct.c_double),        # n_H_halo (output)
            ct.POINTER(ct.c_double),        # B_halo (output)
            ct.POINTER(ct.c_double),        # h_halo (output)
        ]
        self.lib.compute_halo_properties.restype = ct.c_int
        
        # compute_halo_diffusion
        self.lib.compute_halo_diffusion.argtypes = [
            ct.c_ulong,                     # n_gal
            ct.c_uint,                      # n_E
            ct.POINTER(ct.c_double),        # T_CR
            ct.POINTER(ct.c_double),        # h_disc
            ct.POINTER(ct.c_double),        # n_H_disc
            ct.POINTER(ct.c_double),        # sig_gas
            ct.POINTER(ct.c_double),        # f_cal
            ct.POINTER(ct.c_double),        # C
            ct.c_double,                    # chi
            ct.c_double,                    # M_A
            ct.c_double,                    # q_inject
            ct.POINTER(ct.c_double),        # D_e_halo (output)
        ]
        self.lib.compute_halo_diffusion.restype = ct.c_int
        
        # compute_halo_injection
        self.lib.compute_halo_injection.argtypes = [
            ct.c_ulong,                     # n_gal
            ct.c_uint,                      # n_E
            ct.POINTER(ct.c_double),        # h_disc
            ct.POINTER(ct.c_double),        # q_e_disc
            ct.POINTER(ct.c_double),        # D_e_disc
            ct.POINTER(ct.c_double),        # Q_e_halo (output)
        ]
        self.lib.compute_halo_injection.restype = ct.c_int
        
        # Loss time functions
        for func_name in ['compute_synchrotron_loss_time',
                         'compute_plasma_loss_time',
                         'compute_diffusion_loss_time']:
            func = getattr(self.lib, func_name)
            func.argtypes = [
                ct.c_ulong,                 # n_gal
                ct.c_uint,                  # n_E
                ct.POINTER(ct.c_double),    # E_CRe
                ct.POINTER(ct.c_double),    # parameter (B or n_H or h)
                ct.POINTER(ct.c_double),    # tau (output)
            ]
            func.restype = ct.c_int
        
        # solve_halo_steady_state
        self.lib.solve_halo_steady_state.argtypes = [
            ct.c_ulong,                     # n_gal
            ct.c_uint,                      # n_E
            ct.POINTER(ct.c_double),        # Q_e_halo
            ct.POINTER(ct.c_double),        # tau_sync
            ct.POINTER(ct.c_double),        # tau_plasma
            ct.POINTER(ct.c_double),        # tau_diff
            ct.POINTER(ct.c_double),        # tau_IC (can be NULL)
            ct.POINTER(ct.c_double),        # N_e_halo (output)
        ]
        self.lib.solve_halo_steady_state.restype = ct.c_int
        
        # integrate_spectrum
        self.lib.integrate_spectrum.argtypes = [
            ct.c_ulong,                     # n_gal
            ct.c_uint,                      # n_E
            ct.POINTER(ct.c_double),        # E
            ct.POINTER(ct.c_double),        # spec
            ct.POINTER(ct.c_double),        # integrals (output)
        ]
        self.lib.integrate_spectrum.restype = ct.c_int
    
    
    def _load_galaxy_data(self):
        """Load galaxy catalogue"""
        
        print(f"\nLoading galaxy data from {self.galaxy_file}...")
        
        # Read catalogue
        data = []
        with open(self.galaxy_file, 'r') as f:
            # Skip header
            f.readline()
            # Read number of galaxies
            self.n_gal = int(f.readline().strip())
            # Skip column names
            f.readline()
            # Read data
            for line in f:
                parts = line.split()
                if len(parts) >= 4:
                    data.append([float(x) for x in parts[:4]])
        
        data = np.array(data)
        
        self.z = np.ascontiguousarray(data[:, 0], dtype=np.float64)
        self.M_star = np.ascontiguousarray(data[:, 1], dtype=np.float64)
        self.Re = np.ascontiguousarray(data[:, 2], dtype=np.float64)
        self.SFR = np.ascontiguousarray(data[:, 3], dtype=np.float64)
        
        print(f"  ✓ Loaded {self.n_gal} galaxies")
    
    
    def _load_disc_data(self):
        """Load disc ISM properties and CR spectra"""
        
        print(f"\nLoading disc data from {self.disc_dir}...")
        
        # Load ISM properties
        gal_data_file = self.disc_dir / 'gal_data.txt'
        if not gal_data_file.exists():
            raise FileNotFoundError(
                f"Missing {gal_data_file}. "
                f"Run full spectra.c first to generate disc outputs."
            )
        
        data = np.loadtxt(gal_data_file, skiprows=1)
        
        self.h_disc = np.ascontiguousarray(data[:, 0], dtype=np.float64)
        self.n_H_disc = np.ascontiguousarray(data[:, 1], dtype=np.float64)
        self.B_disc = np.ascontiguousarray(data[:, 2], dtype=np.float64)
        self.sig_gas = np.ascontiguousarray(data[:, 3], dtype=np.float64)
        
        print(f"  ✓ Loaded disc ISM properties")
        
        # Load calorimetry fractions
        fcal_file = self.disc_dir / 'fcal.txt'
        if not fcal_file.exists():
            raise FileNotFoundError(f"Missing {fcal_file}")
        
        fcal_data = np.loadtxt(fcal_file, skiprows=1)
        # We only need f_cal[0] (first energy bin) for each galaxy
        self.f_cal_0 = np.ascontiguousarray(fcal_data[:, 0], dtype=np.float64)
        
        print(f"  ✓ Loaded calorimetry fractions")
        
        # Load disc CR spectra
        # For now, generate placeholder C normalization
        # In full version, this would be loaded from disc outputs
        self.C = np.ones(self.n_gal, dtype=np.float64) * 1e-7
        
        print(f"  ✓ Loaded CR normalization")
    
    
    def _create_energy_grids(self):
        """Create logarithmically-spaced energy grids"""
        
        print("\nCreating energy grids...")
        
        # CR kinetic energy [GeV]
        self.T_CR = np.logspace(-3, 8, self.n_E_CR, dtype=np.float64)
        
        # CR electron total energy [GeV]
        m_e_GeV = 0.000510998928
        self.E_CRe = self.T_CR + m_e_GeV
        
        # Gamma-ray energy [GeV]
        self.E_gam = np.logspace(-16, 8, self.n_E_gam, dtype=np.float64)
        
        print(f"  ✓ T_CR: {self.n_E_CR} bins from {self.T_CR[0]:.1e} to {self.T_CR[-1]:.1e} GeV")
        print(f"  ✓ E_γ: {self.n_E_gam} bins from {self.E_gam[0]:.1e} to {self.E_gam[-1]:.1e} GeV")
    
    
    def _initialize_arrays(self):
        """Initialize all result arrays"""
        
        print("\nInitializing result arrays...")
        
        # Halo properties
        self.n_H_halo = np.zeros(self.n_gal, dtype=np.float64)
        self.B_halo = np.zeros(self.n_gal, dtype=np.float64)
        self.h_halo = np.zeros(self.n_gal, dtype=np.float64)
        
        # Diffusion and injection
        self.D_e_disc = np.zeros((self.n_gal, self.n_E_CR), dtype=np.float64)
        self.D_e_halo = np.zeros((self.n_gal, self.n_E_CR), dtype=np.float64)
        self.Q_e_halo = np.zeros((self.n_gal, self.n_E_CR), dtype=np.float64)
        
        # Loss timescales
        self.tau_sync = np.zeros((self.n_gal, self.n_E_CR), dtype=np.float64)
        self.tau_plasma = np.zeros((self.n_gal, self.n_E_CR), dtype=np.float64)
        self.tau_diff = np.zeros((self.n_gal, self.n_E_CR), dtype=np.float64)
        
        # Steady-state spectrum
        self.N_e_halo = np.zeros((self.n_gal, self.n_E_CR), dtype=np.float64)
        
        print(f"  ✓ Initialized arrays for {self.n_gal} galaxies")
    
    
    def compute_halo_properties(self):
        """Compute halo ISM properties from disc properties"""
        
        print("\n" + "="*60)
        print("Computing halo properties...")
        print("="*60)
        
        ret = self.lib.compute_halo_properties(
            self.n_gal,
            self.n_H_disc.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.B_disc.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.h_disc.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.SFR.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.M_star.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.n_H_halo.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.B_halo.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.h_halo.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        
        if ret != 0:
            raise RuntimeError("compute_halo_properties failed")
        
        print(f"\n✓ Halo properties computed")
        print(f"  Example galaxy 0:")
        print(f"    n_H: disc = {self.n_H_disc[0]:.2e} cm⁻³, halo = {self.n_H_halo[0]:.2e} cm⁻³")
        print(f"    B: disc = {self.B_disc[0]:.2e} G, halo = {self.B_halo[0]:.2e} G")
        print(f"    h: disc = {self.h_disc[0]:.2e} pc, halo = {self.h_halo[0]:.2e} pc")
    
    
    def compute_halo_transport(self, chi=1e-4, M_A=2.0, q_inject=2.2):
        """
        Compute full halo transport: diffusion, injection, losses, steady-state
        
        Parameters
        ----------
        chi : float
            Magnetic-to-turbulent pressure ratio (default: 1e-4)
        M_A : float
            Alfvénic Mach number (default: 2.0)
        q_inject : float
            CR injection spectral index (default: 2.2)
        """
        
        print("\n" + "="*60)
        print("Computing halo transport...")
        print("="*60)
        
        # Step 1: Halo diffusion coefficient
        print("\n1. Computing halo diffusion coefficient...")
        ret = self.lib.compute_halo_diffusion(
            self.n_gal,
            self.n_E_CR,
            self.T_CR.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.h_disc.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.n_H_disc.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.sig_gas.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.f_cal_0.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.C.ctypes.data_as(ct.POINTER(ct.c_double)),
            chi, M_A, q_inject,
            self.D_e_halo.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        if ret != 0:
            raise RuntimeError("compute_halo_diffusion failed")
        print(f"   ✓ D_halo computed")
        
        # For injection, we need disc diffusion and disc spectra
        # Simplified: assume disc diffusion same as halo (will be lower in reality)
        # and generate toy disc spectrum
        print("\n2. Computing disc diffusion (simplified)...")
        self.D_e_disc[:] = self.D_e_halo * 0.5  # Placeholder
        
        print("\n3. Generating toy disc CR spectrum...")
        # Simplified power-law spectrum
        for i in range(self.n_gal):
            norm = 1e-10 * self.SFR[i]
            self.q_e_disc = np.zeros((self.n_gal, self.n_E_CR), dtype=np.float64)
            self.q_e_disc[i, :] = norm * (self.E_CRe / 1.0)**(-q_inject)
        
        # Step 2: Halo injection from disc escape
        print("\n4. Computing halo injection from disc escape...")
        ret = self.lib.compute_halo_injection(
            self.n_gal,
            self.n_E_CR,
            self.h_disc.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.q_e_disc.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.D_e_disc.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.Q_e_halo.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        if ret != 0:
            raise RuntimeError("compute_halo_injection failed")
        print(f"   ✓ Q_halo computed")
        
        # Step 3: Loss timescales
        print("\n5. Computing loss timescales...")
        
        ret = self.lib.compute_synchrotron_loss_time(
            self.n_gal, self.n_E_CR,
            self.E_CRe.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.B_halo.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.tau_sync.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        if ret != 0:
            raise RuntimeError("compute_synchrotron_loss_time failed")
        print(f"   ✓ τ_sync computed")
        
        ret = self.lib.compute_plasma_loss_time(
            self.n_gal, self.n_E_CR,
            self.E_CRe.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.n_H_halo.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.tau_plasma.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        if ret != 0:
            raise RuntimeError("compute_plasma_loss_time failed")
        print(f"   ✓ τ_plasma computed")
        
        ret = self.lib.compute_diffusion_loss_time(
            self.n_gal, self.n_E_CR,
            self.h_halo.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.D_e_halo.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.tau_diff.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        if ret != 0:
            raise RuntimeError("compute_diffusion_loss_time failed")
        print(f"   ✓ τ_diff computed")
        
        # Step 4: Solve steady-state
        print("\n6. Solving halo steady-state...")
        ret = self.lib.solve_halo_steady_state(
            self.n_gal, self.n_E_CR,
            self.Q_e_halo.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.tau_sync.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.tau_plasma.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.tau_diff.ctypes.data_as(ct.POINTER(ct.c_double)),
            None,  # tau_IC (not implemented in simplified version)
            self.N_e_halo.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        if ret != 0:
            raise RuntimeError("solve_halo_steady_state failed")
        print(f"   ✓ N_e_halo computed")
        
        print("\n✓ Halo transport complete")
    
        def get_results(self):
        """
        Package all results into a dictionary
        
        Returns
        -------
        dict
            Dictionary containing all computed quantities
        """
        
        return {
            # Galaxy properties
            'n_gal': self.n_gal,
            'z': self.z,
            'M_star': self.M_star,
            'Re': self.Re,
            'SFR': self.SFR,
            
            # Energy grids
            'T_CR': self.T_CR,
            'E_CRe': self.E_CRe,
            'E_gam': self.E_gam,
            
            # Disc properties
            'h_disc': self.h_disc,
            'n_H_disc': self.n_H_disc,
            'B_disc': self.B_disc,
            
            # Halo properties
            'n_H_halo': self.n_H_halo,
            'B_halo': self.B_halo,
            'h_halo': self.h_halo,
            
            # Transport
            'D_e_halo': self.D_e_halo,
            'Q_e_halo': self.Q_e_halo,
            
            # Loss timescales
            'tau_sync': self.tau_sync,
            'tau_plasma': self.tau_plasma,
            'tau_diff': self.tau_diff,
            
            # Steady-state spectrum
            'N_e_halo': self.N_e_halo,
        }
    
    
    def save_results(self, output_dir):
        """
        Save results to files
        
        Parameters
        ----------
        output_dir : str or Path
            Directory to save outputs
        """
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving results to {output_dir}...")
        
        # Halo properties
        np.savetxt(
            output_dir / 'halo_properties.txt',
            np.column_stack([self.n_H_halo, self.B_halo, self.h_halo]),
            header='n_H_halo[cm-3] B_halo[G] h_halo[pc]'
        )
        
        # Energy grids
        np.savetxt(output_dir / 'T_CR.txt', self.T_CR, header='T_CR[GeV]')
        np.savetxt(output_dir / 'E_CRe.txt', self.E_CRe, header='E_CRe[GeV]')
        
        # Transport
        np.savetxt(output_dir / 'D_e_halo.txt', self.D_e_halo,
                  header='D_e_halo[cm2/s] - rows=galaxies, cols=energies')
        np.savetxt(output_dir / 'Q_e_halo.txt', self.Q_e_halo,
                  header='Q_e_halo[CRe/s/GeV]')
        
        # Loss times
        np.savetxt(output_dir / 'tau_sync.txt', self.tau_sync, header='tau_sync[s]')
        np.savetxt(output_dir / 'tau_plasma.txt', self.tau_plasma, header='tau_plasma[s]')
        np.savetxt(output_dir / 'tau_diff.txt', self.tau_diff, header='tau_diff[s]')
        
        # Steady-state spectrum
        np.savetxt(output_dir / 'N_e_halo.txt', self.N_e_halo,
                  header='N_e_halo[CRe/cm3/GeV]')
        
        print("✓ Results saved")


def main():
    """Example usage"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compute halo CR spectra'
    )
    parser.add_argument('galaxy_file', help='Galaxy catalogue')
    parser.add_argument('disc_dir', help='Disc output directory')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--threads', type=int, default=None,
                       help='Number of OpenMP threads')
    
    args = parser.parse_args()
    
    # Initialize
    hc = HaloCalculator(
        galaxy_file=args.galaxy_file,
        disc_dir=args.disc_dir,
        n_threads=args.threads
    )
    
    # Compute halo properties
    hc.compute_halo_properties()
    
    # Compute transport
    hc.compute_halo_transport()
    
    # Save results
    hc.save_results(args.output_dir)
    
    print("\n" + "="*60)
    print("Computation complete!")
    print("="*60)



if __name__ == '__main__':
    main()