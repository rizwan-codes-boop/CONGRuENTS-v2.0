#!/usr/bin/env python3
"""
Python main script for spectra calculations.
Handles I/O and orchestration, delegates heavy computation to C via pybind11.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Import the compiled C module
try:
    import spectra_core
except ImportError:
    print("Error: spectra_core module not found. Please compile it first.")
    sys.exit(1)


def read_tau_ebl(filename):
    """Read tau EBL data from file"""
    with open(filename, 'r') as f:
        # Skip header lines
        for _ in range(2):
            f.readline()
        
        ny_z = int(f.readline().strip())
        f.readline()  # Skip separator
        
        nx_E = int(f.readline().strip())
        f.readline()  # Skip separator
        
        # Read redshift array
        ya_z = np.array([float(x) for x in f.readline().split()])
        
        f.readline()  # Skip separator
        
        # Read energy array
        xa_E = np.array([float(x) for x in f.readline().split()])
        
        f.readline()  # Skip separator
        
        # Read optical depth array
        za_tau = np.array([float(x) for x in f.readline().split()])
        
    return nx_E, ny_z, xa_E, ya_z, za_tau


def read_galaxies(filename):
    """Read galaxy data from file"""
    with open(filename, 'r') as f:
        f.readline()  # Skip header
        n_gal = int(f.readline().strip())
        f.readline()  # Skip separator
        
        data = []
        for i in range(n_gal):
            line = f.readline().strip().split()
            z = float(line[0])
            M_star = float(line[1])
            Re = float(line[2])
            SFR = float(line[3])
            data.append((z, M_star, Re, SFR))
    
    return n_gal, np.array(data)


def write_1d_file(data, label, filename):
    """Write 1D array to file"""
    with open(filename, 'w') as f:
        f.write(f"{label}\n")
        for val in data:
            f.write(f"{val:.6e}\n")


def write_2d_file(data, label, filename):
    """Write 2D array to file"""
    with open(filename, 'w') as f:
        f.write(f"{label}\n")
        for row in data:
            f.write(" ".join(f"{val:.6e}" for val in row) + "\n")


def calculate_calorimetry(gal_data, params):
    """
    Calculate calorimetry fraction and related quantities.
    This is a placeholder - actual implementation will call C functions.
    """
    n_gal = len(gal_data)
    z, M_star, Re, SFR = gal_data.T
    
    # These calculations would be done in C
    # For now, return placeholder arrays
    n_T_CR = params['n_T_CR']
    
    f_cal = np.zeros((n_gal, n_T_CR))
    D__cm2sm1 = np.zeros((n_gal, n_T_CR))
    D_e__cm2sm1 = np.zeros((n_gal, n_T_CR))
    D_e_z2__cm2sm1 = np.zeros((n_gal, n_T_CR))
    
    # Additional calculated quantities
    h__pc = np.zeros(n_gal)
    n_H__cmm3 = np.zeros(n_gal)
    B__G = np.zeros(n_gal)
    B_halo__G = np.zeros(n_gal)
    
    # TODO: Call C functions to compute these
    # This is where the computationally heavy parts would be called
    
    return {
        'f_cal': f_cal,
        'D__cm2sm1': D__cm2sm1,
        'D_e__cm2sm1': D_e__cm2sm1,
        'D_e_z2__cm2sm1': D_e_z2__cm2sm1,
        'h__pc': h__pc,
        'n_H__cmm3': n_H__cmm3,
        'B__G': B__G,
        'B_halo__G': B_halo__G,
    }


def compute_galaxy_spectra(i, gal_data, cal_data, params, interp_objects):
    """
    Compute spectra for a single galaxy.
    This function will call the C module for heavy computation.
    """
    # Extract galaxy parameters
    z, M_star, Re, SFR = gal_data[i]
    
    # Prepare input arrays
    n_T_CR = params['n_T_CR']
    n_E_gam = params['n_E_gam']
    
    # TODO: Call C function to compute spectra
    # result = spectra_core.compute_galaxy(...)
    
    # For now, return placeholder
    result = spectra_core.GalaxyComputeResult()
    
    return result


def main():
    """Main function"""
    if len(sys.argv) < 4:
        print("Usage: python spectra_main.py <galaxy_file> <data_dir> <output_dir>")
        sys.exit(1)
    
    infile = sys.argv[1]
    datadir = sys.argv[2]
    outdir = sys.argv[3]
    
    # Create output directory if it doesn't exist
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    print("Reading tau EBL data...")
    # Read tau EBL - default to Franceschini
    tau_file = os.path.join("input", "tau_Eg_z_Franceschini.txt")
    nx_E, ny_z, xa_E, ya_z, za_tau = read_tau_ebl(tau_file)
    
    print(f"Reading galaxies from {infile}...")
    n_gal, gal_data = read_galaxies(infile)
    print(f"Processing {n_gal} galaxies")
    
    # Parameters
    params = {
        'n_T_CR': 1000,
        'n_E_gam': 500,
        'T_CR_lims__GeV': [1e-3, 1e8],
        'E_CRe_lims__GeV': [1e-3 + 0.511e-3, 1e8 + 0.511e-3],  # Approximate m_e__GeV
        'E_gam_lims__GeV': [1e-16, 1e8],
        'q_p_inject': 2.2,
        'q_e_inject': 2.2,
        'T_p_cutoff__GeV': 1e8,
        'T_e_cutoff__GeV': 1e5,
    }
    
    # Calculate calorimetry
    print("Calculating calorimetry...")
    cal_data = calculate_calorimetry(gal_data, params)
    
    # Write calorimetry results
    write_2d_file(cal_data['f_cal'], "", os.path.join(outdir, "fcal.txt"))
    
    # Load interpolation objects (this would be done in C)
    print("Loading interpolation objects...")
    # interp_objects = load_interp_objects(datadir, params)
    
    # Compute spectra for each galaxy
    print("Computing spectra...")
    results = []
    for i in range(n_gal):
        print(f"Working on galaxy {i+1}/{n_gal}")
        result = compute_galaxy_spectra(i, gal_data, cal_data, params, None)
        results.append(result)
    
    # Write output files
    print("Writing output files...")
    # TODO: Write all the output files
    
    print("Done!")


if __name__ == "__main__":
    main()
