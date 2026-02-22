#!/usr/bin/env python3
"""
================================================================================
RESULTS.PY - Analysis and Visualization for Halo CR Results
================================================================================

This module provides:
  1. Loading and analyzing results from halo calculations
  2. Diagnostic plots (spectra, loss times, etc.)
  3. Comparison with disc emission
  4. Export to publication-ready formats

Usage:
------
    python results.py <output_dir> [options]s
    
    Or in Python:
    
    from results import HaloResults
    
    hr = HaloResults('output_halo/')
    hr.plot_halo_spectra(galaxy_idx=0)
    hr.plot_loss_times(galaxy_idx=0)
    hr.save_summary()

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys


class HaloResults:
    """
    Class for analyzing and visualizing halo CR results
    """
    
    def __init__(self, output_dir):
        """
        Load results from output directory
        
        Parameters
        ----------
        output_dir : str or Path
            Directory containing halo calculation outputs
        """
        
        self.output_dir = Path(output_dir)
        
        if not self.output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")
        
        print("="*60)
        print("Loading Halo CR Results")
        print("="*60)
        print(f"Directory: {self.output_dir}\n")
        
        self._load_data()
        
        print("\n✓ Data loaded successfully")
        print("="*60 + "\n")
    
    
    def _load_data(self):
        """Load all data files"""
        
        # Energy grids
        print("Loading energy grids...")
        self.T_CR = np.loadtxt(self.output_dir / 'T_CR.txt')
        self.E_CRe = np.loadtxt(self.output_dir / 'E_CRe.txt')
        self.n_E = len(self.T_CR)
        print(f"  ✓ {self.n_E} energy bins")
        
        # Halo properties
        print("Loading halo properties...")
        halo_prop = np.loadtxt(self.output_dir / 'halo_properties.txt')
        self.n_H_halo = halo_prop[:, 0]
        self.B_halo = halo_prop[:, 1]
        self.h_halo = halo_prop[:, 2]
        self.n_gal = len(self.n_H_halo)
        print(f"  ✓ {self.n_gal} galaxies")
        
        # Transport
        print("Loading transport...")
        self.D_e_halo = np.loadtxt(self.output_dir / 'D_e_halo.txt')
        self.Q_e_halo = np.loadtxt(self.output_dir / 'Q_e_halo.txt')
        print(f"  ✓ Diffusion and injection")
        
        # Loss timescales
        print("Loading loss timescales...")
        self.tau_sync = np.loadtxt(self.output_dir / 'tau_sync.txt')
        self.tau_plasma = np.loadtxt(self.output_dir / 'tau_plasma.txt')
        self.tau_diff = np.loadtxt(self.output_dir / 'tau_diff.txt')
        print(f"  ✓ Loss times")
        
        # Steady-state spectrum
        print("Loading steady-state spectrum...")
        self.N_e_halo = np.loadtxt(self.output_dir / 'N_e_halo.txt')
        print(f"  ✓ Electron spectrum")
    
    
    def plot_halo_spectrum(self, galaxy_idx=0, ax=None, **kwargs):
        """
        Plot halo CR electron spectrum for a single galaxy
        
        Parameters
        ----------
        galaxy_idx : int
            Galaxy index to plot
        ax : matplotlib.axes.Axes, optional
            Axes to plot on (creates new if None)
        **kwargs : dict
            Additional plotting kwargs
        
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object
        """
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot spectrum
        ax.loglog(self.E_CRe, self.N_e_halo[galaxy_idx, :],
                 label=f'Galaxy {galaxy_idx}', **kwargs)
        
        ax.set_xlabel(r'$E_e$ [GeV]', fontsize=14)
        ax.set_ylabel(r'$N_e(E)$ [CRe cm$^{-3}$ GeV$^{-1}$]', fontsize=14)
        ax.set_title('Halo CR Electron Spectrum', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
    
    
    def plot_loss_times(self, galaxy_idx=0, figsize=(10, 6)):
        """
        Plot all loss timescales for a single galaxy
        
        Parameters
        ----------
        galaxy_idx : int
            Galaxy index to plot
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each loss channel
        ax.loglog(self.E_CRe, self.tau_sync[galaxy_idx, :],
                 label='Synchrotron', lw=2)
        ax.loglog(self.E_CRe, self.tau_plasma[galaxy_idx, :],
                 label='Plasma', lw=2)
        ax.loglog(self.E_CRe, self.tau_diff[galaxy_idx, :],
                 label='Diffusion', lw=2)
        
        # Total loss time (harmonic mean)
        tau_total = 1.0 / (1.0/self.tau_sync[galaxy_idx, :]
                          + 1.0/self.tau_plasma[galaxy_idx, :]
                          + 1.0/self.tau_diff[galaxy_idx, :])
        ax.loglog(self.E_CRe, tau_total, 'k--', label='Total', lw=2)
        
        ax.set_xlabel(r'$E_e$ [GeV]', fontsize=14)
        ax.set_ylabel(r'$\tau_{\rm loss}$ [s]', fontsize=14)
        ax.set_title(f'Halo Loss Timescales - Galaxy {galaxy_idx}', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Add reference times
        yr_s = 3.15576e7
        Gyr_s = 1e9 * yr_s
        ax.axhline(Gyr_s, color='gray', ls=':', alpha=0.5, label='1 Gyr')
        
        return fig, ax
    
    
    def plot_injection_vs_equilibrium(self, galaxy_idx=0, figsize=(10, 6)):
        """
        Compare injection rate with equilibrium spectrum
        
        Parameters
        ----------
        galaxy_idx : int
            Galaxy index to plot
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Injection rate (multiply by effective loss time to get N)
        tau_eff = 1.0 / (1.0/self.tau_sync[galaxy_idx, :]
                        + 1.0/self.tau_plasma[galaxy_idx, :]
                        + 1.0/self.tau_diff[galaxy_idx, :])
        
        ax.loglog(self.E_CRe, self.Q_e_halo[galaxy_idx, :],
                 label='Injection $Q_e$', lw=2, alpha=0.7)
        ax.loglog(self.E_CRe, self.Q_e_halo[galaxy_idx, :] * tau_eff,
                 label=r'$Q_e \times \tau_{\rm eff}$', lw=2, ls='--', alpha=0.7)
        ax.loglog(self.E_CRe, self.N_e_halo[galaxy_idx, :],
                 label='Steady-state $N_e$', lw=2, color='black')
        
        ax.set_xlabel(r'$E_e$ [GeV]', fontsize=14)
        ax.set_ylabel(r'[CRe cm$^{-3}$ GeV$^{-1}$]', fontsize=14)
        ax.set_title(f'Injection vs Equilibrium - Galaxy {galaxy_idx}', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        return fig, ax
    
    
    def plot_diffusion_coefficient(self, galaxy_idx=0, figsize=(8, 6)):
        """
        Plot diffusion coefficient
        
        Parameters
        ----------
        galaxy_idx : int
            Galaxy index to plot
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.loglog(self.E_CRe, self.D_e_halo[galaxy_idx, :], lw=2)
        
        ax.set_xlabel(r'$E_e$ [GeV]', fontsize=14)
        ax.set_ylabel(r'$D_e$ [cm$^2$ s$^{-1}$]', fontsize=14)
        ax.set_title(f'Halo Diffusion Coefficient - Galaxy {galaxy_idx}', fontsize=16)
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    
    def plot_galaxy_comparison(self, galaxy_indices=None, figsize=(12, 8)):
        """
        Compare multiple galaxies in a 2×2 grid
        
        Parameters
        ----------
        galaxy_indices : list of int, optional
            Galaxies to compare (default: first 4)
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : array of matplotlib.axes.Axes
        """
        
        if galaxy_indices is None:
            galaxy_indices = list(range(min(4, self.n_gal)))
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        for idx, gal_idx in enumerate(galaxy_indices):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            
            # Plot spectrum
            ax.loglog(self.E_CRe, self.N_e_halo[gal_idx, :], lw=2)
            
            ax.set_xlabel(r'$E_e$ [GeV]', fontsize=12)
            ax.set_ylabel(r'$N_e$ [CRe cm$^{-3}$ GeV$^{-1}$]', fontsize=12)
            ax.set_title(f'Galaxy {gal_idx}', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add properties
            props_text = (
                f"$n_H$ = {self.n_H_halo[gal_idx]:.2e} cm$^{{-3}}$\n"
                f"$B$ = {self.B_halo[gal_idx]:.2e} G\n"
                f"$h$ = {self.h_halo[gal_idx]:.2e} pc"
            )
            ax.text(0.05, 0.95, props_text,
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.suptitle('Halo CR Spectra Comparison', fontsize=16, y=0.995)
        
        return fig, fig.axes
    
    
    def plot_halo_properties_histogram(self, figsize=(12, 4)):
        """
        Plot histograms of halo properties
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : array of matplotlib.axes.Axes
        """
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Density
        axes[0].hist(np.log10(self.n_H_halo), bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel(r'$\log_{10}(n_H / \mathrm{cm}^{-3})$', fontsize=12)
        axes[0].set_ylabel('Number of galaxies', fontsize=12)
        axes[0].set_title('Halo Density', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Magnetic field
        axes[1].hist(np.log10(self.B_halo * 1e6), bins=20, alpha=0.7,
                    edgecolor='black', color='orange')
        axes[1].set_xlabel(r'$\log_{10}(B / \mu\mathrm{G})$', fontsize=12)
        axes[1].set_ylabel('Number of galaxies', fontsize=12)
        axes[1].set_title('Halo Magnetic Field', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        # Scale height
        axes[2].hist(np.log10(self.h_halo), bins=20, alpha=0.7,
                    edgecolor='black', color='green')
        axes[2].set_xlabel(r'$\log_{10}(h / \mathrm{pc})$', fontsize=12)
        axes[2].set_ylabel('Number of galaxies', fontsize=12)
        axes[2].set_title('Halo Scale Height', fontsize=14)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig, axes
    
    
    def compute_summary_statistics(self):
        """
        Compute summary statistics for all galaxies
        
        Returns
        -------
        dict
            Dictionary of summary statistics
        """
        
        stats = {}
        
        # Halo properties
        stats['n_H_halo'] = {
            'mean': np.mean(self.n_H_halo),
            'median': np.median(self.n_H_halo),
            'std': np.std(self.n_H_halo),
            'min': np.min(self.n_H_halo),
            'max': np.max(self.n_H_halo),
        }
        
        stats['B_halo_muG'] = {
            'mean': np.mean(self.B_halo * 1e6),
            'median': np.median(self.B_halo * 1e6),
            'std': np.std(self.B_halo * 1e6),
            'min': np.min(self.B_halo * 1e6),
            'max': np.max(self.B_halo * 1e6),
        }
        
        stats['h_halo_kpc'] = {
            'mean': np.mean(self.h_halo / 1e3),
            'median': np.median(self.h_halo / 1e3),
            'std': np.std(self.h_halo / 1e3),
            'min': np.min(self.h_halo / 1e3),
            'max': np.max(self.h_halo / 1e3),
        }
        
        # Typical loss times at 1 GeV (index ~333 for default grid)
        idx_1GeV = np.argmin(np.abs(self.E_CRe - 1.0))
        
        stats['tau_sync_1GeV_Gyr'] = {
            'mean': np.mean(self.tau_sync[:, idx_1GeV]) / 3.15576e16,
            'median': np.median(self.tau_sync[:, idx_1GeV]) / 3.15576e16,
        }
        
        stats['tau_plasma_1GeV_Gyr'] = {
            'mean': np.mean(self.tau_plasma[:, idx_1GeV]) / 3.15576e16,
            'median': np.median(self.tau_plasma[:, idx_1GeV]) / 3.15576e16,
        }
        
        stats['tau_diff_1GeV_Gyr'] = {
            'mean': np.mean(self.tau_diff[:, idx_1GeV]) / 3.15576e16,
            'median': np.median(self.tau_diff[:, idx_1GeV]) / 3.15576e16,
        }
        
        return stats
    
    
    def print_summary(self):
        """Print summary statistics"""
        
        stats = self.compute_summary_statistics()
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"\nNumber of galaxies: {self.n_gal}")
        print(f"Energy bins: {self.n_E}")
        
        print("\n" + "-"*60)
        print("HALO PROPERTIES")
        print("-"*60)
        
        print(f"\nDensity [cm⁻³]:")
        print(f"  Mean:   {stats['n_H_halo']['mean']:.2e}")
        print(f"  Median: {stats['n_H_halo']['median']:.2e}")
        print(f"  Range:  {stats['n_H_halo']['min']:.2e} - {stats['n_H_halo']['max']:.2e}")
        
        print(f"\nMagnetic field [μG]:")
        print(f"  Mean:   {stats['B_halo_muG']['mean']:.2f}")
        print(f"  Median: {stats['B_halo_muG']['median']:.2f}")
        print(f"  Range:  {stats['B_halo_muG']['min']:.2f} - {stats['B_halo_muG']['max']:.2f}")
        
        print(f"\nScale height [kpc]:")
        print(f"  Mean:   {stats['h_halo_kpc']['mean']:.2f}")
        print(f"  Median: {stats['h_halo_kpc']['median']:.2f}")
        print(f"  Range:  {stats['h_halo_kpc']['min']:.2f} - {stats['h_halo_kpc']['max']:.2f}")
        
        print("\n" + "-"*60)
        print("TYPICAL LOSS TIMES (at 1 GeV)")
        print("-"*60)
        
        print(f"\nSynchrotron [Gyr]:")
        print(f"  Mean:   {stats['tau_sync_1GeV_Gyr']['mean']:.2f}")
        print(f"  Median: {stats['tau_sync_1GeV_Gyr']['median']:.2f}")
        
        print(f"\nPlasma [Gyr]:")
        print(f"  Mean:   {stats['tau_plasma_1GeV_Gyr']['mean']:.2f}")
        print(f"  Median: {stats['tau_plasma_1GeV_Gyr']['median']:.2f}")
        
        print(f"\nDiffusion [Gyr]:")
        print(f"  Mean:   {stats['tau_diff_1GeV_Gyr']['mean']:.2f}")
        print(f"  Median: {stats['tau_diff_1GeV_Gyr']['median']:.2f}")
        
        print("\n" + "="*60 + "\n")
    
    
    def save_summary(self, filename='summary.txt'):
        """
        Save summary statistics to file
        
        Parameters
        ----------
        filename : str
            Output filename
        """
        
        stats = self.compute_summary_statistics()
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write("="*60 + "\n")
            f.write("HALO CR CALCULATION SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Number of galaxies: {self.n_gal}\n")
            f.write(f"Energy bins: {self.n_E}\n\n")
            
            for key, value in stats.items():
                f.write(f"{key}:\n")
                for stat, val in value.items():
                    f.write(f"  {stat}: {val:.3e}\n")
                f.write("\n")
        
        print(f"✓ Summary saved to {filepath}")
    
    
    def export_figures(self, output_subdir='figures', dpi=150, formats=['png', 'pdf']):
        """
        Generate and save all diagnostic figures
        
        Parameters
        ----------
        output_subdir : str
            Subdirectory for figures
        dpi : int
            Resolution for raster formats
        formats : list of str
            File formats to save
        """
        
        fig_dir = self.output_dir / output_subdir
        fig_dir.mkdir(exist_ok=True)
        
        print(f"\nGenerating figures in {fig_dir}...")
        
        # Example galaxy (first one)
        gal_idx = 0
        
        # 1. Spectrum
        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot_halo_spectrum(gal_idx, ax=ax)
        for fmt in formats:
            fig.savefig(fig_dir / f'spectrum_galaxy{gal_idx}.{fmt}',
                       dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Spectrum")
        
        # 2. Loss times
        fig, ax = self.plot_loss_times(gal_idx)
        for fmt in formats:
            fig.savefig(fig_dir / f'loss_times_galaxy{gal_idx}.{fmt}',
                       dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Loss times")
        
        # 3. Injection vs equilibrium
        fig, ax = self.plot_injection_vs_equilibrium(gal_idx)
        for fmt in formats:
            fig.savefig(fig_dir / f'injection_vs_equilibrium_galaxy{gal_idx}.{fmt}',
                       dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Injection vs equilibrium")
        
        # 4. Diffusion coefficient
        fig, ax = self.plot_diffusion_coefficient(gal_idx)
        for fmt in formats:
            fig.savefig(fig_dir / f'diffusion_coeff_galaxy{gal_idx}.{fmt}',
                       dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Diffusion coefficient")
        
        # 5. Galaxy comparison
        fig, axes = self.plot_galaxy_comparison()
        for fmt in formats:
            fig.savefig(fig_dir / f'galaxy_comparison.{fmt}',
                       dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Galaxy comparison")
        
        # 6. Property histograms
        fig, axes = self.plot_halo_properties_histogram()
        for fmt in formats:
            fig.savefig(fig_dir / f'property_histograms.{fmt}',
                       dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Property histograms")
        
        print(f"\n✓ All figures saved to {fig_dir}/")


def main():
    """Command-line interface"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze and visualize halo CR results'
    )
    parser.add_argument('output_dir', help='Output directory with results')
    parser.add_argument('--galaxy', type=int, default=0,
                       help='Galaxy index for single-galaxy plots')
    parser.add_argument('--summary', action='store_true',
                       help='Print and save summary statistics')
    parser.add_argument('--figures', action='store_true',
                       help='Generate all diagnostic figures')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Load results
    hr = HaloResults(args.output_dir)
    
    # Summary
    if args.summary:
        hr.print_summary()
        hr.save_summary()
    
    # Figures
    if args.figures:
        hr.export_figures()
    
    # Interactive plots
    if args.show:
        hr.plot_halo_spectrum(args.galaxy)
        hr.plot_loss_times(args.galaxy)
        plt.show()


if __name__ == '__main__':
    main()