# -*- coding: utf-8 -*-

import sys,os
import configparser
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.insert(0, os.path.dirname(__file__))
from pyBRIXS.rixs import analysis, rixs

class GetData:
    def __init__(self, ecore, eloss, ecoreindex, spectrum, normalize=True):
        """
        Initialize the GetData class. If multiple spectra are provided, the mean is computed.
        
        Args:
            ecore (np.array): Core excitation energies.
            eloss (np.array): Energy loss values.
            ecoreindex (list): Indices for selecting the core energies.
            spectrum (np.array or list of np.array): Spectrum(s) to analyze. Can be a single spectrum or a list of spectra.
            normalize (bool): Whether to normalize the DDCS by the maximum of each curve.
        """
        self.ecore = ecore
        self.eloss = eloss
        self.ecoreindex = ecoreindex
        self.spectrum = spectrum
        self.normalize = normalize

        # Ensure spectrum is a list (even for a single spectrum)
        if isinstance(spectrum, np.ndarray):
            self.spectrum = [spectrum]  # Wrap single spectrum in a list

        # Call the method to compute DDCS and emission
        self.get_ddcs()

    def get_ddcs(self):
        """
        Calculate the Differential Cross Section (DDCS) for each spectrum (or the mean DDCS if multiple spectra are given).
        """
        self.ddcs = np.zeros((len(self.ecoreindex), len(self.eloss)))
        self.emission = np.zeros((len(self.ecoreindex), len(self.eloss)))

        # Iterate over all core excitation indices
        for idx, j in enumerate(self.ecoreindex):
            for i in range(len(self.eloss)):
                # Iterate over each spectrum (if multiple)
                spectrum_values = np.array([spectrum[j, i] for spectrum in self.spectrum])
                self.ddcs[idx, i] = np.mean(spectrum_values)  # Take mean if multiple spectra
                self.emission[idx, i] = self.ecore[j] - self.eloss[i]

        # Normalize DDCS if specified
        if self.normalize:
            for j in range(len(self.ecoreindex)):
                self.ddcs[j, :] = self.ddcs[j, :] / self.ddcs[j, :].max()

    def write_ddcs(self, outfile):
        """
        Write the DDCS data to an output file.
        
        Args:
            outfile (file object): The file to write the DDCS data to.
        """
        idx = 0
        for j in self.ecoreindex:
            outfile.write("{:<s}\n".format(" "))
            outfile.write("{:<s}\t{:>15.9f} {:>15.9f}\n".format("# ", j, self.ecore[j]))
            for i in range(len(self.eloss)):
                outfile.write("{:> 4.6f} {:> 4.6f}\n".format(self.eloss[i], self.ddcs[idx, i]))
            idx += 1
    
def read_config(cfg_path):
    config = configparser.ConfigParser()
    config.read(cfg_path)

    folder_paths = [f.strip() for f in config.get("paths", "folder_paths").split(",")]
    rixs_files = [folder + '/rixs.h5' for folder in folder_paths]

    broad = config.getfloat("settings", "broad")
    eloss_min = config.getfloat("settings", "eloss_min")
    eloss_max = config.getfloat("settings", "eloss_max")
    eloss_step = config.getfloat("settings", "eloss_step")
    eloss = np.arange(eloss_min, eloss_max, eloss_step)

    ecore_1 = np.array([float(x.strip()) for x in config.get("exc_energy", "omega").split(",")])
    ecoreindex_1 = list(range(len(ecore_1)))

    return rixs_files, broad, eloss, ecore_1, ecoreindex_1

 
def main(cfg_path="input-ddcs.cfg"):
    rixs_files, broad, eloss, ecore_1, ecoreindex_1 = read_config(cfg_path)
    
    output_file = "ddcs_vs_loss_mean" if len(rixs_files) > 1 else "ddcs_vs_loss"

    npts_loss = len(eloss) * 10
    npts_core = len(ecore_1) * 10

    rixs_list = [rixs(file=rixs_file, broad=broad, freq=eloss) for rixs_file in rixs_files]
    analyzed_avg = analysis.average_rixs(rixs_list, ecore_1, np.array([npts_loss, npts_core]))

    spectrum_list = [rixs_instance.spectrum for rixs_instance in rixs_list]
    getdata = GetData(ecore_1, eloss, ecoreindex_1, spectrum_list)

    with open(output_file, 'w') as f:
        getdata.write_ddcs(f)

if __name__ == '__main__':
    main()   
