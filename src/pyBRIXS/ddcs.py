# -*- coding: utf-8 -*-

import argparse
import configparser
from pathlib import Path

import numpy as np

from pyBRIXS.rixs import analysis, rixs


SPECTRUM_MODES = {
    "classic": ("spectrum", ""),
    "coherent": ("spectrum_coh", "_coherent"),
    "incoherent": ("spectrum_incoh", "_incoherent"),
}


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
        self.spectrum = [spectrum] if isinstance(spectrum, np.ndarray) else spectrum
        self.normalize = normalize

        self.get_ddcs()

    def get_ddcs(self):
        """
        Calculate the DDCS for each spectrum, or the mean DDCS if multiple spectra are given.
        """
        spectra = np.asarray(self.spectrum)
        if spectra.ndim != 3:
            raise ValueError("Spectrum data must have shape (runs, excitation energies, energy losses).")
        if not all(spectra[0].shape == spectrum.shape for spectrum in self.spectrum):
            raise ValueError("All spectrum arrays must have the same shape.")
        if max(self.ecoreindex) >= spectra.shape[1]:
            raise ValueError("ecoreindex contains an index outside the spectrum range.")
        if len(self.eloss) != spectra.shape[2]:
            raise ValueError("Number of energy-loss values and spectrum columns differ.")
        self.ddcs = np.mean(spectra[:, self.ecoreindex, :], axis=0)
        self.emission = self.ecore[self.ecoreindex, None] - self.eloss[None, :]

        if self.normalize:
            maxima = self.ddcs.max(axis=1)
            nonzero = maxima != 0
            self.ddcs[nonzero, :] = self.ddcs[nonzero, :] / maxima[nonzero, None]

    def write_ddcs(self, outfile):
        """
        Write the DDCS data to an output file.

        Args:
            outfile (file object): The file to write the DDCS data to.
        """
        for idx, j in enumerate(self.ecoreindex):
            outfile.write("{:<s}\n".format(" "))
            outfile.write("{:<s}\t{:>15.9f} {:>15.9f}\n".format("# ", j, self.ecore[j]))
            for i in range(len(self.eloss)):
                outfile.write("{:> 4.6f} {:> 4.6f}\n".format(self.eloss[i], self.ddcs[idx, i]))


def _split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_modes(config):
    if config.has_option("settings", "modes"):
        modes = _split_csv(config.get("settings", "modes"))
    elif config.has_option("settings", "mode"):
        mode = config.get("settings", "mode").strip()
        modes = list(SPECTRUM_MODES) if mode == "all" else [mode]
    else:
        modes = ["classic", "coherent", "incoherent"]

    unknown = [mode for mode in modes if mode not in SPECTRUM_MODES]
    if unknown:
        raise ValueError("Unknown DDCS mode(s): {}".format(", ".join(unknown)))
    return modes


def read_config(cfg_path):
    config = configparser.ConfigParser()
    read_files = config.read(cfg_path)
    if not read_files:
        raise FileNotFoundError("Could not read config file '{}'.".format(cfg_path))

    folder_paths = _split_csv(config.get("paths", "folder_paths"))
    rixs_files = [str(Path(folder) / "rixs.h5") for folder in folder_paths]

    broad = config.getfloat("settings", "broad")
    eloss_min = config.getfloat("settings", "eloss_min")
    eloss_max = config.getfloat("settings", "eloss_max")
    eloss_step = config.getfloat("settings", "eloss_step")
    eloss = np.arange(eloss_min, eloss_max, eloss_step)

    ecore = np.array([float(x) for x in _split_csv(config.get("exc_energy", "omega"))])
    ecoreindex = list(range(len(ecore)))
    modes = _read_modes(config)

    output_file = config.get(
        "settings",
        "output_file",
        fallback="ddcs_vs_loss_mean" if len(rixs_files) > 1 else "ddcs_vs_loss",
    )
    normalize = config.getboolean("settings", "normalize", fallback=True)
    map_output = config.get("settings", "map_output", fallback="").strip()
    map_grid_scale = config.getint("settings", "map_grid_scale", fallback=10)

    return {
        "rixs_files": rixs_files,
        "broad": broad,
        "eloss": eloss,
        "ecore": ecore,
        "ecoreindex": ecoreindex,
        "modes": modes,
        "output_file": output_file,
        "normalize": normalize,
        "map_output": map_output,
        "map_grid_scale": map_grid_scale,
    }


def _spectra_for_mode(rixs_list, mode):
    attr, _ = SPECTRUM_MODES[mode]
    return [getattr(r, attr) for r in rixs_list if getattr(r, attr) is not None]


def _rixs_for_mode(rixs_list, mode):
    attr, _ = SPECTRUM_MODES[mode]
    return [r for r in rixs_list if getattr(r, attr) is not None]


def write_ddcs_for_mode(rixs_list, ecore, eloss, ecoreindex, output_base, mode, normalize=True):
    spectra = _spectra_for_mode(rixs_list, mode)
    if not spectra:
        return False

    _, suffix = SPECTRUM_MODES[mode]
    getdata = GetData(ecore, eloss, ecoreindex, spectra, normalize=normalize)
    with open(output_base + suffix, "w") as f:
        getdata.write_ddcs(f)
    return True


def export_map_for_mode(rixs_list, ecore, eloss, map_output, map_grid_scale, mode):
    selected_rixs = _rixs_for_mode(rixs_list, mode)
    if not selected_rixs:
        return False

    attr, suffix = SPECTRUM_MODES[mode]
    grid = np.array([len(eloss) * map_grid_scale, len(ecore) * map_grid_scale])
    analyzed = analysis.average_rixs(selected_rixs, ecore, grid=grid, spectrum_attr=attr)
    filepath = "{}{}.npz".format(map_output, suffix)
    avg_rixs = rixs()
    avg_rixs.w = selected_rixs[0].w
    spectra = [getattr(r, attr) for r in selected_rixs]
    setattr(avg_rixs, attr, np.mean(spectra, axis=0))
    analyzed.export(avg_rixs, ecore, filepath)
    return True


def _normalize_modes(modes):
    if modes is None:
        return None
    if "all" in modes:
        return list(SPECTRUM_MODES)
    unknown = [mode for mode in modes if mode not in SPECTRUM_MODES]
    if unknown:
        raise ValueError("Unknown DDCS mode(s): {}".format(", ".join(unknown)))
    return modes


def run(cfg_path="input-ddcs.cfg", modes=None, map_output=None):
    settings = read_config(cfg_path)
    if modes is not None:
        settings["modes"] = _normalize_modes(modes)
    if map_output is not None:
        settings["map_output"] = map_output

    rixs_list = [
        rixs(file=rixs_file, broad=settings["broad"], freq=settings["eloss"])
        for rixs_file in settings["rixs_files"]
    ]

    written = []
    for mode in settings["modes"]:
        if write_ddcs_for_mode(
            rixs_list,
            settings["ecore"],
            settings["eloss"],
            settings["ecoreindex"],
            settings["output_file"],
            mode,
            normalize=settings["normalize"],
        ):
            written.append(mode)

        if settings["map_output"]:
            export_map_for_mode(
                rixs_list,
                settings["ecore"],
                settings["eloss"],
                settings["map_output"],
                settings["map_grid_scale"],
                mode,
            )

    if not written:
        raise ValueError("No requested DDCS mode was present in the RIXS files.")
    return written


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", nargs="?", default="input-ddcs.cfg")
    parser.add_argument(
        "--mode",
        choices=["classic", "coherent", "incoherent", "all"],
        action="append",
        help="Spectrum mode to export. Can be passed multiple times.",
    )
    parser.add_argument(
        "--map-output",
        default=None,
        help="Optional basename for interpolated map export. If omitted, no interpolation is done.",
    )
    args = parser.parse_args(argv)

    run(cfg_path=args.cfg_path, modes=args.mode, map_output=args.map_output)


if __name__ == "__main__":
    main()
