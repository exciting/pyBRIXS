# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np

from pyBRIXS.ddcs import GetData, SPECTRUM_MODES
from pyBRIXS.rixs import analysis, rixs


def find_rixs_files(paths):
    """
    Return rixs.h5 files from folders or direct file paths.
    """
    files = []
    for path in paths:
        path = Path(path).expanduser()
        if path.is_dir():
            path = path / "rixs.h5"
        if not path.exists():
            raise FileNotFoundError("Could not find '{}'.".format(path))
        files.append(path)
    return files


def _require_incident_energies(ecore):
    if ecore is None:
        raise ValueError(
            "Incident energies are required. Pass ecore=omega from the BRIXS input."
        )
    return np.asarray(ecore)


def load_rixs(paths, broad, eloss):
    """
    Load one or more BRIXS rixs.h5 files and calculate spectra.

    Args:
        paths: Folders containing rixs.h5, direct rixs.h5 paths, or both.
        broad: Lorentzian broadening in eV.
        eloss: Energy-loss axis in eV.

    Returns:
        list: Loaded pyBRIXS rixs objects.
    """
    files = find_rixs_files(paths)
    return [rixs(file=str(file), broad=broad, freq=eloss) for file in files]


def available_modes(rixs_list):
    """
    Return available spectrum modes: classic, coherent, incoherent.
    """
    return [
        mode
        for mode, (attr, _) in SPECTRUM_MODES.items()
        if any(getattr(r, attr) is not None for r in rixs_list)
    ]


def _selected_modes(rixs_list, modes):
    present = available_modes(rixs_list)
    if modes in (None, "auto", "all"):
        return present
    if isinstance(modes, str):
        modes = [modes]

    missing = [mode for mode in modes if mode not in present]
    if missing:
        raise ValueError("Requested mode(s) not present: {}".format(", ".join(missing)))
    return list(modes)


def _spectrum_rows(rixs_list, mode):
    attr, _ = SPECTRUM_MODES[mode]
    for item in rixs_list:
        spectrum = getattr(item, attr)
        if spectrum is not None:
            return spectrum.shape[0]
    raise ValueError("Mode '{}' is not present.".format(mode))


def _match_core_energies(ecore, rixs_list, modes):
    nrows = max(_spectrum_rows(rixs_list, mode) for mode in modes)
    if len(ecore) < nrows:
        raise ValueError(
            "The excitation-energy axis has {} entries, but the spectra need {}.".format(
                len(ecore), nrows
            )
        )
    return ecore[:nrows]


def calculate_ddcs(paths, broad, eloss, ecore=None, modes="auto", normalize=True):
    """
    Calculate averaged DDCS curves without interpolation.

    This is the recommended notebook entry point for spectra.

    Returns:
        dict: mode -> GetData object. The intensity array is available as
        result[mode].ddcs.
    """
    rixs_list = load_rixs(paths, broad=broad, eloss=eloss)
    modes = _selected_modes(rixs_list, modes)
    ecore = _require_incident_energies(ecore)
    ecore = _match_core_energies(ecore, rixs_list, modes)
    ecoreindex = list(range(len(ecore)))
    result = {}

    for mode in modes:
        attr, _ = SPECTRUM_MODES[mode]
        spectra = [getattr(r, attr) for r in rixs_list if getattr(r, attr) is not None]
        result[mode] = GetData(
            ecore=ecore,
            eloss=eloss,
            ecoreindex=ecoreindex,
            spectrum=spectra,
            normalize=normalize,
        )

    return result


def calculate_maps(paths, broad, eloss, ecore=None, modes="auto", grid_scale=10):
    """
    Calculate interpolated RIXS maps.

    Use this only when a regular 2D map is needed. Unlike calculate_ddcs(),
    this performs grid interpolation.

    Returns:
        dict: mode -> analysis object.
    """
    rixs_list = load_rixs(paths, broad=broad, eloss=eloss)
    modes = _selected_modes(rixs_list, modes)
    ecore = _require_incident_energies(ecore)
    ecore = _match_core_energies(ecore, rixs_list, modes)
    grid = np.array([len(eloss) * grid_scale, len(ecore) * grid_scale])
    result = {}

    for mode in modes:
        attr, _ = SPECTRUM_MODES[mode]
        selected = [r for r in rixs_list if getattr(r, attr) is not None]
        result[mode] = analysis.average_rixs(
            selected,
            ecore,
            grid=grid,
            spectrum_attr=attr,
        )

    return result


def export_maps(maps_by_mode, output_base="rixs_map_avg"):
    """
    Save interpolated RIXS maps to compressed .npz files.

    For classic data this writes output_base.npz. For coherent/incoherent data
    it writes output_base_coherent.npz and output_base_incoherent.npz.
    """
    written = []
    for mode, rixs_map in maps_by_mode.items():
        _, suffix = SPECTRUM_MODES[mode]
        output_file = Path("{}{}.npz".format(output_base, suffix))
        rixs_map.save(output_file)
        written.append(output_file)
    return written


def load_maps(input_base="rixs_map_avg", modes="auto"):
    """
    Load interpolated RIXS maps from compressed .npz files.

    If modes is "auto", existing files for classic, coherent, and incoherent
    are loaded when present.
    """
    require_all = modes not in (None, "auto", "all")
    if not require_all:
        modes = list(SPECTRUM_MODES)
    elif isinstance(modes, str):
        modes = [modes]
    unknown = [mode for mode in modes if mode not in SPECTRUM_MODES]
    if unknown:
        raise ValueError("Unknown map mode(s): {}".format(", ".join(unknown)))

    maps = {}
    missing = []
    for mode in modes:
        _, suffix = SPECTRUM_MODES[mode]
        input_file = Path("{}{}.npz".format(input_base, suffix))
        if input_file.exists():
            maps[mode] = analysis.load(input_file)
        else:
            missing.append(input_file)

    if not maps:
        raise FileNotFoundError(
            "No map files found. Checked: {}".format(
                ", ".join(str(path) for path in missing)
            )
        )
    if require_all and missing:
        raise FileNotFoundError(
            "Missing requested map file(s): {}".format(
                ", ".join(str(path) for path in missing)
            )
        )
    return maps


def calculate_or_load_maps(
    paths,
    broad,
    eloss,
    ecore=None,
    modes="auto",
    grid_scale=10,
    cache_base="rixs_map_avg",
    force=False,
):
    """
    Load cached maps if available, otherwise calculate and save them.

    This is the convenient notebook entry point when map interpolation is
    expensive and should not be repeated unnecessarily.
    """
    if not force:
        try:
            return load_maps(cache_base, modes=modes)
        except FileNotFoundError:
            pass

    maps = calculate_maps(
        paths,
        broad=broad,
        eloss=eloss,
        ecore=ecore,
        modes=modes,
        grid_scale=grid_scale,
    )
    export_maps(maps, output_base=cache_base)
    return maps


def write_ddcs(ddcs_by_mode, output_base="ddcs_vs_loss"):
    """
    Write DDCS results to text files.

    For classic data this writes output_base. For coherent/incoherent data it
    writes output_base_coherent and output_base_incoherent.
    """
    if hasattr(ddcs_by_mode, "write_ddcs"):
        output_file = Path(output_base)
        with output_file.open("w") as handle:
            ddcs_by_mode.write_ddcs(handle)
        return [str(output_file)]

    written = []
    for mode, ddcs in ddcs_by_mode.items():
        _, suffix = SPECTRUM_MODES[mode]
        output_file = Path("{}{}".format(output_base, suffix))
        with output_file.open("w") as handle:
            ddcs.write_ddcs(handle)
        written.append(str(output_file))
    return written
