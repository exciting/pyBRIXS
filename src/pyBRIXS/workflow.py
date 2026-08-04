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
    if not files:
        raise ValueError("No RIXS files were provided.")
    return files


def _require_incident_energies(ecore, rixs_list):
    if ecore is not None:
        return np.asarray(ecore)

    for item in rixs_list:
        if item.omega is None:
            raise ValueError(
                "The RIXS file '{}' does not contain incident energies. Pass "
                "ecore explicitly when loading legacy files.".format(item.file)
            )

    omega = rixs_list[0].omega
    if not all(np.array_equal(item.omega, omega) for item in rixs_list):
        raise ValueError("The RIXS files contain different omega values.")
    return omega


def load_rixs(paths, broad, eloss, modes=None):
    """
    Load one or more BRIXS rixs.h5 files and calculate spectra.

    Args:
        paths: Folders containing rixs.h5, direct rixs.h5 paths, or both.
        broad: Lorentzian broadening in eV.
        eloss: Energy-loss axis in eV.
        modes: Optional spectrum mode or modes to load. By default all modes
            present in the files are loaded.

    Returns:
        list: Loaded pyBRIXS rixs objects.
    """
    files = find_rixs_files(paths)
    return [
        rixs(file=str(file), broad=broad, freq=eloss, modes=modes)
        for file in files
    ]


def _modes_to_load(modes):
    if modes in (None, "auto", "all"):
        return None
    return modes


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


def calculate_ddcs(
    paths,
    broad,
    eloss,
    ecore=None,
    modes="auto",
    normalize=True,
    output_base=None,
):
    """
    Calculate averaged DDCS curves without interpolation.

    This is the recommended notebook entry point for spectra.

    Returns:
        dict: mode -> GetData object. The intensity array is available as
        result[mode].ddcs.

    If output_base is given, both compressed ``.npz`` cache files and the
    established DDCS text files are written automatically.
    """
    rixs_list = load_rixs(paths, broad=broad, eloss=eloss, modes=_modes_to_load(modes),)
    modes = _selected_modes(rixs_list, modes)
    ecore = _require_incident_energies(ecore, rixs_list)
    ecore = _match_core_energies(ecore, rixs_list, modes)
    ecoreindex = list(range(len(ecore)))
    source_files = tuple(str(Path(item.file).resolve()) for item in rixs_list)
    source_mtime_ns = tuple(Path(item.file).stat().st_mtime_ns for item in rixs_list)
    source_sizes = tuple(Path(item.file).stat().st_size for item in rixs_list)
    result = {}

    for mode in modes:
        attr, _ = SPECTRUM_MODES[mode]
        spectra = [getattr(r, attr) for r in rixs_list if getattr(r, attr) is not None]
        ddcs = GetData(ecore=ecore, eloss=eloss, ecoreindex=ecoreindex, spectrum=spectra,
                       normalize=normalize,)
        ddcs.broad = broad
        ddcs.source_files = source_files
        ddcs.source_mtime_ns = source_mtime_ns
        ddcs.source_sizes = source_sizes
        result[mode] = ddcs

    if output_base is not None:
        export_ddcs(result, output_base=output_base)
        write_ddcs(result, output_base=output_base)

    return result


def export_ddcs(ddcs_by_mode, output_base="ddcs"):
    """Save DDCS results as compressed NumPy cache files."""
    if hasattr(ddcs_by_mode, "ddcs"):
        ddcs_by_mode = {"classic": ddcs_by_mode}

    written = []
    cached_modes = np.asarray(list(ddcs_by_mode), dtype=str)
    for mode, ddcs in ddcs_by_mode.items():
        if mode not in SPECTRUM_MODES:
            raise ValueError("Unknown DDCS mode: {}".format(mode))
        _, suffix = SPECTRUM_MODES[mode]
        output_file = Path("{}{}.npz".format(output_base, suffix))
        np.savez_compressed(
            output_file,
            ecore=ddcs.ecore,
            eloss=ddcs.eloss,
            ddcs=ddcs.ddcs,
            ecoreindex=np.asarray(ddcs.ecoreindex, dtype=int),
            normalize=np.asarray(ddcs.normalize, dtype=bool),
            broad=np.asarray(getattr(ddcs, "broad", np.nan), dtype=float),
            source_files=np.asarray(getattr(ddcs, "source_files", ()), dtype=str),
            source_mtime_ns=np.asarray(
                getattr(ddcs, "source_mtime_ns", ()), dtype=np.int64
            ),
            source_sizes=np.asarray(
                getattr(ddcs, "source_sizes", ()), dtype=np.int64
            ),
            cached_modes=cached_modes,
        )
        written.append(output_file)
    return written


def load_ddcs(input_base="ddcs", modes="auto"):
    """Load DDCS results written by :func:`export_ddcs`."""
    require_all = modes not in (None, "auto", "all")
    if not require_all:
        modes = list(SPECTRUM_MODES)
    elif isinstance(modes, str):
        modes = [modes]

    unknown = [mode for mode in modes if mode not in SPECTRUM_MODES]
    if unknown:
        raise ValueError("Unknown DDCS mode(s): {}".format(", ".join(unknown)))

    results = {}
    missing = []
    for mode in modes:
        _, suffix = SPECTRUM_MODES[mode]
        input_file = Path("{}{}.npz".format(input_base, suffix))
        if not input_file.exists():
            missing.append(input_file)
            continue

        with np.load(input_file, allow_pickle=False) as data:
            result = GetData.from_arrays(
                ecore=data["ecore"],
                eloss=data["eloss"],
                ddcs=data["ddcs"],
                ecoreindex=data["ecoreindex"],
                normalize=data["normalize"].item(),
            )
            result.broad = data["broad"].item() if "broad" in data else np.nan
            result.source_files = (
                tuple(data["source_files"].tolist()) if "source_files" in data else ()
            )
            result.source_mtime_ns = (
                tuple(data["source_mtime_ns"].tolist())
                if "source_mtime_ns" in data else ()
            )
            result.source_sizes = (
                tuple(data["source_sizes"].tolist()) if "source_sizes" in data else ()
            )
            result.cached_modes = (
                tuple(data["cached_modes"].tolist()) if "cached_modes" in data else (mode,)
            )
        results[mode] = result

    if not results:
        raise FileNotFoundError(
            "No DDCS cache files found. Checked: {}".format(
                ", ".join(str(path) for path in missing)
            )
        )
    if require_all and missing:
        raise FileNotFoundError(
            "Missing requested DDCS cache file(s): {}".format(
                ", ".join(str(path) for path in missing)
            )
        )
    return results


def _ddcs_cache_matches(results, paths, broad, eloss, ecore, modes, normalize):
    files = find_rixs_files(paths)
    source_files = tuple(str(path.resolve()) for path in files)
    source_mtime_ns = tuple(path.stat().st_mtime_ns for path in files)
    source_sizes = tuple(path.stat().st_size for path in files)
    expected_eloss = np.asarray(eloss)
    expected_ecore = None if ecore is None else np.asarray(ecore)
    cached_modes = set(next(iter(results.values())).cached_modes)
    if modes in (None, "auto", "all") and set(results) != cached_modes:
        return False

    for result in results.values():
        if not np.array_equal(result.eloss, expected_eloss):
            return False
        if expected_ecore is not None:
            if len(expected_ecore) < len(result.ecore):
                return False
            if not np.array_equal(result.ecore, expected_ecore[:len(result.ecore)]):
                return False
        if result.normalize != bool(normalize):
            return False
        if not np.isfinite(result.broad) or result.broad != broad:
            return False
        if result.source_files != source_files:
            return False
        if result.source_mtime_ns != source_mtime_ns:
            return False
        if result.source_sizes != source_sizes:
            return False
    return True


def calculate_or_load_ddcs(
    paths,
    broad,
    eloss,
    ecore=None,
    modes="auto",
    normalize=True,
    output_base="ddcs",
    force=False,
):
    """Load a matching DDCS cache or calculate and write both output formats."""
    files = find_rixs_files(paths)
    if not force:
        try:
            cached = load_ddcs(output_base, modes=modes)
            if _ddcs_cache_matches(
                cached,
                paths=files,
                broad=broad,
                eloss=eloss,
                ecore=ecore,
                modes=modes,
                normalize=normalize,
            ):
                missing_text = any(
                    not Path("{}{}".format(output_base, SPECTRUM_MODES[mode][1])).exists()
                    for mode in cached
                )
                if missing_text:
                    write_ddcs(cached, output_base=output_base)
                return cached
        except FileNotFoundError:
            pass

    return calculate_ddcs(
        files,
        broad=broad,
        eloss=eloss,
        ecore=ecore,
        modes=modes,
        normalize=normalize,
        output_base=output_base,
    )


def calculate_maps(paths, broad, eloss, ecore=None, modes="auto", grid_scale=10):
    """
    Calculate interpolated RIXS maps.

    Use this only when a regular 2D map is needed. Unlike calculate_ddcs(),
    this performs grid interpolation.

    Returns:
        dict: mode -> analysis object.
    """
    rixs_list = load_rixs(paths, broad=broad, eloss=eloss, modes=_modes_to_load(modes),)
    modes = _selected_modes(rixs_list, modes)
    ecore = _require_incident_energies(ecore, rixs_list)
    ecore = _match_core_energies(ecore, rixs_list, modes)
    grid = np.array([len(eloss) * grid_scale, len(ecore) * grid_scale])
    result = {}

    for mode in modes:
        attr, _ = SPECTRUM_MODES[mode]
        selected = [r for r in rixs_list if getattr(r, attr) is not None]
        result[mode] = analysis.average_rixs(selected, ecore, grid=grid, spectrum_attr=attr,)

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
