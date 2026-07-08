# %% [markdown]
# # pyBRIXS notebook workflow
#
# This example mirrors an interactive Jupyter workflow:
#
# 1. collect one or more `rixs.h5` files,
# 2. calculate spectra for the available modes,
# 3. average DDCS curves without interpolation,
# 4. optionally build interpolated RIXS maps.

# %%
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from pyBRIXS.ddcs import GetData, SPECTRUM_MODES
from pyBRIXS.rixs import analysis, rixs


# %% [markdown]
# ## Choose your dataset
#
# Replace `base_path` and `run_folders` with your own calculation folders.
# Each folder should contain one `rixs.h5`.

# %%
base_path = Path("/path/to/your/calculations")

run_folders = [
    base_path / "30deg-broad05" / f"i2-o{i}"
    for i in range(1, 10)
]

rixs_files = [folder / "rixs.h5" for folder in run_folders]


# %% [markdown]
# For a quick local test with the repository reference data, uncomment one of
# these two blocks instead.

# %%
# rixs_files = [
#     Path("../../BRIXS/test/data/diamond/pathway/rixs_ref.h5"),
# ]

# %%
# rixs_files = [
#     Path("../../BRIXS/test/data/diamond/coherence/rixs_ref.h5"),
# ]


# %% [markdown]
# ## Set analysis parameters

# %%
broad = 0.5
eloss = np.arange(0.0, 15.0, 0.05)


# %% [markdown]
# The core excitation axis is read from the first `rixs.h5`. If your workflow
# uses a shifted or manually selected excitation axis, replace `ecore` here.

# %%
with h5py.File(rixs_files[0], "r") as h5:
    ecore = np.asarray(h5["cevals"])

ecoreindex = list(range(len(ecore)))


# %% [markdown]
# ## Load RIXS data and calculate spectra
#
# This handles both file layouts:
#
# - classic: `oscstr/0001` is a dataset,
# - coherent/incoherent: `oscstr/0001/coherent` and `oscstr/0001/incoherent`.

# %%
rixs_list = [
    rixs(file=str(rixs_file), broad=broad, freq=eloss)
    for rixs_file in rixs_files
]


# %% [markdown]
# Check which spectrum modes are actually present.

# %%
available_modes = [
    mode
    for mode, (attr, _) in SPECTRUM_MODES.items()
    if any(getattr(r, attr) is not None for r in rixs_list)
]

available_modes


# %% [markdown]
# ## Average DDCS curves without interpolation
#
# This is the cheap path. It directly averages `spectrum`, `spectrum_coh`, or
# `spectrum_incoh`. No `griddata` interpolation is used here.

# %%
ddcs_by_mode = {}

for mode in available_modes:
    attr, _ = SPECTRUM_MODES[mode]
    spectra = [getattr(r, attr) for r in rixs_list if getattr(r, attr) is not None]
    ddcs_by_mode[mode] = GetData(
        ecore=ecore,
        eloss=eloss,
        ecoreindex=ecoreindex,
        spectrum=spectra,
        normalize=True,
    )

ddcs_by_mode.keys()


# %% [markdown]
# Plot one excitation index as a quick sanity check.

# %%
mode = available_modes[0]
excitation_index = 0

plt.figure(figsize=(6, 4))
plt.plot(eloss, ddcs_by_mode[mode].ddcs[excitation_index])
plt.xlabel("Energy loss (eV)")
plt.ylabel("Normalized intensity")
plt.title(f"{mode}, excitation {excitation_index}: {ecore[excitation_index]:.3f} eV")
plt.tight_layout()


# %% [markdown]
# Save the averaged DDCS curves in the same text format as `pyBRIXS-ddcs`.

# %%
for mode, ddcs in ddcs_by_mode.items():
    _, suffix = SPECTRUM_MODES[mode]
    output_file = Path(f"ddcs_vs_loss{suffix}")
    with output_file.open("w") as handle:
        ddcs.write_ddcs(handle)
    print(f"Wrote {output_file}")


# %% [markdown]
# ## Optional: build interpolated RIXS maps
#
# This is where interpolation enters. Use this only when you need a regular 2D
# grid for plotting or exporting maps.

# %%
make_maps = False

if make_maps:
    npts_loss = len(eloss) * 10
    npts_core = len(ecore) * 10
    grid = np.array([npts_loss, npts_core])

    maps_by_mode = {}
    for mode in available_modes:
        attr, _ = SPECTRUM_MODES[mode]
        maps_by_mode[mode] = analysis.average_rixs(
            rixs_list,
            ecore,
            grid=grid,
            spectrum_attr=attr,
        )


# %% [markdown]
# Example map plot. Run this cell only after setting `make_maps = True`.

# %%
if make_maps:
    mode = available_modes[0]
    rixs_map = maps_by_mode[mode]

    plt.figure(figsize=(7, 5))
    plt.pcolormesh(rixs_map.xl, rixs_map.y, rixs_map.zl, shading="auto")
    plt.xlabel("Energy loss (eV)")
    plt.ylabel("Excitation energy (eV)")
    plt.title(f"RIXS map: {mode}")
    plt.colorbar(label="Intensity")
    plt.tight_layout()


# %% [markdown]
# Optional map export. This stores `.npz`, not pickle.

# %%
if make_maps:
    for mode, rixs_map in maps_by_mode.items():
        _, suffix = SPECTRUM_MODES[mode]
        output_file = Path(f"analyzed_avg{suffix}.npz")

        avg_rixs = rixs()
        avg_rixs.w = eloss
        attr, _ = SPECTRUM_MODES[mode]
        setattr(avg_rixs, attr, np.mean([getattr(r, attr) for r in rixs_list], axis=0))

        rixs_map.export(avg_rixs, ecore, output_file)
        print(f"Wrote {output_file}")
