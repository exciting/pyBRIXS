# %% [markdown]
# # pyBRIXS notebook workflow
#
# Minimal workflow for a notebook:
#
# - use `calculate_or_load_ddcs(...)` for cached spectra,
# - use `calculate_maps(...)` only when an interpolated RIXS map is needed.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyBRIXS.workflow import (
    calculate_or_load_ddcs,
    calculate_or_load_maps,
)


# %% [markdown]
# ## 1. Choose your dataset
#
# Each entry can be either a folder containing `rixs.h5` or a direct `rixs.h5`
# file. Multiple entries are averaged automatically.

# %%
path_stat = Path("run_LiF/rixs")

rixs_files = [
    path_stat,
]


# %% [markdown]
# Example for several calculations:

# %%
# path_stat = Path("run_LiF/rixs")
# rixs_files = [
#     path_stat / "30deg-broad05" / f"i2-o{i}"
#     for i in range(1, 10)
# ]


# %% [markdown]
# ## 2. Set spectrum parameters

# %%
broad = 0.5
eloss = np.arange(0.0, 15.0, 0.05)
ecore = np.array([653.0, 654.0, 655.0, 656.0])


# %% [markdown]
# ## 3. Calculate or load DDCS spectra
#
# No interpolation happens here. The result is a dictionary:
#
# - `ddcs["classic"]` for normal BRIXS output,
# - `ddcs["coherent"]` and `ddcs["incoherent"]` for coherence output.
#
# A matching `.npz` cache is loaded on later calls. A fresh calculation writes
# both the cache and the established DDCS text format.

# %%
ddcs = calculate_or_load_ddcs(
    rixs_files,
    broad=broad,
    eloss=eloss,
    ecore=ecore,
    modes="auto",
    normalize=True,
    output_base="ddcs_vs_loss",
)

ddcs.keys()


# %% [markdown]
# ## 4. Plot one spectrum

# %%
mode = next(iter(ddcs))
excitation_index = 0

plt.figure(figsize=(6, 4))
plt.plot(eloss, ddcs[mode].ddcs[excitation_index])
plt.xlabel("Energy loss (eV)")
plt.ylabel("Normalized intensity")
plt.title(f"{mode}, excitation index {excitation_index}")
plt.tight_layout()


# %% [markdown]
# ## 5. Optional: calculate RIXS maps
#
# Maps require interpolation. Use this only when you want a 2D map.

# %%
maps = calculate_or_load_maps(
    rixs_files,
    broad=broad,
    eloss=eloss,
    ecore=ecore,
    modes="auto",
    grid_scale=10,
    cache_base="rixs_map_avg",
)

maps.keys()


# %% [markdown]
# Plot one map.

# %%
mode = next(iter(maps))
rixs_map = maps[mode]

plt.figure(figsize=(7, 5))
plt.pcolormesh(rixs_map.xl, rixs_map.y, rixs_map.zl, shading="auto")
plt.xlabel("Energy loss (eV)")
plt.ylabel("Excitation energy (eV)")
plt.title(f"RIXS map: {mode}")
plt.colorbar(label="Intensity")
plt.tight_layout()
