# -*- coding: utf-8 -*-

import sys,os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import matplotlib.gridspec as gridspec

    
def read_data_blocks(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_blocks = []
    current_block = []

    for line in lines:
        stripped_line = line.strip()
        if stripped_line and not stripped_line.startswith('#'): 
            numbers = list(map(float, stripped_line.split()))
            current_block.append(numbers)
        elif not stripped_line:  
            if current_block:
                data_blocks.append(current_block)
                current_block = []
                
    if current_block:
        data_blocks.append(current_block)

    return data_blocks

def plot_blocks(file_path, ax, labels, label, plot_labels=True, line_interval=1, offset=1, color='black'):
    data_blocks = read_data_blocks(file_path)
    all_y = []
    for i, block in enumerate(data_blocks):
        if i % line_interval != 0:
            continue
        block = list(zip(*block))
        x = [-val for val in block[0]]
        y = block[1]

        y = [(val - min(y)) / (max(y) - min(y)) for val in y]

        y = [val + i * offset for val in y]
        all_y.extend(y)
        ax.plot(x, y, color=color, linewidth=2, label=label)

        if plot_labels:
            y_label = y[-1] + 0.3
            ax.text(np.max(x)+2, y_label, f"{labels[i]:.1f} eV", verticalalignment='center')
    if plot_labels:
        ax.text(-0.04, 0.5, "Normalized Intensity [a.u.]", va='center', ha='center',
                rotation=90, transform=ax.transAxes)
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_label_coords(1.15, 0.5)
        ax.set_ylabel("Excitation Energy [eV]", labelpad=10)
        ax.set_xlabel("Energy Loss (eV)")
            
    ax.set_ylim(min(all_y)-0.2,max(all_y)+(max(all_y)-min(all_y))*0.2)    
    ax.tick_params(labelleft=False, left=False)

    handles, legend_labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(legend_labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
