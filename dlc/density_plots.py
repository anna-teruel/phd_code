#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavioral Density plots

Script contributed by Anna Teruel-Sanchis, PhD Student at Universitat de València
NeuroFun Lab (UJI and UV) & Neural Circuits Lab (UV), València, Spain
(https://github.com/anna-teruel)

Created on Sat Jul 23 2022

@author: annateruel
"""

import os
import os.path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_h5List(h5_path):
    """
    h5_path = directory where all "filtered.h5" files are stored
    """
    h5_path = os.chdir(h5_path)
    h5List = []
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith("filtered.h5")]:
            h5List.append(os.path.join(dirpath, filename))
    h5List.sort()
    return h5List


def density_plot(h5List, minutes=None, fps=None, bodypart=None, qualitative=None):
    """
    h5List = variable created with get_h5List function
    fps = frames per second, characteristic of video file. Deatful fps = None
    minutes = number of exact minutes (number of frames) in our study. Deatful minutes = None
    bodypart = the body part you want to plot on the density plot.
    """

    if h5List is None:
        raise ValueError("h5List not found")
    fig, axs = plt.subplots(
        len(h5List), 8, sharex="all", sharey="all", figsize=(15, 15)
    )

    for video in range(len(h5List)):
        video_path = os.path.normpath(h5List[video])
        path_components = video_path.split(os.sep)
        video_name: str = path_components[0]

        df = pd.read_hdf(os.path.join(h5List[video]))

        if minutes is None:
            raise ValueError("No minutes assigned")
        else:
            frames = round(fps * 60 * minutes)
            df = df[:frames]
        # Subset each deeplabcut tracking
        DLCscorer = df.columns[0][0]

        centre = df[DLCscorer]["centre"]
        c = round(np.mean(centre[centre["likelihood"] > 0.90]))
        radi = df[DLCscorer]["radi"]
        r = round(np.mean(radi[radi["likelihood"] > 0.90]))

        rad = (np.sqrt((c[0] - r[0]) ** 2 + (c[1] - r[1]) ** 2)) + 20

        # Animal coordinates
        bp = df[DLCscorer][bodypart]

        data_q = pd.read_excel(os.path.join(qualitative))
        animal = (data_q["animal"])[video]
        grup = (data_q["grup"])[video]
        estimul = (data_q["estimul"])[video]

        # Density Plots

        plt.subplots(figsize=(10, 10))
        sns.set(style="dark")
        ax = sns.kdeplot(
            bp["x"],
            bp["y"],
            shade=True,
            fill=True,
            cbar=True,
            cmap="viridis",
            levels=10,
            alpha=0.9,
        )
        plt.title(animal + grup)
        ax.set_aspect(
            "equal", adjustable="box"
        )  # By default, plots have more pixels along one axis over the other.
        ax.add_patch(
            mpatches.Circle((c["x"], c["y"]), rad, fill=False, edgecolor="#323831")
        )
        plt.savefig("Density_plot_" + animal + grup + ".pdf", format="pdf", dpi=100)


def main():
    h5_path = "/Users/annateruel/frontiers-ceci-2022-07-21/bueno/"
    data_qualitative = (
        "/Users/annateruel/frontiers-ceci-2022-07-21/bueno/data_qualitativa.xlsx"
    )
    h5List = get_h5List(h5_path)
    density_plot(
        h5List, minutes=5, fps=25, bodypart="nose", qualitative=data_qualitative
    )


if __name__ == "__main__":
    main()
