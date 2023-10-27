#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ethogram plots

Script contributed by Anna Teruel-Sanchis, PhD Student at Universitat de València
NeuroFun Lab (UJI and UV) & Neural Circuits Lab (UV), València, Spain
(https://github.com/anna-teruel)

Created on Sat Jul 23 2022

@author: annateruel
"""

import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def ethogram_plot(
    h5List, minutes=None, fps=None, bodypart=None, qualitative=None, roi=bool
):
    """
    h5List = variable created with get_h5List function
    fps = frames per second, characteristic of video file. Deatful fps = None
    minutes = number of exact minutes (number of frames) in our study. Deatful minutes = None
    bodypart = the body part you want to plot on the density plot.
    qualitative = deatful is none. You can add qualitative data to store your results with complete information.
    roi = circle/square (binary option). True for circle, false for square.
    """

    if h5List is None:
        raise ValueError("h5List not found")

    fig, axs = plt.subplots(len(h5List), sharex="all", sharey="all", figsize=(3, 4))

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

        Top = df[DLCscorer]["T"]
        T = round(np.mean(Top[Top["likelihood"] > 0.90]))
        Down = df[DLCscorer]["radi"]
        D = round(np.mean(Down[Down["likelihood"] > 0.90]))

        centre = df[DLCscorer]["centre"]
        c = round(np.mean(centre[centre["likelihood"] > 0.90]))
        radi = df[DLCscorer]["radi"]
        r = round(np.mean(radi[radi["likelihood"] > 0.90]))
        rad = (np.sqrt((c["x"] - r["x"]) ** 2 + (c["y"] - r["y"]) ** 2)) + 20

        # Animal coordinates
        bp = df[DLCscorer][bodypart]

        data_q = pd.read_excel(os.path.join(qualitative))
        video_name = (data_q["video_path"])[video]
        animal = (data_q["animal"])[video]
        # session = ((data_q['session'])[video])
        grup = (data_q["grup"])[video]
        estimul = (data_q["estimul"])[video]
        tipus = (data_q["tipus"])[video]

        results = []
        for coord in range(bp.shape[0]):
            if roi == False:
                if estimul == "esquerra":
                    if bp["x"][coord] < T[0]:
                        results.append([coord])
                elif estimul == "dreta":
                    if bp["x"][coord] > T[0]:
                        results.append([coord])
            elif roi == True:
                if (
                    ((bp["x"][coord] - c["x"]) ** 2) + ((bp["y"][coord] - c["y"]) ** 2)
                ) < (rad**2):
                    results.append([coord])

        r = np.array([elem for singleList in results for elem in singleList])

        # Ethograms
        plt.xlim(0, len(bp))
        plt.ylim(0.5, 1)
        if tipus == "neutre":
            axs[video].eventplot(r, color="#245CA6")
            axs[video].set_title(animal + grup, fontsize=10)
            fig.tight_layout(pad=0.75)
            plt.savefig("Ethogram" + ".pdf", format="pdf", dpi=100)
        elif tipus == "conespecific":
            axs[video].eventplot(r, color="#75A644")
            axs[video].set_title(animal + grup, fontsize=10)
            fig.tight_layout(pad=0.75)
            plt.savefig("Ethogram" + ".pdf", format="pdf", dpi=100)


def main():
    h5_path = "/Users/annateruel/Desktop/frontiers_ceci/frontiers-ceci-2022-07-21/bueno"
    data_qualitative = "/Users/annateruel/Desktop/frontiers_ceci/frontiers-ceci-2022-07-21/bueno/data_qualitativa.xlsx"
    h5List = get_h5List(h5_path)
    ethogram_plot(
        h5List,
        minutes=5,
        fps=25,
        bodypart="nose",
        qualitative=data_qualitative,
        roi=True,
    )


if __name__ == "__main__":
    main()
