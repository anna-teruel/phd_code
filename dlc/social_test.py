#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


def time_roi(h5List, minutes=None, fps=None, dist_cage=None, qualitative=None):
    """
    h5List = variable created with get_h5List function
    fps = frames per second, characteristic of video file. Deatful fps = None
    minutes = number of exact minutes (number of frames) in our study. Deatful minutes = None
    bodypart = the body part you want to plot on the density plot.
    qualitative = deatful is none. You can add qualitative data to store your results with complete information.
    roi = circle/square (binary option). True for circle, false for square.
    data_roi = upload your excel with roi coordinates (area and centre)
    """

    if h5List is None:
        raise ValueError("h5List not found")
    data = np.zeros((len(h5List), 9), dtype=object)

    for video in range(len(h5List)):
        video_path = os.path.normpath(h5List[video])
        path_components = video_path.split(os.sep)
        video_name: str = path_components[0]

        df = pd.read_hdf(os.path.join(h5List[video]))

        frames = fps * 60 * minutes  # 1800 frames in X minutes
        df = df[:frames]

        # Subset each deeplabcut tracking
        DLCscorer = df.columns[0][0]

        # Load corresponding bodypart you want to use
        nose = df[DLCscorer]["nose"]
        rightear = df[DLCscorer]["rightear"]
        leftear = df[DLCscorer]["leftear"]
        head = df[DLCscorer]["head"]

        # Data qualitativa
        data_q = pd.read_excel(os.path.join(qualitative))
        video_name = (data_q["video_name"])[video]
        animal = (data_q["animal"])[video]
        session = (data_q["session"])[video]
        grup = (data_q["grup"])[video]
        estimul_L = (data_q["estimul_L"])[video]
        estimul_R = (data_q["estimul_R"])[video]
        edat = (data_q["edat"])[video]

        # Cage
        sb_TR = df[DLCscorer]["TR"]
        TR = round(np.mean(sb_TR[sb_TR["likelihood"] > 0.80]))
        sb_BL = df[DLCscorer]["BL"]
        BL = round(np.mean(sb_BL[sb_BL["likelihood"] > 0.80]))

        # Glass dish LEFT
        CL = df[DLCscorer]["CL"]
        RL = df[DLCscorer]["RL"]
        CL2 = round(np.mean(CL[CL["likelihood"] > 0.80]))
        RL2 = round(np.mean(RL[RL["likelihood"] > 0.80]))
        radLEFT2 = np.sqrt((CL2["x"] - RL2["x"]) ** 2 + (CL2["y"] - RL2["y"]) ** 2)

        # Glass dish RIGHT
        CR = df[DLCscorer]["CR"]
        RR = df[DLCscorer]["RR"]
        CR2 = round(np.mean(CR[CR["likelihood"] > 0.80]))
        RR2 = round(np.mean(RR[RR["likelihood"] > 0.80]))
        radRIGHT2 = np.sqrt((CR2["x"] - RR2["x"]) ** 2 + (CR2["y"] - RR2["y"]) ** 2)

        resultsLEFT = []
        resultsRIGHT = []
        for coord in range(head.shape[0]):
            if (
                (
                    (
                        ((nose["x"][coord] - CR2["x"]) ** 2)
                        + ((nose["y"][coord] - CR2["y"]) ** 2)
                    )
                    < (radRIGHT2**2)
                )
                and (
                    (
                        ((rightear["x"][coord] - CR2["x"]) ** 2)
                        + ((rightear["y"][coord] - CR2["y"]) ** 2)
                    )
                    < (radRIGHT2**2)
                )
                and (
                    (
                        ((leftear["x"][coord] - CR2["x"]) ** 2)
                        + ((leftear["y"][coord] - CR2["y"]) ** 2)
                    )
                    < (radRIGHT2**2)
                )
                and (
                    (
                        ((head["x"][coord] - CR2["x"]) ** 2)
                        + ((head["y"][coord] - CR2["y"]) ** 2)
                    )
                    < (radRIGHT2**2)
                )
            ):
                resultsRIGHT.append([coord])
            elif (
                (
                    (
                        ((nose["x"][coord] - CL2["x"]) ** 2)
                        + ((nose["y"][coord] - CL2["y"]) ** 2)
                    )
                    < (radLEFT2**2)
                )
                and (
                    (
                        ((rightear["x"][coord] - CL2["x"]) ** 2)
                        + ((rightear["y"][coord] - CL2["y"]) ** 2)
                    )
                    < (radLEFT2**2)
                )
                and (
                    (
                        ((leftear["x"][coord] - CL2["x"]) ** 2)
                        + ((leftear["y"][coord] - CL2["y"]) ** 2)
                    )
                    < (radLEFT2**2)
                )
                and (
                    (
                        ((head["x"][coord] - CL2["x"]) ** 2)
                        + ((head["y"][coord] - CL2["y"]) ** 2)
                    )
                    < (radLEFT2**2)
                )
            ):
                resultsLEFT.append([coord])

        timeR = len(resultsRIGHT)
        timeR_sec = timeR / fps
        rR = np.array([elem for singleList in resultsRIGHT for elem in singleList])

        timeL = len(resultsLEFT)
        timeL_sec = timeL / fps
        rL = np.array([elem for singleList in resultsLEFT for elem in singleList])

        # discrimination index = (new-old)/(new+old)
        # if grup == "B":
        #     dis_index = (timeR_sec - timeL_sec) / (timeR_sec + timeL_sec)
        # elif grup == "A":
        #     dis_index = (timeL_sec - timeR_sec) / (timeL_sec + timeR_sec)

        # cálculo velocidad y distancia total recorrida
        if dist_cage is None:
            raise ValueError(
                "No dist_cage assigned. Required for pixels to cm conversion"
            )
        else:
            # distance in pixels between two points from the cage
            dx = TR["x"] - BL["x"]
            dy = BL["y"] - TR["y"]
            conversion = dx / dist_cage  # pixels in 1 cm

            # velocity and total distance are calculated from the "spine 2" tracking,
            # which represents de body center of mass.
            vel_pixels = np.sqrt((np.diff(head["x"])) ** 2 + (np.diff(head["y"])) ** 2)
            vel_cm = vel_pixels / conversion

            total_dist = np.sum(vel_cm)
            vel = vel_cm * fps
            avg_vel = np.mean(vel)

        # data[video] = [
        #     video_name,
        #     animal,
        #     edat,
        #     session,
        #     grup,
        #     timeR_sec,
        #     timeL_sec,
        #     total_dist,
        #     avg_vel,
        # ]

        # Create subplots and store the axis object
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.set(style="dark")

# Create the KDE plot
        sns.kdeplot(
            x = head["x"], y = head["y"], shade=True, cmap="plasma", fill=True, cbar=True, ax=ax
        )
        ax.set_aspect(
            "equal", adjustable="box"
        )  # By default, plots have more pixels along one axis over the other.
        # ax.add_patch(
        #    mpatches.Rectangle((TR['x'],TR['y']),dx,dy, angle= 90, edgecolor='#323831', facecolor='none'))
        ax.add_patch(
            mpatches.Circle(
                (CL2["x"], CL2["y"]), radLEFT2, fill=False, edgecolor="#364FA3"
            )
        )  # azul izquierda
        ax.add_patch(
            mpatches.Circle(
                (CR2["x"], CR2["y"]), radRIGHT2, fill=False, edgecolor="#75A33E"
            )
        )  # verde derecha
        plt.title(video_name)
        plt.savefig(
            "Density_plot"
            + "_"
            + str(animal)
            + "_"
            + str(edat)
            + "_"
            + str(session)
            + "_"
            + str(grup)
            + ".pdf",
            format="pdf",
            dpi=100,
        )

        # if grup == "A":  # esquerra és el que es canvia
        #     fig, axs = plt.subplots(2, sharex="all", sharey="all")
        #     plt.xlim(0, len(head))
        #     plt.ylim(0.5, 1.5)
        #     axs[0].eventplot(rR, color="#6640F7")  # estimul vell AZUL
        #     axs[0].set_title(animal + grup, fontsize=10)

        #     axs[1].eventplot(rL, color="#B6F734")  # estimul nou VERDE
        #     axs[1].set_title(animal + grup, fontsize=10)

        #     fig.tight_layout(pad=0.75)
        #     plt.savefig(
        #         "Ethogram"
        #         + "_"
        #         + str(animal)
        #         + "_"
        #         + str(edat)
        #         + "_"
        #         + str(session)
        #         + ".pdf",
        #         format="pdf",
        #         dpi=100,
        #     )
        # elif grup == "B":  # dreta és el que es canvia
        #     fig, axs = plt.subplots(2, sharex="all", sharey="all")
        #     plt.xlim(0, len(head))
        #     plt.ylim(0.5, 1.5)
        #     axs[0].eventplot(rL, color="#6640F7")  # estimul vell AZUL
        #     axs[0].set_title(animal + grup, fontsize=10)

        #     axs[1].eventplot(rR, color="#B6F734")  # estimul nou VERDE
        #     axs[1].set_title(animal + grup, fontsize=10)

        #     fig.tight_layout(pad=0.75)
        #     plt.savefig(
        #         "Ethogram"
        #         + "_"
        #         + str(animal)
        #         + "_"
        #         + str(edat)
        #         + "_"
        #         + str(session)
        #         + ".pdf",
        #         format="pdf",
        #         dpi=100,
        #     )

    np.savetxt("data_results.csv", data, delimiter=",", fmt="%s")


def main():
    h5_path = '/Users/annateruel/Desktop/h5s-social-test_2022/'
    data_qualitative = '/Users/annateruel/Desktop/h5s-social-test_2022/data_q.xlsx'
    h5List = get_h5List(h5_path)
    time_roi(h5List, minutes=5, fps=15, dist_cage=22, qualitative=data_qualitative)


if __name__ == "__main__":
    main()
