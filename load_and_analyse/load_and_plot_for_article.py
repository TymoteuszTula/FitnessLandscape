# load_and_plot_for_article.py

import pickle
import sys
sys.path.append("./bin/")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

if __name__ == "__main__":

    foldername_input_feynman1 = "./run/input/feynman/feynman1/run16042022/"
    foldername_input_feynman2 = "./run/input/feynman/feynman2/run17042022/"
    foldername_input_feynman3 = "./run/input/feynman/feynman3/run17042022/"

    foldername_output_feynman1 = "./run/output/feynman/feynman1/run16042022/"
    foldername_output_feynman2 = "./run/output/feynman/feynman2/run17042022/"
    foldername_output_feynman3 = "./run/output/feynman/feynman3/run17042022/"

    file_no_f1 = 200
    file_no_f2 = 176
    file_no_f3 = 352

    data = []

    # Load files from feynman

    for i in range(file_no_f1):
        filename_i = "input" + str(i+1) + ".txt"
        filename_o = "output" + str(i+1) + ".pickle"
        with open(foldername_output_feynman1 + filename_o, "rb") as fh:
            data.append(pickle.load(fh))
        with open(foldername_input_feynman1 + filename_i, 'r') as fh:
            content = fh.readlines()
            ham_type = content[4][10:-1]
            rand_type = content[5][11:-1]
            data[-1]["ham_type"] = ham_type
            data[-1]["rand_type"] = rand_type

    # Load files from feynman2

    for i in range(file_no_f2):
        filename_i = "input" + str(i+1) + ".txt"
        filename_o = "output" + str(i+1) + ".pickle"
        with open(foldername_output_feynman2 + filename_o, "rb") as fh:
            data.append(pickle.load(fh))
        with open(foldername_input_feynman2 + filename_i, 'r') as fh:
            content = fh.readlines()
            ham_type = content[4][10:-1]
            rand_type = content[5][11:-1]
            data[-1]["ham_type"] = ham_type
            data[-1]["rand_type"] = rand_type

    # Load files from feynman3

    for i in range(file_no_f3):
        filename_i = "input" + str(i+1) + ".txt"
        filename_o = "output" + str(i+1) + ".pickle"
        with open(foldername_output_feynman3 + filename_o, "rb") as fh:
            data.append(pickle.load(fh))
        with open(foldername_input_feynman3 + filename_i, 'r') as fh:
            content = fh.readlines()
            ham_type = content[4][10:-1]
            rand_type = content[5][11:-1]
            data[-1]["ham_type"] = ham_type
            data[-1]["rand_type"] = rand_type

    # Write data in different categories

    rand_types = [("ham_randNN", 7), ("ham_rand_randDelta", 7), ("state_rand", 8)]
    temp_types = [("0", 1), ("1/2Gap", 0), ("2Gap", 0), ("1/4Bandwidth", 0)]
    L = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    Sij_diff = {rtype[0]: {ttype[0]: {L_i: [] for L_i in L[:rtype[1]+ttype[1]]} for ttype in temp_types} for rtype in rand_types}
    Sq_diff = {rtype[0]: {ttype[0]: {L_i: [] for L_i in L[:rtype[1]+ttype[1]]} for ttype in temp_types} for rtype in rand_types}
    dist = {rtype[0]: {ttype[0]: {L_i: [] for L_i in L[:rtype[1]+ttype[1]]} for ttype in temp_types} for rtype in rand_types}
    ham_dist = {rtype[0]: {ttype[0]: {L_i: [] for L_i in L[:rtype[1]+ttype[1]]} for ttype in temp_types} for rtype in rand_types[:2]}

    for i in range(file_no_f1 + file_no_f2 + file_no_f3):
        if data[i]["info"]["temp_type"] == "value":
            temp_label = "0"
        elif data[i]["info"]["temp_type"] == "gap":
            if data[i]["info"]["temp_mul"] < 1:
                temp_label = "1/2Gap"
            else:
                temp_label = "2Gap"
        else:
            temp_label = "1/4Bandwidth"
        rand_label = data[i]["rand_type"]
        L_label = data[i]["info"]["L"]

        Sij_diff[rand_label][temp_label][L_label] += data[i]["diffSij"]
        Sq_diff[rand_label][temp_label][L_label] += data[i]["diffSq"]

        dist[rand_label][temp_label][L_label] += data[i]["dist"]
        if rand_label != "state_rand":
            ham_dist[rand_label][temp_label][L_label] += data[i]["ham_dist"]

    # Transform data to arrays (to see if they have equal number of data points)

    Sij_diff_ar = {rtype: {ttype: {L_i: np.array(Sij_diff[rtype][ttype][L_i]) / L_i**2 for L_i in Sij_diff[rtype][ttype].keys()} 
                        for ttype in Sij_diff[rtype].keys()} for rtype in Sij_diff.keys()}
    Sq_diff_ar = {rtype: {ttype: {L_i: np.array(Sq_diff[rtype][ttype][L_i]) / L_i for L_i in Sq_diff[rtype][ttype].keys()} 
                        for ttype in Sq_diff[rtype].keys()} for rtype in Sq_diff.keys()}

    dist_ar = {rtype: {ttype: {L_i: np.array(dist[rtype][ttype][L_i]) for L_i in dist[rtype][ttype].keys()} 
                        for ttype in dist[rtype].keys()} for rtype in dist.keys()}
    ham_dist_ar = {rtype: {ttype: {L_i: np.array(ham_dist[rtype][ttype][L_i]) for L_i in ham_dist[rtype][ttype].keys()} 
                        for ttype in ham_dist[rtype].keys()} for rtype in ham_dist.keys()}

    # for rtype in Sij_diff.keys():
    #     for ttype in Sij_diff[rtype].keys():
    #         for L_i in Sij_diff[rtype][ttype].keys():
    #             if Sq_diff_ar[rtype][ttype][L_i].size != 16000:
    #                 print("Something's wrong!")

    # Prepare data for figures

    Y_plots = [[Sq_diff_ar["ham_randNN"]["0"], Sq_diff_ar["ham_rand_randDelta"]["0"], Sq_diff_ar["state_rand"]["0"]],
                [Sq_diff_ar["ham_randNN"]["1/2Gap"], Sq_diff_ar["ham_rand_randDelta"]["1/2Gap"], Sq_diff_ar["state_rand"]["1/2Gap"]],
                [Sq_diff_ar["ham_randNN"]["2Gap"], Sq_diff_ar["ham_rand_randDelta"]["2Gap"], Sq_diff_ar["state_rand"]["2Gap"]],
                [Sq_diff_ar["ham_randNN"]["1/4Bandwidth"], Sq_diff_ar["ham_rand_randDelta"]["1/4Bandwidth"], Sq_diff_ar["state_rand"]["1/4Bandwidth"]]]

    X_plots = [[ham_dist_ar["ham_randNN"]["0"], ham_dist_ar["ham_rand_randDelta"]["0"], dist_ar["state_rand"]["0"]],
                [ham_dist_ar["ham_randNN"]["1/2Gap"], ham_dist_ar["ham_rand_randDelta"]["1/2Gap"], dist_ar["state_rand"]["1/2Gap"]],
                [ham_dist_ar["ham_randNN"]["2Gap"], ham_dist_ar["ham_rand_randDelta"]["2Gap"], dist_ar["state_rand"]["2Gap"]],
                [ham_dist_ar["ham_randNN"]["1/4Bandwidth"], ham_dist_ar["ham_rand_randDelta"]["1/4Bandwidth"], dist_ar["state_rand"]["1/4Bandwidth"]]]

    L_init = 4

    L_range = [[L[L_init:8], L[L_init:8], L[L_init:9]], 
               [L[L_init:7], L[L_init:7], L[L_init:8]],
               [L[L_init:7], L[L_init:7], L[L_init:8]],
               [L[L_init:7], L[L_init:7], L[L_init:8]]]


    # Dist vs HamDist plots

    Y_DHD_plots = [[dist_ar["ham_randNN"]["0"], dist_ar["ham_rand_randDelta"]["0"]],
                   [dist_ar["ham_randNN"]["1/2Gap"], dist_ar["ham_rand_randDelta"]["1/2Gap"]],
                   [dist_ar["ham_randNN"]["2Gap"], dist_ar["ham_rand_randDelta"]["2Gap"]],
                   [dist_ar["ham_randNN"]["1/4Bandwidth"], dist_ar["ham_rand_randDelta"]["1/4Bandwidth"]]]

    X_DHD_plots = [[ham_dist_ar["ham_randNN"]["0"], ham_dist_ar["ham_rand_randDelta"]["0"]],
                   [ham_dist_ar["ham_randNN"]["1/2Gap"], ham_dist_ar["ham_rand_randDelta"]["1/2Gap"]],
                   [ham_dist_ar["ham_randNN"]["2Gap"], ham_dist_ar["ham_rand_randDelta"]["2Gap"]],
                   [ham_dist_ar["ham_randNN"]["1/4Bandwidth"], ham_dist_ar["ham_rand_randDelta"]["1/4Bandwidth"]]]

    L_DHD_range = [[L[L_init:8], L[L_init:8]],
                   [L[L_init:7], L[L_init:7]],
                   [L[L_init:7], L[L_init:7]],
                   [L[L_init:7], L[L_init:7]]]

    # Find largest value in each column

    value0 = 1
    value1 = 3
    value2 = 1.5

    valuex0 = 0.04
    valuex1 = 0.04
    valuex2 = 0.7

    value_max = [value0, value1, value2]
    value_xmax = [valuex0, valuex1, valuex2]

    scale = False

    ###############################################################################################################
    # Plotting figures
    ###############################################################################################################
    case = "Second Draft"
    if case == "First Draft":
        mpl.rcParams.update({'font.size': 6})

        # Establish figures

        figSq, axsSq = plt.subplots(4, 3, figsize=(10, 7))
        plt.subplots_adjust(left = 0.075, right=0.95, top=0.95, bottom=0.075, wspace=0.15, hspace=0.15)
        figDHD, axsDHD = plt.subplots(4, 2, figsize=(7, 7))

        # First figure

        for i in range(4):
            axsSq[i][0].set_ylabel(r"$\Delta S_{H}^{H_t}$")
            axsSq[i][2].set_ylabel(r"$\Delta S_{\Psi}^{H_t}$")
            for j in range(3):
                for l in L_range[i][j]:
                    axsSq[i][j].plot(X_plots[i][j][l], Y_plots[i][j][l], 'o',label="L="+str(l), ms=1, alpha=0.2)
                    if scale:
                        axsSq[i][j].set_ylim(-0.05 * value_max[j], 1.05 * value_max[j])
                        axsSq[i][j].set_xlim(-0.05 * value_xmax[j], 1.05 * value_xmax[j])
                axsSq[i][j].legend()
                axsSq[i][j].grid()
            
        axsSq[3][0].set_xlabel("dist")
        axsSq[3][1].set_xlabel("dist")
        axsSq[3][2].set_xlabel("dist")
        axsSq[0][0].set_title("Random Hamiltonian - NN")
        axsSq[0][1].set_title("Random Hamiltonian")
        axsSq[0][2].set_title("Random state")
        axsSq[0][0].text(-0.2, 0.8, r"$T=0$", transform=axsSq[0,0].transAxes, rotation="vertical")
        axsSq[1][0].text(-0.2, 0.8, r"$T=\frac{1}{2}E_{gap}$", transform=axsSq[1,0].transAxes, rotation="vertical")
        axsSq[2][0].text(-0.2, 0.8, r"$T=2E_{gap}$", transform=axsSq[2,0].transAxes, rotation="vertical")
        axsSq[3][0].text(-0.2, 0.8, r"$T=\frac{1}{4}E_{bandwidth}$", transform=axsSq[3,0].transAxes, rotation="vertical")

        # Second figure

        for i in range(4):
            axsDHD[i][0].set_ylabel(r"Dist state")
            for j in range(2):
                for l in L_range[i][j]:
                    axsDHD[i][j].plot(X_DHD_plots[i][j][l], Y_DHD_plots[i][j][l], 'o',label="L="+str(l), ms=1, alpha=0.2)
                axsDHD[i][j].legend()
                axsDHD[i][j].grid()
            
        axsDHD[3][0].set_xlabel("Dist ham")
        axsDHD[3][1].set_xlabel("Dist ham")
        axsDHD[0][0].set_title("Random Hamiltonian - NN")
        axsDHD[0][1].set_title("Random Hamiltonian")
        axsDHD[0][0].text(-0.2, 0.8, r"$T=0$", transform=axsDHD[0,0].transAxes, rotation="vertical")
        axsDHD[1][0].text(-0.2, 0.8, r"$T=\frac{1}{2}E_{gap}$", transform=axsDHD[1,0].transAxes, rotation="vertical")
        axsDHD[2][0].text(-0.2, 0.8, r"$T=2E_{gap}$", transform=axsDHD[2,0].transAxes, rotation="vertical")
        axsDHD[3][0].text(-0.2, 0.8, r"$T=\frac{1}{4}E_{bandwidth}$", transform=axsDHD[3,0].transAxes, rotation="vertical")

        plt.show()
    
    if case == "Second Draft":

        # x labels
        dist_ham_label = r"$|| \frac{H_{rand}}{Tr H_{rand}} - \frac{H_{init}}{Tr H_{init}} ||$"
        dist_label = r"$1 - \langle \psi_{rand} | \psi_{init}\rangle$"
        dist_dm_label = r"Tr$[(\rho_{rand} - \rho_{init})^\dag (\rho_{rand} - \rho_{init})]$"

        x_labels_f1 = [[dist_ham_label, dist_label],
                       [dist_ham_label, dist_dm_label],
                       [dist_ham_label, dist_dm_label],
                       [dist_ham_label, dist_dm_label]]

        x_labels_f2 = [dist_ham_label, dist_ham_label, dist_ham_label, dist_ham_label]
        y_labels_f2 = [dist_label, dist_dm_label, dist_dm_label, dist_dm_label]

        mpl.rcParams.update({'font.size': 6})

        # Establish figures

        gs_kw1 = dict(left=0.1, right=0.95, top=0.95, bottom=0.075, wspace=0.21, hspace=0.3)
        gs_kw2 = dict(left=0.2, right=0.975, top=0.95, bottom=0.075, wspace=0.21, hspace=0.3)

        figSq, axsSq = plt.subplots(4, 2, figsize=(7, 7), gridspec_kw=gs_kw1)
        #plt.subplots_adjust(left = 0.075, right=0.95, top=0.95, bottom=0.075, wspace=0.15, hspace=0.15)
        figDHD, axsDHD = plt.subplots(4, 1, figsize=(3.5, 7), gridspec_kw=gs_kw2)

        # First figure

        for i in range(4):
            axsSq[i][0].set_ylabel(r"$\Delta S_{H}^{H_t} / L$", labelpad=0.1)
            axsSq[i][1].set_ylabel(r"$\Delta S_{\Psi}^{H_t} / L$", labelpad=0.1)
            for j in range(2):
                j_p = 2 * j
                for l in L_range[i][j_p]:
                    axsSq[i][j].plot(X_plots[i][j_p][l], Y_plots[i][j_p][l], 'o',label="L="+str(l), ms=1, alpha=0.2)
                    if scale:
                        axsSq[i][j].set_ylim(-0.05 * value_max[j], 1.05 * value_max[j])
                        axsSq[i][j].set_xlim(-0.05 * value_xmax[j], 1.05 * value_xmax[j])
                # lgnd = axsSq[i][j].legend()
                h = []
                hl = []
                for l_i, l in enumerate(L_range[i][j_p]):
                    # Dummy Line2D objects
                    h.append(Line2D([0], [0], marker='o', ms=4, c="C"+str(l_i), linestyle='None'))
                    hl.append("L="+str(l))
                axsSq[i][j].grid()
                axsSq[i][j].legend(h, hl)
                axsSq[i][j].set_xlabel(x_labels_f1[i][j], labelpad=0.1)
            
        #axsSq[0][0].set_title("Random Hamiltonian - NN")
        #axsSq[0][1].set_title("Random S")
        axsSq[0][0].text(-0.2, 0.8, r"$T=0$", transform=axsSq[0,0].transAxes, rotation="vertical")
        axsSq[1][0].text(-0.2, 0.8, r"$T=\frac{1}{2}E_{gap}$", transform=axsSq[1,0].transAxes, rotation="vertical")
        axsSq[2][0].text(-0.2, 0.8, r"$T=2E_{gap}$", transform=axsSq[2,0].transAxes, rotation="vertical")
        axsSq[3][0].text(-0.2, 0.8, r"$T=\frac{1}{4}E_{bandwidth}$", transform=axsSq[3,0].transAxes, rotation="vertical")

        # Second figure

        ColorsF2 = ["b", "r", "g", "blueviolet"]

        for i in range(4):
            h = []
            hl = []
            for l_i, l in  enumerate(L_range[i][0]):
                axsDHD[i].plot(X_DHD_plots[i][0][l], Y_DHD_plots[i][0][l], 'x',label="L="+str(l), ms=1, alpha=0.2,
                                c=ColorsF2[l_i])
            
            for l_i, l in enumerate(L_range[i][0]):
                # Dummy Line2D objects
                h.append(Line2D([0], [0], marker='x', ms=4, c=ColorsF2[l_i], linestyle='None'))
                hl.append("L="+str(l))
            axsDHD[i].legend(h, hl)
            axsDHD[i].grid()
            axsDHD[i].set_xlabel(x_labels_f2[i], labelpad=0.1)
            axsDHD[i].set_ylabel(y_labels_f2[i], labelpad=0.1)
        
        axsDHD[0].text(-0.2, 0.8, r"$T=0$", transform=axsDHD[0].transAxes, rotation="vertical")
        axsDHD[1].text(-0.2, 0.8, r"$T=\frac{1}{2}E_{gap}$", transform=axsDHD[1].transAxes, rotation="vertical")
        axsDHD[2].text(-0.2, 0.8, r"$T=2E_{gap}$", transform=axsDHD[2].transAxes, rotation="vertical")
        axsDHD[3].text(-0.2, 0.8, r"$T=\frac{1}{4}E_{bandwidth}$", transform=axsDHD[3].transAxes, rotation="vertical")

        plt.show()





    

