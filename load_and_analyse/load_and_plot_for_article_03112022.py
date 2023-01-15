# load_and_plot_for_article_20052022.py

import pickle
import sys
sys.path.append("./bin/")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

if __name__ == "__main__":

    foldername_input_feynman1 = "./run/input/feynman/feynman2/run20052022/"
    foldername_input_feynman2 = "./run/input/feynman/feynman3/run20052022/"
    foldername_input_feynman3 = "./run/input/feynman/feynman4/run20052022/"

    foldername_output_feynman1 = "./run/output/feynman/feynman2/run20052022/"
    foldername_output_feynman2 = "./run/output/feynman/feynman3/run20052022/"
    foldername_output_feynman3 = "./run/output/feynman/feynman4/run20052022/"

    file_no_f1 = 80
    file_no_f2 = 80
    file_no_f3 = 80

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

    rand_types = [("ham_randNN", 4), ("state_rand", 4)]
    temp_types = [("0", 1), ("1/2Gap", 1), ("1/4Bandwidth", 1)]
    L = [4, 6, 8, 10]

    Init_Ham = "NNRand"

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

        if data[i]["ham_type"] == Init_Ham:

            Sij_diff[rand_label][temp_label][L_label] += data[i]["diffSij"]
            Sq_diff[rand_label][temp_label][L_label] += data[i]["Sq_int"]

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

    # Change dist_ar to correct

    for l_i in L:
        dist_ar["ham_randNN"]["1/2Gap"][l_i] = dist_ar["ham_randNN"]["1/2Gap"][l_i][:,0,0]
        dist_ar["ham_randNN"]["1/4Bandwidth"][l_i] = dist_ar["ham_randNN"]["1/4Bandwidth"][l_i][:,0,0]


    # for rtype in Sij_diff.keys():
    #     for ttype in Sij_diff[rtype].keys():
    #         for L_i in Sij_diff[rtype][ttype].keys():
    #             if Sq_diff_ar[rtype][ttype][L_i].size != 16000:
    #                 print("Something's wrong!")

    # Prepare data for figures

    Y_plots = [[Sq_diff_ar["ham_randNN"]["0"], Sq_diff_ar["state_rand"]["0"]],
                [Sq_diff_ar["ham_randNN"]["1/2Gap"], Sq_diff_ar["state_rand"]["1/2Gap"]],
                [Sq_diff_ar["ham_randNN"]["1/4Bandwidth"], Sq_diff_ar["state_rand"]["1/4Bandwidth"]]]

    X_plots = [[ham_dist_ar["ham_randNN"]["0"], dist_ar["state_rand"]["0"]],
                [ham_dist_ar["ham_randNN"]["1/2Gap"], dist_ar["state_rand"]["1/2Gap"]],
                [ham_dist_ar["ham_randNN"]["1/4Bandwidth"], dist_ar["state_rand"]["1/4Bandwidth"]]]

    #L_init = 4

    L_range = [[L for i in range(2)] for j in range(3)]


    # Dist vs HamDist plots

    Y_DHD_plots = [[dist_ar["ham_randNN"]["0"]],
                   [dist_ar["ham_randNN"]["1/2Gap"]],
                   [dist_ar["ham_randNN"]["1/4Bandwidth"]]]

    X_DHD_plots = [[ham_dist_ar["ham_randNN"]["0"]],
                   [ham_dist_ar["ham_randNN"]["1/2Gap"]],
                   [ham_dist_ar["ham_randNN"]["1/4Bandwidth"]]]

    L_DHD_range = [[L],
                   [L],
                   [L]]

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
    case = "First Draft"
    if case == "First Draft":
        mpl.rcParams.update({'font.size': 8})

        # Establish figures

        gridspec_1 = {"left": 0.15, "right": 0.95, "top": 0.9, "bottom": 0.1, "wspace": 0.25, "hspace": 0.33}
        gridspec_2 = {"left": 0.25, "right": 0.95, "top": 0.93, "bottom": 0.08, "wspace": 0.18, "hspace": 0.33}

        figSq, axsSq = plt.subplots(3, 2, figsize=(10, 8), gridspec_kw=gridspec_1)
        #plt.subplots_adjust(left = 0.15, right=0.95, top=0.9, bottom=0.1, wspace=0.25, hspace=0.33)
        figDHD, axsDHD = plt.subplots(3, 1, figsize=(3, 8), gridspec_kw=gridspec_2)

        # First figure
        #figSq.suptitle("Initial Random NN Hamiltonian")
        #figDHD.suptitle("Initial Random NN Hamiltonian")

        for i in range(3):
            # axsSq[i][0].set_ylabel(r"$\Delta S_{H}^{H_t}$")
            # axsSq[i][1].set_ylabel(r"$\Delta S_{\Psi}^{H_t}$")
            axsSq[i][0].set_ylabel(r"$\sum_{i,j}\sum_{\alpha, \beta} |\rho_{i,j}^{[init];\alpha, \beta} - \rho_{i,j}^{[rand];\alpha ,\beta}|^2$")
            axsSq[i][1].set_ylabel(r"$\sum_{i,j}\sum_{\alpha, \beta} |\rho_{i,j}^{[init];\alpha, \beta} - \rho_{i,j}^{[rand];\alpha ,\beta}|^2$")
            for j in range(2):
                for l in L_range[i][j]:
                    axsSq[i][j].plot(X_plots[i][j][l], Y_plots[i][j][l], 'o',label="L="+str(l), ms=1, alpha=0.2)
                    if scale:
                        axsSq[i][j].set_ylim(-0.05 * value_max[j], 1.05 * value_max[j])
                        axsSq[i][j].set_xlim(-0.05 * value_xmax[j], 1.05 * value_xmax[j])
                axsSq[i][j].legend()
                axsSq[i][j].grid()
            
        axsSq[2][0].set_xlabel(r"$|| \frac{H_{rand}}{Tr H_{rand}} - \frac{H_{init}}{Tr H_{init}} ||$")
        axsSq[1][0].set_xlabel(r"$|| \frac{H_{rand}}{Tr H_{rand}} - \frac{H_{init}}{Tr H_{init}} ||$")
        axsSq[0][0].set_xlabel(r"$|| \frac{H_{rand}}{Tr H_{rand}} - \frac{H_{init}}{Tr H_{init}} ||$")
        #axsSq[2][1].set_xlabel(r"Tr$[(\rho_{rand} - \rho_{init})^\dag (\rho_{rand} - \rho_{init})]$")
        #axsSq[1][1].set_xlabel(r"Tr$[(\rho_{rand} - \rho_{init})^\dag (\rho_{rand} - \rho_{init})]$")
        axsSq[1][1].set_xlabel(r"Tr$[(\rho_{rand} - \rho_{init})^2]$")
        axsSq[2][1].set_xlabel(r"Tr$[(\rho_{rand} - \rho_{init})^2]$")
        axsSq[0][1].set_xlabel(r"$1 - |\langle \psi_{rand} | \psi_{init}\rangle|^2$")
        axsSq[0][0].set_title("Random Hamiltonian - NN")
        axsSq[0][1].set_title("Random state")
        axsSq[0][0].text(-0.3, 0.8, r"$T=0$", transform=axsSq[0,0].transAxes, rotation="vertical")
        axsSq[1][0].text(-0.3, 0.8, r"$T=\frac{1}{2}E_{gap}$", transform=axsSq[1,0].transAxes, rotation="vertical")
        axsSq[2][0].text(-0.3, 0.8, r"$T=\frac{1}{4}E_{bandwidth}$", transform=axsSq[2,0].transAxes, rotation="vertical")

        # Second figure

        for i in range(3):
            for j in range(1):
                for l in L_range[i][j]:
                    axsDHD[i].plot(X_DHD_plots[i][j][l], Y_DHD_plots[i][j][l], 'o',label="L="+str(l), ms=1, alpha=0.2)
                axsDHD[i].legend()
                axsDHD[i].grid()

        axsDHD[0].set_ylabel(r"$1 - |\langle \psi_{rand} | \psi_{init}\rangle|^2$")
        axsDHD[1].set_ylabel(r"Tr$[(\rho_{rand} - \rho_{init})^2]$")
        axsDHD[2].set_ylabel(r"Tr$[(\rho_{rand} - \rho_{init})^2]$")
        axsDHD[2].set_xlabel(r"$|| \frac{H_{rand}}{Tr H_{rand}} - \frac{H_{init}}{Tr H_{init}} ||$")
        #axsDHD[1].set_xlabel(r"$|| \frac{H_{rand}}{Tr H_{rand}} - \frac{H_{init}}{Tr H_{init}} ||$")
        #axsDHD[0].set_xlabel(r"$|| \frac{H_{rand}}{Tr H_{rand}} - \frac{H_{init}}{Tr H_{init}} ||$")
        # axsDHD[0].set_title("Random Hamiltonian - NN")
        axsDHD[0].text(-0.2, 1.08, r"$T=0$", transform=axsDHD[0].transAxes, rotation="horizontal")
        axsDHD[1].text(-0.2, 1.08, r"$T=\frac{1}{2}E_{gap}$", transform=axsDHD[1].transAxes, rotation="horizontal")
        axsDHD[2].text(-0.2, 1.08, r"$T=\frac{1}{4}E_{bandwidth}$", transform=axsDHD[2].transAxes, rotation="horizontal")

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