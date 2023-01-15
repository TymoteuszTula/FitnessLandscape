# load_and_plot_add_Sqs.py

import pickle
import sys
sys.path.append("./bin/")
sys.path.append(".")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import scipy 
from hamiltonians import NNHamiltonian
from randomizer import RandomizerHamiltonianNNRandomDelta, RandomizerStateRandomDelta
from stability_analysis_class import StabilityAnalysisSparse
from tools import SijCalculator
from exact_diagonalisation_code_sparse import create_sx_sparse, create_sy_sparse, create_sz_sparse
from math import sqrt, pi

def xhat(q1, q2):
    qhat = q1/sqrt(q1**2 + q2**2)
    return qhat

def returnSq_int(Sq_initial):

    qx = np.linspace(-2*pi, 2*pi, 100)
    qy = np.linspace(-2*pi, 2*pi, 100)

    Sq_proper = {"Sxx": np.zeros((qx.size, qy.size), dtype=complex), 
                     "Syy": np.zeros((qx.size, qy.size), dtype=complex),
                     "Sxy": np.zeros((qx.size, qy.size), dtype=complex),
                     "Syx": np.zeros((qx.size, qy.size), dtype=complex)}

    for i, qx_i in enumerate(qx):
        for j, qy_j in enumerate(qy):
            qxhat = xhat(qx_i, qy_j)
            qyhat = xhat(qy_j, qx_i)
            Sq_proper["Sxx"][i][j] = (1-qxhat**2) * Sq_initial["Sxx"][i][j]
            Sq_proper["Syy"][i][j] = (1-qyhat**2) * Sq_initial["Syy"][i][j]
            Sq_proper["Sxy"][i][j] = -qxhat * qyhat * Sq_initial["Sxy"][i][j]
            Sq_proper["Syx"][i][j] = -qxhat * qyhat * Sq_initial["Syx"][i][j]

    return np.real(Sq_proper["Sxx"] + Sq_proper["Syy"] + Sq_initial["Szz"] +
                   Sq_proper["Sxy"] + Sq_proper["Syx"])

if __name__ == "__main__":

    foldername_input = './run/input/feynman/test/test_with_Sqs/'
    foldername_output = './run/output/feynman/test/test_with_Sqs/'

    file_no = 4

    data = []

    for i in range(file_no):
        filename_i = "input" + str(i+1) + ".txt"
        filename_o = "output" + str(i+1) + ".pickle"
        with open(foldername_output + filename_o, 'rb') as fh:
            data.append(pickle.load(fh))
        with open(foldername_input + filename_i, "r") as fh:
            content = fh.readlines()
            ham_type = content[4][10:-1]
            rand_type = content[5][11:-1]
            data[-1]["ham_type"] = ham_type
            data[-1]["rand_type"] = rand_type

    rand_types = ["state_rand"]
    temp_types = ["0", "1Gap"]
    L = [4]
    In_Ham = ["NNAf", "NNRand"]

    Sq_diff = {rtype: {ttype: {L_i: [] for L_i in L} for ttype in temp_types} for rtype in In_Ham}
    Sq_int_diff ={rtype: {ttype: {L_i: [] for L_i in L} for ttype in temp_types} for rtype in In_Ham}
    dist = {rtype: {ttype: {L_i: [] for L_i in L} for ttype in temp_types} for rtype in In_Ham}
    dist_ham = {rtype: {ttype: {L_i: [] for L_i in L} for ttype in temp_types} for rtype in In_Ham}

    for i in range(file_no):

        if data[i]["info"]["temp_mul"] < 1/4:
            temp_label = "0"
        else:
            temp_label = "1Gap"

        rand_label = data[i]["ham_type"]
        L_label = data[i]["info"]["L"]

        Sq_diff[rand_label][temp_label][L_label] += data[i]["diffSq"]
        Sq_int_diff[rand_label][temp_label][L_label] += data[i]["Sq_int"]
        dist[rand_label][temp_label][L_label] += data[i]["dist"]
        # if rand_label != "state_rand":
        #     dist_ham[rand_label][temp_label][L_label] += data[i]["ham_dist"]


    X = [[dist["NNAf"]["0"][4], dist["NNRand"]["0"][4]],
         [dist["NNAf"]["1Gap"][4], dist["NNRand"]["1Gap"][4]]]

    Y1 = [[Sq_int_diff["NNAf"]["0"][4], Sq_int_diff["NNRand"]["0"][4]],
         [Sq_int_diff["NNAf"]["1Gap"][4], Sq_int_diff["NNRand"]["1Gap"][4]]]

    ###############################################################################################################
    # Plotting figures
    ###############################################################################################################

    # fig, axs = plt.subplots(2, 2)
    
    # for i in range(2):
    #     for j in range(2):
    #         axs[i][j].plot(np.array(X[i][j]), np.array(Y1[i][j]), 'o', ms=1, alpha=0.3)

    #         axs[i][j].grid()


    ###############################################################################################################
    # Create Hamiltonians
    ###############################################################################################################
    corr = ["Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz", "Syx", "Szx", "Szy"]
    # NNAf 
    L = 9
    h = [[0,0,0]]
    J_onsite = np.zeros((3, 3))
    J_nnn = np.zeros((3, 3))
    J_nn = [[1 + 0.25, 0, 0], [0, 1 - 0.25, 0], [0, 0, 1]]

    np.random.seed(42)
    h_rand = [np.random.rand(3)]
    J_onsite_rand = np.random.rand(3, 3)
    J_onsite_rand = 1 / 2 * J_onsite_rand.conj().T @ J_onsite_rand
    J_nnn_rand = np.random.rand(3, 3)
    J_nnn_rand = 1 / 2 * J_nnn_rand.conj().T @ J_nnn_rand
    J_nn_rand = np.random.rand(3, 3)
    J_nn_rand = 1 / 2 * J_nn_rand.conj().T @ J_nn_rand
    np.random.seed()

    ham_NNAf = NNHamiltonian(L, h, J_onsite, J_nn, J_nnn, temp=0)
    ham_NNRand = NNHamiltonian(L, h_rand, J_onsite_rand, J_nn_rand, J_nnn_rand, temp=0)

    eigvals_NNAf = SijCalculator.find_eigvals(ham_NNAf.get_init_ham().todense())
    eigvals_NNRand = SijCalculator.find_eigvals(ham_NNRand.get_init_ham().todense())

    gap_NNAf = SijCalculator.find_gap(eigvals_NNAf)
    gap_NNRand = SijCalculator.find_gap(eigvals_NNRand)

    ham_NNAf_ft = NNHamiltonian(L, h, J_onsite, J_nn, J_nnn, temp=gap_NNAf)
    ham_NNRand_ft = NNHamiltonian(L, h_rand, J_onsite_rand, J_nn_rand, J_nnn_rand, temp=gap_NNRand)

    # Randomizers
    delta = 0.5
    rand_NNAf = RandomizerStateRandomDelta(ham_NNAf, delta, no_of_processes=8)
    rand_NNRand = RandomizerStateRandomDelta(ham_NNRand, delta, no_of_processes=8)
    rand_NNAf_ft = RandomizerStateRandomDelta(ham_NNAf_ft, delta, no_of_processes=8)
    rand_NNRand_ft = RandomizerStateRandomDelta(ham_NNRand_ft, delta, no_of_processes=8)

    # StAnClasses
    st_an_spNNAf = StabilityAnalysisSparse(ham_NNAf, rand_NNAf, corr, False,
                                            temp_mul=0, temp_type='value', no_qpoints=100,
                                            save_Sqs=True)
    st_an_spNNRand = StabilityAnalysisSparse(ham_NNRand, rand_NNRand, corr, False,
                                            temp_mul=0, temp_type='value', no_qpoints=100,
                                            save_Sqs=True)

    st_an_spNNAf_ft = StabilityAnalysisSparse(ham_NNAf_ft, rand_NNAf_ft, corr, False,
                                            temp_mul=0.01, temp_type='value', no_qpoints=100,
                                            save_Sqs=True)
    st_an_spNNRand_ft = StabilityAnalysisSparse(ham_NNRand_ft, rand_NNRand_ft, corr, False,
                                            temp_mul=0.01, temp_type='value', no_qpoints=100,
                                            save_Sqs=True)

    no_of_simulation = 5000

    # Run simultaions
    st_an_spNNAf.run(no_of_simulation)
    st_an_spNNRand.run(0)
    st_an_spNNAf_ft.run(0)
    st_an_spNNRand_ft.run(0)

    # Plotting figures

    fig1, axs1 = plt.subplots(1,1)
    plt.subplots_adjust(left = 0.15, right=0.95, top=0.9, bottom=0.15, wspace=0.4, hspace=0.4)

    x_chosen = st_an_spNNAf.dist 
    y_chosen = st_an_spNNAf.diffSij

    # x_chosen = st_an_spNNAf_ft.dist 
    # y_chosen = st_an_spNNAf_ft.diffSij

    argsorts = np.argsort(y_chosen)
    ly_c = argsorts[5 * no_of_simulation // 10]
    lyxr_c = argsorts[1 * no_of_simulation // 10]

    # ly_c = np.argmax(y_chosen)
    # ind_tsm = np.argpartition(y_chosen, 2)[:3]
    # lyxr_c = ind_tsm[np.argmax(np.array(y_chosen)[ind_tsm])]

    #axs1.plot(np.array(X[1][0]), np.array(Y1[1][0]), 'o', ms=1, alpha=0.3)
    axs1.plot(x_chosen, y_chosen, 'o', ms=1, alpha=0.3, c="C0")
    axs1.plot([0], [0], 'o', ms=5, alpha=1, c="k")
    axs1.plot(x_chosen[ly_c], y_chosen[ly_c], 's', ms=5, alpha=1, c="k")
    axs1.plot(x_chosen[lyxr_c], y_chosen[lyxr_c], '^', ms=5, alpha=1, c="k")
    axs1.text(x_chosen[ly_c] + 0.0015, y_chosen[ly_c] -0.0015, "2")
    axs1.text(x_chosen[lyxr_c] + 0.0015, y_chosen[lyxr_c] - 0.0015, "1")
    #axs1.set_xlabel(r"$|| \frac{H_{rand}}{Tr H_{rand}} - \frac{H_{init}}{Tr H_{init}} ||$")
    axs1.set_xlabel(r"$1 - |\langle \psi_{rand} | \psi_{init}\rangle|^2$")
    axs1.set_ylabel(r"$\Delta S_{\psi}^{H_t}$")

    # SWITCH CASE -------------#
    #case = "show scattering functions"
    case = "show real functions"
    ############################    
    
    if case == "show scattering functions":

        fig2, axs2 = plt.subplots(1,3)
        fig3, axs3 = plt.subplots(1,1)

        orig_SQ = returnSq_int(st_an_spNNAf_ft.Sq_in)
        SQ1 = returnSq_int(st_an_spNNAf_ft.Sqs[ly_c])
        SQ2 = returnSq_int(st_an_spNNAf_ft.Sqs[lyxr_c])

        max_values = [np.max(orig_SQ), np.max(SQ1), np.max(SQ2)]
        max_value = max(max_values)

        min_values = [np.min(orig_SQ), np.min(SQ1), np.min(SQ2)]
        min_value = min(min_values)

        axs2[0].imshow(orig_SQ, cmap="Blues", vmin=min_value, vmax=max_value)
        axs2[0].set_xticks([0, 24, 49, 74, 99])
        axs2[0].set_xticklabels([r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"])
        axs2[0].set_yticks([0, 24, 49, 74, 99])
        axs2[0].set_yticklabels([r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"])
        axs2[0].set_title("Orig")
        axs2[0].set_xlim()
        axs2[0].plot([0, 99], [74, 74], c='k')
        axs2[0].set_xlabel(r"$q_x$")
        axs2[0].set_ylabel(r"$q_y$", labelpad=0)

        axs2[1].imshow(SQ1, cmap="Blues", vmin=min_value, vmax=max_value)
        axs2[1].set_xticks([0, 24, 49, 74, 99])
        axs2[1].set_xticklabels([r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"])
        axs2[1].set_yticks([0, 24, 49, 74, 99])
        axs2[1].set_yticklabels([r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"])
        axs2[1].set_title("1")
        axs2[1].set_xlabel(r"$q_x$")
        axs2[1].set_ylabel(r"$q_y$", labelpad=0)

        axs2[2].imshow(SQ2, cmap="Blues", vmin=min_value, vmax=max_value)
        axs2[2].set_xticks([0, 24, 49, 74, 99])
        axs2[2].set_xticklabels([r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"])
        axs2[2].set_yticks([0, 24, 49, 74, 99])
        axs2[2].set_yticklabels([r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"])
        axs2[2].set_title("2")
        axs2[2].set_xlabel(r"$q_x$")
        axs2[2].set_ylabel(r"$q_y$", labelpad=0)

        y_cut_orig = orig_SQ[:,25]
        y_cut_1 = SQ1[:,25]
        y_cut_2 = SQ2[:,25]

        axs3.plot(y_cut_orig, c='k', label="Orig")
        axs3.plot(y_cut_1, label="1", c='b', linestyle='--')
        axs3.plot(y_cut_2, label="2", c='b', linestyle=':')
        axs3.legend()
        axs3.set_xticks([0, 25, 50, 75, 100])
        axs3.set_xticklabels([r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"])
        axs3.set_ylabel(r"$\Delta S_{H}^{H_t}$")
        axs3.set_xlabel(r"$q_x$")

        plt.show()

    if case == "show real functions":

        fig2, axs2 = plt.subplots(3, 3)
        
        orig_SIJ = st_an_spNNAf.Sij_in
        SIJ1 = st_an_spNNAf.Sijs[ly_c]
        SIJ2 = st_an_spNNAf.Sijs[lyxr_c]

        # orig_SIJ = st_an_spNNAf_ft.Sij_in
        # SIJ1 = st_an_spNNAf_ft.Sijs[ly_c]
        # SIJ2 = st_an_spNNAf_ft.Sijs[lyxr_c]

        lab2coords = [["Sxx", "Sxy", "Sxz"],
                      ["Syx", "Syy", "Syz"],
                      ["Szx", "Szy", "Szz"]]

        lab2coords2 = [["XX", "XY", "XZ"],
                       ["YX", "YY", "YZ"],
                       ["ZX", "ZY", "ZZ"]]

        yl_ondiag = 0.3
        yl_offdiag = 0.05

        ylims = [[yl_ondiag, yl_offdiag, yl_offdiag],
                 [yl_offdiag, yl_ondiag, yl_offdiag],
                 [yl_offdiag, yl_offdiag, yl_ondiag]]

        yticks_ondiag = [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]
        yticks_offdiag = [-0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04]

        yticks  = [[yticks_ondiag, yticks_offdiag, yticks_offdiag],
                   [yticks_offdiag, yticks_ondiag, yticks_offdiag],
                   [yticks_offdiag, yticks_offdiag, yticks_ondiag]]

        colors = [['b', 'r', 'r'], ['r', 'b', 'r'], ['r', 'r', 'b']]

        y_orig = [[orig_SIJ[lab2coords[i][j]][0] for i in range(3)] for j in range(3)]
        y_1 = [[SIJ1[lab2coords[i][j]][0] for i in range(3)] for j in range(3)]
        y_2 = [[SIJ2[lab2coords[i][j]][0] for i in range(3)] for j in range(3)]

        for i in range(3):
            for j in range(3):
                axs2[i,j].plot(y_orig[i][j], c='k', linestyle="-", marker='o')
                axs2[i,j].plot(y_1[i][j], c=colors[i][j], linestyle=':', marker='s')
                axs2[i,j].plot(y_2[i][j], c=colors[i][j], linestyle='--', marker='^')
                axs2[i,j].text(0.5, 0.5, lab2coords2[i][j], alpha=0.2, zorder=-5, transform=axs2[i,j].transAxes,
                                horizontalalignment='center', verticalalignment='center', fontsize=80, c=colors[i][j])
                axs2[i,j].set_ylim(-ylims[i][j], ylims[i][j])
                axs2[i,j].set_xticks(np.arange(L))
                axs2[i,j].set_xlim(-0.5, L - 0.5)
                axs2[i,j].set_yticks(yticks[i][j])
                axs2[i,j].grid()

        # Create custom legend
        legend_elements = [Line2D([0], [0], marker='o', color='k', label='Orig', markersize=8, linestyle='-'),
                            Line2D([0], [0], marker='^', color='k', label='1',markersize=8, linestyle='--'),
                            Line2D([0], [0], marker='s', color='k', label='2', markersize=8, linestyle=':')]

        axs2[0,2].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1.05))

        # Create labels
        fig2.text(0.5, 0.04, 'Position j', ha='center', fontsize='large')
        fig2.text(0.04, 0.5, r'$\langle S_{L/2}^\alpha S_{L/2 + j}^\beta \rangle$', va='center',
                       rotation='vertical', fontsize='large')

        fig2.suptitle("Real correlators - comparison")
    

        plt.show()