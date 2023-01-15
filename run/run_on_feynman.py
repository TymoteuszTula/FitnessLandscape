# run_on_feynman.py

import numpy as np
import sys
import os
sys.path.append("./bin/")
from hamiltonians import NNHamiltonian, RandomHamiltonianTI
from randomizer import RandomizerState, RandomizerHamiltonianNN, RandomizerHamiltonianRandom
from randomizer import RandomizerHamiltonianNNRandomDelta, RandomizerStateRandomDelta
from randomizer import RandomizerHamiltonianRandomRandomDelta
from stability_analysis_class import StabilityAnalysisSparse
from tools import SijCalculator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no', type=int, nargs='+')
parser.add_argument('--range', type=str)
parser.add_argument('--input_folder', type=str)

args = parser.parse_args()

if args.no !=None:
    no_file = args.no
elif args.range != None:
    rng = args.range.split("-")
    if len(rng) == 2:
        no_file = list(range(int(rng[0]), int(rng[1]) + 1))
    else:
        raise ValueError("Range should have a form min-max")
else:
    raise ValueError("Either Range or no must be specified")

# Open relevant input file and unpack the data

par_values = {"ham_type": "NNAf", "rand_type": "ham", "L": "4", "temp": "0", "h_max": "1",
              "J_max": "1", "delta": "0-1-0-0", "no_of_processes": "4", "no_of_samples": "5000",
              "output_prefix": "output", "save_rhams": "False", "temp_type": "value",
              "no_qpoints": "100", "save_Sqs": "False", "save_Sijs": "False", "save_Sqints": "False",
              "save_rham_params": "False"}

foldername_input = './run/input/feynman/' + args.input_folder

np.random.seed(42)
h_rand = [np.random.rand(3)]
J_onsite_rand = np.random.rand(3, 3)
J_onsite_rand = 1 / 2 * J_onsite_rand.conj().T @ J_onsite_rand
J_nnn_rand = np.random.rand(3, 3)
J_nnn_rand = 1 / 2 * J_nnn_rand.conj().T @ J_nnn_rand
J_nn_rand = np.random.rand(3, 3)
J_nn_rand = 1 / 2 * J_nn_rand.conj().T @ J_nn_rand
np.random.seed()

h = [[0, 0, 0]]
J_onsite = np.zeros((3, 3))
J_nnn = np.zeros((3, 3))
J_nn = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

corr = ["Sxx", "Sxy", "Sxz", "Syx", "Syy", "Syz", "Szx", "Szy", "Szz"]

for i in no_file:
    filename_input = 'input' + str(i) + '.txt'
    with open(foldername_input + filename_input, "r") as fh:
        first_line = fh.readline()
        if first_line[:-1] != "#Feynman":
            raise ValueError
        for next_line in fh:
            par = next_line.split("=")
            if par[0] in par_values:
                par_values[par[0]] = par[1][1:-1]

    ham_type = str(par_values["ham_type"])
    rand_type = str(par_values["rand_type"])
    L = int(par_values["L"])
    temp = float(par_values["temp"])
    h_max = float(par_values["h_max"])
    J_max = float(par_values["J_max"])
    no_of_processes = int(par_values["no_of_processes"])
    no_of_samples = int(par_values["no_of_samples"])
    output_filename = str(par_values["output_prefix"])
    delta = str(par_values["delta"]).split("-")
    delta = [float(delta[i]) for i in range(len(delta))]
    temp_type = str(par_values["temp_type"])
    no_qpoints = int(par_values["no_qpoints"])

    if par_values["save_rhams"] == "True":
        save_rhams = True
    else:
        save_rhams = False

    if par_values["save_Sqs"] == "True":
        save_Sqs = True
    else:
        save_Sqs = False

    if par_values["save_Sijs"] == "True":
        save_Sijs = True
    else:
        save_Sijs = False

    if par_values["save_Sqints"] == "True":
        save_Sqints = True
    else:
        save_Sqints = False

    if par_values["save_rham_params"] == "True":
        save_rham_params = True
    else:
        save_rham_params = False

    # Change in chosing temperature

    if ham_type == "NNAf":
        ham = NNHamiltonian(L, h, J_onsite, J_nn, J_nnn, temp=0)
    elif ham_type == "NNRand":
        ham = NNHamiltonian(L, h_rand, J_onsite_rand, J_nn_rand, J_nnn_rand, temp=0)
    elif ham_type == "Rand":
        ham = RandomHamiltonianTI(L, temp=0, h_max=h_max, J_max=J_max)
    else:
        raise ValueError("Inproper ham_type name")

    if temp_type == "value":
        ham.temp = temp
    else:
        H_in = ham.get_init_ham().todense()
        eigvals = SijCalculator.find_eigvals(H_in)
        if temp_type == "gap":
            gap = SijCalculator.find_gap(eigvals)
            ham.temp = temp * gap
        elif temp_type == "bandwidth":
            bandwidth = SijCalculator.find_bandwidth(eigvals)
            ham.temp = temp * bandwidth
        else:
            raise ValueError("Inproper temperature type")

    # if rand_type == "ham":
    #     if ham_type == "NNAf":
    #         rand = RandomizerHamiltonianNN(ham, delta, no_of_processes)
    #     elif ham_type == "Rand":
    #         rand = RandomizerHamiltonianRandom(ham, delta, no_of_processes)
    #     elif ham_type == "RandNN":
    #         rand = RandomizerHamiltonianNNRandomDelta(ham, delta, no_of_processes)
    # elif rand_type == "state":
    #     if 
    #     rand = RandomizerState(ham, delta[0], no_of_processes)

    if rand_type == "hamNN":
        rand = RandomizerHamiltonianNN(ham, delta, no_of_processes)
    elif rand_type == "ham_rand":
        rand = RandomizerHamiltonianRandom(ham, delta, no_of_processes)
    elif rand_type == "ham_randNN":
        rand = RandomizerHamiltonianNNRandomDelta(ham, delta, no_of_processes)
    elif rand_type == "ham_rand_randDelta":
        rand = RandomizerHamiltonianRandomRandomDelta(ham, delta, no_of_processes)
    elif rand_type == "state":
        rand = RandomizerState(ham, delta, no_of_processes)
    elif rand_type == "state_rand":
        rand = RandomizerStateRandomDelta(ham, delta, no_of_processes)
    else:
        raise ValueError("Inproper rand_type name")

    stabsparse = StabilityAnalysisSparse(ham, rand, corr, save_rhams, 
                                         temp_mul=temp, temp_type=temp_type,
                                         no_qpoints=no_qpoints,
                                         save_Sqs=save_Sqs,
                                         save_Sijs=save_Sijs,
                                         save_rham_params=save_rham_params,
                                         save_Sqints=save_Sqints)
    foldername = './run/output/feynman/'
    output_split = output_filename.split("/")
    if len(output_split) != 1:
        path = foldername
        for j in range(len(output_split) - 1):
            path += output_split[j] + "/"
    
    if not os.path.exists(path):
        os.makedirs(path)

    stabsparse.run(no_of_samples)
    stabsparse.save_random_samples(foldername, output_filename + str(i) + '.pickle')