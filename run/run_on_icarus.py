# run_on_icarus.py

import sys
import os
sys.path.append("./bin/")
from hamiltonians import NNHamiltonian, RandomHamiltonianTI
from randomizer import RandomizerState, RandomizerHamiltonianNN, RandomizerHamiltonianRandom
from randomizer import RandomizerHamiltonianNNRandomDelta, RandomizerStateRandomDelta
from randomizer import RandomizerHamiltonianRandomRandomDelta
from stability_analysis_class import StabilityAnalysisSparse
import argparse
import numpy as np

parser = argparse.ArgumentParser()
par_values = {"ham_type": "NNAf", "rand_type": "ham", "L": "4", "temp": "0", "h_max": "1",
              "J_max": "1", "delta": "0-1-0-0", "no_of_processes": "4", "no_of_samples": "5000",
              "output_prefix": "output", "save_rhams": "False"}
parser.add_argument('no', type=str)
parser.add_argument('--input_folder', type=str)
args = parser.parse_args()

no = args.no

foldername_input = "./run/input/icarus/" + args.input_folder

h = [[0, 0, 0]]
J_onsite = np.zeros((3, 3))
J_nnn = np.zeros((3, 3))
J_nn = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
corr = ["Sxx", "Sxy", "Sxz", "Syx", "Syy", "Syz", "Szx", "Szy", "Szz"]

with open(foldername_input + "input" + no + '.txt', 'r') as fh:
    first_line = fh.readline()
    if first_line[:-1] != "#Icarus":
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

if par_values["save_rhams"] == "True":
    save_rhams = True
else:
    save_rhams = False

# Run simulation

if ham_type == "NNAf":
    ham = NNHamiltonian(L, h, J_onsite, J_nn, J_nnn, temp=temp)
elif ham_type == "Rand":
    ham = RandomHamiltonianTI(L, temp=temp, h_max=h_max, J_max=J_max)
else:
    raise ValueError("Inproper ham_type name")

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

stabsparse = StabilityAnalysisSparse(ham, rand, corr)
foldername = './run/output/icarus/'
output_split = output_filename.split("/")
if len(output_split) != 1:
    path = foldername
    for j in range(len(output_split) - 1):
        path += output_split[j]

if not os.path.exists(path):
    os.makedirs(path)

stabsparse.run(no_of_samples)
stabsparse.save_random_samples(foldername, output_filename + no + '.pickle')