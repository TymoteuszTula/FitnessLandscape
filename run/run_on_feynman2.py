# run_on_feynman2.py

import numpy as np
import sys
from bin.stability_analysis_class import StabilityAnalysisSparse
sys.path.append("./bin/")
from hamiltonians import NNHamiltonian, RandomHamiltonianTI
from randomizers import RandomizerState, RandomizerHamiltonianNN, RandomizerHamiltonianRandom
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no', type=int, nargs='+')
parser.add_argument('--range', type=str)

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
              "output_prefix": "output"}

foldername_input = './run/input/feynman2/'

h = [[0, 0, 0]]
J_onsite = np.zeros((3, 3))
J_nnn = np.zeros((3, 3))
J_nn = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
corr = ["Sxx", "Sxy", "Sxz", "Syx", "Syy", "Syz", "Szx", "Szy", "Szz"]

for i in no_file:
    filename_input = 'input' + str(i) + '.txt'
    with open(foldername_input + filename_input, "r") as fh:
        first_line = fh.readline()
        if first_line != "#Feynman2":
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
    output_filename = str(par_values["output_filename"])
    delta = str(par_values["delta"]).split("-")
    delta = [float(delta[i]) for i in range(len(delta))]

    # Run simulation

    if ham_type == "NNAf":
        ham = NNHamiltonian(L, h, J_onsite, J_nn, J_nnn, temp=temp)
    elif ham_type == "Rand":
        ham = RandomHamiltonianTI(L, temp=temp, h_max=h_max, J_max=J_max)
    else:
        raise ValueError("Inproper ham_type name")

    if rand_type == "ham":
        if ham_type == "NNAf":
            rand = RandomizerHamiltonianNN(ham, delta, no_of_processes)
        elif ham_type == "Rand":
            rand = RandomizerHamiltonianRandom(ham, delta, no_of_processes)
    elif rand_type == "state":
        rand = RandomizerState(ham, delta[0], no_of_processes)

    stabsparse = StabilityAnalysisSparse(ham, rand, corr)
    foldername = './run/output/feynman2/'

    stabsparse.run(no_of_samples)
    stabsparse.save_random_samples(foldername, output_filename + str(i) + '.pickle')