# tests.py

from hamiltonians import NNHamiltonian, GeneralHamiltonian, RandomHamiltonianTI
from randomizer import RandomizerHamiltonianNN, RandomizerHamiltonianRandom, RandomizerState
from randomizer import RandomizerStateRandomDelta
from stability_analysis_class import StabilityAnalysisSparse
import numpy as np
import matplotlib.pyplot as plt

corr = ["Sxx", "Sxy", "Sxz", "Syx", "Syy", "Syz", "Szx", "Szy", "Szz"]

def return_NNHam(temp):
    L = 4
    h = [[0, 0, 0]]
    J_onsite = np.zeros((3, 3))
    J_nn = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    J_nnn = np.zeros((3, 3))
    
    return NNHamiltonian(L, h, J_onsite, J_nn, J_nnn, temp)

def return_RandHam(temp):
    L = 4
    h_max = 2
    J_max = 2

    return RandomHamiltonianTI(L, temp, h_max=h_max, J_max=J_max)

def return_NNRand(ham):
    delta = [0, 1, 0, 0]
    no_of_processes = 4

    return RandomizerHamiltonianNN(ham, delta, no_of_processes)

def return_RandRand(ham):
    delta = [1, 1]
    no_of_processes = 4

    return RandomizerHamiltonianRandom(ham, delta, no_of_processes)

def return_RandState(ham):
    delta = 1
    no_of_processes = 4

    return RandomizerState(ham, delta, no_of_processes)

def run_NNHamRandNN():
    ham = return_NNHam(0)
    randomizer = return_NNRand(ham)
    stabAn = StabilityAnalysisSparse(ham, randomizer, corr)
    stabAn.run(1000)
    print("Run time for NNHamNNRand: " + str(stabAn.time[0]))

def run_NNHamRandRand():
    ham = return_NNHam(0)
    randomizer = return_RandRand(ham)
    stabAn = StabilityAnalysisSparse(ham, randomizer, corr)
    stabAn.run(1000)
    print("Run time for NNHamRandRand: " + str(stabAn.time[0]))

def run_NNHamRandState():
    ham = return_NNHam(0)
    randomizer = return_RandState(ham)
    stabAn = StabilityAnalysisSparse(ham, randomizer, corr)
    stabAn.run(1000)
    print("Run time for NNHamRandState: " + str(stabAn.time[0]))

def run_RandRandNN():
    ham = return_RandHam(0)
    randomizer = return_NNRand(ham)
    stabAn = StabilityAnalysisSparse(ham, randomizer, corr)
    stabAn.run(1000)
    print("Run time for RandRandNN: " + str(stabAn.time[0]))

def run_RandRandRand():
    ham = return_RandHam(0)
    randomizer = return_RandRand(ham)
    stabAn = StabilityAnalysisSparse(ham, randomizer, corr)
    stabAn.run(1000)
    print("Run time for RandRandRand: " + str(stabAn.time[0]))

def run_NNHamRandNN_ft():
    ham = return_NNHam(1)
    randomizer = return_NNRand(ham)
    stabAn = StabilityAnalysisSparse(ham, randomizer, corr)
    stabAn.run(1000)
    print("Run time for NNHamNNRand: " + str(stabAn.time[0]))

def run_and_save_RandRandState():
    ham = return_RandHam(0)
    randomizer = return_RandState(ham)
    stabAn = StabilityAnalysisSparse(ham, randomizer, corr)
    stabAn.run(1000)
    print("Run time for NNHamNNRand: " + str(stabAn.time[0]))
    stabAn.save_random_samples('./test_runs/', 'test_0.pickle')



if __name__ == "__main__":

    case = 1

    if case == 0:

        run_NNHamRandNN()
        run_NNHamRandRand()
        run_NNHamRandState()

        #run_RandRandNN()
        run_RandRandRand()

        run_NNHamRandNN_ft()

        run_and_save_RandRandState()

    if case == 1:

        L = 4
        h = [[0, 0, 0]]
        J_nn = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        J_nnn = np.zeros((3, 3))
        J_onsite = np.zeros((3, 3))
        delta_1 = [0, 0.01, 0, 0]
        delta_2 = 1
        no_of_processes = 4

        ham_zt = NNHamiltonian(L, h, J_onsite, J_nn, J_nnn, temp=0)
        ham_ft = NNHamiltonian(L, h, J_onsite, J_nn, J_nnn, temp=1)

        rand1 = RandomizerHamiltonianNN(ham_zt, delta_1, no_of_processes)
        rand2 = RandomizerHamiltonianNN(ham_ft, delta_1, no_of_processes)
        rand3 = RandomizerStateRandomDelta(ham_zt, delta_2, no_of_processes)
        rand4 = RandomizerStateRandomDelta(ham_ft, delta_2, no_of_processes)

        sclass = [StabilityAnalysisSparse(ham_zt, rand1, corr),
                  StabilityAnalysisSparse(ham_ft, rand2, corr),
                  StabilityAnalysisSparse(ham_zt, rand3, corr),
                  StabilityAnalysisSparse(ham_ft, rand4, corr)]

        foldername = "./test_runs/"
        filenames = ["test0.pickle", "test1.pickle", "test2.pickle", "test3.pickle"]

        data = []

        for i in range(4):

            sclass[i].run(1000)
            print(sclass[i].time[0])
            sclass[i].save_random_samples(foldername, filename=filenames[i])
            data.append(sclass[i].load_random_samples(foldername, filename=filenames[i]))

        fig, axs = plt.subplots(2, 2)

        axs[0,0].scatter(data[0]["dist"], data[0]["diffSq"], alpha=0.2, s=2)
        axs[0,1].scatter(data[1]["dist"], data[1]["diffSq"], alpha=0.2, s=2)
        axs[1,0].scatter(data[2]["dist"], data[2]["diffSq"], alpha=0.2, s=2)
        axs[1,1].scatter(data[3]["dist"], data[3]["diffSq"], alpha=0.2, s=2)

        
        Sij_test = [sclass[0].Sq_in["Sxx"][i] for i in range(L)]

        plt.figure(2)
        plt.plot(Sij_test, 'o')
        plt.show()



        

        

