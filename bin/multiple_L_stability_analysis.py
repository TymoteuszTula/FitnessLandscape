# multiple_L_stability_analysis.py

import numpy as np
from exact_diagonalisation_code_sparse import create_states, create_sx_sparse, create_sy_sparse
from exact_diagonalisation_code_sparse import create_sz_sparse, create_hamiltonian_sparse
import matplotlib.pyplot as plt
import scipy.sparse as sprs
import scipy.sparse.linalg as sprsla
from stability_analysis_gs import calculate_Sij, calculate_Sij_dm
import time
from multiprocessing import Pool
from random import random

def calculate_ham_sparse(L, states, h, J_onsite, J_nn, J_nnn, bc="infinite"):

    J = [[[] for i in range(L)] for j in range(L)]
    J_zero = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for i in range(L):
        for j in range(L):
            if i == j:
                J[i][j] = J_onsite
            else:
                if bc == 'finite':
                    if abs(i - j) == 1:
                        J[i][j] = J_nn
                    elif abs(i - j) == 2:
                        J[i][j] = J_nnn
                    else:
                        J[i][j] = J_zero
                elif bc == "infinite":
                    if min(abs(i - j), abs(L + i - j), abs(L - i + j)) == 1:
                        J[i][j] = J_nn
                    elif min(abs(i - j), abs(L + i - j), abs(L - i + j)) == 2:
                        J[i][j] = J_nnn
                    else:
                        J[i][j] = J_zero

    return create_hamiltonian_sparse(states, params_input={"L": L, "J": J, "h": h})

def find_gs_sparse(ham):
    return sprsla.eigsh(ham, 1)

def return_dm_sparse(ham, beta=1):
    re = sprsla.expm(-beta * ham.tocsc())
    return re / re.trace()

def calculate_Sij_dm_sparse(L, dm, SX, SY, SZ):
    
    Sij = {"Sxx": np.zeros((L, L), dtype=complex), "Sxy": np.zeros((L, L), dtype=complex), "Sxz": np.zeros((L, L), dtype=complex),
           "Syx": np.zeros((L, L), dtype=complex), "Syy": np.zeros((L, L), dtype=complex), "Syz": np.zeros((L, L), dtype=complex),
           "Szx": np.zeros((L, L), dtype=complex), "Szy": np.zeros((L, L), dtype=complex), "Szz": np.zeros((L, L), dtype=complex)}

    for i in range(L):
        for j in range(L):
            dm_trace = dm.tocsr().trace()

            Sij["Sxx"][i, j] = (dm @ SX[i] @ SX[j]).tocsr().trace() / dm_trace
            Sij["Sxy"][i, j] = (dm @ SX[i] @ SY[j]).tocsr().trace() / dm_trace
            Sij["Sxz"][i, j] = (dm @ SX[i] @ SZ[j]).tocsr().trace() / dm_trace

            Sij["Syx"][i, j] = (dm @ SY[i] @ SX[j]).tocsr().trace() / dm_trace
            Sij["Syy"][i, j] = (dm @ SY[i] @ SY[j]).tocsr().trace() / dm_trace
            Sij["Syz"][i, j] = (dm @ SY[i] @ SZ[j]).tocsr().trace() / dm_trace

            Sij["Szx"][i, j] = (dm @ SZ[i] @ SX[j]).tocsr().trace() / dm_trace
            Sij["Szy"][i, j] = (dm @ SZ[i] @ SY[j]).tocsr().trace() / dm_trace
            Sij["Szz"][i, j] = (dm @ SZ[i] @ SZ[j]).tocsr().trace() / dm_trace

    return Sij

def generate_random_parallel_ham(delta, L, states, h, J_onsite, J_nn, J_nnn, bc, temp, Sij_init,
                                 state_in, SX, SY, SZ, corr, H_in):
    ranH_J_onsite = delta[0] * (np.random.rand(3, 3) - 0.5)
    ranH_J_nn = delta[1] * (np.random.rand(3, 3) - 0.5)
    ranH_J_nnn = delta[2] * (np.random.rand(3, 3) - 0.5)
    ranH_J_onsite = 1 / 2 * ranH_J_onsite.T @ ranH_J_onsite
    ranH_J_nn = 1 / 2 * ranH_J_nn.T @ ranH_J_nn
    ranH_J_nnn = 1 / 2 * ranH_J_nnn.T @ ranH_J_nnn
    ranH_h = delta[3] * (np.random.rand(3) - 0.5)

    H_plus_delta = calculate_ham_sparse(L, states, h + ranH_h, J_onsite + ranH_J_onsite,
                                        J_nn + ranH_J_nn, J_nnn + ranH_J_nnn, bc=bc)
    if temp == 0:
        _, gs = find_gs_sparse(H_plus_delta)
        state_rand = gs[:,0]
        Sij_rand = calculate_Sij(L, state_rand, SX, SY, SZ)
        S_total = 0
        for corr_i in corr:
            S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
        dist = np.sum(np.abs(state_rand.conj() * state_in)**2)
        energy = (state_rand[np.newaxis].conj() @ H_in @ state_rand[np.newaxis].T)[0,0]
    else:
        state_rand = return_dm_sparse(H_plus_delta, 1/temp)
        Sij_rand = calculate_Sij_dm_sparse(L, state_rand, SX, SY, SZ)
        S_total = 0
        for corr_i in corr:
            S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
        dist = np.abs(np.trace((state_in - state_rand).conj().T @ (state_in - state_rand)))
        energy = (H_in @ state_rand).trace()
    
    dist_ham = sprsla.norm(H_in / H_in.trace() - H_plus_delta / H_plus_delta.trace())

    return {"Sij": S_total, "dist": dist, "energy": energy, "dist_ham": dist_ham}

# DIfferent overload

def generate_random_parallel_ham(params):

    delta = params["delta"]
    L = params["L"]
    states = params["states"]
    h = params["h"]
    J_onsite = params["J_onsite"]
    J_nn = params["J_nn"]
    J_nnn = params["J_nnn"]
    bc = params["bc"]
    temp = params["temp"]
    Sij_init = params["Sij_init"]
    state_in = params["state_in"]
    SX = params["SX"]
    SY = params["SY"]
    SZ = params["SZ"]
    corr = params["corr"]
    H_in = params["H_in"]
    en_in = params["en_in"]

    ranH_J_onsite = delta[0] * (np.random.rand(3, 3) - 0.5)
    ranH_J_nn = delta[1] * (np.random.rand(3, 3) - 0.5)
    ranH_J_nnn = delta[2] * (np.random.rand(3, 3) - 0.5)
    ranH_J_onsite = 1 / 2 * ranH_J_onsite.T @ ranH_J_onsite
    ranH_J_nn = 1 / 2 * ranH_J_nn.T @ ranH_J_nn
    ranH_J_nnn = 1 / 2 * ranH_J_nnn.T @ ranH_J_nnn
    ranH_h = delta[3] * (np.random.rand(3) - 0.5)

    H_plus_delta = calculate_ham_sparse(L, states, h + ranH_h, J_onsite + ranH_J_onsite,
                                        J_nn + ranH_J_nn, J_nnn + ranH_J_nnn, bc=bc)
    if temp == 0:
        en_rand, gs = find_gs_sparse(H_plus_delta)
        state_rand = gs[:,0]
        Sij_rand = calculate_Sij(L, state_rand, SX, SY, SZ)
        S_total = 0
        for corr_i in corr:
            S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
        dist = (np.abs(state_rand[np.newaxis].conj() @ state_in[np.newaxis].T)**2)[0,0]
        dist = 1 - dist
        energy = (state_rand[np.newaxis].conj() @ H_in @ state_rand[np.newaxis].T)[0,0] - en_in
        
    else:
        en_rand, _ = find_gs_sparse(H_plus_delta)
        state_rand = return_dm_sparse(H_plus_delta, 1/temp)
        Sij_rand = calculate_Sij_dm_sparse(L, state_rand, SX, SY, SZ)
        S_total = 0
        for corr_i in corr:
            S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
        dist = np.abs(((state_in - state_rand).conj().T @ (state_in - state_rand)).trace())
        energy = (H_in @ state_in).trace() - (H_in @ state_rand).trace() 

    H_in_m = H_in - en_in[0] * sprs.eye(len(states))
    H_rand_m = H_plus_delta - en_rand[0] * sprs.eye(len(states))
    dist_ham = sprsla.norm(H_in_m / H_in_m.trace() - H_rand_m / H_rand_m.trace())

    return {"Sij": S_total, "dist": dist, "energy": energy, "dist_ham": dist_ham}

def generate_random_parallel_state(delta, L, temp, Sij_init, state_in, SX, SY, SZ, corr, H_in):

    state_no = 2 ** L

    if temp == 0:
        state_rand = state_in + delta * (np.random.rand(state_no) + 1j * np.random.rand(state_no) - 0.5 * (1 + 1j))
        state_rand = state_rand / np.sqrt(np.sum(state_rand.conj() * state_rand))
        Sij_rand = calculate_Sij(L, state_rand, SX, SY, SZ)
        S_total = 0
        for corr_i in corr:
            S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
        dist = np.sum(np.abs(state_rand.conj() * state_in)**2)
        energy = (state_rand[np.newaxis].conj() @ H_in @ state_rand[np.newaxis].T)[0,0] 
    else:
        # Here state_in must be dense
        ranp = delta * (np.random.rand(state_no, state_no) - 0.5 + 1j * np.random.rand(state_no, state_no) -0.5j)
        ranp = ranp.conj().T @ ranp
        state_rand = state_in + ranp
        state_rand = state_rand / np.trace(state_rand)
        Sij_rand = calculate_Sij_dm(L, state_rand, SX, SY, SZ)
        S_total = 0
        for corr_i in corr:
            S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
        dist = np.abs(np.trace((state_in - state_rand).conj().T @ (state_in - state_rand)))
        energy = (H_in @ state_rand).trace()

    return {"Sij": S_total, "dist": dist, "energy": energy}

# Different overload

def generate_random_parallel_state(params):

    delta = params["delta"]
    L = params["L"]
    temp = params["temp"]
    Sij_init = params["Sij_init"]
    state_in = params["state_in"]
    SX = params["SX"]
    SY = params["SY"]
    SZ = params["SZ"]
    corr = params["corr"]
    H_in = params["H_in"]
    en_in = params["en_in"]

    state_no = 2 ** L

    if temp == 0:
        state_rand = state_in + delta * (np.random.rand(state_no) + 1j * np.random.rand(state_no) - 0.5 * (1 + 1j))
        state_rand = state_rand / np.sqrt(np.sum(state_rand.conj() * state_rand))
        Sij_rand = calculate_Sij(L, state_rand, SX, SY, SZ)
        S_total = 0
        for corr_i in corr:
            S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
        #dist = np.sum(np.abs(state_rand.conj() * state_in)**2)
        dist = (np.abs(state_rand[np.newaxis].conj() @ state_in[np.newaxis].T)**2)[0,0]
        dist = 1 - dist
        energy = (state_rand[np.newaxis].conj() @ H_in @ state_rand[np.newaxis].T)[0,0] - en_in
    else:
        # Here state_in must be dense
        ranp = delta * (np.random.rand(state_no, state_no) - 0.5 + 1j * np.random.rand(state_no, state_no) -0.5j)
        ranp = ranp.conj().T @ ranp
        state_rand = state_in + ranp
        state_rand = state_rand / np.trace(state_rand)
        Sij_rand = calculate_Sij_dm(L, state_rand, SX, SY, SZ)
        S_total = 0
        for corr_i in corr:
            S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
        dist = np.abs(((state_in - state_rand).conj().T @ (state_in - state_rand)).trace())
        energy = (H_in @ state_rand).trace()

    return {"Sij": S_total, "dist": dist, "energy": energy}

def generate_random_Sij_sparse(L, h, J_onsite, J_nn, J_nnn, delta, corr, no_of_samples, temp = 0, 
                               rand_ham=True, bc="infinite", parallel=True, no_of_processes=4,
                               rand_delta=True):
    states = create_states(L)
    H_in = calculate_ham_sparse(L, states, h, J_onsite, J_nn, J_nnn, bc)
    SX = create_sx_sparse(states, L)
    SY = create_sy_sparse(states, L)
    SZ = create_sz_sparse(states, L)
    if temp == 0:
        en_in, gs = find_gs_sparse(H_in)
        state_in = gs[:,0]
        Sij_in = calculate_Sij(L, state_in, SX, SY, SZ)
    else:
        en_in, _ = find_gs_sparse(H_in)
        state_in = return_dm_sparse(H_in, 1/temp)
        Sij_in = calculate_Sij_dm_sparse(L, state_in, SX, SY, SZ)

    if parallel:
        if rand_ham:
            if not isinstance(delta, list):
                raise ValueError("If change in Hamiltonian, delta has to be a list of 4 elements")
            elif len(delta) != 4:
                raise ValueError("If change in Hamiltonian, delta has to be a list of 4 elements")
            
            with Pool(processes=no_of_processes) as pool:
                #iter = no_of_samples * [(delta, L, states, h, J_onsite, J_nn, J_nnn, bc, temp, 
                #                         Sij_in, state_in, SX, SY, SZ, corr, H_in)]
                # iter = (no_of_samples * [delta], no_of_samples * [L], no_of_samples * [states], 
                #         no_of_samples * [J_onsite], no_of_samples * [J_nn], no_of_samples * [J_nnn],
                #         no_of_samples * [bc], no_of_samples * [temp], no_of_samples * [Sij_in], 
                #         no_of_samples * [state_in], no_of_samples * [SX], no_of_samples * [SY], 
                #         no_of_samples * [SZ], no_of_samples * [corr], no_of_samples * [H_in])
                params = {"delta": delta, "L": L, "states": states, "h": h, "J_onsite": J_onsite,
                          "J_nn": J_nn, "J_nnn": J_nnn, "bc": bc, "temp": temp, 
                          "Sij_init": Sij_in, "state_in": state_in, "SX": SX, "SY": SY, "SZ": SZ,
                          "corr": corr, "H_in": H_in, "en_in": en_in}
                iter = no_of_samples * [params]
                data = pool.map(generate_random_parallel_ham, iter)
                en = [data[i]["energy"] for i in range(no_of_samples)]
                dist = [data[i]["dist"] for i in range(no_of_samples)]
                diffSij = [data[i]["Sij"] for i in range(no_of_samples)]
                dist_ham = [data[i]["dist_ham"] for i in range(no_of_samples)]

                return en, dist, diffSij, dist_ham

        else:
            with Pool(processes=no_of_processes) as pool:
                # Change dm to dense in this case
                if temp != 0:
                    state_in = state_in.todense()
                if rand_delta:
                    #delta_r = delta * np.random.rand(no_of_samples)
                    #iter = [(delta_r[i], L, temp, Sij_in, state_in, SX, SY, SZ, corr, H_in) for i in range(no_of_samples)]
                    # iter = (delta * np.random.rand(no_of_samples), no_of_samples * [L], no_of_samples * [temp], 
                    #         no_of_samples * [Sij_in], no_of_samples * state_in, no_of_samples * SX,
                    #         no_of_samples * [SY], no_of_samples * [SZ], no_of_samples * [corr],
                    #         no_of_samples * [corr], no_of_samples * [H_in])
                    iter = [{"delta": delta * random(), "L": L, "temp": temp, 
                                "Sij_init": Sij_in, "state_in": state_in, "SX": SX, "SY": SY, "SZ": SZ,
                                "corr": corr, "H_in": H_in, "en_in": en_in} for i in range(no_of_samples)]
                else:
                    #iter = no_of_samples * [(delta, L, temp, Sij_in, state_in, SX, SY, SZ, corr, H_in)]
                    # iter = (no_of_samples * [delta], no_of_samples * [L], no_of_samples * [temp], 
                    #         no_of_samples * [Sij_in], no_of_samples * state_in, no_of_samples * SX,
                    #         no_of_samples * [SY], no_of_samples * [SZ], no_of_samples * [corr],
                    #         no_of_samples * [corr], no_of_samples * [H_in])
                    params = {"delta": delta , "L": L, "temp": temp, 
                                "Sij_init": Sij_in, "state_in": state_in, "SX": SX, "SY": SY, "SZ": SZ,
                                "corr": corr, "H_in": H_in, "en_in": en_in}
                    iter = no_of_samples * [params]
                data = pool.map(generate_random_parallel_state, iter)
                en = [data[i]["energy"] for i in range(no_of_samples)]
                dist = [data[i]["dist"] for i in range(no_of_samples)]
                diffSij = [data[i]["Sij"] for i in range(no_of_samples)]

                return en, dist, diffSij

    else:

        en = []
        dist = []
        diffSij = []

        if rand_ham:
            if not isinstance(delta, list):
                raise ValueError("If change in Hamiltonian, delta has to be a list of 4 elements")
            elif len(delta) != 4:
                raise ValueError("If change in Hamiltonian, delta has to be a list of 4 elements")

            dist_ham = []

            params = {"delta": delta, "L": L, "states": states, "h": h, "J_onsite": J_onsite,
                          "J_nn": J_nn, "J_nnn": J_nnn, "bc": bc, "temp": temp, 
                          "Sij_init": Sij_in, "state_in": state_in, "SX": SX, "SY": SY, "SZ": SZ,
                          "corr": corr, "H_in": H_in, "en_in": en_in}
            
            for i in range(no_of_samples):
                data = generate_random_parallel_ham(params)
                en.append(data["energy"])
                dist.append(data["dist"])
                diffSij.append(data["Sij"])
                dist_ham.append(data["dist_ham"])
            
            return en, dist, diffSij, dist_ham

        else:

            params = {"delta": delta , "L": L, "temp": temp, 
                                "Sij_init": Sij_in, "state_in": state_in, "SX": SX, "SY": SY, "SZ": SZ,
                                "corr": corr, "H_in": H_in}
            if temp != 0:
                state_in = state_in.todense()
            for i in range(no_of_samples):
                if rand_delta:
                    delta_c = delta * random()
                else:
                    delta_c = delta
                data = generate_random_parallel_state(params)
                en.append(data["energy"])
                dist.append(data["dist"])
                diffSij.append(data["Sij"])

            return en, dist, diffSij


    

if __name__ == "__main__":

    # -------------- # Switch # -------------- #
    case = "check_generate_works"
    # ---------------------------------------- #

    if case == "test_sparse":
        L = 8
        states = create_states(L)

        h = [[0, 0, 0]]
        J_onsite = np.zeros((3, 3))
        J_nn = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        J_nnn = np.zeros((3, 3))


        start_ham = time.time()
        ham = calculate_ham_sparse(L, states, h, J_onsite, J_nn, J_nnn, bc="infinite")
        stop_ham = time.time()

        start_find_gs = time.time()
        eigvals, eig = find_gs_sparse(ham)
        stop_find_gs = time.time()

        SX = create_sx_sparse(states, L)
        SY = create_sy_sparse(states, L)
        SZ = create_sz_sparse(states, L)
        
        start_find_dm = time.time()
        dm = return_dm_sparse(ham)
        stop_find_dm = time.time()

        start_gs = time.time()
        Sij = calculate_Sij(L, eig[:,0], SX, SY, SZ)
        stop_gs = time.time()

        start_dm = time.time()
        Sij_dm = calculate_Sij_dm_sparse(L, dm, SX, SY, SZ)
        stop_dm = time.time()

        print("##### Profiling #####")
        print("# Time for calculating hamiltonian: " + str(stop_ham-start_ham))
        print("# Time for calculating gs:          " + str(stop_find_gs-start_find_gs))
        print("# Time for calculating dm:          " + str(stop_find_dm-start_find_dm))
        print("# Time for calculating Sij from gs: " + str(stop_gs-start_gs))
        print("# Time for calculating Sij from dm: " + str(stop_dm-start_dm))
        print("##### ~~~~~~~~~ #####")
        print("# Size of dm:                       " + str(dm.count_nonzero()))
        print("# Maximum size of matrices:         " + str(2 ** (2 * L)))
        print("# Size of dm in bytes vs sparse:    " + str(80 * 2 ** (2 * L)) + " | " + str(dm.count_nonzero() * (32 + 32 + 80)))

    if case == "check_generate_works":

        L = 4
        h = [[0, 0, 0]]
        J_onsite = np.zeros((3, 3))
        J_nn = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        J_nnn = np.zeros((3, 3))
        delta_state = 0.1
        delta_ham = [0, 10, 0, 0]
        corr = ["Sxx", "Syy", "Szz", "Sxy", "Syx", "Sxz", "Szx", "Syz", "Szy"]
        no_of_samples = 2000
        temp = 1
        no_of_processes = 10

        params_ham_par_gs = (L, h, J_onsite, J_nn, J_nnn, delta_ham, corr, no_of_samples, 0,
                             True, "infinite", True, no_of_processes, True)
        params_ham_seq_gs = (L, h, J_onsite, J_nn, J_nnn, delta_ham, corr, no_of_samples, 0,
                             True, "infinite", False, no_of_processes, True)
        params_ham_par_dm = (L, h, J_onsite, J_nn, J_nnn, delta_ham, corr, no_of_samples, temp,
                             True, "infinite", True, no_of_processes, True)
        params_ham_seq_dm = (L, h, J_onsite, J_nn, J_nnn, delta_ham, corr, no_of_samples, temp,
                             True, "infinite", False, no_of_processes, True)

        params_st_par_gs = (L, h, J_onsite, J_nn, J_nnn, delta_state, corr, no_of_samples, 0,
                             False, "infinite", True, no_of_processes, True)
        params_st_seq_gs = (L, h, J_onsite, J_nn, J_nnn, delta_state, corr, no_of_samples, 0,
                             False, "infinite", False, no_of_processes, True)
        params_st_par_dm = (L, h, J_onsite, J_nn, J_nnn, delta_state, corr, no_of_samples, temp,
                             False, "infinite", True, no_of_processes, True)
        params_st_seq_dm = (L, h, J_onsite, J_nn, J_nnn, delta_state, corr, no_of_samples, temp,
                             False, "infinite", False, no_of_processes, True)

        start_ham_par_gs = time.time()
        en_ham_par_gs, dist_ham_par_gs, diffSij_ham_par_gs, dist_ham_ham_par_gs = generate_random_Sij_sparse(*params_ham_par_gs)
        stop_ham_par_gs = time.time()
        #en_ham_seq_gs, dist_ham_seq_gs, diffSij_ham_seq_gs, dist_ham_ham_seq_gs = generate_random_Sij_sparse(*params_ham_seq_gs)
        stop_ham_seq_gs = time.time()
        en_ham_par_dm, dist_ham_par_dm, diffSij_ham_par_dm, dist_ham_ham_par_dm = generate_random_Sij_sparse(*params_ham_par_dm)
        stop_ham_par_dm = time.time()
        #en_ham_seq_dm, dist_ham_seq_dm, diffSij_ham_seq_dm, dist_ham_ham_seq_dm = generate_random_Sij_sparse(*params_ham_seq_dm)
        stop_ham_seq_dm = time.time()

        start_st_par_gs = time.time()
        en_st_par_gs, dist_st_par_gs, diffSij_st_par_gs = generate_random_Sij_sparse(*params_st_par_gs)
        stop_st_par_gs = time.time()
        #en_st_seq_gs, dist_st_seq_gs, diffSij_st_seq_gs = generate_random_Sij_sparse(*params_st_seq_gs)
        stop_st_seq_gs = time.time()
        en_st_par_dm, dist_st_par_dm, diffSij_st_par_dm = generate_random_Sij_sparse(*params_st_par_dm)
        stop_st_par_dm = time.time()
        #en_st_seq_dm, dist_st_seq_dm, diffSij_st_seq_dm = generate_random_Sij_sparse(*params_st_seq_dm)
        stop_st_seq_dm = time.time()

        print("######### Compare times:         ")
        print("# HamParGs - " + str(stop_ham_par_gs - start_ham_par_gs))
        print("# HamSqGs  - " + str(stop_ham_seq_gs - stop_ham_par_gs))
        print("# HamParDm - " + str(stop_ham_par_dm - stop_ham_seq_gs))
        print("# HamSqDm  - " + str(stop_ham_seq_dm - stop_ham_par_dm))
        print("# StParGs  - " + str(stop_st_par_gs - start_st_par_gs))
        print("# StSqGs   - " + str(stop_st_seq_gs - stop_st_par_gs))
        print("# StParDm  - " + str(stop_st_par_dm - stop_st_seq_gs))
        print("# StSqDm   - " + str(stop_st_seq_dm - stop_st_par_dm))
        print("######### Total time: " + str(stop_st_seq_dm - start_ham_par_gs))

        plt.rcParams.update({'font.size': 8})

        fig, axs = plt.subplots(2, 4)

        sizems = 2

        #axs[0,0].set_title("SijVsDist_Ham_GS")
        axs[0,0].scatter(dist_ham_par_gs, diffSij_ham_par_gs, alpha=0.2, s=sizems)
        axs[0,0].set_xlabel(r"$1 - \langle \psi_{rand} | \psi_{init}\rangle$")
        axs[0,0].grid()
        
        #axs[1,0].set_title("SijVsDist_St_GS")
        axs[1,0].scatter(dist_st_par_gs, diffSij_st_par_gs, alpha=0.2, s=sizems)
        axs[1,0].set_xlabel(r"$1 - \langle \psi_{rand} | \psi_{init}\rangle$")
        axs[1,0].grid()

        #axs[0,1].set_title("SijVsDist_Ham_DM")
        axs[0,1].scatter(dist_ham_par_dm, diffSij_ham_par_dm, alpha=0.2, s=sizems)
        axs[0,1].set_xlabel(r"Tr$[(\rho_{rand} - \rho_{init})^\dag (\rho_{rand} - \rho_{init})]$")
        axs[0,1].grid()

        #axs[1,1].set_title("SijVsDist_St_DM")
        axs[1,1].scatter(dist_st_par_dm, diffSij_st_par_dm, alpha=0.2, s=sizems)
        axs[1,1].set_xlabel(r"Tr$[(\rho_{rand} - \rho_{init})^\dag (\rho_{rand} - \rho_{init})]$")
        axs[1,1].grid()

        #axs[0,2].set_title("SijVsDist_Ham_DM")
        axs[0,2].scatter(dist_ham_par_gs, diffSij_ham_par_gs, alpha=0.2, s=sizems)
        axs[0,2].scatter(dist_ham_par_dm, diffSij_ham_par_dm, alpha=0.2, s=sizems)
        axs[0,2].set_xlabel(r"Dist")
        axs[0,2].grid()

        #axs[1,2].set_title("SijVsDist_St_DM")
        axs[1,2].scatter(dist_st_par_gs, diffSij_st_par_gs, alpha=0.2, s=sizems)
        axs[1,2].scatter(dist_st_par_dm, diffSij_st_par_dm, alpha=0.2, s=1)
        axs[1,2].set_xlabel(r"Dist")
        axs[1,2].grid()

        #axs[0,3].set_title("SijVsDisHam_Ham_GS")
        axs[0,3].scatter(dist_ham_ham_par_gs, diffSij_ham_par_gs, alpha=0.2, s=sizems)
        axs[0,3].set_xlabel(r"$||\frac{H_{init}}{Tr H_{init}} - \frac{H_{rand}}{Tr H_{rand}} ||$")
        axs[0,3].grid()

        #axs[1,3].set_title("SijVsDisHam_Ham_DM")
        axs[1,3].scatter(dist_ham_ham_par_dm, diffSij_ham_par_dm, alpha=0.2, s=sizems)
        axs[1,3].set_xlabel(r"$||\frac{H_{init}}{Tr H_{init}} - \frac{H_{rand}}{Tr H_{rand}} ||$")
        axs[1,3].grid()

        for i in range(2):
            for j in range(1):
                axs[i,j].set_ylabel(r"$\sum_{i,j}\sum_{\alpha, \beta} |C_{rand}^{\alpha, \beta}(i,j) - C_{init}^{\alpha, \beta}(i,j)|^2 $")

        plt.show()

        



