# tools.py

import numpy as np
import scipy.sparse.linalg as sprsla
from math import pi

class SijCalculator:
    r"""Prototype of class designed to return Sij correlation functions"""

    round_to = 10

    def __init__(self):
        pass

    def return_Sij(L, dm, SX, SY, SZ, temp):
        if temp == 0:
            gs2 = dm[np.newaxis]

            Sij = {"Sxx": np.zeros((L, L), dtype=complex), "Sxy": np.zeros((L, L), dtype=complex), "Sxz": np.zeros((L, L), dtype=complex),
                "Syx": np.zeros((L, L), dtype=complex), "Syy": np.zeros((L, L), dtype=complex), "Syz": np.zeros((L, L), dtype=complex),
                "Szx": np.zeros((L, L), dtype=complex), "Szy": np.zeros((L, L), dtype=complex), "Szz": np.zeros((L, L), dtype=complex)}

            for i in range(L):
                for j in range(L):
                    Sij["Sxx"][i, j] = (gs2.conj() @ SX[i] @ SX[j] @ gs2.T)[0,0]
                    Sij["Sxy"][i, j] = (gs2.conj() @ SX[i] @ SY[j] @ gs2.T)[0,0]
                    Sij["Sxz"][i, j] = (gs2.conj() @ SX[i] @ SZ[j] @ gs2.T)[0,0]

                    Sij["Syx"][i, j] = (gs2.conj() @ SY[i] @ SX[j] @ gs2.T)[0,0]
                    Sij["Syy"][i, j] = (gs2.conj() @ SY[i] @ SY[j] @ gs2.T)[0,0]
                    Sij["Syz"][i, j] = (gs2.conj() @ SY[i] @ SZ[j] @ gs2.T)[0,0]

                    Sij["Szx"][i, j] = (gs2.conj() @ SZ[i] @ SX[j] @ gs2.T)[0,0]
                    Sij["Szy"][i, j] = (gs2.conj() @ SZ[i] @ SY[j] @ gs2.T)[0,0]
                    Sij["Szz"][i, j] = (gs2.conj() @ SZ[i] @ SZ[j] @ gs2.T)[0,0]

            return Sij
        else:
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

    def return_SijSq(L, dm, SX, SY, SZ, temp):
        if temp == 0:
            gs2 = dm[np.newaxis]

            Sij = {"Sxx": np.zeros((L, L), dtype=complex), "Sxy": np.zeros((L, L), dtype=complex), "Sxz": np.zeros((L, L), dtype=complex),
                "Syx": np.zeros((L, L), dtype=complex), "Syy": np.zeros((L, L), dtype=complex), "Syz": np.zeros((L, L), dtype=complex),
                "Szx": np.zeros((L, L), dtype=complex), "Szy": np.zeros((L, L), dtype=complex), "Szz": np.zeros((L, L), dtype=complex)}

            Sq = {}

            for i in range(L):
                for j in range(L):
                    Sij["Sxx"][i, j] = (gs2.conj() @ SX[i] @ SX[j] @ gs2.T)[0,0]
                    Sij["Sxy"][i, j] = (gs2.conj() @ SX[i] @ SY[j] @ gs2.T)[0,0]
                    Sij["Sxz"][i, j] = (gs2.conj() @ SX[i] @ SZ[j] @ gs2.T)[0,0]

                    Sij["Syx"][i, j] = (gs2.conj() @ SY[i] @ SX[j] @ gs2.T)[0,0]
                    Sij["Syy"][i, j] = (gs2.conj() @ SY[i] @ SY[j] @ gs2.T)[0,0]
                    Sij["Syz"][i, j] = (gs2.conj() @ SY[i] @ SZ[j] @ gs2.T)[0,0]

                    Sij["Szx"][i, j] = (gs2.conj() @ SZ[i] @ SX[j] @ gs2.T)[0,0]
                    Sij["Szy"][i, j] = (gs2.conj() @ SZ[i] @ SY[j] @ gs2.T)[0,0]
                    Sij["Szz"][i, j] = (gs2.conj() @ SZ[i] @ SZ[j] @ gs2.T)[0,0]

                if i == 0:
                    k = np.arange(0, 2*pi, 2*pi/L)
                    j = np.arange(0, L)
                    kj = k[np.newaxis].T @ j[np.newaxis]
                    Sq["Sxx"] = (np.exp(1j * kj) @ Sij["Sxx"][0,:][np.newaxis].T)[:,0]
                    Sq["Sxy"] = (np.exp(1j * kj) @ Sij["Sxy"][0,:][np.newaxis].T)[:,0]
                    Sq["Sxz"] = (np.exp(1j * kj) @ Sij["Sxz"][0,:][np.newaxis].T)[:,0]

                    Sq["Syx"] = (np.exp(1j * kj) @ Sij["Syx"][0,:][np.newaxis].T)[:,0]
                    Sq["Syy"] = (np.exp(1j * kj) @ Sij["Syy"][0,:][np.newaxis].T)[:,0]
                    Sq["Syz"] = (np.exp(1j * kj) @ Sij["Syz"][0,:][np.newaxis].T)[:,0]

                    Sq["Szx"] = (np.exp(1j * kj) @ Sij["Szx"][0,:][np.newaxis].T)[:,0]
                    Sq["Szy"] = (np.exp(1j * kj) @ Sij["Szy"][0,:][np.newaxis].T)[:,0]
                    Sq["Szz"] = (np.exp(1j * kj) @ Sij["Szz"][0,:][np.newaxis].T)[:,0]

            return Sij, Sq
        else:
            Sij = {"Sxx": np.zeros((L, L), dtype=complex), "Sxy": np.zeros((L, L), dtype=complex), "Sxz": np.zeros((L, L), dtype=complex),
            "Syx": np.zeros((L, L), dtype=complex), "Syy": np.zeros((L, L), dtype=complex), "Syz": np.zeros((L, L), dtype=complex),
            "Szx": np.zeros((L, L), dtype=complex), "Szy": np.zeros((L, L), dtype=complex), "Szz": np.zeros((L, L), dtype=complex)}

            Sq = {}

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

                if i == 0:
                    k = np.arange(0, 2*pi, 2*pi/L)
                    j = np.arange(0, L)
                    kj = k[np.newaxis].T @ j[np.newaxis]
                    Sq["Sxx"] = (np.exp(1j * kj) @ Sij["Sxx"][0,:][np.newaxis].T)[:,0]
                    Sq["Sxy"] = (np.exp(1j * kj) @ Sij["Sxy"][0,:][np.newaxis].T)[:,0]
                    Sq["Sxz"] = (np.exp(1j * kj) @ Sij["Sxz"][0,:][np.newaxis].T)[:,0]

                    Sq["Syx"] = (np.exp(1j * kj) @ Sij["Syx"][0,:][np.newaxis].T)[:,0]
                    Sq["Syy"] = (np.exp(1j * kj) @ Sij["Syy"][0,:][np.newaxis].T)[:,0]
                    Sq["Syz"] = (np.exp(1j * kj) @ Sij["Syz"][0,:][np.newaxis].T)[:,0]

                    Sq["Szx"] = (np.exp(1j * kj) @ Sij["Szx"][0,:][np.newaxis].T)[:,0]
                    Sq["Szy"] = (np.exp(1j * kj) @ Sij["Szy"][0,:][np.newaxis].T)[:,0]
                    Sq["Szz"] = (np.exp(1j * kj) @ Sij["Szz"][0,:][np.newaxis].T)[:,0]

            return Sij, Sq


    def return_Sij_dm_not_sparse(L, dm, SX, SY, SZ):

        Sij = {"Sxx": np.zeros((L, L), dtype=complex), "Sxy": np.zeros((L, L), dtype=complex), "Sxz": np.zeros((L, L), dtype=complex),
        "Syx": np.zeros((L, L), dtype=complex), "Syy": np.zeros((L, L), dtype=complex), "Syz": np.zeros((L, L), dtype=complex),
        "Szx": np.zeros((L, L), dtype=complex), "Szy": np.zeros((L, L), dtype=complex), "Szz": np.zeros((L, L), dtype=complex)}

        for i in range(L):
            for j in range(L):
                dm_trace = np.trace(dm)

                Sij["Sxx"][i, j] = np.trace(dm @ SX[i] @ SX[j])/ dm_trace
                Sij["Sxy"][i, j] = np.trace(dm @ SX[i] @ SY[j]) / dm_trace
                Sij["Sxz"][i, j] = np.trace(dm @ SX[i] @ SZ[j])/ dm_trace

                Sij["Syx"][i, j] = np.trace(dm @ SY[i] @ SX[j])/ dm_trace
                Sij["Syy"][i, j] = np.trace(dm @ SY[i] @ SY[j]) / dm_trace
                Sij["Syz"][i, j] = np.trace(dm @ SY[i] @ SZ[j]) / dm_trace

                Sij["Szx"][i, j] = np.trace(dm @ SZ[i] @ SX[j])/ dm_trace
                Sij["Szy"][i, j] = np.trace(dm @ SZ[i] @ SY[j])/ dm_trace
                Sij["Szz"][i, j] = np.trace(dm @ SZ[i] @ SZ[j])/ dm_trace

        return Sij

    def return_SijSq_dm_not_sparse(L, dm, SX, SY, SZ):

        Sij = {"Sxx": np.zeros((L, L), dtype=complex), "Sxy": np.zeros((L, L), dtype=complex), "Sxz": np.zeros((L, L), dtype=complex),
        "Syx": np.zeros((L, L), dtype=complex), "Syy": np.zeros((L, L), dtype=complex), "Syz": np.zeros((L, L), dtype=complex),
        "Szx": np.zeros((L, L), dtype=complex), "Szy": np.zeros((L, L), dtype=complex), "Szz": np.zeros((L, L), dtype=complex)}
        
        Sq = {}

        for i in range(L):
            for j in range(L):
                dm_trace = np.trace(dm)

                Sij["Sxx"][i, j] = np.trace(dm @ SX[i] @ SX[j])/ dm_trace
                Sij["Sxy"][i, j] = np.trace(dm @ SX[i] @ SY[j]) / dm_trace
                Sij["Sxz"][i, j] = np.trace(dm @ SX[i] @ SZ[j])/ dm_trace

                Sij["Syx"][i, j] = np.trace(dm @ SY[i] @ SX[j])/ dm_trace
                Sij["Syy"][i, j] = np.trace(dm @ SY[i] @ SY[j]) / dm_trace
                Sij["Syz"][i, j] = np.trace(dm @ SY[i] @ SZ[j]) / dm_trace

                Sij["Szx"][i, j] = np.trace(dm @ SZ[i] @ SX[j])/ dm_trace
                Sij["Szy"][i, j] = np.trace(dm @ SZ[i] @ SY[j])/ dm_trace
                Sij["Szz"][i, j] = np.trace(dm @ SZ[i] @ SZ[j])/ dm_trace

            if i == 0:
                    k = np.arange(0, 2*pi, 2*pi/L)
                    j = np.arange(0, L)
                    kj = k[np.newaxis].T @ j[np.newaxis]
                    Sq["Sxx"] = (np.exp(1j * kj) @ Sij["Sxx"][0,:][np.newaxis].T)[:,0]
                    Sq["Sxy"] = (np.exp(1j * kj) @ Sij["Sxy"][0,:][np.newaxis].T)[:,0]
                    Sq["Sxz"] = (np.exp(1j * kj) @ Sij["Sxz"][0,:][np.newaxis].T)[:,0]

                    Sq["Syx"] = (np.exp(1j * kj) @ Sij["Syx"][0,:][np.newaxis].T)[:,0]
                    Sq["Syy"] = (np.exp(1j * kj) @ Sij["Syy"][0,:][np.newaxis].T)[:,0]
                    Sq["Syz"] = (np.exp(1j * kj) @ Sij["Syz"][0,:][np.newaxis].T)[:,0]

                    Sq["Szx"] = (np.exp(1j * kj) @ Sij["Szx"][0,:][np.newaxis].T)[:,0]
                    Sq["Szy"] = (np.exp(1j * kj) @ Sij["Szy"][0,:][np.newaxis].T)[:,0]
                    Sq["Szz"] = (np.exp(1j * kj) @ Sij["Szz"][0,:][np.newaxis].T)[:,0]

        return Sij, Sq
        
    def find_gs_sparse(ham):
        return sprsla.eigsh(ham, 1)

    def find_eigvals(ham):
        return sprsla.eigsh(ham, return_eigenvectors=False)

    def find_gap(eigvals):
        round_to = SijCalculator.round_to
        gs_energy = np.inf
        fes_energy = np.inf
        for ev in eigvals:
            ev_round = round(ev, round_to)
            if ev_round < gs_energy:
                fes_energy = gs_energy
                gs_energy = ev_round
            elif ev_round < fes_energy and ev_round != gs_energy:
                fes_energy = ev_round
        return fes_energy - gs_energy
                
    def find_bandwidth(eigvals):
        gs_energy = np.min(eigvals)
        les_energy = np.max(eigvals)
        return les_energy - gs_energy

    def return_dm_sparse(ham, beta=1):
        re = sprsla.expm(-beta * ham.tocsc())
        return re / re.trace()