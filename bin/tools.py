# tools.py

import numpy as np
import scipy.sparse.linalg as sprsla
import scipy.linalg as spla
from math import pi, nan, sqrt, cos, sin, nan

class SijCalculator:
    r"""Prototype of class designed to return Sij correlation functions"""

    round_to = 10

    def __init__(self):
        pass

    def qhat(q1, q2):
		""" calculate the norm vector along the direction of the given x and y components"""
        if (q1==0 and q2==0):
            return (0,0)
        else:
            invnorm=1.0/sqrt(q1**2 + q2**2)
            qxhat = q1*invnorm
            qyhat = q2*invnorm
        return (qxhat, qyhat)

    def calculate_Sq(L, Sij, no_ofqpoints=100):
        Sq = {}
        ip = np.arange(L)
        jp = np.arange(L)

        ip = np.repeat(ip[np.newaxis], L, axis=0)
        jp = np.repeat(jp[np.newaxis].T, L, axis=1)

        imj = np.repeat((ip-jp)[np.newaxis], no_ofqpoints, axis=0)
        k_tile = np.transpose(np.tile(np.linspace(0, 2*pi, no_ofqpoints), (L, L, 1)), (2, 0, 1))

        exp_fac = np.exp(1j * imj * k_tile)

        Sq["Sxx"] = np.tensordot(exp_fac, Sij["Sxx"]) / L
        Sq["Sxy"] = np.tensordot(exp_fac, Sij["Sxy"]) / L
        Sq["Sxz"] = np.tensordot(exp_fac, Sij["Sxz"]) / L

        Sq["Syx"] = np.tensordot(exp_fac, Sij["Syx"]) / L
        Sq["Syy"] = np.tensordot(exp_fac, Sij["Syy"]) / L
        Sq["Syz"] = np.tensordot(exp_fac, Sij["Syz"]) / L

        Sq["Szx"] = np.tensordot(exp_fac, Sij["Szx"]) / L
        Sq["Szy"] = np.tensordot(exp_fac, Sij["Szy"]) / L
        Sq["Szz"] = np.tensordot(exp_fac, Sij["Szz"]) / L

        return Sq

    def get_radius(L, a):
        theta_n = 2*pi/L
        return a / sqrt(2 * (1-cos(theta_n)))

    def returnLambdaMatrices(L):

        theta_diff = 2 * pi /L

        thetas = np.arange(theta_diff/2, 2*pi+theta_diff/2, 2*pi/L)
        Lambdas = [[[cos(thetas[i]), sin(thetas[i]), 0],
                    [-sin(thetas[i]), cos(thetas[i]), 0],
                    [0, 0, 1]] for i in range(L)]

        return Lambdas

    def calculate_Sq_2d(L, Sij, no_ofqpoints=100):
        Sq = {}
        Rs = []
        a = 1
        p = SijCalculator.get_radius(L, a)

        theta_diff = 2 * pi/L

        theta_range = np.arange(theta_diff/2, 2*pi+theta_diff/2, theta_diff)

        Rs = np.array([p * np.array([cos(theta_range[i]), sin(theta_range[i])]) for i in range(L)])

        Rxs = Rs[:,0]
        Rys = Rs[:,1]

        kx = np.linspace(-2*pi, 2*pi, no_ofqpoints)
        ky = np.linspace(-2*pi, 2*pi, no_ofqpoints)

        Rxs_diff = np.repeat(Rxs[np.newaxis], L, axis=0)
        Rxs_diff = Rxs_diff - Rxs_diff.T
        Rys_diff = np.repeat(Rys[np.newaxis], L, axis=0)
        Rys_diff = Rys_diff - Rys_diff.T

        imj_x = np.repeat(Rxs_diff[np.newaxis], no_ofqpoints, axis=0)
        imj_y = np.repeat(Rys_diff[np.newaxis], no_ofqpoints, axis=0)

        k_tile_x = np.transpose(np.tile(kx, (L, L, 1)), (2, 0, 1))
        k_tile_y = np.transpose(np.tile(ky, (L, L, 1)), (2, 0, 1))

        exp_fac_x = np.exp(1j * imj_x * k_tile_x)
        exp_fac_x = np.transpose(np.repeat(exp_fac_x[np.newaxis], no_ofqpoints, axis=0), (1, 0, 2, 3))

        exp_fac_y = np.exp(1j * imj_y * k_tile_y)
        exp_fac_y = np.repeat(exp_fac_y[np.newaxis], no_ofqpoints, axis=0)

        exp_fac = exp_fac_x * exp_fac_y

        # Transform Sij into true values

        Lambdas = np.array(SijCalculator.returnLambdaMatrices(L))

        Sij_trans = np.array([[[[Sij["Sxx"][i][j], Sij["Sxy"][i][j], Sij["Sxz"][i][j]],
                                [Sij["Syx"][i][j], Sij["Syy"][i][j], Sij["Syz"][i][j]],
                                [Sij["Szx"][i][j], Sij["Szy"][i][j], Sij["Szz"][i][j]]]
                                for i in range(L)] for j in range(L)])

        Sij_trans = [[Lambdas[i] @ Sij_trans[i][j] @ Lambdas[j].T for i in range(L)] for j in range(L)]

        Sij_new = {"Sxx": [[Sij_trans[i][j][0][0] for i in range(L)] for j in range(L)], 
                   "Sxy": [[Sij_trans[i][j][0][1] for i in range(L)] for j in range(L)],
                   "Sxz": [[Sij_trans[i][j][0][2] for i in range(L)] for j in range(L)],
                   "Syx": [[Sij_trans[i][j][1][0] for i in range(L)] for j in range(L)],
                   "Syy": [[Sij_trans[i][j][1][1] for i in range(L)] for j in range(L)],
                   "Syz": [[Sij_trans[i][j][1][2] for i in range(L)] for j in range(L)],
                   "Szx": [[Sij_trans[i][j][2][0] for i in range(L)] for j in range(L)], 
                   "Szy": [[Sij_trans[i][j][2][1] for i in range(L)] for j in range(L)],
                   "Szz": [[Sij_trans[i][j][2][2] for i in range(L)] for j in range(L)]}

        # Sq["Sxx"] = np.tensordot(exp_fac, Sij_new["Sxx"]) / L
        # Sq["Sxy"] = np.tensordot(exp_fac, Sij_new["Sxy"]) / L
        # Sq["Sxz"] = np.tensordot(exp_fac, Sij_new["Sxz"]) / L

        # Sq["Syx"] = np.tensordot(exp_fac, Sij_new["Syx"]) / L
        # Sq["Syy"] = np.tensordot(exp_fac, Sij_new["Syy"]) / L
        # Sq["Syz"] = np.tensordot(exp_fac, Sij_new["Syz"]) / L

        # Sq["Szx"] = np.tensordot(exp_fac, Sij_new["Szx"]) / L
        # Sq["Szy"] = np.tensordot(exp_fac, Sij_new["Szy"]) / L
        # Sq["Szz"] = np.tensordot(exp_fac, Sij_new["Szz"]) / L

        Sq["Sxx"] = np.tensordot(exp_fac, Sij_new["Sxx"]) / L
        Sq["Sxy"] = np.tensordot(exp_fac, Sij_new["Sxy"]) / L
        Sq["Sxz"] = np.tensordot(exp_fac, Sij_new["Sxz"]) / L

        Sq["Syx"] = np.tensordot(exp_fac, Sij_new["Syx"]) / L
        Sq["Syy"] = np.tensordot(exp_fac, Sij_new["Syy"]) / L
        Sq["Syz"] = np.tensordot(exp_fac, Sij_new["Syz"]) / L

        Sq["Szx"] = np.tensordot(exp_fac, Sij_new["Szx"]) / L
        Sq["Szy"] = np.tensordot(exp_fac, Sij_new["Szy"]) / L
        Sq["Szz"] = np.tensordot(exp_fac, Sij_new["Szz"]) / L

        return Sq

    def calculate_exp_fac(L, no_ofqpoints):
        Rs = []
        a = 1
        p = SijCalculator.get_radius(L, a)
        
        # positions of atoms in the molecule:
        # differences between angles
        theta_diff = 2 * pi/L

        theta_range = np.arange(theta_diff/2, 2*pi+theta_diff/2, theta_diff)
		# actual positions
        Rs = np.array([p * np.array([cos(theta_range[i]), sin(theta_range[i])]) for i in range(L)])

        Rxs = Rs[:,0]
        Rys = Rs[:,1]
		
		# choose some range of momenta to look at (in general we could include some multiplier for overall range):
        kx = np.linspace(-2*pi, 2*pi, no_ofqpoints)
        ky = np.linspace(-2*pi, 2*pi, no_ofqpoints)

		# make vector of pairwise differences in x and y-positions:
        Rxs_diff = np.repeat(Rxs[np.newaxis], L, axis=0)
        Rxs_diff = Rxs_diff - Rxs_diff.T
        Rys_diff = np.repeat(Rys[np.newaxis], L, axis=0)
        Rys_diff = Rys_diff - Rys_diff.T

		
        imj_x = np.repeat(Rxs_diff[np.newaxis], no_ofqpoints, axis=0)
        imj_y = np.repeat(Rys_diff[np.newaxis], no_ofqpoints, axis=0)

        k_tile_x = np.transpose(np.tile(kx, (L, L, 1)), (2, 0, 1))
        k_tile_y = np.transpose(np.tile(ky, (L, L, 1)), (2, 0, 1))

        exp_fac_x = np.exp(1j * imj_x * k_tile_x)
        exp_fac_x = np.transpose(np.repeat(exp_fac_x[np.newaxis], no_ofqpoints, axis=0), (1, 0, 2, 3))

        exp_fac_y = np.exp(1j * imj_y * k_tile_y)
        exp_fac_y = np.repeat(exp_fac_y[np.newaxis], no_ofqpoints, axis=0)

        exp_fac = exp_fac_x * exp_fac_y

        return exp_fac

    def calculate_Lambdas(L):
        
        Lambdas = np.array(SijCalculator.returnLambdaMatrices(L))

        return Lambdas

    def calculate_Sq_2d_with_int(L, Sij, no_ofqpoints, exp_fac, Lambdas):

        Sq = {}

        Sij_trans = np.array([[[[Sij["Sxx"][i][j], Sij["Sxy"][i][j], Sij["Sxz"][i][j]],
                                [Sij["Syx"][i][j], Sij["Syy"][i][j], Sij["Syz"][i][j]],
                                [Sij["Szx"][i][j], Sij["Szy"][i][j], Sij["Szz"][i][j]]]
                                for i in range(L)] for j in range(L)])

        Sij_trans = [[Lambdas[i] @ Sij_trans[i][j] @ Lambdas[j].T for i in range(L)] for j in range(L)]

        Sij_new = {"Sxx": [[Sij_trans[i][j][0][0] for i in range(L)] for j in range(L)], 
                   "Sxy": [[Sij_trans[i][j][0][1] for i in range(L)] for j in range(L)],
                   "Sxz": [[Sij_trans[i][j][0][2] for i in range(L)] for j in range(L)],
                   "Syx": [[Sij_trans[i][j][1][0] for i in range(L)] for j in range(L)],
                   "Syy": [[Sij_trans[i][j][1][1] for i in range(L)] for j in range(L)],
                   "Syz": [[Sij_trans[i][j][1][2] for i in range(L)] for j in range(L)],
                   "Szx": [[Sij_trans[i][j][2][0] for i in range(L)] for j in range(L)], 
                   "Szy": [[Sij_trans[i][j][2][1] for i in range(L)] for j in range(L)],
                   "Szz": [[Sij_trans[i][j][2][2] for i in range(L)] for j in range(L)]}

        # Sq["Sxx"] = np.tensordot(exp_fac, Sij_new["Sxx"]) / L
        # Sq["Sxy"] = np.tensordot(exp_fac, Sij_new["Sxy"]) / L
        # Sq["Sxz"] = np.tensordot(exp_fac, Sij_new["Sxz"]) / L

        # Sq["Syx"] = np.tensordot(exp_fac, Sij_new["Syx"]) / L
        # Sq["Syy"] = np.tensordot(exp_fac, Sij_new["Syy"]) / L
        # Sq["Syz"] = np.tensordot(exp_fac, Sij_new["Syz"]) / L

        # Sq["Szx"] = np.tensordot(exp_fac, Sij_new["Szx"]) / L
        # Sq["Szy"] = np.tensordot(exp_fac, Sij_new["Szy"]) / L
        # Sq["Szz"] = np.tensordot(exp_fac, Sij_new["Szz"]) / L

        Sq["Sxx"] = np.tensordot(exp_fac, Sij_new["Sxx"]) / L
        Sq["Sxy"] = np.tensordot(exp_fac, Sij_new["Sxy"]) / L
        Sq["Sxz"] = np.tensordot(exp_fac, Sij_new["Sxz"]) / L

        Sq["Syx"] = np.tensordot(exp_fac, Sij_new["Syx"]) / L
        Sq["Syy"] = np.tensordot(exp_fac, Sij_new["Syy"]) / L
        Sq["Syz"] = np.tensordot(exp_fac, Sij_new["Syz"]) / L

        Sq["Szx"] = np.tensordot(exp_fac, Sij_new["Szx"]) / L
        Sq["Szy"] = np.tensordot(exp_fac, Sij_new["Szy"]) / L
        Sq["Szz"] = np.tensordot(exp_fac, Sij_new["Szz"]) / L

        qx = np.linspace(-2*pi, 2*pi, no_ofqpoints)
        qy = np.linspace(-2*pi, 2*pi, no_ofqpoints)

        Sq_proper = {"Sxx": np.zeros((qx.size, qy.size), dtype=complex), 
                     "Syy": np.zeros((qx.size, qy.size), dtype=complex),
                     "Sxy": np.zeros((qx.size, qy.size), dtype=complex),
                     "Syx": np.zeros((qx.size, qy.size), dtype=complex)}

        for i, qx_i in enumerate(qx):
            for j, qy_j in enumerate(qy):
                (qxhat, qyhat) = SijCalculator.qhat(qx_i, qy_j)
                Sq_proper["Sxx"][i][j] = (1-qxhat**2) * Sq["Sxx"][i][j]
                Sq_proper["Syy"][i][j] = (1-qyhat**2) * Sq["Syy"][i][j]
                Sq_proper["Sxy"][i][j] = -qxhat * qyhat * Sq["Sxy"][i][j]
                Sq_proper["Syx"][i][j] = -qxhat * qyhat * Sq["Syx"][i][j]

        Sq_int = np.real(Sq_proper["Sxx"] + Sq_proper["Syy"] + Sq["Szz"] + 
                        Sq_proper["Sxy"] + Sq_proper["Syx"])

        return Sq, Sq_int


    def transform_correlations(L, i, j, Sij, Rs, thetas):

        Sij_mat = np.array([[Sij["Sxx"][i][j], Sij["Sxy"][i][j], Sij["Sxz"][i][j]],
                            [Sij["Syx"][i][j], Sij["Syy"][i][j], Sij["Syz"][i][j]],
                            [Sij["Szx"][i][j], Sij["Szy"][i][j], Sij["Szz"][i][j]]])

        Lambda_alpha = np.array([[cos(thetas[i]), -sin(thetas[i]), 0],
                                 [sin(thetas[i]), cos(thetas[i]), 0],
                                 [0, 0, 1]])

        Lambda_beta = np.array([[cos(thetas[j]), -sin(thetas[j]), 0],
                                 [sin(thetas[j]), cos(thetas[j]), 0],
                                 [0, 0, 1]])

        Sij_new = Lambda_alpha @ Sij_mat @ Lambda_beta

        return Sij_new

    def calculate_Sq_2d_simple(L, Sij, no_ofqpoints=100):

        Sq = {}
        Sq["Sxx"] = np.zeros((no_ofqpoints, no_ofqpoints))
        Sq["Syy"] = np.zeros((no_ofqpoints, no_ofqpoints))
        Sq["Szz"] = np.zeros((no_ofqpoints, no_ofqpoints))
        Sq["Sxy"] = np.zeros((no_ofqpoints, no_ofqpoints))
        Sq["Syx"] = np.zeros((no_ofqpoints, no_ofqpoints))

        # Function that given i,j L and Sij, transforms it into relevant correlations
        
        a = 1
        p = SijCalculator.get_radius(L, a)

        theta_range = np.arange(0, 2*pi, 2*pi/L)

        Rs = np.array([p * np.array([cos(theta_range[i]), sin(theta_range[i])]) for i in range(L)])

        Qx = np.linspace(-2*pi, 2*pi, no_ofqpoints)
        Qy = np.linspace(-2*pi, 2*pi, no_ofqpoints)

        for qx in range(no_ofqpoints):
            for qy in range(no_ofqpoints):
                Sqxx = 0
                Sqyy = 0
                Sqzz = 0
                Sqxy = 0
                Sqyx = 0

                for delta in range(L):
                    for deltap in range(L):
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

    def return_Sq2(L, dm, SX, SY, SZ, temp, no_ofqpoints, exp_fac, Lambdas):
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

                    
            Sq, Sq_int = SijCalculator.calculate_Sq_2d_with_int(L, Sij, no_ofqpoints, exp_fac, Lambdas)

            return Sij, Sq, Sq_int
        else:
            Sij = {"Sxx": np.zeros((L, L), dtype=complex), "Sxy": np.zeros((L, L), dtype=complex), "Sxz": np.zeros((L, L), dtype=complex),
            "Syx": np.zeros((L, L), dtype=complex), "Syy": np.zeros((L, L), dtype=complex), "Syz": np.zeros((L, L), dtype=complex),
            "Szx": np.zeros((L, L), dtype=complex), "Szy": np.zeros((L, L), dtype=complex), "Szz": np.zeros((L, L), dtype=complex)}

            Sq = {}

            for i in range(L):
                for j in range(L):
                    dm_trace = dm.tocsr().trace()

                    try:
                        Sij["Sxx"][i, j] = (dm @ SX[i] @ SX[j]).tocsr().trace() / dm_trace
                        Sij["Sxy"][i, j] = (dm @ SX[i] @ SY[j]).tocsr().trace() / dm_trace
                        Sij["Sxz"][i, j] = (dm @ SX[i] @ SZ[j]).tocsr().trace() / dm_trace

                        Sij["Syx"][i, j] = (dm @ SY[i] @ SX[j]).tocsr().trace() / dm_trace
                        Sij["Syy"][i, j] = (dm @ SY[i] @ SY[j]).tocsr().trace() / dm_trace
                        Sij["Syz"][i, j] = (dm @ SY[i] @ SZ[j]).tocsr().trace() / dm_trace

                        Sij["Szx"][i, j] = (dm @ SZ[i] @ SX[j]).tocsr().trace() / dm_trace
                        Sij["Szy"][i, j] = (dm @ SZ[i] @ SY[j]).tocsr().trace() / dm_trace
                        Sij["Szz"][i, j] = (dm @ SZ[i] @ SZ[j]).tocsr().trace() / dm_trace
                    except:
                        Sij["Sxx"][i, j] = nan
                        Sij["Sxy"][i, j] = nan
                        Sij["Sxz"][i, j] = nan

                        Sij["Syx"][i, j] = nan
                        Sij["Syy"][i, j] = nan
                        Sij["Syz"][i, j] = nan

                        Sij["Szx"][i, j] = nan
                        Sij["Szy"][i, j] = nan
                        Sij["Szz"][i, j] = nan

            #Sq = SijCalculator.calculate_Sq_2d(L, Sij)

            Sq, Sq_int = SijCalculator.calculate_Sq_2d_with_int(L, Sij, no_ofqpoints, exp_fac, Lambdas)

            return Sij, Sq, Sq_int

    def returnSq2_dm_not_sparse(L, dm, SX, SY, SZ, no_ofqpoints, exp_fac, Lambdas):
        Sij = {"Sxx": np.zeros((L, L), dtype=complex), "Sxy": np.zeros((L, L), dtype=complex), "Sxz": np.zeros((L, L), dtype=complex),
        "Syx": np.zeros((L, L), dtype=complex), "Syy": np.zeros((L, L), dtype=complex), "Syz": np.zeros((L, L), dtype=complex),
        "Szx": np.zeros((L, L), dtype=complex), "Szy": np.zeros((L, L), dtype=complex), "Szz": np.zeros((L, L), dtype=complex)}
        
        Sq = {}

        dm_trace = np.trace(dm)
        if dm_trace == (nan + 1j * nan):
            print("dm_trace not correct")

        for i in range(L):
            for j in range(L):
                
                Sij["Sxx"][i, j] = np.trace(dm @ SX[i] @ SX[j])/ dm_trace
                Sij["Sxy"][i, j] = np.trace(dm @ SX[i] @ SY[j]) / dm_trace
                Sij["Sxz"][i, j] = np.trace(dm @ SX[i] @ SZ[j])/ dm_trace

                Sij["Syx"][i, j] = np.trace(dm @ SY[i] @ SX[j])/ dm_trace
                Sij["Syy"][i, j] = np.trace(dm @ SY[i] @ SY[j]) / dm_trace
                Sij["Syz"][i, j] = np.trace(dm @ SY[i] @ SZ[j]) / dm_trace

                Sij["Szx"][i, j] = np.trace(dm @ SZ[i] @ SX[j])/ dm_trace
                Sij["Szy"][i, j] = np.trace(dm @ SZ[i] @ SY[j])/ dm_trace
                Sij["Szz"][i, j] = np.trace(dm @ SZ[i] @ SZ[j])/ dm_trace


        Sq, Sq_int = SijCalculator.calculate_Sq_2d_with_int(L, Sij, no_ofqpoints, exp_fac, Lambdas)

        return Sij, Sq, Sq_int

        
    def find_gs_sparse(ham):
        return sprsla.eigsh(ham, 1, which="SA")

    def find_eigvals(ham):
        return np.linalg.eigvals(ham)

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

    # def return_dm_not_sparse(ham, beta=1):
    #     re = spla.expm(-beta * ham.todense())
    #     return re / re.trace()

    def return_dm_not_sparse(ham, beta=1):
        ham_dense = ham.todense()
        eigvals, eigvecs = np.linalg.eigh(-beta * ham_dense)
        # trexp = np.sum(np.exp(-beta * eigvals))

        # if np.isnan(trexp):
        #     print("not correct")

        # state = eigvecs @ np.diag(np.exp(-beta * eigvals) / trexp) @ eigvecs.conj().T

        m = np.max(eigvals)
        logTr = m + np.log(np.sum(np.exp(eigvals-m)))
        logLamBar = eigvals - np.ones((eigvals.size,)) * logTr
        LamBar = np.exp(logLamBar)

        dm = eigvecs @ np.diag(LamBar) @ eigvecs.conj().T

        return dm
