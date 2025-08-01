# exact_diagonalisation_code_sparse.py

import numpy as np
from exact_diagonalisation_code import apply_op
import scipy.sparse as sprs
import time

def create_hamiltonian_sparse(params_input={}, print_time=False):
    start_time = time.time()

    default_params = {"L": 3, "J": [[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]], "h": [[0, 0, 0]]}
    params = {}
    params.update(default_params)
    params.update(params_input)

    L = params["L"]
    J = params["J"]
    h = params["h"]

    H_spar = sprs.lil_matrix((2 ** L, 2 ** L), dtype=complex)

    for i_row in range(2 ** L):
        
        j_max = L
        
        for i in range(j_max):

            if len(h) == 1:
                h_c = h[0]
            else:
                h_c = h[i]

            res_z_temp = apply_op("Sz", i % L, i_row)
            res_x_temp = apply_op("Sx", i % L, i_row)
            res_y_temp = apply_op("Sy", i % L, i_row)

            H_spar[res_z_temp[0], i_row] += h_c[2] * res_z_temp[1]
            H_spar[res_x_temp[0], i_row] += h_c[0] * res_x_temp[1]
            H_spar[res_y_temp[0], i_row] += h_c[1] * res_y_temp[1] 

            for j in range(j_max):

                if len(J) == 1:
                    J_c = J[0][0]
                else:
                    J_c = J[i][j]

                res_zz_temp = apply_op("Sz", j % L, res_z_temp[0])
                res_xz_temp = apply_op("Sx", j % L, res_z_temp[0])
                res_yz_temp = apply_op("Sy", j % L, res_z_temp[0])

                res_zx_temp = apply_op("Sz", j % L, res_x_temp[0])
                res_xx_temp = apply_op("Sx", j % L, res_x_temp[0])
                res_yx_temp = apply_op("Sy", j % L, res_x_temp[0])

                res_zy_temp = apply_op("Sz", j % L, res_y_temp[0])
                res_xy_temp = apply_op("Sx", j % L, res_y_temp[0])
                res_yy_temp = apply_op("Sy", j % L, res_y_temp[0])

                H_spar[res_zz_temp[0], i_row] += J_c[2][2] * res_z_temp[1] * res_zz_temp[1]
                H_spar[res_xz_temp[0], i_row] += J_c[2][0] * res_z_temp[1] * res_xz_temp[1]
                H_spar[res_yz_temp[0], i_row] += J_c[2][1] * res_z_temp[1] * res_yz_temp[1]

                H_spar[res_zx_temp[0], i_row] += J_c[0][2] * res_x_temp[1] * res_zx_temp[1]
                H_spar[res_xx_temp[0], i_row] += J_c[0][0] * res_x_temp[1] * res_xx_temp[1]
                H_spar[res_yx_temp[0], i_row] += J_c[0][1] * res_x_temp[1] * res_yx_temp[1]

                H_spar[res_zy_temp[0], i_row] += J_c[1][2] * res_y_temp[1] * res_zy_temp[1]
                H_spar[res_xy_temp[0], i_row] += J_c[1][0] * res_y_temp[1] * res_xy_temp[1]
                H_spar[res_yy_temp[0], i_row] += J_c[1][1] * res_y_temp[1] * res_yy_temp[1]

    H_spar = H_spar.transpose()
    H_spar = H_spar.tobsr()
    H_spar.eliminate_zeros()

    end_time = time.time()
    if (print_time):
        print("Time for executing create_hamiltonian_sparse: ",end_time-start_time)
    return H_spar

def create_sz_sparse(L):

    SZ = []
    for j in range(L):
        sz = sprs.eye(2**L, format='lil', dtype=complex)
        for i in range(2 ** L):
            bit = (i >> j) & 1
            if bit == 0:
                sz[i, i] = -1

        SZ.append(1/2 * sz.tobsr())

    return SZ

def create_sx_sparse(L):

    SX = []
    for j in range(L):
        sx = sprs.lil_matrix((2**L, 2**L), dtype=complex)
        for i in range(2 ** L):
            indx = apply_op("Sx", j, i)
            sx[i, indx[0]] = indx[1]

        SX.append(sx.tobsr())

    return SX

def create_sy_sparse(L):

    SY = []
    for j in range(L):
        sy = sprs.lil_matrix((2**L, 2**L), dtype=complex)
        for i in range(2 ** L):
            indx = apply_op("Sy", j, i)
            sy[i, indx[0]] = indx[1]

        SY.append(sy.tobsr())

    return SY


            
