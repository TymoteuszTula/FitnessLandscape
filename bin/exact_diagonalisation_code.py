# ED_compare.py

import numpy as np

def create_states(L=3):
    state_no = np.arange(2 ** L)

    states = []

    for state_i in state_no:
        str_state = ('{:0' + str(L) + 'b}').format(state_i)
        states.append([2 * int(j) - 1 for j in str_state])
        
	# this list is not needed. We know the binary numbers from 0 to 2^L-1 represent the spin configurations!
	
    return states

# def apply_op(states, op, j, i):
#     """ apply the operator described by op to the j-th spin in the basis state i
# 		op : string
# 		  name of the operator to apply - Sx, Sy or Sz
# 		j -> j_spin: position of spin on which to act with the operator
# 		i -> state_idx: index of the many-body basis state on which to act
# 		returns: tuple (idx, amplitude)
# 		  where idx = index of the target state
# 		        amplitude = operator matrix element connecting i and idx
	
#     """

# 	# use bit-wise operations:
# 	# state_idx
# 	# (note DiagHam implements a separate object which represents the Hilbert space)
# 	# here: state = state_idx
# 	# test if the j-th bit is set:
# 	# bit = (state >> j_spin) & 1
# 	# flip the j_spin-th bit:
# 	# state = state ^ (1 << j_spin)

#     if op == "Sz":
#         res = (i, 1/2 * states[i][j])

#     elif op == "Sx":
#         if states[i][j] == -1:
#             state_temp = states[i].copy() 
#             state_temp[j] = 1
#             idx_temp = states.index(state_temp)

#             res = (idx_temp, 1/2)
#         else:
#             state_temp = states[i].copy() 
#             state_temp[j] = -1
#             idx_temp = states.index(state_temp)

#             res = (idx_temp, 1/2)
#     elif op == "Sy":
#         if states[i][j] == -1:
#             state_temp = states[i].copy() 
#             state_temp[j] = 1
#             idx_temp = states.index(state_temp)

#             res = (idx_temp, -1/2 * 1j)
#         else:
#             state_temp = states[i].copy()
#             state_temp[j] = -1
#             idx_temp = states.index(state_temp)

#             res = (idx_temp, 1/2 * 1j)

#     return res

def apply_op(op, j, i):
    bit = (i >> j) & 1
    if op == "Sz":
        res = (i, 1/2 * (2 * bit - 1))

    elif op == "Sx":
        i_new = i ^ (1 << j)
        res = (i_new, 1/2)
    
    elif op == "Sy":
        i_new = i ^ (1 << j)
        res = (i_new, 1/2 * 1j * (2 * bit - 1))

    return res


def create_hamiltonian(params_input={}):

    default_params = {"L": 3, "Jz": 1, "Jy": 1, "Jx": 1, "hx": 0, "hy": 0, "hz": 0, "D": 0, "E": 0, "bc": "finite"}
    params = {}
    params.update(default_params)
    params.update(params_input)

    L = params["L"]
    Jz = params["Jz"]
    Jx = params["Jx"]
    Jy = params["Jy"]
    hz = params["hz"]
    hx = params["hx"]
    hy = params["hy"]
    D = params["D"]
    E = params["E"]
    bc = params["bc"]

    H = []

    for i in range(2 ** L):
        current_row = 1j * np.zeros(2 ** L)

        res = []

        if bc == "finite":
            j_max = L - 1
        elif bc == "infinite":
            j_max = L

        for j in range(j_max):

            res_z_temp1 = apply_op("Sz", j % L, i)
            res_z_temp2 = apply_op("Sz", (j + 1) % L, res_z_temp1[0])
            res_zz_temp = apply_op("Sz", j % L, res_z_temp1[0])

            res.append((res_z_temp2[0], Jz * res_z_temp1[1] * res_z_temp2[1]))
            res.append((res_zz_temp[0], D * res_z_temp1[1] * res_zz_temp[1]))

            res_x_temp1 = apply_op("Sx", j % L, i)
            res_x_temp2 = apply_op("Sx", (j + 1) % L, res_x_temp1[0])
            res_xx_temp = apply_op("Sx", j % L, res_x_temp1[0])

            res.append((res_x_temp2[0], Jx * res_x_temp1[1] * res_x_temp2[1]))
            res.append((res_xx_temp[0], E * res_x_temp1[1] * res_xx_temp[1]))

            res_y_temp1 = apply_op("Sy", j % L, i)
            res_y_temp2 = apply_op("Sy", (j + 1) % L, res_y_temp1[0])
            res_yy_temp = apply_op("Sy", j % L, res_y_temp1[0])

            res.append((res_y_temp2[0], Jy * res_y_temp1[1] * res_y_temp2[1]))
            res.append((res_yy_temp[0], -E * res_y_temp1[1] * res_yy_temp[1]))

            res.append((res_z_temp1[0], -hz * res_z_temp1[1]))
            res.append((res_x_temp1[0], -hx * res_x_temp1[1]))
            res.append((res_y_temp1[0], -hy * res_y_temp1[1]))

        if bc == "finite":
            res_z_temp1 = apply_op("Sz", L-1, i)
            res_zz_temp = apply_op("Sz", L-1, res_z_temp1[0])
            res_x_temp1 = apply_op("Sx", L-1, i)
            res_xx_temp = apply_op("Sx", L-1, res_x_temp1[0])
            res_y_temp1 = apply_op("Sy", L-1, i)
            res_yy_temp = apply_op("Sy", L-1, res_y_temp1[0])

            res.append((res_zz_temp[0], D * res_z_temp1[1] * res_zz_temp[1]))
            res.append((res_xx_temp[0], E * res_x_temp1[1] * res_xx_temp[1]))
            res.append((res_yy_temp[0], -E * res_y_temp1[1], res_yy_temp[1]))

            res.append((res_z_temp1[0], -hz * res_z_temp1[1]))
            res.append((res_x_temp1[0], -hx * res_x_temp1[1]))
            res.append((res_y_temp1[0], -hy * res_y_temp1[1]))

        k_idx = 0
        for id in [res[k][0] for k in range(len(res))]:
            current_row[id] += res[k_idx][1]
            k_idx += 1

        H.append(current_row)

    H = np.array(H)

    return H

def create_hamiltonian_2(params_input={}):

    default_params = {"L": 3, "J": [[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]], "h": [[0, 0, 0]]}
    params = {}
    params.update(default_params)
    params.update(params_input)

    L = params["L"]
    J = params["J"]
    h = params["h"]

    H = []

    for i_row in range(2 ** L):
        current_row = 1j * np.zeros(2 ** L)

        res = []

        j_max = L

        for i in range(j_max):

            if len(h) == 1:
                h_c = h[0]
            else:
                h_c = h[i]

            res_z_temp = apply_op("Sz", i % L, i_row)
            res_x_temp = apply_op("Sx", i % L, i_row)
            res_y_temp = apply_op("Sy", i % L, i_row)

            res.append((res_z_temp[0], -h_c[2] * res_z_temp[1]))
            res.append((res_x_temp[0], -h_c[0] * res_x_temp[1])) 
            res.append((res_y_temp[0], -h_c[1] * res_y_temp[1]))  

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

                res.append((res_zz_temp[0], J_c[2][2] * res_z_temp[1] * res_zz_temp[1]))
                res.append((res_xz_temp[0], J_c[2][0] * res_z_temp[1] * res_xz_temp[1]))
                res.append((res_yz_temp[0], J_c[2][1] * res_z_temp[1] * res_yz_temp[1]))

                res.append((res_zx_temp[0], J_c[0][2] * res_x_temp[1] * res_zx_temp[1]))
                res.append((res_xx_temp[0], J_c[0][0] * res_x_temp[1] * res_xx_temp[1]))
                res.append((res_yx_temp[0], J_c[0][1] * res_x_temp[1] * res_yx_temp[1]))

                res.append((res_zy_temp[0], J_c[1][2] * res_y_temp[1] * res_zy_temp[1]))
                res.append((res_xy_temp[0], J_c[1][0] * res_y_temp[1] * res_xy_temp[1]))
                res.append((res_yy_temp[0], J_c[1][1] * res_y_temp[1] * res_yy_temp[1]))

        k_idx = 0
        for id in [res[k][0] for k in range(len(res))]:
            current_row[id] += res[k_idx][1]
            k_idx += 1

        H.append(current_row)

    H = np.array(H)

    return H
            


def create_sz(states, L):

    SZ = []

    for j in range(L):

        sz = np.eye(2 ** L)
    
        for i in range(2 ** L):
            bit = (i >> j) & 1
            if bit == 0:
                sz[i, i] = -1

        SZ.append( 1 / 2 * sz)

    return SZ

def create_sx(L):

    SX = []

    for j in range(L):

        sx = np.zeros((2 ** L, 2 ** L))

        for i in range(2 ** L):
            indx = apply_op("Sx", j, i)
            sx[i, indx[0]] = 1
        
        SX.append( sx)

    return SX

def create_sy(L):

    SY = []

    for j in range(L):

        sy = np.zeros((2 ** L, 2 ** L)) + 1j * np.zeros((2 ** L, 2 ** L))

        for i in range(2 ** L):
            indx = apply_op("Sy", j, i)
            sy[i, indx[0]] = indx[1]

        SY.append( sy)

    return SY
                
