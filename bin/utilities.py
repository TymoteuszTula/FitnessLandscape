
import numpy as np
import scipy.sparse.linalg as sprsla
import scipy.linalg as spla
from math import pi, nan, sqrt, cos, sin, nan
from cmath import exp

# Distance between two S(q)s
# This takes two arrays mat_tar and mat_rnd that represent
# the values of S(q) and S'(q) on a discrete lattice of values
# of q, assumed to be uniformly spaced. It calculates
# approximately the r.m.s. difference between S(q) and S'(q)
# and divides by the r.m.s. deviation of S(q).
# Note that we don't need to know the q-values because
# the ration taken at the end cancels the norms of the
# integrals.
#
def dist_S(mat_tar,mat_rnd):
    # Individual deviations:
    dev=mat_tar-mat_rnd
    # Squared deviations:
    dev=np.power(dev,2)
    # Average of all squared deviations w.r.t. q:
    dev = np.average(dev)
    # Take the square root:
    dev = sqrt(dev)
    # Normalize:
    dev = dev/np.std(mat_tar)
    #
    return dev
