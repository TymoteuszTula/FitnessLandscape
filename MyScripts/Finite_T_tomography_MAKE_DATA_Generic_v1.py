#!/usr/bin/env python
# coding: utf-8

# # Plots for paper - make data

## this version of notebook used couplings which are drawn from Jmax*(1-2*normal) distribution for each spin component.

# GENERIC LIBRARIES:

import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy 
import scipy.sparse.linalg as sprsla
from math import sqrt, pi

def get_filename(parameters):
    """ construct a unique file-name corresponding to the given parameters """
          
    filename = "L_{}_rand_H_Jm_{}_hm_{}_seed_{}_T_{}_d_{}_dist_{}".format(parameters['L'],
			parameters['J_max_value'],
    	    parameters['h_max_value'],
            parameters['rand_init'],
            parameters['temp'],
            parameters['delta'],
			parameters['random_distribution_type']['shape']
            )
    if (parameters['temp']!=0):
        filename += "_dm_"+parameters['random_distribution_type']['rhoMethod']
        if (parameters['random_distribution_type']['rhoMethod']=='projectEigenvalues'):
            filename += "_minev_{}".format(parameters['random_distribution_type']['rhoParams']['min_ev'])
            
    return filename

if __name__ == '__main__':
    
	#     ( we assume a symbolic link to the
	#       CorrelationConnection directory,
	#       named thus, in the working directory )
	sys.path.append("../CorrelationConnection/bin/")
		
    

	# CorrelationConnection LIBRARIES:

	import hamiltonians
	import randomizer
	import stability_analysis_class
	import tools
	import exact_diagonalisation_code_sparse
	import utilities


	# SET THE SEED FOR THE RANDOM NUMBER GENERATOR:

	rand_init=41

	# CHOICE OF TYPE OF REFERENCE HAMILTONIAN:
	#  note this notebook is hard-coded for random type 'RN'!
	ham_type='RN' # <-- 'NN': nearest-neighbour, 'RN': random

	# PARAMETERS FOR THE REFERENCE MODEL:

	# General parameters:

	L=8 # <-- Number of spins
	temp = 1.0              # <-- Temperature
	extra_params = {        # <-- additional parameters for stability analysis
		'geometry': 'line'  # <-- defining possible geometry choices:
							#     'ring' or 'line'
	}
	bc = "finite"         # <-- boundary conditions
							#     (closed = "infinite", open = "finite")

	# Parameters applying to NN hamiltonians
	# (note: not used if we go for random hamiltonian)
	#        - no need to comment them out):

	#h = [[0, 0, 0]]                              # <-- Applied field
	#J_onsite = np.zeros((3, 3))                  # <-- Onsite interactions (superfluous for S=1/2)
	#J_nnn = np.zeros((3, 3))                     # <-- nnn interactions
	#J_nn = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # <-- nn interactions

	# Parameters applying to random hamiltonians
	# (note: not used if we go for NN hamiltonian
	#        - no need to comment them out):

	sigma = 1.0
	h_max_value = sigma # <-- maximum value of field
	J_max_value = sigma # <-- maximum value of coupling constant

	# SET RANDOMIZER PARAMETERS:

	delta= 2.0          # <-- Variation of random samples away from reference state/Hamiltonian
	# vary delta
	vary_delta=True # <-- turning this on changes distributions to sawtooth distrib around the reference value, otherwise uniform
	
	# control the type of distribution used to generate random states:
	random_distribution_type = {\
		'shape':  'uniformCircle', # allowed values: 'uniformSquare' (previous default), 'uniformCircle', 'normal'
		'rhoMethod': 'squareRandom', # allowed values: 'squareRandom' (previous default), 'projectEigenvalues'
		'rhoParams': { # further parameters controlling density matrix generation
		          'min_ev': 0.0 # minimal allowed value for eigenvalues of the density matrix in projectEigenvalues modes
				}
		}


	# <-- Number of processes to use (parallelization)
	no_of_processes= 40 # <-- Number of processes to use (parallelization)

	# SET PARAMETERS FOR FITNESS LANDSCAPE:

	save_rhams= "False"     # <-- Do not save random Hamiltonians/states
	temp_type= "value"      # <-- Temperature given explicitly

	# SET PARAMETERS FOR CALCULATION:

	no_of_samples= 1000 #
	

	### INITIALIZE RANDOM NUMBER GENERATOR:

	np.random.seed(rand_init)

	# CREATE THE REFERENCE MODEL:

	### Build set of random couplings for a long-range random Hamiltonian

	# Initialise J:
	# First, we create an list of length L of empty lists:
	J = np.empty((L, 0)).tolist()


	# Then, we populate each list within the list with another list of length L:
	for i in range(L):
		J[i]=np.empty((L, 0)).tolist()

	# We now have a list of list of empty lists: J[i][j] gives the interaction
	#                              between the spin at i and the spin at j (not yet populated)
	# Fill J:
	# We populate each element J[i][j] with a matrix giving the 9 interactions between
	# the x,y,z components of the spin at i and the x,y,z components of the spin at j:
	for i in range(L):
		for j in range(L):
			J[i][j]=J_max_value*(1.0-2.0*np.random.randn(3,3))+0.0 # <-- normal distribution with
														 #     STD = sigma and
														 #     average = 0.0


	# Note due to the structure of the Hamiltonian and the spin commutation relations,
	# we can assume J[i][j][alpha][beta] to have certain symmetries, for instance
	# J[i][i][alpha][beta]=-J[i][i][beta][alpha]
	# J[i][j][alpha][beta]=J[j][i][beta][alpha]
	# However, we do not need to impose those symmetries - the Hamiltonian will take
	# care of that. See random_hamiltonians.lyx
	#
	# Now create the random field:
	#
	h = [h_max_value * np.random.randn(3)]


	# In[8]:


	# Now create an instance of the GeneralHamiltonian class:
	HAMILTONIAN = hamiltonians.GeneralHamiltonian(L, h, J, temp, bc)
	# Note: the parameters required to generate an instance of the class
	# GeneralHamiltonian in the module hamiltonians can be found in the
	# definition of the __init__ procedure of that class in hamiltonians.py.
	#HAMILTONIAN = hamiltonians.RandomHamiltonianTI(
	#    L, temp, bc="infinite", h=None, J=None, h_max=h_max_value, J_max=J_max_value)


	# CREATE RANDOMIZER:

	RANDOMIZER = randomizer.RandomizerStateRandomDelta(
		HAMILTONIAN, delta, no_of_processes, random_distribution_type)


	# In[10]:


	# CREATE FINTESS LANDSCAPE ANALYSIS:

	corr = ["Sxx", "Sxy", "Sxz", "Syx", "Syy", "Syz", "Szx", "Szy", "Szz"] # <-- Correlators that need to be calculated=
	STABILITY_ANALYSIS = \
	stability_analysis_class.StabilityAnalysisSparse(
		HAMILTONIAN, RANDOMIZER, \
		corr, save_rhams,\
		temp_mul=temp, temp_type=temp_type, \
		extra_params=extra_params
	)


	# In[11]:


	# CALCULATE:

	dist, en, diffSij, diffSq, Sqs, Sijs, Sqints, Sq_int=STABILITY_ANALYSIS.generate_random_Sij_sparse(no_of_samples)


	# In[12]:


	# STORE ALL PARAMETERS FOR LATER REFERENCE:

	parameters={
		#Parameters:
		'rand_init': rand_init,
		'ham_type': ham_type,
		'L':L,
		'temp':temp,
		'vary_delta':vary_delta,
		'random_distribution_type':random_distribution_type,
		'extra_params':extra_params,
		'bc':bc,
		'h':h,
	#    'J_onsite':J_onsite,
	#    'J_nnn':J_nnn,
	#    'J_nn':J_nn,
		'J':J,
		'h_max_value':h_max_value,
		'J_max_value':J_max_value,
		'delta':delta,
		'no_of_processes':no_of_processes,
		'save_rhams':save_rhams,
		'temp_type':temp_type,
		'no_of_samples':no_of_samples}
	calculations={
		'HAMILTONIAN':HAMILTONIAN,
		'RANDOMIZER':RANDOMIZER,
		'STABILITY_ANALYSIS':STABILITY_ANALYSIS}
	results={
		'dist':dist,
		'en':en,
		'diffSij':diffSij,
		'diffSq':diffSq,
		#'Sqs':Sqs, # Sqs is by far the largest field (correlators of individual spin components), so omit this here
		'Sijs':Sijs,
		'Sqints':Sqints,
		'Sq_int':Sq_int}
	data={
		'parameters':parameters,
		'calculations':calculations,
		'results':results
	}


	# Save data:

	# generate filename based on parameters
	filename=get_filename(parameters)

	outpath='../data/'+filename+'.pickle'
	with open(outpath, 'wb') as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
		print ("Data saved to %s"%(outpath))



