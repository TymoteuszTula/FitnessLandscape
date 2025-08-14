# Fitness Landscape

This python code contains the method to generate spin correlators, structure factors and neutron scattering functions for general magnetic Hamiltonians. It also can be used to generate multiple samples of those observables for a random deviations from the reference ground state in order to study the fitness landscape of neutron scattering functions in the quantum state space. 

This code was used to generate data presented in our article: 

T. Tula, J. Quintanilla, G. Möller, *Fitness landscape for quantum state tomography from neutron scattering*, [arXiv:2412.10502](https://arxiv.org/abs/2412.10502)

## Setting up your environment

It is advised to set up a conda environment to ensure compatibility before using this software. The file ```corrconn_env.yml``` has been provided for this purpose. 

### Instructions

To find out which environments are currently available in your system (the asterisk indicates which one is active):

conda info --envs

If corrconn appears on the list, activate it using 

```conda activate corrconn```

You can later go back to your default enviornment using 

```conda deactivate```

If corrconn does not appear on the list, you must install it first:

```conda env create --file corrconn_env.yml```

## Getting started

Check out the tutorial in the docs folder (a Jupyter notebook).
