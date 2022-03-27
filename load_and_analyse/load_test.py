# load_test.py

import sys
sys.path.append('./bin/')
import matplotlib.pyplot as plt
from stability_analysis_class import StabilityAnalysisSparse

foldername = './run/output/feynman/'
filename = 'output_test0.pickle'

heh = 1

data = StabilityAnalysisSparse.load_random_samples(heh, foldername, filename)

plt.figure()
plt.plot(data["dist"], data["diffSq"], "-o")
plt.show()

