import paths
import sys
import pickle
import _fig_params as fp
import numpy as np
import scipp as sc
import corner
import matplotlib.pyplot as plt
from dynesty.utils import Results

sys.path.append(str(paths.code))
from functions import line_to_D

with open( paths.data /'pLET_aniso_model.pkl', 'rb') as f:
    ns_res = pickle.load(f)

ns_res = Results(ns_res)
samples_eq = ns_res.samples_equal()

for_corner = np.array([line_to_D(samples_eq[:,0])*10**8, samples_eq[:,1] / 0.658212, samples_eq[:,2] / 0.658212]).T

fig = plt.figure(figsize=(fp.PAGE_WIDTH*0.5, fp.PAGE_WIDTH*0.5))
corner.corner(for_corner, labels=[r'$D^*$ / Å$^2$ps$^{-1}$', r'$D_{s}$ / ps$^{-1}$', r'$D_{t}$ / ps$^{-1}$'], show_titles=True, title_fmt=".2f", fig=fig)

plt.savefig(paths.figures /'si_aniso_corner.pdf', bbox_inches = 'tight')
plt.close()