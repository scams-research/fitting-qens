import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipp as sc

from dynesty.utils import Results
from matplotlib.colors import LogNorm
from cmcrameri import cm
import paths
import _fig_params as fp

sys.path.append(str(paths.code))
from functions import line_to_D

with open(paths.data / "pLET_aniso_model.pkl", "rb") as f:
    ns_res = pickle.load(f)
ns_res = Results(ns_res)

fig = plt.figure(figsize=(fp.COLUMN_WIDTH, 0.8))
gs = plt.GridSpec(1, 3, figure=fig, hspace=0.4)
axes = [fig.add_subplot(gs[i]) for i in range(3)]
ax1, ax2, ax3 = axes

res_samples = ns_res.samples_equal()
ax1.hist(
    line_to_D(res_samples[:, 0]) * 10**8,
    density=True,
    bins=fp.NBINS,
    color=fp.colors[0],
    label=r"P - LET",
    alpha=0.6,
)
ax1.set_xlabel(r"$D_{\mathrm{fick}}\,|\,S_{\mathrm{inc}}$  / Å$^2$ps$^{-1}$")
ax1.set_xticks([0.178, 0.185])

ax2.hist(
    res_samples[:, 2] / 0.658212,
    density=True,
    bins=fp.NBINS,
    color=fp.colors[1],
    label=r"P - LET",
    alpha=0.6,
)
ax2.set_xlabel(r"$D_t\,|\,S_{\mathrm{inc}}$ / ps$^{-1}$")
# ax2.set_xticks([0.03, 0.05])

ax3.hist(
    res_samples[:, 1] / 0.658212,
    density=True,
    bins=fp.NBINS,
    color=fp.colors[1],
    alpha=0.6,
)
ax3.set_xlabel(r"$D_s\,|\,S_{\mathrm{inc}}$ / ps$^{-1}$")
# ax3.set_xticks([0.41, 0.45])

for ax in [ax1, ax2, ax3]:
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])

for i, ax in enumerate([ax1, ax2, ax3]):
    ax.text(0.01, 1.02, fp.ALPHABET[i], transform=ax.transAxes, ha="center", va="bottom")

for i, ax in enumerate([ax1, ax2, ax3]):
    print(ax.get_window_extent().x1 - ax.get_window_extent().x0)
    print(ax.get_window_extent().y1 - ax.get_window_extent().y0)

plt.savefig(paths.figures / "hists.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close()