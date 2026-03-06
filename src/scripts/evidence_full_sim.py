import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dynesty.utils import Results
import sys
import paths
import _fig_params as fp

sys.path.append(str(paths.code) + "/iris_analysis")
sys.path.append(str(paths.code))
from functions import line_to_D

def load_ns_group(paths, prefix, runs=("050", "075", "100", "125", "150")):
    ns_runs = {}

    for run in runs:
        fname = paths.data / f"{prefix}_full_mod_{run}mev.pkl"
        with open(fname, "rb") as f:
            ns_runs[run] = Results(pickle.load(f))

    logz = np.array([ns_runs[r].logz[-1] for r in runs])

    return ns_runs, logz

ns_runs, logz = load_ns_group(paths, "aniso")
iso_ns_runs, iso_logz = load_ns_group(paths, "iso")

with open(paths.data / "model_gauss_twoexp.pkl", "rb") as f:
    result = pickle.load(f)
gt = Results(result)
samples = gt.samples_equal()
D_perp, D_parr = samples[:, 1], samples[:, 0]

fig = plt.figure(figsize=(fp.COLUMN_WIDTH, 3.26))

gs = gridspec.GridSpec(
    nrows=2,
    ncols=3,
    height_ratios=[2.4, 1],
    hspace=0.4,
    wspace=0.15,
)

# Top: spans all 3 columns
ax0 = fig.add_subplot(gs[0, :])

# Bottom row: three small axes
ax1 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax2 = fig.add_subplot(gs[1, 2])

x = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
y = logz - iso_logz

ax0.scatter(x, y, color=fp.colors[3], s=8)
ax0.set_xlim([0.4, 1.6])
ax0.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5])

ax0.axhline(5, color="k", lw=1, ls="--")
ax0.set_xlabel(r"$\omega$-dynamic range / $\pm$ meV")
ax0.set_ylabel(r"$\ln\left[p\,\left(m_1 \mid \mathbf{D}\right)\right] - \ln\left[p\,\left(m_0 \mid \mathbf{D}\right)\right]$")
ax0.text(1.6, 4.2, "Strong evidence threshold", ha='right')

ns = ns_runs["125"]
samples = ns.samples_equal() 

for i, axi in enumerate([ax1, ax2, ax3]):
    axi.spines["left"].set_visible(False)
    axi.set_yticks([])

ax1.hist(line_to_D(samples[:, 0]) * 10**8, bins=fp.NBINS, density=True, alpha=0.6, color=fp.colors[0]) 
ax1.axvline(line_to_D(0.093) * 10**8, color="k", ls="--")
ax1.set_xlabel(r"$D_{\mathrm{fick}}\,|\,S^{\mathrm{sim}}_{\mathrm{inc}}$ / Å$^2$ps$^{-1}$")
ax1.set_xticks([0.12, 0.16])

ax2.hist(samples[:, 1] / 0.658212, bins=fp.NBINS, density=True, alpha=0.6, color=fp.colors[1])
ax2.axvline(np.mean(D_parr), color="k", ls="--")
ax2.set_xlabel(r"$D_s\,|\,S^{\mathrm{sim}}_{\mathrm{inc}}$ / ps$^{-1}$")
ax2.set_xlim(0.15, 0.35)

ax3.hist(samples[:, 2] / 0.658212, bins=fp.NBINS, density=True, alpha=0.6, color=fp.colors[1])
ax3.axvline(np.mean(D_perp), color="k", ls="--")
ax3.set_xlabel(r"$D_t\,|\,S^{\mathrm{sim}}_{\mathrm{inc}}$ / ps$^{-1}$")
ax3.set_xlim(None, 0.1)
ax3.set_xticks([0.02, 0.08])

for i, ax in enumerate([ax0, ax1, ax3, ax2]):
    ax.text(0.01, 1.04, fp.ALPHABET[i], transform=ax.transAxes, ha="center", va="bottom")

for i, ax in enumerate([ax0, ax2]):
    print(ax.get_window_extent().x1 - ax.get_window_extent().x0)
    print(ax.get_window_extent().y1 - ax.get_window_extent().y0)

plt.savefig(paths.figures / "Evidence_full_sim.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close()
