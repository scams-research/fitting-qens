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
from plet_data import PletData


def remove_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.axes.get_xaxis().set_visible(False)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.axes.get_yaxis().set_visible(False)

def get_data(energy, q_lim_low, q_lim_high, q_bins, omega_lim=1.25):

    incident = {197: 1.97, 360: 3.60}

    data = PletData(
        paths.data / f"pLET_benzene_290K_{energy}_inc.nxspe",
        incident[energy],
        omega_lims=[-omega_lim, omega_lim],
        q_lims=[q_lim_low, q_lim_high],
    )
    empty = PletData(
        paths.data / f"pLET_empty_{energy}_inc.nxspe",
        incident[energy],
        omega_lims=[-omega_lim, omega_lim],
        q_lims=[q_lim_low, q_lim_high],
    )
    res = PletData(
        paths.data / f"pLET_benzene_260K_{energy}_inc.nxspe",
        incident[energy],
        omega_lims=[-omega_lim, omega_lim],
        q_lims=[q_lim_low, q_lim_high],
    )

    q_bins = sc.linspace(
        "q", data.q.min().values, data.q.max().values, q_bins, unit=sc.Unit("1/angstrom")
    )

    data.bin_q(q_bins)
    empty.bin_q(q_bins)
    res.bin_q(q_bins)

    data.data -= empty.data
    res.data -= empty.data

    omega_masked = data.omega_mid.values[np.invert(data.data.masks["omega"].values)]
    q_mid = data.q_mid.values[np.invert(data.data.masks["q"].values)]

    if energy == 360:
        exp = data.masked / 2
    else:
        exp = data.masked 

    samples = pickle.load(open(paths.data / f"pLET_aniso_model_samples_{energy}.pkl", "rb"))
    model_mean = samples.mean(axis=1)
    res_ratio = exp / model_mean

    return exp, model_mean, res_ratio, omega_masked, q_mid

def plot(ax1, exp, ax3, res_ratio, ax4, model_mean, omega_masked, q_mid, q_ticks):
    im1 = ax1.imshow(
        exp,
        cmap="viridis",
        norm=LogNorm(vmin=np.nanmin(exp), vmax=np.nanmax(exp)),
        extent=[
            omega_masked[0],
            omega_masked[-1],
            q_mid[0],
            q_mid[-1],
        ],
        aspect="auto",
        origin="lower",
    )
    ax1.set_xticks([])
    ax1.set_yticks(q_ticks)
    cbar1 = combined_fig.colorbar(
        im1, ax=ax1, location="right", fraction=0.1, pad=0.05, shrink=1
    )
    cbar1.set_ticks([1, 5])
    cbar1.set_ticklabels(["1", "5"])

    im3 = ax3.imshow(
        res_ratio,
        cmap=cm.vik,
        vmin=0.6,
        vmax=1.4,
        extent=[
            omega_masked[0],
            omega_masked[-1],
            q_mid[0],
            q_mid[-1],
        ],
        aspect="auto",
        origin="lower",
    )
    ax3.set_yticks(q_ticks)
    cbar2 = combined_fig.colorbar(im3, ax=ax3, location="right", fraction=0.1, pad=0.05, shrink=1)
    cbar2.set_ticks([0.7, 1.0, 1.3])
    cbar2.set_ticklabels(["0.7", "1.0", "1.3"])

    im4 = ax4.imshow(
        model_mean,
        norm=LogNorm(vmin=np.nanmin(exp), vmax=np.nanmax(exp)),
        cmap="viridis",
        extent=[
            omega_masked[0],
            omega_masked[-1],
            q_mid[0],
            q_mid[-1],
        ],
        aspect="auto",
        origin="lower",
    )
    ax4.set_yticks([])
    cbar1 = combined_fig.colorbar(
        im1, ax=ax4, location="right", fraction=0.1, pad=0.05, shrink=1
    )
    cbar1.set_ticks([1, 5])
    cbar1.set_ticklabels(["1", "5"])

    remove_ax(_)

    pos = [0.5, 1.02]
    ax1.text(
        pos[0],
        pos[1],
        "Experiment",
        transform=ax1.transAxes,
        ha="center",
        va="bottom",
    )

    ax3.text(
        pos[0], pos[1], "Ratio", transform=ax3.transAxes, ha="center", va="bottom"
    )

    ax4.text(
        pos[0], pos[1], "Model", transform=ax4.transAxes, ha="center", va="bottom"
    )

    lab_y = r"$Q$ / Å$^{-1}$"
    ax1.set_ylabel(lab_y)
    ax3.set_ylabel(lab_y)

    lab_x = r"$\omega$ / meV"
    ax3.set_xlabel(lab_x)
    ax4.set_xlabel(lab_x)

exp, model_mean, res_ratio, omega_masked, q_mid = get_data(197, 0.55, 1.58, 30)

combined_fig = plt.figure(figsize=(fp.PAGE_WIDTH*0.8, 2.8))
subfigs = combined_fig.subfigures(1, 2)
ax = subfigs[0].subplots(2, 2)
subfigs[0].subplots_adjust(hspace=0.15, wspace=0.2)
ax1, _, ax3, ax4 = ax.ravel()

plot(ax1, exp, ax3, res_ratio, ax4, model_mean, omega_masked, q_mid, [0.6, 1.0, 1.4])


exp, model_mean, res_ratio, omega_masked, q_mid = get_data(360, 0.6, 1.8, 31)

axl = subfigs[1].subplots(2, 2)
subfigs[1].subplots_adjust(hspace=0.15, wspace=0.2)
ax5, _, ax7, ax8 = axl.ravel()

plot(ax5, exp, ax7, res_ratio, ax8, model_mean, omega_masked, q_mid, [0.8, 1.2, 1.6])

subfigs[0].suptitle(r"$E_i = 1.97\,\mathrm{meV}$", ha="center", va="bottom", y=0.93, fontsize=fp.FONTSIZE)
subfigs[1].suptitle(r"$E_i = 3.60\,\mathrm{meV}$", ha="center", va="bottom", y=0.93, fontsize=fp.FONTSIZE)

for i, ax in enumerate([ax1, ax3, ax4, ax5, ax7, ax8]):
    print(ax.get_window_extent().x1 - ax.get_window_extent().x0)
    print(ax.get_window_extent().y1 - ax.get_window_extent().y0)

plt.savefig(paths.figures / "heatmap.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close()
