import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipp as sc

from dynesty.utils import Results
import paths
import _fig_params as fp

sys.path.append(str(paths.code))
from plet_data import PletData


def get_model(energy, q_lim_low, q_lim_high, q_bins, omega_lim=1.25):

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

    q_bins = sc.linspace(
        "q", data.q.min().values, data.q.max().values, q_bins, unit=sc.Unit("1/angstrom")
    )

    data.bin_q(q_bins)
    empty.bin_q(q_bins)

    data.data -= empty.data


    omega_masked = data.omega_mid.values[np.invert(data.data.masks["omega"].values)]
    q_mid = data.q_mid.values[np.invert(data.data.masks["q"].values)]
    samples = pickle.load(open(paths.data / f"pLET_aniso_model_samples_{energy}.pkl", "rb"))


    return q_mid, omega_masked, data, samples

fig = plt.figure(figsize=(fp.PAGE_WIDTH, 8))
gs = plt.GridSpec(4, 3, figure=fig, hspace=0.1)
axes = [fig.add_subplot(gs[i, j]) for i in range(4) for j in range(3)]

credible_intervals = [[16, 84], [2.5, 97.5], [0.15, 99.85]]
alphas = [0.8, 0.6, 0.4]

q_mid197, omega_masked197, data197, samples197 = get_model(197, 0.55, 1.58,30)

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
    ax.set_zorder(0)


fig = plt.figure(figsize=(fp.PAGE_WIDTH, 1.37 * 5)) 
gs = plt.GridSpec(5, 3, figure=fig, hspace=0.1)
axes = [fig.add_subplot(gs[i, j]) for i in range(5) for j in range(3)]

credible_intervals = [[16, 84], [2.5, 97.5], [0.15, 99.85]]
alphas = [0.8, 0.6, 0.4]


for i in range(len(q_mid197)):
    ax = axes[i]
    for ci, alpha in zip(credible_intervals, alphas):
        y_lower, y_upper = np.percentile(samples197[i], ci, axis=0)
        ax.fill_between(omega_masked197, y_lower, y_upper, color=fp.colors[1], alpha=alpha, lw=0, label = 'Anisotropic model')
    if i < 11:
        ax.set_xticks([])

    # Plot data
    ax.errorbar(omega_masked197,data197.masked[i],yerr = data197.errors[i], marker='.', ls = 'none', ms = 1, color = fp.colors[0],alpha=0.3, label = 'P-LET') # , label = f"q={q_mid[x]:.4f}"
    ax.text(0.33, 0.06, f"$Q={q_mid197[i]:.4f}$ Å$^{{-1}}$", transform=ax.transAxes)
    ax.set_ylim(0, None)


axes[11].set_xlabel(r'$\omega$ / meV')
axes[12].set_xlabel(r'$\omega$ / meV')
axes[13].set_xlabel(r'$\omega$ / meV')
axes[11].set_zorder(100)

for ax in axes:
    ax.spines[['right', 'top']].set_visible(False)

remove_ax(axes[14])

plt.savefig(paths.figures /'si_model197_fit.pdf', bbox_inches='tight')
plt.close()