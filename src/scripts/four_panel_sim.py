import numpy as np
from dynesty.utils import Results
import sys
import pickle
import matplotlib.pyplot as plt
import paths
import _fig_params as fp

sys.path.append(str(paths.code))
from autocorr_models import model_BB

def timecoeff_to_d(tau_perp, tau_parr):
    D_perp = 1 / (6 * tau_perp)
    D_parr = 1 / (4 * tau_parr) - D_perp / 2
    return D_perp, D_parr

loaded = np.loadtxt(paths.data / "autocorrelation_samples.txt", skiprows=1)

aved_autocorr0_290 = loaded[1:, 0][:-50:2]
ac_std0_290 = loaded[1:, 1][:-50:2]
aved_autocorr90_290 = loaded[1:, 2][:-50:2]
ac_std90_290 = loaded[1:, 3][:-50:2]
time = np.linspace(0, 7.5, 150)[1::2]
yerr = ac_std0_290 

with open(paths.data / "model_gauss_twoexp.pkl", "rb") as f:
    result = pickle.load(f)
res = Results(result)
samples = res.samples_equal()
curves = model_BB(time[:, np.newaxis], *(samples.T))  

D_msd = np.loadtxt(paths.data / "kinisi_MSD")
D_dt = np.loadtxt(paths.data / "kinisi_t")
D_distribution = np.loadtxt(paths.data / "kinisi_distribution")
D_kinisi = np.loadtxt(paths.data / "kinisi_D")

fig = plt.figure(figsize=(fp.COLUMN_WIDTH, 1.68 * 1.96))
gs = plt.GridSpec(2, 1, figure=fig, hspace=0.4)
axes = [fig.add_subplot(gs[i]) for i in range(2)]
ax1, ax3 = axes

ax1.plot(D_dt / 1e6, D_msd / 1e3, "k-", label='MSD$(t\,)$')
for i, ci in enumerate(fp.CREDIBLE_INTERVALS):
    ax1.fill_between(
        D_dt / 1e6,
        *np.percentile(D_distribution, ci[:2], axis=1) / 1e3,
        alpha=ci[-1],
        color=fp.colors[0],
        lw=0,
    )
ax1.fill_between([], [], [], color=fp.colors[0], alpha=0.6, label=r'$p\,\left[m\,|\,\mathrm{MSD}(t\,)\right]$')
ax1.set_xlabel(r"$t$ / $10^3$ ps")
ax1.set_xlim([0, 10])
ax1.set_xticks([0, 5, 10.1])
ax1.set_ylabel(r"MSD$(t\,)$ / $10^{3}$ Å$^2$")
ax1.set_ylim(0, None)
ax1.set_yticks([0, 4, 8])
leg = ax1.legend()
leg.legend_handles[1].set_linewidth(0)

ax2 = fig.add_axes([0.65, 0.64, 0.25, 0.15])
ax2.hist(D_kinisi * 1e4, bins=fp.NBINS, density=True, color=fp.colors[0], alpha=0.6)
ax2.set_xlabel(r"$D^*\,|\,\mathrm{MSD}(t\,)$ / Å$^{2}$ps$^{-1}$")
ax2.set_xticks([0.135, 0.145])
ax2.set_yticks([])
ax2.spines["left"].set_visible(False)
ax2.patch.set_alpha(0.0) 

for c in curves:
    ax3.plot(time, c.mean(1), color=fp.colors[1])

ax3.errorbar(
    time,
    aved_autocorr90_290,
    yerr=ac_std90_290,
    fmt=".",
    ls="",
    label=r"$\theta = {0^{\circ}}$",
    zorder=10,
    color=fp.colors[2]
)
ax3.errorbar(
    time,
    aved_autocorr0_290,
    yerr=yerr,
    fmt=".",
    ls="",
    label=r"$\theta = {90^{\circ}}$",
    zorder=10,
    color=fp.colors[3]
)
ax3.set_xlabel(r"$t$ / ps")
ax3.set_xlim([0, 7.6])
ax3.set_xticks([0, 2.5, 5, 7.5])
ax3.set_ylabel(r"$G_{\theta}(t\,)$")
ax3.set_yticks([0, 0.5, 1])
ax3.set_ylim(0, None)
ax3.legend(ncol=1, bbox_to_anchor=(0.24, 0.3))

ax4 = fig.add_axes([0.37, 0.28, 0.25, 0.15])
ax4.hist(samples[:, 1], color=fp.colors[1], alpha=0.6, bins=fp.NBINS)
ax4.set_xlabel(r"$D_t\,|\,G_{\theta}(t\,)$ / ps$^{-1}$")
ax4.set_yticks([])
ax4.spines["left"].set_visible(False)
ax4.patch.set_alpha(0.0) 

ax5 = fig.add_axes([0.65, 0.28, 0.25, 0.15])
ax5.hist(samples[:, 0], color=fp.colors[1], alpha=0.6, bins=fp.NBINS)
ax5.set_xlabel(r"$D_s\,|\,G_{\theta}(t\,)$ / ps$^{-1}$")
ax5.set_xticks([0.18, 0.22])
ax5.set_yticks([])
ax5.spines["left"].set_visible(False)
ax5.patch.set_alpha(0.0) 

# Add panel labels below x-axis
for i, ax in enumerate(axes):
    ax.text(0.01, 1.04, fp.ALPHABET[i], transform=ax.transAxes, ha="center", va="bottom")

for i, ax in enumerate([ax1, ax3]):
    print(ax.get_window_extent().x1 - ax.get_window_extent().x0)
    print(ax.get_window_extent().y1 - ax.get_window_extent().y0)

plt.savefig(paths.figures / "four_panel_sim.pdf", bbox_inches='tight', pad_inches=0.01)
plt.close()
