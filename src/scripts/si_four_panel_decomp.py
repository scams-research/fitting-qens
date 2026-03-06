import sys
import paths
import matplotlib.pyplot as plt
import numpy as np
import _fig_params as fp
from cmcrameri import cm
from scipy.signal import fftconvolve
from scipy.special import spherical_jn

sys.path.append(str(paths.code))
from functions import MDANSEdata

rot = MDANSEdata(paths.data / "rotation_only_incoh_mdanse.csv")
rot.parse(energy_lim=4.13456)
rot.scippbin(bins=12)
rot_intens = rot.binned.values.max(axis=1)  # / DWF

trans = MDANSEdata(paths.data / "translation_only_incoh_mdanse.csv")
trans.parse(energy_lim=4.13456)
trans.scippbin(bins=12)

sim_incoh = MDANSEdata(paths.data / "incoh_mdanse.csv")
sim_incoh.parse(energy_lim=4.13456)
sim_incoh.scippbin(bins=12)

q_mid = (
    sim_incoh.binned.coords["q"].values[:-1]
    + np.diff(sim_incoh.binned.coords["q"].values) / 2
)
reconv = np.zeros(rot.binned.values.shape)
for q in range(12):
    reconv[q] = fftconvolve(rot.binned.values[q], trans.binned.values[q], mode="same")

EISF_iso = spherical_jn(0, q_mid * 2.48) ** 2
res_ratio = (sim_incoh.binned.values * 1550) / (reconv)

fig = plt.figure(figsize=(fp.PAGE_WIDTH, 1.37))
gs = plt.GridSpec(1, 4, figure=fig, hspace=0.5)
ax1, ax2, ax3, ax4 = [fig.add_subplot(gs[i]) for i in range(4)]
q = 2

ax1.plot(
    rot.energy,
    trans.binned.values[q],
    label=r"$S_{inc}^{T}(Q,\omega)$",
    color="#CC78BC",
)
ax1.plot(
    rot.energy,
    rot.binned.values[q],
    label=r"$S_{inc}^{R}(Q,\omega)$",
    color="#029E73",
)
ax1.set_xlabel("$\omega$ / meV")
ax1.set_xlim([-0.4, 0.4])
ax1.set_xticks([-0.3, 0, 0.3])
ax1.set_ylabel("Intensity")
ax1.set_ylim([0, 4])
ax1.set_yticks([])
ax1.text(0.5, 1.04, r"$Q = 0.738\,\mathrm{Å}^{-1}$", transform=ax1.transAxes, ha="center", va="bottom")

ax2.plot(
    q_mid,
    rot_intens / np.max(rot_intens) * 0.569,
    marker="o",
    ls='',
    label=r"$\delta(\omega)$",
    color="#0173B2",
    zorder=10
)
ax2.plot(
    q_mid,
    EISF_iso,
    label=r"$j_0^2(Q,2.48\,\mathrm{Å})$",
    color="#029E73",
)
ax2.set_xlabel(r"$Q$ / Å$^{-1}$")
ax2.set_xlim(0.4, 1.85)
ax2.set_xticks([0.6, 1.1, 1.6])
ax2.set_ylabel("Normalised intensity")
ax2.set_yticks([])
ax2.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.1, 1.1))

for i in range(6):
    ax3.plot(
        sim_incoh.energy,
        sim_incoh.binned.values[i] * 1550 / 2000,
        color="#CC78BC"
    )
    ax3.plot(sim_incoh.energy, reconv[i] / 2000, color="#029E73", ls="--")
ax3.set_xlabel(r"$\omega$ / meV")
ax3.set_xlim(-0.4, 0.4)
ax3.set_xticks([-0.3, 0, 0.3])
ax3.set_ylabel("Intensity")
ax3.set_ylim(0.04, None)
ax3.set_yticks([])

im3 = ax4.imshow(
    res_ratio,
    cmap=cm.vik,
    vmin=0.6,
    vmax=1.4,
    extent=[
        sim_incoh.energy[0],
        sim_incoh.energy[-1],
        q_mid[0],
        q_mid[-1],
    ],
    aspect="auto",
    origin="lower",
)
ax4.set_xlabel(r"$\omega$ / meV")
ax4.set_xlim([-0.4, 0.4])
ax4.set_xticks([-0.3, 0, 0.3])
ax4.set_ylabel(r"$Q$ / Å$^{-1}$")
ax4.set_yticks([0.6, 1.1, 1.6])
cbar2 = fig.colorbar(im3, ax=ax4, location="right", fraction=0.1, pad=0.05, shrink=1)
cbar2.set_ticks([0.7, 1.0, 1.3])

# Add panel labels below x-axis
for ax, label in zip([ax1, ax2, ax3, ax4], ["a", "b", "c", "d"]):
    ax.text(
        0.01, 1.04, label, transform=ax.transAxes, ha="center", va="bottom"
    )

for i, ax in enumerate([ax1, ax2]):
    print(ax.get_window_extent().x1 - ax.get_window_extent().x0)
    print(ax.get_window_extent().y1 - ax.get_window_extent().y0)

plt.savefig(paths.figures / "si_four_panel_decomp.pdf", bbox_inches="tight")
plt.close()
