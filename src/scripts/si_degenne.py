import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipp as sc

import sys
import paths
import _fig_params as fp

sys.path.append(str(paths.code)+ "/iris_analysis")
sys.path.append(str(paths.code))

from iris import IRISData
from functions import second_moment_analyser, limit_integral, MDANSEdata

resolution = IRISData(paths.data / "resolution_IRIS00107073.nxs", omega_lims=[-0.4, 0.4])
exp290 = IRISData(paths.data / "benzene_IRIS00107089.nxs", omega_lims=[-0.4, 0.4])
background = IRISData(paths.data / "bkg_IRIS00107097.nxs", omega_lims=[-0.4, 0.4])

q_bins = sc.linspace( "q", resolution.data.coords["q"].min(), resolution.data.coords["q"].max(), 13)
exp_bin_norm = np.histogram(exp290.q, bins=q_bins.values)[0]

resolution.bin_q(q_bins)
exp290.bin_q(q_bins)
background.bin_q(q_bins)

resolution.data -= background.data
exp290.data -= background.data

res_int = np.trapezoid(resolution.masked, axis=1) / 90

sim290_tot = MDANSEdata(paths.data / "total_mdanse.csv")
sim290_tot.parse()
sim290_tot.scippbin(bins=12)
sim290_tot.convolve(resolution, norm_factor=res_int)

sim290_incoh = MDANSEdata(paths.data / "incoh_mdanse.csv")
sim290_incoh.parse()
sim290_incoh.scippbin(bins=12)
sim290_incoh.convolve(resolution, norm_factor=res_int)


sim290_coh = MDANSEdata(paths.data / "coh_mdanse.csv")
sim290_coh.parse()
sim290_coh.scippbin(bins=12)
sim290_coh.convolve(resolution, norm_factor=res_int)


mid_q = 0.5 * (q_bins.values[1:] + q_bins.values[:-1])

# Structural integralsf
struc_lims = [-0.05, 0.05]
sim_int = limit_integral(exp290.omega, sim290_tot.convolved * 1.203e7, struc_lims)
sim_incoh = limit_integral(
    exp290.omega, sim290_incoh.convolved * 0.22 * 1.75, struc_lims
)
exp_int, exp_int_err = limit_integral(
    exp290.omega,
    exp290.masked / res_int[:, np.newaxis],
    struc_lims,
    s_qw_err=exp290.errors / res_int[:, np.newaxis],
)

# Second moment analyser
moment_exp, err_exp = second_moment_analyser(
    exp290.omega, exp290.masked, s_qw_err=exp290.errors
)
moment_sim = second_moment_analyser(exp290.omega, sim290_tot.convolved, s_qw_err=None)
moment_incoh = second_moment_analyser(exp290.omega, sim290_incoh.convolved)
moment_coh = second_moment_analyser(exp290.omega, sim290_coh.convolved)

fig = plt.figure(figsize=(fp.COLUMN_WIDTH, 1.37))
gs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.5)

ax1 = fig.add_subplot(gs[0, 0])



prop_sq = 1 / (2 * moment_exp**0.5)
ax1.errorbar(
    mid_q,
    moment_exp**0.5,
    prop_sq * err_exp,
    linestyle="",
    marker=".",
    color=fp.colors[0],
    zorder=10,
    label = "Experiment"
)
ax1.plot(
    mid_q,
    moment_incoh**0.5,
    linestyle="-",
    marker="",
    color=fp.colors[1],
    label = "Incoherent"
)
ax1.plot(
    mid_q,
    moment_sim**0.5,
    linestyle="-",
    marker="",
    color=fp.colors[2],
    label = "Total"
)

ax1.plot(
    mid_q,
    moment_coh**0.5,
    linestyle="-",
    marker="",
    color=fp.colors[3],
    label = "Coherent"
)

ax1.set_xlabel(r"$Q$ / $\mathrm{Å}^{-1}$")
ax1.set_ylabel("$m^{0.5}_2$  / meV")
ax1.set_ylim(None, 0.25)
ax1.set_yticks([0.15, 0.2, 0.25])

fig.legend(loc='outside upper center', ncol=4, bbox_to_anchor=(0.5, 1.05))

for i, ax in enumerate([ax1]):
    ax.set_xlim(0.4, 1.9)


for i, ax in enumerate([ax1]):
    print(ax.get_window_extent().x1 - ax.get_window_extent().x0)
    print(ax.get_window_extent().y1 - ax.get_window_extent().y0)


plt.savefig(paths.figures / 'si_degenne.pdf', bbox_inches='tight', pad_inches=0.01)
plt.close()
