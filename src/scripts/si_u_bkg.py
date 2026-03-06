import paths
import pickle
import sys
import _fig_params as fp
import numpy as np
import scipp as sc
import matplotlib.pyplot as plt
from dynesty.utils import Results

sys.path.append(str(paths.code))
from plet_data import PletData

def get_qmid(energy, q_lim_low, q_lim_high, q_bins, omega_lim=1.25):
    incident = {197: 1.97, 360: 3.60}
    data = PletData(
        paths.data / f"pLET_benzene_290K_{energy}_inc.nxspe",
        incident[energy],
        omega_lims=[-omega_lim, omega_lim],
        q_lims=[q_lim_low, q_lim_high],
    )
    q_bins = sc.linspace(
        "q", data.q.min().values, data.q.max().values, q_bins, unit=sc.Unit("1/angstrom")
    )
    data.bin_q(q_bins)
    q_mid = data.q_mid.values[np.invert(data.data.masks["q"].values)]
    return q_mid


q_mid197 = get_qmid(197, 0.55, 1.58, 30)
q_mid360 = get_qmid(360, 0.6, 1.8, 31)

with open( paths.data /'pLET_aniso_model.pkl', 'rb') as f:
    ns_res = pickle.load(f)
ns_res = Results(ns_res)

fig = plt.figure(figsize=(fp.PAGE_WIDTH*1/3, 1.37))
gs = plt.GridSpec(1, 2, figure=fig, wspace=0.4)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

n_global = 4

credible_intervals = [[16, 84], [2.5, 97.5], [0.15, 99.85]]
alphas = [0.8, 0.6, 0.4]

ax1.hist(ns_res.samples_equal()[:,n_global-1], bins = fp.NBINS, color = "#0173B2", alpha = 0.6, label = "3.60 meV")
ax1.set_xlabel(r"$\langle u ^2 \rangle$ / Å$^2$")
ax1.set_yticks([])
ax1.spines['left'].set_visible(False)

bkg360 = ns_res.samples_equal()[:,n_global+1:n_global+1+q_mid360.shape[0]]
for ci, alpha in zip(credible_intervals, alphas):
    y_lower, y_upper = np.percentile(bkg360, ci, axis=0)
    ax2.fill_between(q_mid360, y_lower, y_upper, color="#029E73", alpha=alpha, lw=0)

bkg197 = ns_res.samples_equal()[:,n_global+q_mid360.shape[0]+2:]
for ci, alpha in zip(credible_intervals, alphas):
    y_lower, y_upper = np.percentile(bkg197, ci, axis=0)
    ax2.fill_between(q_mid197, y_lower, y_upper, color="#CC78BC", alpha=alpha, lw=0)
ax2.set_xlabel(r'$Q$ / Å$^{-1}$')
ax2.set_ylabel('Background / arb. units')
ax2.set_ylim(0, None)

for ax in [ax1,ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

for ax, label in zip([ax1,ax2], [ 'a', 'b']):
    ax.text(0.01, 1.02, label, transform=ax.transAxes, ha='center', va='bottom')

for i, ax in enumerate([ax1, ax2]):
    print(ax.get_window_extent().x1 - ax.get_window_extent().x0)
    print(ax.get_window_extent().y1 - ax.get_window_extent().y0)

plt.savefig(paths.figures /'si_u_bkg.pdf', bbox_inches='tight')
plt.close()