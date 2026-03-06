import numpy as np
from dynesty.utils import Results
import pickle
import matplotlib.pyplot as plt
import paths
import _fig_params as fp

with open(paths.data / "rotation_only_model.pkl", "rb") as f:
    aniso = pickle.load(f)
aniso_q = Results(aniso)
aniso_samples = aniso_q.samples_equal()

with open(paths.data / "model_gauss_twoexp.pkl", "rb") as f:
    result = pickle.load(f)
gt = Results(result)
samples = gt.samples_equal()
D_perp, D_parr = samples[:, 1], samples[:, 0]

fig = plt.figure(figsize=(fp.COLUMN_WIDTH * 2 / 3, 0.8))
gs = plt.GridSpec(1, 2, figure=fig, hspace=0.4)
axes = [fig.add_subplot(gs[i]) for i in range(2)]
ax1, ax2 = axes

ax1.hist(
    aniso_samples[:, 1] / 0.658212,
    density=True,
    bins=fp.NBINS,
    color=fp.colors[1],
    alpha=0.6,
)
ax1.axvline(np.mean(D_perp), color="k", ls="--")
ax1.set_xlabel(r"$D_t\,|\,S^{\mathrm{sim}}_{\mathrm{inc},R}$ / ps$^{-1}$")
ax1.set_xticks([0.04, 0.06])

ax2.hist(
    aniso_samples[:, 0] / 0.658212,
    density=True,
    bins=fp.NBINS,
    color=fp.colors[1],
    alpha=0.6,
)
ax2.axvline(np.mean(D_parr), color="k", ls="--")
ax2.set_xlabel(r"$D_s\,|\,S^{\mathrm{sim}}_{\mathrm{inc},R}$ / ps$^{-1}$")
ax2.set_xticks([0.22, 0.26])

# Add panel labels below x-axis
for i, ax in enumerate(axes):
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.text(0.01, 1.04, fp.ALPHABET[i], transform=ax.transAxes, ha="center", va="bottom")

for i, ax in enumerate(axes):
    print(ax.get_window_extent().x1 - ax.get_window_extent().x0)
    print(ax.get_window_extent().y1 - ax.get_window_extent().y0)

plt.savefig(paths.figures / 'rot_sim_ds.pdf', bbox_inches="tight", pad_inches=0.01)
plt.close()
