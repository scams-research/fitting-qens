import sys
import paths
import matplotlib.pyplot as plt
import numpy as np
import _fig_params as fp

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

heatmap = np.loadtxt(paths.data / "angular_density_2d.txt")
cutoff = np.linspace(3, 8, 35)
cutoff_mid = 0.5 * (cutoff[:-1] + cutoff[1:])
n_angle_bins = 46
angle_bins = np.linspace(0, 90, n_angle_bins + 1)
angle_mid = 0.5 * (angle_bins[:-1] + angle_bins[1:])

g_r_from_angles = np.sum(heatmap, axis=1) 
pheta_dist = np.sum(heatmap, axis=0)

fig = plt.figure(figsize = (fp.PAGE_WIDTH*0.4, fp.PAGE_WIDTH*0.403))

gs = plt.GridSpec(2, 2, figure=fig, hspace=0.1, wspace=0.1)
ax = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

remove_ax(ax[1])

ax[2].imshow(heatmap.T, extent=[cutoff[0], cutoff[-1], angle_bins[0], angle_bins[-1]], aspect='auto', origin='lower')
ax[2].set_xlabel(r'$r$ / Å')
ax[2].set_ylabel(r'$\theta$ / $^\circ$')
ax[2].set_ylim([0, 90])
ax[2].set_yticks([0, 30, 60, 90])

ax[0].plot(cutoff_mid, g_r_from_angles / 4.1, color=fp.colors[0]) # Normalise to RDF
ax[0].set_xticks([])  
ax[0].set_ylabel('$g(r)$')
ax[0].set_yticks([0, 1, 2])

ax[3].plot(pheta_dist / np.trapezoid(pheta_dist), angle_mid,color=fp.colors[0])
ax[3].set_xlabel(r'$g(\theta)$')
ax[3].set_yticks([])

for i, ax in enumerate([ax[0], ax[2], ax[3]]):
    print(ax.get_window_extent().x1 - ax.get_window_extent().x0)
    print(ax.get_window_extent().y1 - ax.get_window_extent().y0)
    
plt.savefig(paths.figures / "si_angular_density.pdf", bbox_inches = 'tight')
plt.close()