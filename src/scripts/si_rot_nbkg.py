import paths
import pickle
import _fig_params as fp
import numpy as np
import matplotlib.pyplot as plt
from dynesty.utils import Results

res = {}
logz = np.zeros(13)
res_iso = {}
logz_iso = np.zeros(13)

for x in range(13):
    with open(paths.data / f"rotation_only_aniso_model_nbkg{x:02d}.pkl", "rb") as f:
        ns_res = pickle.load(f)
    ns_res_aniso = Results(ns_res)

    with open(paths.data / f"rotation_only_iso_model_nbkg{x:02d}.pkl", "rb") as f:
        ns_res = pickle.load(f)
    ns_res_iso = Results(ns_res)

    logz[x] = ns_res_aniso.logz[-1]
    logz_iso[x] = ns_res_iso.logz[-1]

    res[f"bkg_{x}"] = ns_res_aniso
    res_iso[f"bkg_{x}"] = ns_res_iso

fig, ax0 = plt.subplots(figsize=(fp.PAGE_WIDTH*0.5, 1.37))

ax0.plot(range(13), -logz, marker='o', ls='', color="#CC78BC")
ax0.plot(range(13), -logz_iso, marker='o', ls='', color="#029E73")

ax0.set_xlabel("No. background terms")
ax0.set_xlim([0, None])
ax0.set_xticks([0, 4, 8, 12])
ax0.set_ylabel(r"$- \ln\left[p\,\left(m \mid \mathbf{D}\right)\right]$")
ax0.set_yscale("log")
ax0.set_yticks([100, 1000])

print(ax0.get_window_extent().x1 - ax0.get_window_extent().x0)
print(ax0.get_window_extent().y1 - ax0.get_window_extent().y0)

plt.savefig(paths.figures / "si_rot_nbkg.pdf", bbox_inches="tight")
plt.close()
