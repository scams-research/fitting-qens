import pickle
import matplotlib.pyplot as plt
from dynesty.utils import Results
import corner

import paths
import _fig_params as fp

names = ["model_nogauss_oneexp", 'model_nogauss_twoexp', 'model_nogauss_oneexpgauss', 'model_nogauss_twoexpgauss', 'model_gauss_oneexp', 'model_gauss_twoexp', 'model_gauss_oneexpgauss', 'model_gauss_twoexpgauss']
max = 0

for i in names:
    with open(paths.data / f'{i}.pkl', 'rb') as f:
        res = Results(pickle.load(f))
        if res.logz[-1] > max:
            max = res.logz[-1]
            max_i = i
for i in names:
    with open(paths.data / f'{i}.pkl', 'rb') as f:
        res = Results(pickle.load(f))
    with open(paths.output / f'{i}_evidence.txt', 'w') as f:
        if i == max_i:
            f.write(r"$\mathbf{" + f"{res.logz[-1]:.1f}" + r"}\;\;\;" + f"[{res.samples_equal().shape[1]}]$" + r"\unskip")
        else:   
            f.write(fr"${res.logz[-1]:.1f}\;\;\;[{res.samples_equal().shape[1]}]$" + r"\unskip")


with open(paths.data / f'{max_i}.pkl', 'rb') as f:
    res = Results(pickle.load(f))

with open(paths.output / f'best_raf_evidence.txt', 'w') as f:
    f.write(fr"${res.logz[-1]:.1f}$" + r"\unskip")

samples = res.samples_equal()

labels = ["$D_s$  / ps$^{-1}$", "$D_t$  / ps$^{-1}$", r"$A_{\mathcal{N}}$", r"$\sigma$  / ps$^{-1}$", "$A_2$", r"$\tau_1$  / ps", "$A_3$", r"$\tau_2$  / ps"]

fig = plt.figure(figsize=(fp.PAGE_WIDTH, fp.PAGE_WIDTH))
corner.corner(samples, labels=labels, show_titles=True, title_fmt=".2f", fig=fig)
plt.savefig(paths.figures / "si_raf_corner.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close()