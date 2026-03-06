import numpy as np
import pickle
from dynesty.utils import Results  
import paths
import sys

sys.path.append(str(paths.code))
from functions import line_to_D

D_kinisi = np.loadtxt(paths.data / "kinisi_D") * 1e4
with open(paths.output / "kinisi_D.txt", "w") as f:
    f.write(r"\qty{" + fr"{D_kinisi.mean():.3f} \pm {D_kinisi.std():.3f}" + r"}{\angstrom\squared\per\pico\second}\unskip")

with open(paths.data / "model_gauss_twoexp.pkl", "rb") as f:
    result = pickle.load(f)
res = Results(result)
samples = res.samples_equal()
D_s, D_t = samples[:, 0], samples[:, 1]
with open(paths.output / "D_rot_diff.txt", "w") as f:
    f.write(r"\qty{" + fr"{D_s.mean() - D_t.mean():.2f}" + r"}{\per\pico\second}\unskip")
with open(paths.output / "D_rot_ratio.txt", "w") as f:
    f.write(r"\num{" + fr"{D_s.mean() / D_t.mean():.1f}" + r"}\unskip")

with open(paths.data / "rotation_only_model.pkl", "rb") as f:
    result = pickle.load(f)
res = Results(result)
with open(paths.output / "aniso_rot_only_evidence.txt", "w") as f:
    f.write(fr"${res.logz[-1]:.1f}\unskip$")

with open(paths.data / "rotation_only_model_iso.pkl", "rb") as f:
    result = pickle.load(f)
res = Results(result)
with open(paths.output / "iso_rot_only_evidence.txt", "w") as f:
    f.write(fr"${res.logz[-1]:.1f}\unskip$")

with open(paths.data / "pLET_aniso_model.pkl", "rb") as f:
    ns_res = pickle.load(f)
ns_res = Results(ns_res)
samples = ns_res.samples_equal()
D_fick = line_to_D(samples[:, 0]) * 10**8
with open(paths.output / "D_fick_let.txt", "w") as f:
    f.write(r"\qty{" + fr"{D_fick.mean():.3f} \pm {D_fick.std():.3f}" + r"}{\angstrom\squared\per\pico\second}\unskip")
D_t = samples[:, 2] / 0.658212
D_s = samples[:, 1] / 0.658212
with open(paths.output / "D_ratio_let.txt", "w") as f:
    f.write(r"\num{" + fr"{D_s.mean() / D_t.mean():.0f}" + r"}\unskip")
with open(paths.output / "evidence_aniso_let.txt", "w") as f:
    f.write(fr"${ns_res.logz[-1]:.1f}\unskip$")

with open(paths.data / "pLET_iso_model.pkl", "rb") as f:
    ns_res = pickle.load(f)
ns_res = Results(ns_res)
with open(paths.output / "evidence_iso_let.txt", "w") as f:
    f.write(fr"${ns_res.logz[-1]:.1f}\unskip$")