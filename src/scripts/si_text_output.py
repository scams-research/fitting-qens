import numpy as np
import pickle
from scipy.special import gamma
from dynesty.utils import Results  
import paths

with open(paths.data / "lit_approach.pkl", "rb") as f:
    result = pickle.load(f)
res = Results(result)
with open(paths.output / "lit_approach_evidence.txt", "w") as f:
    f.write(fr"${res.logz[-1]:.1f}\unskip$")

samples = res.samples_equal().mean(0)
t1 = samples[0] / samples[1] * gamma(1 / samples[1])
t2 = samples[2] / samples[3] * gamma(1 / samples[3])
with open(paths.output / "lit_approach_t90.txt", "w") as f:
    f.write(r"\qty{" + fr"{t1:.2f}" + r"}{\pico\second}\unskip")
with open(paths.output / "lit_approach_t0.txt", "w") as f:
    f.write(r"\qty{" + fr"{t2:.2f}" + r"}{\pico\second}\unskip")
D_t = 1 / (6 * t2)
e = 4*(t1 - 0.25* t2) / 3
D_s = ((1/e) - 2 * D_t) / 4
with open(paths.output / "lit_approach_Ds.txt", "w") as f:
    f.write(r"\qty{" + fr"{D_s:.2f}" + r"}{\per\pico\second}\unskip")
with open(paths.output / "lit_approach_Dt.txt", "w") as f:
    f.write(r"\qty{" + fr"{D_t:.2f}" + r"}{\per\pico\second}\unskip")
