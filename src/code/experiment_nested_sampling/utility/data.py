import sys
import paths

import numpy as np
import scipp as sc

sys.path.append(str(paths.code))
from plet_data import PletData


omega_lim = 1.25
q_lim_low = 0.6
q_lim_high = 1.8

data360 = PletData(
    str(paths.data) + "/benzene_290_360_inc.nxspe",
    3.60,
    omega_lims=[-omega_lim, omega_lim],
    q_lims=[q_lim_low, q_lim_high],
)
empty360 = PletData(
    str(paths.data) + "/empty_360_inc.nxspe",
    3.60,
    omega_lims=[-omega_lim, omega_lim],
    q_lims=[q_lim_low, q_lim_high],
)
res360 = PletData(
    str(paths.data) + "/benzene_260_360_inc.nxspe",
    3.60,
    omega_lims=[-omega_lim, omega_lim],
    q_lims=[q_lim_low, q_lim_high],
)


q_bins360 = sc.linspace(
    "q", data360.q.min().values, data360.q.max().values, 31, unit=sc.Unit("1/angstrom")
)
data360.bin_q(q_bins360)
empty360.bin_q(q_bins360)
res360.bin_q(q_bins360)

omega_masked360 = data360.omega_mid.values[
    np.invert(data360.data.masks["omega"].values)
]
q_mid360 = data360.q_mid.values[np.invert(data360.data.masks["q"].values)]
n_bkg1 = q_mid360.shape[0]


data360.data -= empty360.data
res360.data -= empty360.data
data360.data /= 2

q_lim_low = 0.55
q_lim_high = 1.58

data197 = PletData(
    str(paths.data) + "/benzene_290_197_inc.nxspe",
    1.97,
    omega_lims=[-omega_lim, omega_lim],
    q_lims=[q_lim_low, q_lim_high],
)
empty197 = PletData(
    str(paths.data) + "/empty_197_inc.nxspe",
    1.97,
    omega_lims=[-omega_lim, omega_lim],
    q_lims=[q_lim_low, q_lim_high],
)
res197 = PletData(
    str(paths.data) + "/benzene_260_197_inc.nxspe",
    1.97,
    omega_lims=[-omega_lim, omega_lim],
    q_lims=[q_lim_low, q_lim_high],
)


q_bins197 = sc.linspace(
    "q", data197.q.min().values, data197.q.max().values, 30, unit=sc.Unit("1/angstrom")
)
data197.bin_q(q_bins197)
empty197.bin_q(q_bins197)
res197.bin_q(q_bins197)

omega_masked197 = data197.omega_mid.values[
    np.invert(data197.data.masks["omega"].values)
]
q_mid197 = data197.q_mid.values[np.invert(data197.data.masks["q"].values)]

data197.data -= empty197.data
res197.data -= empty197.data
