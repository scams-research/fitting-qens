import h5py
import scipp as sc
import numpy as np

q_values = np.array(
    [
        0.44168133382170255,
        0.4836718818802158,
        0.5252494850177617,
        0.5667096154060947,
        0.6078712434417826,
        0.6487126881042931,
        0.689059119666155,
        0.7291972609506315,
        0.7688009107323014,
        0.8081515132390025,
        0.8470764389591491,
        0.8854097486319593,
        0.9234238587847176,
        0.960809830116343,
        0.9978333468728936,
        1.0343312759325145,
        1.0702843927798718,
        1.105540290816107,
        1.1403495168734994,
        1.1745580878433999,
        1.2080214609561863,
        1.2409774444254584,
        1.2731582061631168,
        1.3047924394761778,
        1.3357394008561196,
        1.365982789642908,
        1.395395800040148,
        1.42418745626771,
        1.4521237695746756,
        1.4794032699998334,
        1.505903525914927,
        1.5316105788683587,
        1.5564177547948364,
        1.580501346447981,
        1.6037524423088392,
        1.6260752149772537,
        1.6476283017392186,
        1.6683135349991718,
        1.6880463869882285,
        1.7069670896628295,
        1.724921875155287,
        1.742038339552496,
        1.7582372219489453,
        1.7735099899245481,
        1.7877958583340137,
        1.8011471901015434,
        1.8136025596776764,
        1.82510265230754,
        1.8356414105595806,
        1.8451787261224841,
        1.8537823761437875,
    ]
)


class IRISData:
    def __init__(self, file, omega_lims=[-0.5, 0.5]):
        file_open = h5py.File(file)

        I = file_open["mantid_workspace_1"]["workspace"]["values"][:]
        dI = file_open["mantid_workspace_1"]["workspace"]["errors"][:]
        omega = file_open["mantid_workspace_1"]["workspace"]["axis1"][:]
        q = q_values[
            file_open["mantid_workspace_1"]["workspace"]["axis2"][:].astype(int)
        ]

        file_open.close()

        I_sc = sc.array(
            values=I,
            variances=dI**2,
            dims=["q", "omega"],
            unit=(sc.units.angstrom / sc.units.meV),
        )
        omega_sc = sc.midpoints(
            sc.array(values=omega, dims=["omega"], unit=sc.units.meV)
        )
        q_sc = sc.array(values=q, dims=["q"], unit=(1 / sc.units.angstrom).unit)
        self.data = sc.DataArray(data=I_sc, coords={"omega": omega_sc, "q": q_sc})
        self.data.masks["omega"] = (
            self.data.coords["omega"] < omega_lims[0] * sc.units.meV
        ) | (self.data.coords["omega"] > omega_lims[1] * sc.units.meV)

    @property
    def masked(self):
        return self.data.data.values[:, np.invert(self.data.masks["omega"].values)]

    @property
    def errors(self):
        return np.sqrt(
            self.data.data.variances[:, np.invert(self.data.masks["omega"].values)]
        )

    def plot(self):
        return sc.plot(self.data)

    @property
    def omega(self):
        return self.data.coords["omega"].values[
            np.invert(self.data.masks["omega"].values)
        ]

    @property
    def q(self):
        return self.data.coords["q"].values

    def bin_q(self, q_bins):
        # Binning factor ensures every bin is normalised to have the same amount of summed points.
        binning_factor = np.histogram(self.q, bins=q_bins.values)[0]
        self.binning_factor = sc.array(values=binning_factor, dims=["q"])
        self.data = self.data.hist({"q": q_bins}).transpose() / self.binning_factor
