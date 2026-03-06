import numpy as np
import scipp as sc
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from scipy import constants

class MDANSEdata:
    def __init__(self, file):
        self.data = np.loadtxt(file, delimiter=",", skiprows=6)

    def parse(self, energy_lim=0.4):
        """
        extract energy, Q, and S(Q, ω) data from MDANSE csv output

        Parameters
        ----------
        filename : str
            Path to the MDANSE CSV file.
        energy_lim : float, optional
            Energy range limit (±meV) to crop the energy axis

        Returns
        -------
        energy : ndarray
            Energy transfer values (meV), cropped to ±energy_lim.
        s_qw : ndarray
            S(Q, ω) values with shape (n_q, n_energy).
        q : ndarray
            Momentum transfer (Q) values. A^-1
        """
        energy = self.data[0, 1:]
        # No zero q vector
        s_qw = self.data[1:, 1:]

        # -0.4 to 0.4 mev energy range
        lims = len(energy) - np.abs(energy - energy_lim).argmin()

        self.energy = energy[lims:-lims]
        self.s_qw = s_qw[:, lims:-lims]
        self.q = self.data[1:, 0]

        self.sc_qw = sc.DataArray(
            data=sc.array(dims=["q", "omega"], values=self.s_qw),
            coords={
                "q": sc.array(dims=["q"], values=self.q),
                "omega": sc.array(dims=["omega"], values=self.energy),
            },
        )

    def scippbin(self, bins, ignore_bin_err=False):
        """
        Bin S(Q, ω) data along the Q-axis using Scipp.

        Parameters
        ----------
        bins : int
            Number of Q bins to apply.
        """
        if not ignore_bin_err:
            assert np.isclose(self.q[0], 0.441, atol=1e-2), (
                "AssertionError: Q binning incorrect! Ensure MDANSE is set to use 4.41,18.68,0.2798"
            )
            assert np.isclose(self.q[-1], 1.868, atol=1e-2), (
                "AssertionError: Q binning incorrect! Ensure MDANSE is set to use 4.41,18.68,0.2798"
            )

        self.q_bins = sc.linspace(
            "q", self.sc_qw.coords["q"].min(), self.sc_qw.coords["q"].max(), bins + 1
        )

        # Binning factor ensures every bin is normalised to have the same amount of summed points.
        binning_factor = np.histogram(self.q, bins=self.q_bins.values)[0]
        self.binning_factor = sc.array(values=binning_factor, dims=["q"])
        self.binned = (
            self.sc_qw.hist({"q": self.q_bins}).transpose() / self.binning_factor
        )

    def convolve(self, resolution, norm_factor, binned=True):
        """
        Convolves simulation with resolution function

        Parameters
        ----------
        resolution: IRISdata class object
            1D resolution function

        binned: Bool
            True if scippbin has been called previously
        """
        if binned:
            assert self.binned.values.shape[0] == resolution.masked.shape[0]

            bin_conv_trimmed = np.zeros(
                (self.binned.values.shape[0], len(resolution.omega))
            )

            for i in range(self.binned.values.shape[0]):
                res_i = resolution.masked[i] / norm_factor[i]
                conv_full = np.convolve(
                    res_i / res_i.sum(), self.binned.values[i], mode="full"
                )
                trim = (len(conv_full) - len(resolution.omega)) // 2
                bin_conv_trimmed[i] = conv_full[trim:-trim] if trim > 0 else conv_full

            self.convolved = bin_conv_trimmed

        if not binned:
            assert self.sc_qw.values.shape[0] == resolution.masked.shape[0]

            conv_trimmed = np.zeros((self.sc_qw.values.shape[0], len(resolution.omega)))

            for i in range(self.sc_qw.values.shape[0]):
                res_i = resolution.masked[i] / norm_factor[i]
                conv_full = np.convolve(
                    res_i / res_i.sum(), self.sc_qw.values[i], mode="full"
                )
                trim = (len(conv_full) - len(resolution.omega)) // 2
                conv_trimmed[i] = conv_full[trim:-trim] if trim > 0 else conv_full

            self.convolved = conv_trimmed


# -------------------- General Analysis functions -----------------------------------


def second_moment_analyser(energy, s_qw, s_qw_err=None):
    """
    Calculate the second moment of S(q,ω).

    Parameters
    ----------
    energy: ndarray
         Array of energy values.
    s_qw: ndarray
        2D array where each row corresponds to the scattering function at a specific energy.

    Returns
    ----------
        second_moment: ndarray
            Array of second moment values as a function of q
    """
    s_qe = np.zeros(s_qw.shape[0])
    e2s_qe = np.zeros(s_qw.shape[0])

    if s_qw_err is not None:
        err_total = np.zeros(s_qw.shape[0])

    for i, sqw in enumerate(s_qw):
        s_qe[i] = trapezoid(y=sqw, x=energy)
        e2s_qe[i] = trapezoid(y=(energy**2 * sqw), x=energy)
        if s_qw_err is not None:
            _, s_qe_err = trap_uncertainty(x=energy, y=sqw, var_y=s_qw_err[i])
            _, e2s_qe_err = trap_uncertainty(
                x=energy, y=(energy**2 * sqw), var_y=(energy**2 * s_qw_err[i])
            )
            err_total[i] = np.sqrt(
                (e2s_qe[i] / s_qe[i] ** 2) ** 2 * s_qe_err**2
                + (1 / s_qe[i]) ** 2 * e2s_qe_err**2
            )

    second_moment = e2s_qe / s_qe

    if s_qw_err is not None:
        return second_moment, err_total
    else:
        return second_moment


def limit_integral(energy, s_qw, limits, s_qw_err=None):
    """
    Numerical integration of S(q,ω) with limits

    Parameters
    ----------
    energy: ndarray
         Array of energy values.
    s_qw: ndarray
        2D array where each row corresponds to the scattering function at a specific energy.
    limits: ndarray
        2 limits i.e. [-0.02,0.02]

    Returns
    ----------
        integral: ndarray
            Integral within limits as a function of q
    """
    lims = np.where((limits[0] < energy) & (energy < limits[1]))[0]
    s_qw_trimmed = s_qw[:, lims]

    if s_qw_err is not None:
        s_qw_err_trimmed = s_qw_err[:, lims]
        int_err = np.zeros(s_qw_trimmed.shape[0])

    integral = np.zeros(s_qw_trimmed.shape[0])

    for i, sqw in enumerate(s_qw_trimmed):
        integral[i] = trapezoid(y=sqw, x=energy[lims])
        if s_qw_err is not None:
            _, int_err[i] = trap_uncertainty(
                x=energy[lims], y=sqw, var_y=s_qw_err_trimmed[i]
            )

    if s_qw_err is not None:
        return integral, int_err
    else:
        return integral


def trap_uncertainty(x, y, var_y):
    """
    Compute trapezoidal integral and propagated uncertainty for unevenly spaced data.
    Based off:
    https://stats.stackexchange.com/questions/214850/propagate-errors-in-measured-points-to-simpsons-numerical-integral

    Parameters
    ----------
    x : array-like
        The x-values (must be in increasing order).
    y : array-like
        The corresponding y-values.
    var_y : array-like
        variances of y-values.

    Returns
    -------
    integral : float
        The computed trapezoidal integral.
    sigma_integral : float
        The propagated uncertainty in the integral.
    """
    N = len(x)
    h = np.diff(x)

    # Trapezoidal integral
    integral = np.sum(0.5 * h * (y[:-1] + y[1:]))

    # Error propagation
    sigma_I_sq = 0.0
    sigma_I_sq += (0.5 * h[0]) ** 2 * var_y[0]
    sigma_I_sq += (0.5 * h[-1]) ** 2 * var_y[-1]

    for j in range(1, N - 1):
        coeff = 0.5 * (h[j - 1] + h[j])
        sigma_I_sq += coeff**2 * var_y[j]

    sigma_integral = np.sqrt(sigma_I_sq)

    # Assert solution is similar to scipy implementation
    assert np.isclose(integral, trapezoid(x=x, y=y))

    return integral, sigma_integral


# -------------------- Rotational simulation functions - benzene------------------------------


def normalized(a, axis=-1, order=2):
    """
    normalise vector over final axis, shamelessly stolen from stack overflow
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def calc_autocorr(vecs_t0, vecs_t):
    """
    Calculates rotational autocorrelation based on the second order legendre polynomial:
    G_pheta(x) = <0.5*[3*cos^2(a(t)) - 1]>
    """
    cos_at = np.einsum("ij,ij->i", vecs_t0, vecs_t)
    autocorr_x = 0.5 * (3 * cos_at**2 - 1)
    autocorr = np.mean(autocorr_x)
    return autocorr


def vector_maker(atom_positions):
    """
    Choses bezene vectors based on:

    #     0 - 1
    #   5       2
    #     4 - 3

    Parameters
    ----------
    atom_positions : ndarray
        Array of shape (num_molecules, num_atoms, 3) containing 3d atom positions

    Returns
    -------
    vec_0 : ndarray
        Vector in the plan of benzene
    vec_90 : ndarray
        Vector orthogonal to benzene plane
    """
    # Vector between 4th and 0th benzene, straight line down benzene
    vec_0 = normalized(atom_positions[:, 4] - atom_positions[:, 0])
    # Vector between 1st and 2nd benzene- within the plane of the benzene
    vec_1 = normalized(atom_positions[:, 1] - atom_positions[:, 2])
    # Orthogonal to vec_0
    vec_2 = normalized(atom_positions[:, 4] - atom_positions[:, 1])
    # Orthogonal vector to benzene plane
    vec_90 = normalized(np.cross(vec_0, vec_1))
    return vec_0, vec_90, vec_2


def rotational_ac_calc(
    universe,
    window_size=50,
    atom_types="type 1 2 3 4 5 6 7 8 9 10 11 12",
    N_frames=None,
):
    """
    Calculates rotational autocorrelation functions for benzene based on the second order legendre polynomial:

    G_pheta(x) = <0.5*[3*cos^2(a(t)) - 1]>

    where a(t) = a(t) dot a(0)

    Parameters
    ----------
    Universe : MDA universe
    window_size : int
        size of window to calculate correlation function over

    Returns
    -------
    aved_autocorr0: ndarray
        Average autocorrelation function over time for vector in plane
    ac_std0: ndarray
        standard deviation of autocorrelation function over time for vector in plane
    aved_autocorr90: ndarray
        Average autocorrelation function over time for vector orthogonal to plane
    ac_std90: ndarray
        standard deviation of autocorrelation function over time for vector orthogonal to plane

    """
    if N_frames is None:
        N_frames = len(universe.trajectory)

    rotational_autocorr0 = np.zeros(N_frames)
    rotational_autocorr90 = np.zeros(N_frames)
    rotational_autocorr2 = np.zeros(N_frames)

    universe.trajectory[0]
    benzs = universe.select_atoms(atom_types).fragments

    for ts in tqdm(range(N_frames)):
        universe.trajectory[ts]
        if ts % window_size == 0:
            atom_pos_t0 = np.array([frag.positions for frag in benzs])
            vec0_t0, vec90_t0, vec2_t0 = vector_maker(atom_pos_t0)

        atom_pos_t = np.array(
            [frag.positions for frag in benzs]
        )  # Shape (num_molecules, num_atoms, 3)
        vec0_t, vec90_t, vec2_t = vector_maker(atom_pos_t)

        rotational_autocorr0[ts] = calc_autocorr(vec0_t0, vec0_t)
        rotational_autocorr90[ts] = calc_autocorr(vec90_t0, vec90_t)
        rotational_autocorr2[ts] = calc_autocorr(vec2_t0, vec2_t)

    if N_frames % 2 != 0:
        reshaped_autocorr0 = rotational_autocorr0[:-1].reshape(-1, window_size)
        aved_autocorr0 = np.mean(reshaped_autocorr0, axis=0)
        ac_std0 = np.std(reshaped_autocorr0, axis=0)

        reshaped_autocorr90 = rotational_autocorr90[:-1].reshape(-1, window_size)
        aved_autocorr90 = np.mean(reshaped_autocorr90, axis=0)
        ac_std90 = np.std(reshaped_autocorr90, axis=0)

        reshaped_autocorr2 = rotational_autocorr2[:-1].reshape(-1, window_size)
        aved_autocorr2 = np.mean(reshaped_autocorr2, axis=0)
        ac_std2 = np.std(reshaped_autocorr2, axis=0)
    else:
        reshaped_autocorr0 = rotational_autocorr0.reshape(-1, window_size)
        aved_autocorr0 = np.mean(reshaped_autocorr0, axis=0)
        ac_std0 = np.std(reshaped_autocorr0, axis=0)

        reshaped_autocorr90 = rotational_autocorr90.reshape(-1, window_size)
        aved_autocorr90 = np.mean(reshaped_autocorr90, axis=0)
        ac_std90 = np.std(reshaped_autocorr90, axis=0)

        reshaped_autocorr2 = rotational_autocorr2.reshape(-1, window_size)
        aved_autocorr2 = np.mean(reshaped_autocorr2, axis=0)
        ac_std2 = np.std(reshaped_autocorr2, axis=0)

    return aved_autocorr0, ac_std0, aved_autocorr90, ac_std90, aved_autocorr2, ac_std2


def ps_to_mev(time):
    """
    Translates characteristic time from autocorrelation function to HWHM energy through the relation
    E = h_bar * omega

    """
    freq = 1 / (time * 1e-12)
    Joules = freq * (constants.h / (2 * np.pi))
    mev = Joules / 1.60218e-22
    return mev


# ----------------------- Modelling diffusion ------------


def line_to_D(slope):
    """
    gradient of HWHM vs Q**2 and unit conversion

    https://docs.mantidproject.org/v6.1.0/fitting/fitfunctions/FickDiffusion.html

    Also see Hall ross model paper
    """
    D_mev_A2 = slope
    D_J_A2 = D_mev_A2 * 1.60218e-22
    # Should this be reduced planks constant - Yes (bee 1988)
    D_A2_s = D_J_A2 / (constants.h / (2 * np.pi))
    D_m2_s = D_A2_s * (1e-20)
    return D_m2_s


def D_to_line(D_m2_s):
    """
    Diffusion coefficient to straight line gradient
    """
    D_A2_s = D_m2_s / (1e-20)
    D_JA2 = D_A2_s * (constants.h / (2 * np.pi))
    D_mevA2 = D_JA2 / 1.60218e-22
    slope = D_mevA2
    return slope


def flat_model(Q2, const):
    return const * np.ones_like(Q2)


def flat_line_fitter(Q2, HWHM):
    popt, pcov = curve_fit(flat_model, Q2, HWHM)
    Q2_fit = np.linspace(min(Q2), max(Q2), 100)
    print(popt)
    HWHM_fit = flat_model(Q2_fit, *popt)
    return Q2_fit, HWHM_fit
