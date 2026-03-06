import h5py
import scipp as sc
import numpy as np

q = {1.24: sc.arange('q', 0.03175, 1.70968+(0.00675/2), 0.00675, unit=sc.Unit('1/angstrom')),
     1.97: sc.arange('q', 0.04001, 2.15495+(0.00851/2), 0.00851, unit=sc.Unit('1/angstrom')),
     3.60: sc.arange('q', 0.05409, 2.91309+(0.0115/2), 0.0115, unit=sc.Unit('1/angstrom')),
     8.61: sc.arange('q', 0.08365, 4.50511+(0.01779/2), 0.01779, unit=sc.Unit('1/angstrom')),
     3.62: sc.arange('q', 0.13002, 2.88972+(0.0115/2), 0.0115, unit=sc.Unit('1/angstrom')) # only vanadium 
     }

omega = {1.24: sc.arange('omega', -0.992, 0.992+(0.0062/2), 0.0062, unit=sc.Unit('meV')),
         1.97: sc.arange('omega', -1.576, 1.576+(0.00985/2), 0.00985, unit=sc.Unit('meV')),
         3.60: sc.arange('omega', -2.88, 2.88+(0.018/2), 0.018, unit=sc.Unit('meV')),
         8.61: sc.arange('omega', -6.88, 6.88+(0.04305/2), 0.04305, unit=sc.Unit('meV')),
         3.62: sc.arange('omega', -2.7, 2.7+(0.018/2), 0.018, unit=sc.Unit('meV')),  # only vanadium
         }

class PletData:
    def __init__(self, filename, energy, omega_lims = [-1,1], q_lims = [0.4,1.7]):
        """
        Load PLET data from a nxspe file.

        :filename: str
            Path to the nxspe file.
        :energy: float
            Incident energy in meV.
        """
        self.energy = energy    
        if 'inc' in str(filename):
            self.data_type = 'inc'
        elif 'coh' in str(filename):
            self.data_type = 'coh'
        for key in q.keys():
            if str(key) in str(filename):
                self.energy = key
                

   
        self.q = q[self.energy]
        self.omega = omega[self.energy]

        file = h5py.File(filename, 'r')
        id = list(file.keys())[0]

        self.data = sc.DataArray(data=sc.array(dims=['q', 'omega'], 
                                               values=file[id]['data']['data'][:],
                                               variances=file[id]['data']['error'][:] ** 2),
                                 coords={'q': self.q, 'omega': self.omega})
        
        self.q_mid = (self.q[:-1] + self.q[1:]) / 2
        self.omega_mid = self.omega[:-1] + self.omega[1:] / 2

        self.data.masks['omega'] = (
            self.omega_mid < omega_lims[0] * sc.units.meV) | (
                self.omega_mid > omega_lims[1] * sc.units.meV)

        self.data.masks['q'] = (
            self.q_mid < q_lims[0] * sc.Unit('1/angstrom')) | (
                self.q_mid > q_lims[1] * sc.Unit('1/angstrom'))
        
        self.q_lims = q_lims
    
    def bin_q(self, q_bins):
        # Binning factor ensures every bin is normalised to have the same amount of summed points.
        binning_factor = np.histogram(self.q.values, bins=q_bins.values)[0]
        self.binning_factor = sc.array(values = binning_factor, dims = ['q'])
        self.data = sc.rebin(self.data, q=q_bins)
        self.data = self.data / self.binning_factor.values[1,np.newaxis]

        self.q = self.data.coords['q']
        self.q_mid = (self.q[:-1] + self.q[1:]) / 2
        self.data.masks['q'] = (
            self.q_mid < self.q_lims[0] * sc.Unit('1/angstrom')) | (
                self.q_mid > self.q_lims[1] * sc.Unit('1/angstrom'))

    @property
    def masked(self):
        return self.data.data.values[np.invert(self.data.masks['q'].values)][:,np.invert(self.data.masks['omega'].values)]
    
    @property
    def errors(self):
        return np.sqrt(self.data.data.variances[np.invert(self.data.masks['q'].values)][:,np.invert(self.data.masks['omega'].values)])
    
    def plot(self):
        return sc.plot(self.data)
    