import numpy as np
import iris
import iris.coords
import iris.util
import gc
from salmon.utils import calculus

class WaveProcessor:
    def __init__(self):
        # Physical parameters
        self.g = 9.8
        self.beta = 2.3e-11
        self.radea = 6.371e6
        self.spd = 86400.
        self.ww = 2 * np.pi / self.spd
        self.R = 6371229.0 # Earth radius consistent with UM
        self.deg2rad = np.pi / 180.
        
        # Methodology parameters
        self.kmin = 2 
        self.kmax = 40
        self.pmin = 2.0 # days
        self.pmax = 30.0 # days
        self.y0 = 6.0 # meridional trapping scale (degrees)
        self.wave_names = np.array(['Kelvin', 'WMRG', 'R1', 'R2'])
        
        self.y0real = 2 * np.pi * self.radea * self.y0 / 360.0
        self.ce = 2 * self.y0real ** 2 * self.beta
        self.g_on_c = self.g / self.ce
        self.c_on_g = self.ce / self.g

    def uz_to_qr(self, u_data, z_data):
        """Transform (u, z) to (q, r) Riemann variables."""
        q = z_data * self.g_on_c + u_data
        r = z_data * self.g_on_c - u_data
        return q, r

    def filt_project(self, qf, rf, vf, lats):
        """Filter and project spectral coefficients onto wave modes."""
        nf, nz, nlat, nk = qf.shape
        
        # Frequencies and wavenumbers
        f = np.fft.fftfreq(nf, 0.25) # assuming 6-hourly data (0.25 days)
        fmin = np.where(f >= 1./self.pmax)[0][0]
        fmax = np.where(f > 1./self.pmin)[0][0] - 1
        
        f1p, f2p = fmin, fmax + 1
        f1n, f2n = nf - fmax, nf - fmin + 1
        k1p, k2p = self.kmin, self.kmax + 1
        k1n, k2n = nk - self.kmax, nk - self.kmin + 1
        
        # Parabolic cylinder functions
        spi2 = np.sqrt(2 * np.pi)
        dsq = np.array([spi2, spi2, 2 * spi2, 6 * spi2])
        d = np.zeros([dsq.size, nlat])
        y = lats / self.y0
        ysq = y ** 2
        d[0, :] = np.exp(-ysq / 4.0)
        d[1, :] = y * d[0, :]
        d[2, :] = (ysq - 1.0) * d[0, :]
        d[3, :] = y * (ysq - 3.0) * d[0, :]
        dlat = np.abs(lats[0] - lats[1])
        
        # Spectral coefficients for modes
        qf_Kel = np.zeros([nf, nz, nk], dtype=complex)
        qf_mode = np.zeros([4, nf, nz, nk], dtype=complex)
        rf_mode = np.zeros([4, nf, nz, nk], dtype=complex)
        vf_mode = np.zeros([4, nf, nz, nk], dtype=complex)
        
        # Transpose qf, rf, vf to [f, z, k, lat] for summation
        qft = np.transpose(qf, [0, 1, 3, 2])
        rft = np.transpose(rf, [0, 1, 3, 2])
        vft = np.transpose(vf, [0, 1, 3, 2])
        
        for m in range(4):
            # Projections
            weights = d[m, :] * dlat / (dsq[m] * self.y0)
            if m == 0:
                qf_Kel[f1n:f2n, :, k1p:k2p] = np.sum(qft[f1n:f2n, :, k1p:k2p, :] * weights, axis=-1)
                qf_Kel[f1p:f2p, :, k1n:k2n] = np.sum(qft[f1p:f2p, :, k1n:k2n, :] * weights, axis=-1)
            
            # General modes
            for f_slice, k_slice in [(slice(f1n, f2n), slice(k1n, k2n)), (slice(f1p, f2p), slice(k1p, k2p))]:
                qf_mode[m, f_slice, :, k_slice] = np.sum(qft[f_slice, :, k_slice, :] * weights, axis=-1)
                rf_mode[m, f_slice, :, k_slice] = np.sum(rft[f_slice, :, k_slice, :] * weights, axis=-1)
                vf_mode[m, f_slice, :, k_slice] = np.sum(vft[f_slice, :, k_slice, :] * weights, axis=-1)
        
        # Reconstruct waves in spectral space
        uf_wave = np.zeros([4, nf, nz, nlat, nk], dtype=complex)
        zf_wave = np.zeros([4, nf, nz, nlat, nk], dtype=complex)
        vf_wave = np.zeros([4, nf, nz, nlat, nk], dtype=complex)
        
        for w, wname in enumerate(self.wave_names):
            for j in range(nlat):
                if wname == 'Kelvin':
                    uf_wave[w, :, :, j, :] = 0.5 * qf_Kel * d[0, j]
                    zf_wave[w, :, :, j, :] = 0.5 * qf_Kel * d[0, j] * self.c_on_g
                elif wname == 'WMRG':
                    uf_wave[w, :, :, j, :] = 0.5 * qf_mode[1, :, :, :] * d[1, j]
                    zf_wave[w, :, :, j, :] = 0.5 * qf_mode[1, :, :, :] * d[1, j] * self.c_on_g
                    vf_wave[w, :, :, j, :] = vf_mode[0, :, :, :] * d[0, j]
                elif wname == 'R1':
                    uf_wave[w, :, :, j, :] = 0.5 * (qf_mode[2, :, :, :] * d[2, j] - rf_mode[0, :, :, :] * d[0, j])
                    zf_wave[w, :, :, j, :] = 0.5 * (qf_mode[2, :, :, :] * d[2, j] + rf_mode[0, :, :, :] * d[0, j]) * self.c_on_g
                    vf_wave[w, :, :, j, :] = vf_mode[1, :, :, :] * d[1, j]
                elif wname == 'R2':
                    uf_wave[w, :, :, j, :] = 0.5 * (qf_mode[3, :, :, :] * d[3, j] - rf_mode[1, :, :, :] * d[1, j])
                    zf_wave[w, :, :, j, :] = 0.5 * (qf_mode[3, :, :, :] * d[3, j] + rf_mode[1, :, :, :] * d[1, j]) * self.c_on_g
                    vf_wave[w, :, :, j, :] = vf_mode[2, :, :, :] * d[2, j]
                    
        return uf_wave, zf_wave, vf_wave

    def makes_5d_cube(self, data, time_coord, press_coord, lat_coord, lon_coord):
        """Create a 5D cube with realization, wave_name, time, pressure, lat, lon dimensions."""
        var_cube = iris.cube.Cube(data)
        wave_coord = iris.coords.DimCoord(range(len(self.wave_names)), long_name='wave_name')
        
        # Mapping wave names to indices via attributes
        wave_attrs = {name: i for i, name in enumerate(self.wave_names)}
        wave_coord.attributes = wave_attrs
        
        var_cube.add_dim_coord(wave_coord, 0)
        var_cube.add_dim_coord(time_coord, 1)
        var_cube.add_dim_coord(press_coord, 2)
        var_cube.add_dim_coord(lat_coord, 3)
        var_cube.add_dim_coord(lon_coord, 4)
        
        return var_cube

    def derivative(self, cube, axisname):
        """Calculate spatial derivative of a cube."""
        if axisname == 'latitude':
            lats = cube.coord('latitude').points
            axis_index = cube.coord_dims('latitude')[0]
            dlat = np.diff(lats) * self.deg2rad
            dy = self.R * np.sin(dlat) # This is actually R * dlat for simple grid
            # Correcting dy calculation:
            dy = self.R * dlat
            dcube = calculus.differentiate(cube, 'latitude')
            dcube /= iris.util.broadcast_to_shape(dy, dcube.shape, (axis_index,))
            return dcube
        elif axisname == 'longitude':
            lats = cube.coord('latitude').points
            lons = cube.coord('longitude').points
            lat_index = cube.coord_dims('latitude')[0]
            lon_index = cube.coord_dims('longitude')[0]
            dlon = (lons[1] - lons[0]) * self.deg2rad
            dx = np.array([self.R * np.cos(lat * self.deg2rad) * dlon for lat in lats])
            dcube = calculus.differentiate(cube, 'longitude')
            dcube /= iris.util.broadcast_to_shape(dx, dcube.shape, (lat_index,))
            return dcube
        return cube
