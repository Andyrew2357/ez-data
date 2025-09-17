from . import xr_tools
from .extract_gap import mu, tl_model, lf_model, phase_correct
from .ezplt import errorplot, style_xr_xlabel

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import Tuple, Type

def _get_attr_parm(ds: xr.Dataset, p, type_: Type = float):
    if not p in ds.attrs:
        raise ValueError(f"'{p}' is not a dataset attribute")
    attr = ds.attrs[p]
    if isinstance(attr, type_):
        return attr
    elif 'value' in attr and isinstance(attr['value'], type_):
        return attr['value']
    else:
        raise ValueError(f"Cannot locate '{p}' value of the right type.")

class GappedState():
    def __init__(self, ds: xr.Dataset, # must be aligned to n and D properly
                 chi_g: float = None, var_chi_g: float = None, 
                 f: float = None, Cb: float = None, 
                 gamma: float = None, var_gamma: float = None,
                 label: str = '',
                 k_chi_r: str = 'Cex',
                 k_chi_i: str = 'Closs',
                 k_vt: str = 'Vtg', 
                 k_vb: str = 'Vbg',
                 k_x: str = 'n', 
                 k_y: str = 'D'):
        self.ds = ds
        self.chi_g = chi_g or _get_attr_parm(ds, 'chi_g', float)
        self.var_chi_g = var_chi_g or _get_attr_parm(ds, 'var_chi_g', float)
        self.f = f or _get_attr_parm(ds, 'f', float)
        self.cb = Cb or _get_attr_parm(ds, 'Cb', float)
        self.gamma = gamma or _get_attr_parm(ds, 'gamma', float)
        self.var_gamma = var_gamma or _get_attr_parm(ds, 'var_gamma', float)
        self.label = label
        self.k_chi_r, self.k_chi_i = k_chi_r, k_chi_i
        self.k_vt, self.k_vb = k_vt, k_vb
        self.k_x, self.k_y = k_x, k_y

    def apply_rough_x_mask(self, msk: xr.DataArray):
        self._rough_x_mask = msk

    def apply_rough_y_mask(self, msk: xr.DataArray):
        self._rough_y_mask = msk

    def refine_state_mask(self, nsigma: float = 2.0):
        self._state_fit = self.ds.where(
            self._rough_x_mask & self._rough_y_mask
        )[self.k_chi_r].ez.fit_peaks(self.k_x, self.k_y)
        self._refined_mask = (self._state_fit['peak_center'] - nsigma * \
                              self._state_fit['peak_sigma'] < self.ds[self.k_x]) & \
                             (self._state_fit['peak_center'] + nsigma * \
                              self._state_fit['peak_sigma'] > self.ds[self.k_x]) & \
                              self._rough_x_mask & self._rough_y_mask

    def apply_rough_band_mask(self, msk: xr.DataArray):
        self._rough_band_mask = msk

    def _get_state_mask(self) -> xr.DataArray:
        if hasattr(self, '_refined_mask'):
            return self._refined_mask
        else:
            return self._rough_x_mask & self._rough_y_mask

    def _get_band_mask(self) -> xr.DataArray:
        stmsk = self._get_state_mask()
        return self._rough_band_mask & stmsk.any(dim = self.k_x) & ~stmsk
    
    def _get_state(self, drop: bool = False) -> xr.Dataset:
        return self.ds.where(self._get_state_mask(), drop = drop)
    
    def _get_band(self, drop: bool = False) -> xr.Dataset:
        return self.ds.where(self._get_band_mask(), drop = drop)

    def calc_gap(self, neg_to_nan: bool = False
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = self._get_state(drop = True).transpose(self.k_y, self.k_x)
        band = self._get_band(drop = True).transpose(self.k_y, self.k_x)
        self._mu, self._mu_unc = mu(
            vt = state[self.k_vt].values,
            vb = state[self.k_vb].values,
            chi_r = state[self.k_chi_r].values,
            chi_i = state[self.k_chi_i].values,
            chi_g = self.chi_g,
            chi_b = band[self.k_chi_r].mean(dim = self.k_x).values.reshape(-1, 1),
            var_chi_r = band[self.k_chi_r].var(dim = self.k_x).values.reshape(-1, 1),
            var_chi_i = band[self.k_chi_i].var(dim = self.k_x).values.reshape(-1, 1),
            var_chi_g = self.var_chi_g,
            var_chi_b = band[self.k_chi_r].var(dim = self.k_x).values.reshape(-1, 1),
            cb = self.cb,
            gamma = self.gamma,
            var_gamma = self.var_gamma,
            omega = 2* np.pi * self.f,
            mask_nans = True,
            neg_to_nan = neg_to_nan
        )
        self._Y = state[self.k_y].values

    def plot_masks(self, cmap: str | colors.Colormap = 'coolwarm', norm = None, 
                   band_color: Tuple[float, float, float] = None, ax = None,
                   **kwargs):
        band_color = band_color or (1, 0, 0)
        self._get_state()[self.k_chi_r].plot.pcolormesh(
            ax = ax, x = self.k_x, y = self.k_y, cmap = cmap, norm = norm, 
            **kwargs)
        self._get_band()[self.k_chi_r].plot.pcolormesh(
            ax = ax, x = self.k_x, y = self.k_y, 
            cmap = colors.ListedColormap([(*band_color, 0.5)]), 
            add_colorbar = False)

    def plot_gap(self, ax = None, **kwargs):
        kwargs.setdefault('errstyle', 'shade')
        errorplot(x=self._Y.flatten(), 
                  y = 1e3*self._mu[:,-1].flatten(), 
                  yerr = 1e3*self._mu_unc[:,-1].flatten(), 
                  ax = ax, label = self.label, **kwargs)
        
        style_xr_xlabel(self.ds[self.k_y], ax=ax)
        if ax is None:
            plt.ylabel(r'$\Delta\mu$ [meV]')
        else:
            ax.set_ylabel(r'$\Delta\mu$ [meV]')

def extract_cqAR(ds: xr.Dataset,
                 chi_g: float                = None,
                 chi_b: float | xr.DataArray = None,
                 Cb: float                   = None, 
                 gamma: float                = None,
                 f: float                    = None,
                 k_chi_r: str                = 'Cex',
                 k_chi_i: str                = 'Closs',
                 model: str                  = 'tl_model',
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract quantum capacitance and AR (sheet resistance * area) from 
    penetration capacitance data.
    """

    chi_g = chi_g or _get_attr_parm(ds, 'chi_g', float)
    chi_b = chi_b or _get_attr_parm(ds, 'chi_b', float)
    Cb = Cb or _get_attr_parm(ds, 'Cb', float)
    gamma = gamma or _get_attr_parm(ds, 'gamma', float)
    f = f or _get_attr_parm(ds, 'f', float)
    if model == 'tl_model':
        return tl_model(ds[k_chi_r].values, ds[k_chi_i].values, 
                        chi_g, chi_b, Cb, gamma, 2*np.pi*f) 
    elif model == 'lf_model':
        return lf_model(ds[k_chi_r].values, ds[k_chi_i].values, 
                        chi_g, chi_b, Cb, gamma)
    else:
        raise ValueError(
            f"'{model}' is not a supported model. Use 'tl_model' or 'lf_model'"
        )
    
def extract_compressibility(ds: xr.Dataset,
                            chi_g: float                = None,
                            chi_b: float | xr.DataArray = None,
                            Cb: float                   = None, 
                            gamma: float                = None,
                            f: float                    = None,
                            k_chi_r: str                = 'Cex',
                            k_chi_i: str                = 'Closs',
                            model: str                  = 'tl_model',
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract cq, AR. Then report dmu/dn and AR with dmu/dn in units of eVnm^2
    """

    cq, AR = extract_cqAR(ds, chi_g, chi_b, Cb, gamma, f, 
                          k_chi_r, k_chi_i, model)
    f =  1.602e-1 # (nm/m)^2 / q
    return f / cq, AR
    

@xr.register_dataset_accessor("ezgap")
class ezDatasetAccessor():
    def __init__(self, xr_obj: xr.Dataset):
        self._obj = xr_obj

    def cqAR(self,
             chi_g: float                = None,
             chi_b: float | xr.DataArray = None,
             Cb: float                   = None, 
             gamma: float                = None,
             f: float                    = None,
             k_chi_r: str                = 'Cex',
             k_chi_i: str                = 'Closs',
             model: str                  = 'tl_model') -> xr.Dataset:
        """Accessor to extract_cqAR"""
        ds = self._obj.copy()
        cq, AR = extract_cqAR(ds, chi_g, chi_b, Cb, gamma, f, 
                              k_chi_r, k_chi_i, model)
        ds['cq'] = xr.DataArray(cq, dims = ds[k_chi_r].dims, 
            attrs = {'long_name': R'$c_q$', 'units': R'F$\cdot$m$^{-2}$'})
        ds['AR'] = xr.DataArray(AR, dims = ds[k_chi_r].dims, 
            attrs = {'long_name': R'$AR_s$', 'units': R'm$^2\Omega$'})
        return ds
    
    def compressibility(self,
                        chi_g: float                = None,
                        chi_b: float | xr.DataArray = None,
                        Cb: float                   = None, 
                        gamma: float                = None,
                        f: float                    = None,
                        k_chi_r: str                = 'Cex',
                        k_chi_i: str                = 'Closs',
                        model: str                  = 'tl_model') -> xr.Dataset:
        """Accessor to extract_compressibility"""
        ds = self._obj.copy()
        dmu_dn, AR = extract_compressibility(ds, chi_g, chi_b, Cb, gamma, f, 
                                             k_chi_r, k_chi_i, model)
        ds['dmu_dn'] = xr.DataArray(dmu_dn, dims = ds[k_chi_r].dims, 
            attrs = {'long_name': R'$\frac{d\mu}{dn}$', 
                     'units': R'eV$\cdot$nm$^{-2}$'})
        ds['AR'] = xr.DataArray(AR, dims = ds[k_chi_r].dims, 
            attrs = {'long_name': R'$AR_s$', 'units': R'm$^2\Omega$'})
        return ds

    def phase_correct(self,
                      X_spur: float | xr.DataArray, 
                      Y_spur: float | xr.DataArray,
                      k_chi_r: str = 'Cex', k_chi_i: str = 'Closs',
                      ) -> xr.Dataset:
        """Accessor to phase_correct"""
        chi = (k_chi_r, k_chi_i)
        return self._obj.ez.transform(
            lambda x, y: phase_correct(x, y, X_spur, Y_spur), 
            chi, chi, xr_output_type='data_vars',
        )
