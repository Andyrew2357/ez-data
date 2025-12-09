"""Tools for plotting; compatible with xarray objects"""

import warnings
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.colorbar import Colorbar
from matplotlib.container import ErrorbarContainer
from matplotlib.collections import LineCollection, FillBetweenPolyCollection

import xarray as xr
import numpy as np

from typing import List, Tuple

DARR = List[float] | xr.DataArray

def make_algebraic(arr: DARR | None):
    """
    Prepare a DARR type so that it supports algebraic operations with other
    DARR objects along with other basic functions.
    """

    if arr is None or isinstance(arr, xr.DataArray):
        return
    arr = np.array(arr)

def get_arr_values(arr: DARR) -> np.ndarray:
    """Get the values of a DARR type as a numpy array."""

    make_algebraic(arr)
    return arr.values if isinstance(arr, xr.DataArray) else arr

def get_xr_label(da: xr.DataArray) -> str:
    """Get the label appropriate of a DataArray"""

    label = da.attrs.get('long_name', '')
    units = da.attrs.get('units', '')
    label += f' [{units}]' if units else ''
    return label

def style_xr_xlabel(da: DARR, ax: Axes = None):
    """Set the xlabel corresponding to a DataArray"""

    if not isinstance(da, xr.DataArray):
        return
    xlabel = get_xr_label(da)
    if ax is None:
        plt.xlabel(xlabel)
    else:
        ax.set_xlabel(xlabel)

def style_xr_ylabel(da: DARR, ax: Axes = None):
    """Set the ylabel corresponding to a DataArray"""

    if not isinstance(da, xr.DataArray):
        return
    ylabel = get_xr_label(da)
    if ax is None:
        plt.ylabel(ylabel)
    else:
        ax.set_ylabel(ylabel)

def style_xr_colorbar(da: DARR, cbar: Colorbar):
    """Set the colorbar label corresponding to a DataArray"""
    
    if not isinstance(da, xr.DataArray):
        return
    zlabel = get_xr_label(da)
    cbar.set_label(zlabel)

SHADE_EPLOT = Tuple[List[Line2D], FillBetweenPolyCollection] | \
    Tuple[List[Line2D], FillBetweenPolyCollection, FillBetweenPolyCollection]
def errorplot(x: DARR, y: DARR, 
              xerr: DARR = None, yerr: DARR = None, 
              errstyle: str = 'bar', 
              ax: Axes = None, 
              ekwargs: dict = None, 
              **kwargs) -> ErrorbarContainer | SHADE_EPLOT:
    """
    Make a plot with errors. The errorstyle can either be 'bar' or 'shade'.
    'bar' will plot with errorbars, while 'shade' will continuously shade the 
    region surrounding a line
    """
    
    ekwargs = ekwargs or {}
    chart = ax or plt.gca()
    make_algebraic(x)
    make_algebraic(y)
    make_algebraic(xerr)
    make_algebraic(yerr)

    if errstyle == 'bar':
        result = chart.errorbar(x, y, xerr = xerr, yerr = yerr, **kwargs)
    elif errstyle == 'shade':
        lines = chart.plot(x, y, **kwargs)
        if not 'alpha' in ekwargs:
            ekwargs['alpha'] = 0.5 * (lines[-1].get_alpha() or 1.0)
        curve_color = lines[-1].get_color()
        ekwargs.setdefault('color', curve_color)

        result = [lines]
        if xerr is not None:
            result.append(chart.fill_betweenx(y, x - xerr, x + xerr, **ekwargs))
        if yerr is not None:
            result.append(chart.fill_between(x, y - yerr, y + yerr, **ekwargs))
    else:
        raise ValueError(
            f"Unrecognized errstyle: '{errstyle}'. Try 'bar' or 'shade'"
        )

    style_xr_xlabel(x)
    style_xr_ylabel(y)
    return tuple(result)

WATERFALL = Tuple[LineCollection, Colorbar] | LineCollection
def waterfall(x: DARR, y: DARR, z: DARR,
              ax: Axes = None,
              cmap: str | colors.Colormap = 'RdBu',
              norm: cm.ScalarMappable = None,
              cbar: bool = True,
              logz: bool = False,
              vmin: float = None, vmax: float = None,
              **kwargs) -> WATERFALL:
    """
    Create a waterfall plot 
    (series of lines plots with colors corresponding to some third variable)
    """

    chart = ax or plt.gca()
    make_algebraic(x)
    make_algebraic(y)
    make_algebraic(z)

    xval = get_arr_values(x)
    yval = get_arr_values(y)
    zval = get_arr_values(z).ravel()

    if xval.ndim != 2 or yval.ndim != 2:
        raise ValueError(f"x and y should be two dimensional")
    if xval.shape != yval.shape:
        raise ValueError(f"x and y must have the same shape")
    
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    if norm is None:
        vmin = vmin or float(zval.min())
        vmax = vmax or float(zval.max())
        if logz:
            norm = colors.LogNorm(vmin, vmax)
        else:
            norm = colors.Normalize(vmin, vmax)

    if zval.size == xval.shape[0]:
        segs = [np.column_stack([xval[i, :], yval[i, :]]) 
                for i in range(xval.shape[0])]
    elif zval.size == xval.shape[1]:
        segs = [np.column_stack([xval[:, i], yval[:, i]]) 
                for i in range(xval.shape[1])]
    else:
        raise ValueError(f"Provided z does not match the dimensions of x and y")
    
    # create LineCollection so we have colormap linkage
    lc = LineCollection(segs, array=zval, cmap=cmap, norm=norm, **kwargs)
    chart.add_collection(lc)
    chart.autoscale()

    style_xr_xlabel(x, chart)
    style_xr_ylabel(y, chart)

    if cbar:
        c_bar = plt.colorbar(lc, ax=chart)
        style_xr_colorbar(z, c_bar)
        return lc, c_bar
    
    return lc

def _check_and_fix_shape(x: DARR, y: DARR, Z: DARR) -> DARR:
    """
    Ensure Z has a shape compatible with coordinates x, y for pcolormesh.
    Handles both 1D (x, y) and 2D (meshgrid) coordinate cases.
    """

    if x.ndim == 1 and y.ndim == 1:
        expected = (len(y), len(x))
    elif x.ndim == 2 and y.ndim == 2:
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape when 2D.")
        expected = x.shape
    else:
        raise ValueError("x and y must either both be 1D or both be 2D.")

    if Z.shape == expected:
        return Z
    elif Z.shape == expected[::-1]:
        return Z.transpose()
    else:
        raise ValueError(
            f"Incompatible shapes: Z {Z.shape}, x {x.shape}, y {y.shape}, "
            f"expected {expected} (or its transpose)."
        )

class ComplexMeshController():
    def __init__(self, x: DARR, y: DARR, *, 
                 zx: DARR = None, zy: DARR = None, 
                 zr: DARR = None, zt: DARR = None,
                 axes   : Tuple[Axes, Axes]          = None,
                 figsize: Tuple[float, float]        = (12, 5),
                 cmaps  : Tuple[str | colors.Colormap, 
                            str | colors.Colormap]   = ('RdBu_r', None),
                 norms  : Tuple[cm.ScalarMappable, 
                            cm.ScalarMappable]       = (None, None),
                 vmins  : Tuple[float, float]        = (None, None), 
                 vmaxs  : Tuple[float, float]        = (None, None),
                 cbar   : Tuple[bool, bool]          = (True, True),
                 shading: str                        = 'auto',
                 connect_scroll_on_init: bool        = True):
        """
        ezplt-style constructor for interactive complex pcolormesh.

        Provide either (zx, zy) = (real, imag) OR (zr, zt) = (magnitude, phase).
        x, y, zx, zy, zr, zt may be numpy arrays or xarray.DataArray.
        """

        # store original inputs for labeling
        self._orig_x, self._orig_y = x, y
        self._orig_zs = []
        self.x, self.y = get_arr_values(x), get_arr_values(y)
        self.phase = 0.0
        self.shading = shading

        norms = norms or (None, None)
        vmins = vmins or (None, None)
        vmaxs = vmaxs or (None, None)

        # build and store base complex array Z0 once
        if zx is not None and zy is not None:
            self.mode = 'realimag'
            rx, ix = get_arr_values(zx), get_arr_values(zy)
            rx = _check_and_fix_shape(self.x, self.y, rx)
            ix = _check_and_fix_shape(self.x, self.y, ix)
            self.Z0 = rx + 1j * ix
            self._orig_zs = [zx, zy]
        elif zr is not None and zt is not None:
            self.mode = 'magphase'
            mag, ph = get_arr_values(zr), get_arr_values(zt)
            mag = _check_and_fix_shape(self.x, self.y, mag)
            ph = _check_and_fix_shape(self.x, self.y, ph)
            self.Z0 = mag * np.exp(1j * ph)
            self._orig_zs = [zr, zt]
            if cmaps[1] is None:
                cmaps[1] = 'colorwheel'
        else:
            raise ValueError("Must provide either (zx, zy) or (zr, zt).")

        # figure/axes
        if axes is None:
            self.fig, self.axes = plt.subplots(1, 2, figsize = figsize)
        else:
            self.fig = axes[0].figure
            self.axes = axes

        self._autoscale = [False, False]
        for i in (0, 1):
            if norms[i] is None and vmins[i] is None and vmaxs[i] is None:
                self._autoscale[i] = True

        # initial draw
        comps = self._compute_components(0.0)
        self.meshes, self.cbars = [], []
        for i, (ax, data, zorig) in enumerate(zip(self.axes, comps, 
                                                  self._orig_zs)):
            norm = norms[i] or (colors.Normalize(vmin=vmins[i], vmax=vmaxs[i]) 
                if (vmins[i] is not None or vmaxs[i] is not None) else None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                qm = ax.pcolormesh(self.x, self.y, data, shading=self.shading, 
                                   cmap=cmaps[i], norm=norm)
            style_xr_xlabel(self._orig_x, ax)
            style_xr_ylabel(self._orig_y, ax)

            cb = None
            if cbar[i]:
                cb = self.fig.colorbar(qm, ax=ax)
                style_xr_colorbar(zorig, cb)

            self.meshes.append(qm)
            self.cbars.append(cb)

        self.suptitle = self.fig.suptitle(fR"$\phi$ = {self.phase:.2f} rad")
        plt.tight_layout()

        if connect_scroll_on_init:
            self.connect_scroll()

    def _compute_components(self, phi):
        Z = self.Z0 * np.exp(1j * phi)
        if self.mode == 'realimag':
            return (np.real(Z), np.imag(Z))
        else:
            return (np.abs(Z), np.angle(Z))

    def update_phase(self, phi):
        self.phase = phi % (2 * np.pi)
        comps = self._compute_components(self.phase)
        for i, (qm, data) in enumerate(zip(self.meshes, comps)):
            qm.set_array(data.ravel())

            if self._autoscale[i]:
                try:
                    qm.autoscale()
                except Exception:
                    try:
                        if hasattr(qm, "norm") and qm.norm is not None:
                            qm.norm.autoscale(qm.get_array())
                    except Exception:
                        pass
                if self.cbars[i] is not None:
                    try:
                        self.cbars[i].update_normal(qm)
                    except Exception:
                        pass

        self.suptitle.set_text(fR"$\phi$ = {self.phase:.2f} rad")
        self.fig.canvas.draw_idle()

    def connect_scroll(self, step=0.01):
        try:
            self.fig.canvas.capture_scroll = True
        except Exception:
            pass
        def on_scroll(event):
            if event.button == 'up':
                self.update_phase(self.phase + step)
            elif event.button == 'down':
                self.update_phase(self.phase - step)
        self.fig.canvas.mpl_connect('scroll_event', on_scroll)

    def add_slider(self, step=0.01, start=0.0, end=2*np.pi):
        import ipywidgets as widgets
        slider = widgets.FloatSlider(value=self.phase, min=start, max=end, 
                                     step=step, description=R"$\phi$")
        def _on_change(change):
            if change.get('name') == 'value':
                self.update_phase(change['new'])
        slider.observe(_on_change, names='value')
        from IPython.display import display
        display(slider)
        return slider

def ccolormesh(x: DARR, y: DARR, *, 
               zx: DARR = None, zy: DARR = None, 
               zr: DARR = None, zt: DARR = None, 
               **kwargs) -> ComplexMeshController:
    """Convenience wrapper for ComplexMeshController."""
    return ComplexMeshController(x, y, zx=zx, zy=zy, zr=zr, zt=zt, **kwargs)


"""Accessors for xarray Datasets and DataArrays"""

@xr.register_dataset_accessor("ezplt")
class ezpltDatasetAccessor():
    def __init__(self, xr_obj: xr.Dataset):
        self._obj = xr_obj

    def _get_arr(self, a: str | DARR) -> DARR:
        if isinstance(a, str):
            try:
                return self._obj[a]
            except:
                raise ValueError(f"'{a}' is unrecognized.")
        return a

    def errorplot(self, 
                  x: str | DARR, y: str | DARR,
                  xerr: str | DARR = None, yerr: str | DARR = None,
                  errstyle: str = 'bar', 
                  ax: Axes = None, 
                  ekwargs: dict = None, **kwargs
                  ) -> ErrorbarContainer | SHADE_EPLOT:
        """Accessor to errorplot"""
        
        x = self._get_arr(x)
        y = self._get_arr(y)
        xerr = self._get_arr(xerr)
        yerr = self._get_arr(yerr)
        return errorplot(x, y, xerr, yerr, errstyle, ax, ekwargs, **kwargs)
    
    def waterfall(self,
                  x: str | DARR, y: str | DARR, z: str | DARR,
                  reduce_zdim: str = None,
                  ax: Axes = None, 
                  cmap: str | colors.Colormap = 'RdBu', 
                  norm: cm.ScalarMappable = None,
                  cbar: bool = True,
                  logz: bool = False, 
                  vmin: float = None, vmax: float = None, 
                  **kwargs) -> WATERFALL:
        """Accessor to waterfall"""

        x = self._get_arr(x)
        y = self._get_arr(y)
        z = self._get_arr(z)
        if reduce_zdim:
            z = z.mean(dim = reduce_zdim, keep_attrs = True)

        return waterfall(x, y, z, ax, 
                         cmap, norm, cbar, logz, vmin, vmax, **kwargs)
    
    def ccolormesh(self, x: str | DARR, y: str | DARR, 
                   zx: str | DARR = None, zy: str | DARR = None, 
                   zr: str | DARR = None, zt: str | DARR = None, 
                   **kwargs):
        """Accessor to ccolormesh"""
        x = self._get_arr(x)
        y = self._get_arr(y)
        zx = self._get_arr(zx)
        zy = self._get_arr(zy)
        zr = self._get_arr(zr)
        zt = self._get_arr(zt)
        return ccolormesh(x, y, zx=zx, zy=zy, zr=zr, zt=zt, **kwargs)
    
@xr.register_dataarray_accessor("ezplt")
class ezpltDataArrayAccessor():
    def __init__(self, xr_obj: xr.DataArray):
        self._obj = xr_obj

    def _get_arr(self, a: str | DARR) -> DARR:
        if isinstance(a, str):
            try:
                return self._obj[a]
            except:
                raise ValueError(f"'{a}' is unrecognized.")
        return a

    def errorplot(self, 
                  x: str | DARR = None, y: str | DARR = None,
                  xerr: str | DARR = None, yerr: str | DARR = None,
                  errstyle: str = 'bar', 
                  ax: Axes = None, 
                  ekwargs: dict = None, **kwargs
                  ) -> ErrorbarContainer | SHADE_EPLOT:
        """Accessor to errorplot"""
        
        if x is None and y is None:
            raise ValueError("One of 'x' or 'y' must be provided.")
        x = self._obj if x is None else self._get_arr(x)
        y = self._obj if y is None else self._get_arr(y)
        xerr = self._get_arr(xerr)
        yerr = self._get_arr(yerr)
        return errorplot(x, y, xerr, yerr, errstyle, ax, ekwargs, **kwargs)
    
    def waterfall(self,
                  x: str | DARR = None, y: str | DARR = None, 
                  z: str | DARR = None, reduce_zdim: str = None,
                  ax: Axes = None, 
                  cmap: str | colors.Colormap = 'RdBu', 
                  norm: cm.ScalarMappable = None,
                  cbar: bool = True,
                  logz: bool = False, 
                  vmin: float = None, vmax: float = None, 
                  **kwargs) -> WATERFALL:
        """Accessor to waterfall"""

        if (x is None and (y is None or z is None)) or \
            (y is None and (x is None or z is None)) or \
            (z is None and (y is None or x is None)):
            raise ValueError("At most one of 'x', 'y', and 'z' can be None.")
        x = self._obj if x is None else self._get_arr(x)
        y = self._obj if y is None else self._get_arr(y)
        z = self._obj if z is None else self._get_arr(z)
        if reduce_zdim:
            z = z.mean(dim = reduce_zdim, keep_attrs = True)

        return waterfall(x, y, z, ax, 
                         cmap, norm, cbar, logz, vmin, vmax, **kwargs)
