"""Tools for plotting; compatible with xarray objects"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.colorbar import Colorbar
from matplotlib.container import ErrorbarContainer
from matplotlib.collections import FillBetweenPolyCollection

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
    chart = ax or plt
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

WATERFALL = List[List[Line2D]] | Tuple[List[List[Line2D]], Colorbar]
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

    chart = ax or plt
    make_algebraic(x)
    make_algebraic(y)
    make_algebraic(z)

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    if norm is None:
        vmin = vmin or float(z.min())
        vmax = vmax or float(z.max())
        if logz:
            norm = colors.LogNorm(vmin, vmax)
        else:
            norm = colors.Normalize(vmin, vmax)

    sm = cm.ScalarMappable(norm, cmap)

    xval = get_arr_values(x)
    yval = get_arr_values(y)
    zval = get_arr_values(z)
    
    if xval.ndim != 2 or yval.ndim != 2:
        raise ValueError(f"x and y should be two dimensional")
    if xval.shape != yval.shape:
        raise ValueError(f"x and y must have the same shape")
    if not zval.ndim in [1, 2]:
        raise ValueError(f"z should be 1 or 2 dimensional")
    
    zval = zval.flatten()

    Nlines = len(zval)
    shape = xval.shape
    if not Nlines in shape:
        raise ValueError(f"Provided z does not match the dimensions of x and y")
    
    lines = []
    if Nlines == shape[0]:
        for i in range(Nlines):
            lines.append(
                chart.plot(x[i, :], y[i, :], color = cmap(norm(z[i])), **kwargs)
            )
    else:
        for i in range(Nlines):
            lines.append(
                chart.plot(x[:, i], y[:, i], color = cmap(norm(z[i])), **kwargs)
            )

    style_xr_xlabel(x)
    style_xr_ylabel(y)

    if cbar:
        if ax:
            c_bar = plt.colorbar(sm, ax = ax)
        else:
            c_bar = plt.colorbar(sm, ax = plt.gca())
        style_xr_colorbar(z, c_bar)
        return lines, c_bar
    
    return lines

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
            z = z.mean(dim = reduce_zdim)

        return waterfall(x, y, z, ax, 
                         cmap, norm, cbar, logz, vmin, vmax, **kwargs)
    
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
            z = z.mean(dim = reduce_zdim)

        return waterfall(x, y, z, ax, 
                         cmap, norm, cbar, logz, vmin, vmax, **kwargs)
