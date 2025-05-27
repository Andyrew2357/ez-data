import xarray as xr
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.xarray
from typing import Dict


def snap_to_grid(ds: xr.Dataset, xs: str, ys: str, zs: str, 
                 axes: Dict[str, list]) -> xr.Dataset:
    """
    Snap data to a rectilinear grid and fill missing bins with NaN.
    (axes are assumed to be strictly increasing or decreasing).
    """
    
    xbins = np.array(axes[xs])
    if xbins[1] < xbins[0]:
        xbins = np.flip(xbins)
    small_bin_x = np.min(np.abs(np.diff(xbins)))
    xbins = np.r_[2*xbins[0] - xbins[1], xbins] + 0.5*small_bin_x
        
    ybins = np.array(axes[ys])
    if ybins[1] < ybins[0]:
        ybins = np.flip(ybins)
    small_bin_y = np.min(np.abs(np.diff(ybins)))
    ybins = np.r_[2*ybins[0] - ybins[1], ybins] + 0.5*small_bin_y

    df = pd.DataFrame({
        xs: pd.cut(ds[xs].values, bins = xbins, labels = xbins[1:], 
                   include_lowest = False, right = True).astype(float),
        ys: pd.cut(ds[ys].values, bins = ybins, labels = ybins[1:],
                   include_lowest = False, right = True).astype(float),
        zs: ds[zs]
    })

    grouped = df.groupby([ys, xs])[zs].mean().unstack()
    grid_y = grouped.index.values - 0.5*small_bin_y
    grid_x = grouped.columns.values - 0.5*small_bin_x
    grid_z = grouped.values

    return xr.Dataset(
        {zs: (["y", "x"], grid_z)},
        coords = {"x": grid_x, "y": grid_y}
    )

def get_label(ds, s: str):
    label = ds[s].attrs.get("long_name", s)
    units = ds[s].attrs.get("units", "")
    if units:
        label += f" [{units}]"
    return label

def plot_2d_data(ds: xr.Dataset, xs: str, ys: str, zs: str, 
                 axes: Dict[str, list], plot_mode: str = 'image',
                 plot_kwargs: dict = None, cbar_kwargs: dict = None):
    """Plot two dimensional data snapped to provided axes"""
    # Common plot_kwargs: cmap, clim, logz
    
    xlabel = get_label(ds, xs)
    ylabel = get_label(ds, ys)
    zlabel = get_label(ds, zs)

    gridded = snap_to_grid(ds, xs, ys, zs, axes)

    opt = dict(
        x = "x", y = "y",
        xlabel = xlabel, 
        ylabel = ylabel
    )
    plot_kwargs = {**plot_kwargs, **opt}
    plot_kwargs['colorbar'] = cbar_kwargs is not None

    match plot_mode:
        case 'image':
            plot = gridded[zs].hvplot.image(**plot_kwargs)
        case 'quad':
            plot_kwargs['line_color'] = None
            plot_kwargs['line_width'] = 0
            plot = gridded[zs].hvplot.quadmesh(**plot_kwargs)
        case _:
            raise ValueError(f"{plot_mode} is not implemented as a plot mode.")

    if plot_kwargs.get('colorbar', False) and not cbar_kwargs:
        cbar_kwargs = {}
    if cbar_kwargs:
        if hv.Store.current_backend == "bokeh" and \
            not hasattr(cbar_kwargs, 'title'):
            cbar_kwargs['title'] = zlabel
        plot = plot.opts(colorbar_opts = cbar_kwargs)

    return plot
