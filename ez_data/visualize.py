import xarray as xr
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import holoviews as hv
import hvplot.xarray
from typing import Union

def configure_plotting_theme():
    """Optional default style"""
    import holoviews as hv
    hv.extension("bokeh")
    hv.opts.defaults(
        hv.opts.QuadMesh(show_frame=False, line_color=None, toolbar='above')
    )

def snap_and_grid_for_plotting(ds: xr.Dataset, x: str, y: str, z: str, 
                        resolution: Union[float, dict] = 1e-3) -> xr.Dataset:
    """
    Fast gridding using pandas grouping to snap and bin data
    """
    if isinstance(resolution, dict):
        res_x = resolution.get(x, 1e-3)
        res_y = resolution.get(y, 1e-3)
    else:
        res_x = res_y = resolution

    df = pd.DataFrame({
        x: (ds[x].values / res_x).round() * res_x,
        y: (ds[y].values / res_y).round() * res_y,
        z: ds[z].values,
    })

    grouped = df.groupby([y, x])[z].mean().unstack()
    grid_y = grouped.index.values
    grid_x = grouped.columns.values
    grid_z = grouped.values

    return xr.Dataset(
        {z: (["y", "x"], grid_z)},
        coords={"x": grid_x, "y": grid_y}
    )

def safe_interpolation(ds: xr.Dataset, x: str, y: str, z: str, 
                       resolution: Union[float, dict] = 1e-3, 
                       method: str = "linear") -> xr.Dataset:
    """
    Safe linear interpolation using scipy griddata, masking where data is not 
    supported.
    Resolution works like in snap_and_grid_for_plotting: defines step size along 
    each axis.
    """
    df = ds[[x, y, z]].to_dataframe().dropna()
    points = df[[x, y]].values
    values = df[z].values

    if isinstance(resolution, dict):
        dx = resolution.get(x, 1e-3)
        dy = resolution.get(y, 1e-3)
    else:
        dx = dy = resolution

    x_lin = np.arange(df[x].min(), df[x].max() + dx, dx)
    y_lin = np.arange(df[y].min(), df[y].max() + dy, dy)
    X, Y = np.meshgrid(x_lin, y_lin)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    Z = griddata(points, values, grid_points, method=method)
    mask = griddata(points, np.ones_like(values), grid_points, method="nearest")
    Z[mask != 1] = np.nan

    return xr.Dataset(
        {z: (["y", "x"], Z.reshape(len(y_lin), len(x_lin)))},
        coords={"x": x_lin, "y": y_lin}
    )

def plot_snapped_image(ds, x, y, z, resolution=1e-3, clim=None, logz=False, 
                          cmap="plasma", zero_center=False, cbar=True, 
                          interpolation=None, function='image'):
    """
    Plot snapped image with optional color scaling and colormap customization
    Automatically includes axis labels and colorbar label 
    (with units, if available)
    interpolation: None (default grid), or 'linear' for safe interpolation
    """
    if interpolation == "linear":
        gridded = safe_interpolation(ds, x, y, z, resolution=resolution)
    else:
        gridded = snap_and_grid_for_plotting(ds, x, y, z, resolution)

    z_label = ds[z].attrs.get("long_name", z)
    z_units = ds[z].attrs.get("units", "")
    if z_units:
        z_label += f" [{z_units}]"

    x_label = ds[x].attrs.get("long_name", x)
    x_units = ds[x].attrs.get("units", "")
    if x_units:
        x_label += f" [{x_units}]"

    y_label = ds[y].attrs.get("long_name", y)
    y_units = ds[y].attrs.get("units", "")
    if y_units:
        y_label += f" [{y_units}]"

    opts = dict(
        x = "x", y = "y", 
        cmap    = cmap,
        xlabel  = x_label, 
        ylabel  = y_label,
        colorbar = cbar,
    )
    if clim:
        opts["clim"] = clim
    if logz:
        opts["logz"] = True

    match function:
        case 'image':
            plot = gridded[z].hvplot.image(**opts)
        case 'quadmesh':
            opts['line_color'] = None
            opts['line_width'] = 0
            plot = gridded[z].hvplot.quadmesh(**opts)
        case _:
            raise ValueError(f"{function} is not a valid plot function.")
   
    if hv.Store.current_backend == "bokeh" and cbar:
        plot = plot.opts(colorbar_opts={'title': z_label})
    return plot
