import xarray as xr
from matplotlib import axes
from matplotlib import colors

def get_label(da: xr.DataArray, name: str = "") -> str:
    """Get a formatted label with units for plotting"""
    label = da.attrs.get("long_name", name)
    units = da.attrs.get("units", "")
    if units:
        label += f" [{units}]"
    return label

# MAYBE OVERLOAD TO WORK WITH DATAFRAME AS WELL?
def plot_pcolormesh(ax: axes.Axes, ds: xr.Dataset, xs: str, ys: str, zs: str,
            cmap: str = "plasma", logz: bool = False, vmin = None, vmax = None,
            xscale: str = 'linear', yscale: str = 'linear', cbar: bool = True,
            plot_kwargs: dict = {}, cbar_kwargs: dict = {}):
    """
    Plot 2D structured data using holoviews, faithfully representing sampling. 
    """
    
    if xs not in ds.data_vars:
        raise ValueError(f"Variable '{xs}' not found in dataset.")
    if ys not in ds.data_vars:
        raise ValueError(f"Variable '{ys}' not found in dataset.")
    if zs not in ds.data_vars:
        raise ValueError(f"Variable '{zs}' not found in dataset.")

    # Get the arrays (either data variables or coordinates)
    x = ds[xs]
    y = ds[ys]
    z = ds[zs]

    # Create labels
    xlabel = get_label(ds, xs)
    ylabel = get_label(ds, ys)
    zlabel = get_label(ds, zs)

    norm = colors.LogNorm(vmin, vmax) if logz else None

    opts = {
        'cmap': cmap,
        'norm': norm,    
    }
    if not logz:
        opts = {**opts, 'vmin': vmin, 'vmax': vmax}

    opts.update(**plot_kwargs)
    pc = ax.pcolormesh(x, y, z, **opts)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    opts = {
        'ax'    : ax,
        'label' : zlabel
    }
    opts.update(**cbar_kwargs)
    if cbar:
        cb = ax.figure.colorbar(pc, **opts)
    else:
        cb = None

    return ax, cb
