import numpy as np
import pandas as pd
import xarray as xr
from typing import Callable, List, Tuple

XR_OBJ = xr.Dataset | xr.DataArray

def align_dims(obj: XR_OBJ, replace: bool = False, **alignment) -> XR_OBJ:
    """
    Often some coordinates will be aligned with a dimension. In this case, we
    can collapse the coordinate array down to that dimension and rename the
    dimension, giving us a coordinate that is both logical and physical. The new
    coord retains the attrs of the original coordinate. If replace is False, 
    instead of a strict replacement, we create a new coordinate with the same 
    name as the original plus a trailing '_'.
    """

    rename = alignment if replace else {k: v+'_' for k, v in alignment.items()}

    for dim, cd in alignment.items():
        if cd not in obj.coords:
            continue

        og_cd = obj.coords[cd]
        
        if dim not in og_cd.dims:
            continue

        other_dims = [d for d in og_cd.dims if d != dim]
        new_cd_data = og_cd.mean(dim = other_dims) if other_dims else og_cd
        
        # create new coordinate with preserved attributes
        new_cd = xr.DataArray(
            new_cd_data.values,
            dims = [dim],
            attrs = og_cd.attrs
        )
        
        # add the new coordinate to the dataset
        obj = obj.assign_coords({rename[dim]: new_cd})
        
        if replace:
            obj = obj.drop_vars(cd)
    
    return obj

def smart_sel(obj: XR_OBJ, tolerance: float = None, **coords) -> XR_OBJ:
    """
    Intelligently select from imperfectly aligned physical coordinates.

    **coords : {coord_name: scalar or slice}
        Coordinate selections, exactly like xarray.sel, but:
        Scalars drop ONE dimension (the one the coord varies over).
        Slices keep dims but mask by range.
    """

    indexers = {}

    for cname, selector in coords.items():
        if cname not in obj.coords:
            raise KeyError(f"Coordinate '{cname}' not found.")

        coord = obj.coords[cname]
        c_dims = coord.dims
        c_vals = coord.values

        # ---- decide which dim to cut along ------------------------
        if len(c_dims) == 1:
            cut_dim = c_dims[0]
        else:
            # compute a single variance number per candidate dim
            dim_var = {}
            for d in c_dims:
                axis = coord.get_axis_num(d)
                v = np.nanvar(c_vals, axis = axis)
                dim_var[d] = float(np.nanmean(v))
            cut_dim = max(dim_var, key = dim_var.get)

        cut_axis = coord.get_axis_num(cut_dim)

        # ===========================================================
        # Scalar selection  ->  drop 'cut_dim'
        # ===========================================================
        if np.isscalar(selector):
            target = float(selector)
            diff = np.abs(c_vals - target)

            if tolerance is not None:
                diff = np.where(diff <= tolerance, diff, np.inf)
                if np.all(np.isinf(diff)):
                    raise ValueError(
                        f"No '{cname}' values within ±{tolerance} of {target}"
                    )

            # argmin along cut axis -> indices of shape 'remaining dims'
            idx = np.argmin(diff, axis = cut_axis)

            # remaining dims (in order):
            rem_dims = tuple(d for d in c_dims if d != cut_dim)
            idx_da = xr.DataArray(idx, dims = rem_dims)
            indexers[cut_dim] = idx_da

        # ===========================================================
        # Slice selection  ->  keep 'cut_dim' but trim its extent
        # ===========================================================
        elif isinstance(selector, slice):
            lo = -np.inf if selector.start is None else selector.start
            hi =  np.inf if selector.stop  is None else selector.stop
            mask = (c_vals >= lo) & (c_vals <= hi)

            # We'll find indices along the cut dimension (cut_axis) where *any*
            # value along the other dims is within the desired range
            collapsed = np.any(mask, axis=tuple(i for i in range(mask.ndim) \
                                                if i != cut_axis))

            idxs = np.where(collapsed)[0]
            if len(idxs) == 0:
                raise ValueError(
                    f"Slice {selector} selects no data on '{cname}'."
                )

            indexers[cut_dim] = slice(idxs.min(), idxs.max() + 1)

        else:
            raise TypeError(
                f"Selector for '{cname}' must be scalar or slice, "
                f"got {type(selector)}."
            )

    return obj.isel(**indexers)

def preserve_attrs(obj, col, data, dims, xr_output_type):
    # get attrs if variable/coord exists
    attrs = {}
    if col in obj:
        attrs = obj[col].attrs.copy()
    # create new DataArray with attrs
    da = xr.DataArray(data, dims = dims, attrs = attrs)
    if xr_output_type == 'coords':
        obj = obj.assign_coords({col: da})
    else:
        obj[col] = da
    return obj

def apply_transform(obj: pd.DataFrame | xr.Dataset, 
                    func: Callable[[float | Tuple[float]], float | Tuple[float]], 
                    input_cols: str | List[str], 
                    output_cols: str | List[str], 
                    xr_output_type: str = 'coords'
                    ) -> pd.DataFrame | xr.Dataset:
    """
    Parameters:
        obj: pd.DataFrame or xr.Dataset
        func: callable, e.g. f(x1, x2, ...) -> (y1, ..., ym)
        input_cols: list of column/variable names to use as input
        output_cols: list of names for the output columns/variables
        xr_output_type: 'data_vars' or 'coords' (for xr.Dataset only)
    Returns:
        Transformed object (same type as input)
    """

    if isinstance(obj, pd.DataFrame):
        inputs = [obj[col].values for col in input_cols]
        results = func(*inputs)
        if not isinstance(results, tuple):
            results = (results,)
        for i, col in enumerate(output_cols):
            obj[col] = results[i]
        return obj

    elif isinstance(obj, xr.Dataset):
        arrays = [obj[var].values for var in input_cols]
        # all arrays are broadcastable to a common shape
        results = func(*arrays)
        if not isinstance(results, tuple):
            results = (results,)
        dims = obj[input_cols[0]].dims
        for i, col in enumerate(output_cols):
            data = results[i]
            preserve_attrs(obj, col, data, dims, xr_output_type)

        return obj

    else:
        raise ValueError("Input must be a pandas DataFrame or xarray.Dataset")


def apply_linear_transform(obj: pd.DataFrame | xr.Dataset, 
                           matrix: np.ndarray, 
                           input_cols: str | List[str], 
                           output_cols: str | List[str], 
                           xr_output_type: str = 'coords'
                           ) -> pd.DataFrame | xr.Dataset:
    """    
    Parameters:
        obj: pd.DataFrame or xr.Dataset
        matrix: numpy.ndarray of shape (m, n), n = len(input_cols)
        input_cols: list of column/variable names to use as input
        output_cols: list of names for the output columns/variables (length m)
        xr_output_type: 'data_vars' or 'coords' (for xr.Dataset only)
    Returns:
        Transformed object (same type as input)
    """

    if isinstance(obj, pd.DataFrame):
        X = obj[input_cols].values  # shape (n_samples, n_inputs)
        Y = np.dot(X, matrix.T)     # shape (n_samples, n_outputs)
        for i, col in enumerate(output_cols):
            obj[col] = Y[:, i]
        return obj

    elif isinstance(obj, xr.Dataset):
        # broadcast all input variables to a common shape
        arrays = [obj[var].values for var in input_cols]
        # stack all input arrays along a new last axis
        stacked = np.stack(arrays, axis = -1)  # shape (..., n_inputs)
        # reshape to (N, n_inputs)
        orig_shape = stacked.shape[:-1]
        n_inputs = stacked.shape[-1]
        flat = stacked.reshape(-1, n_inputs)
        Y = np.dot(flat, matrix.T)  # shape (N, n_outputs)
        # reshape back to original shape + n_outputs
        Y = Y.reshape(*orig_shape, matrix.shape[0])
        for i, col in enumerate(output_cols):
            data = Y[..., i]
            dims = obj[input_cols[0]].dims  # Use dims from first input var
            preserve_attrs(obj, col, data, dims, xr_output_type)
            
        return obj

    else:
        raise ValueError("Input must be a pandas DataFrame or xarray.Dataset")

def bin_to_grid(data: XR_OBJ | pd.DataFrame, reduce: str | Callable = 'mean', 
                **bins) -> XR_OBJ | pd.DataFrame:
    """
    Bin irregular data to a regular N-dimensional grid using physical coords.

    Parameters:
        data: The input data to be binned.
        reduce: Reduction function to use. 
            Options: 'mean', 'median', 'max', 'min' or a callable.
        **bins: dict[str, int | array-like]
            Keyword arguments mapping coordinate names to either number of bins 
            or explicit bin edges.

    Returns:
        Binned result with the same type as data passed.
    """

    if isinstance(data, (xr.DataArray, xr.Dataset)):
        df = data.to_dataframe().reset_index()
        coords = list(bins.keys())
        dims_to_strip = [d for d in data.dims if d not in data.coords]
        binned = _bin_dataframe(df, coords, reduce, bins)
        result = _rebuild_xarray(binned, data, coords)

        # Clean up: remove any dim-like leftovers not used as bins or coords
        if isinstance(data, xr.Dataset):
            leftover_dim_vars = [
                var for var in dims_to_strip
                if var in result.data_vars or var in result.coords
            ]
        else:
            leftover_dim_vars = dims_to_strip
            
        return result.drop_vars(leftover_dim_vars, errors="ignore")
    
    elif isinstance(data, pd.DataFrame):
        coords = list(bins.keys())
        return _bin_dataframe(data, coords, reduce, bins)
    
    else:
        raise TypeError(
            "Input must be xr.DataArray, xr.Dataset, or pd.DataFrame"
        )

def _bin_dataframe(df: pd.DataFrame, coords: List[str], 
                   reduce: str | Callable, binspec: dict) -> pd.DataFrame:
    # Parse bin edges
    bin_edges = {}
    for dim in coords:
        if np.isscalar(binspec[dim]):
            n = int(binspec[dim])
            bin_edges[dim] = np.linspace(df[dim].min(), df[dim].max(), n + 1)
        else:
            bin_edges[dim] = np.asarray(binspec[dim])

    # Assign bin indices for each coordinate
    for dim in coords:
        df[f"_bin_{dim}"] = pd.cut(df[dim], bins=bin_edges[dim], labels = False)

    # Drop rows where any bin assignment failed
    df = df.dropna(subset = [f"_bin_{dim}" for dim in coords]).copy()
    for dim in coords:
        df[f"_bin_{dim}"] = df[f"_bin_{dim}"].astype(int)

    # Choose reduction function
    if isinstance(reduce, str):
        if reduce not in {"mean", "median", "max", "min"}:
            raise ValueError(f"Unknown reduce method '{reduce}'")
        use_func_directly = False
        agg_func = reduce  # string

    elif callable(reduce):
        use_func_directly = True
        agg_func = reduce

    else:
        raise TypeError("reduce must be a string or a callable")

    # Group by bin indices
    group_cols = [f"_bin_{dim}" for dim in coords]
    if use_func_directly:
        grouped = df.groupby(group_cols).agg(lambda g: agg_func(g)).reset_index()
    else:
        grouped = df.groupby(group_cols).agg(agg_func).reset_index()
        
    # Add physical coordinate columns back into grouped df
    for dim in coords:
        edges = bin_edges[dim]
        centers = 0.5 * (edges[:-1] + edges[1:])
        grouped[dim] = grouped[f"_bin_{dim}"].map(lambda i: centers[i])

    return grouped

def _rebuild_xarray(df: pd.DataFrame, original: XR_OBJ, coords: List[str]
                    ) -> XR_OBJ:
    # Extract bin index columns
    bin_cols = [f"_bin_{dim}" for dim in coords]

    # Separate data vars from coords
    other_coords = [
        name for name in original.coords
        if name not in coords and name in df.columns
    ]
    data_vars = [
        col for col in df.columns
        if col not in coords + bin_cols + other_coords
    ]

    # Coordinate axis values from bin centers
    coord_values = {dim: np.sort(df[dim].unique()) for dim in coords}
    shape = tuple(len(coord_values[dim]) for dim in coords)
    dims = tuple(coords)

    # Get flat index from bin indices
    multi_idx = tuple(df[f"_bin_{dim}"].values for dim in coords)
    flat_idx = np.ravel_multi_index(multi_idx, shape)

    def fill_array(values, dtype=float, fill_value = np.nan):
        arr = np.full(np.prod(shape), fill_value, dtype = dtype)
        arr[flat_idx] = values
        return arr.reshape(shape)

    # Fill data_vars and extra coordinates
    arrays = {var: (dims, fill_array(df[var].values)) for var in data_vars}
    extra_coords = {
        var: (dims, fill_array(df[var].values))
        for var in other_coords
    }

    ds = xr.Dataset(arrays, coords={**coord_values, **extra_coords})

    # Restore attributes
    if isinstance(original, xr.Dataset):
        for var in ds.data_vars:
            if var in original:
                ds[var].attrs = original[var].attrs.copy()
        for coord in coords + other_coords:
            if coord in original.coords:
                ds.coords[coord].attrs = original.coords[coord].attrs.copy()
        ds.attrs = original.attrs.copy()
        return ds

    elif isinstance(original, xr.DataArray):
        var_name = list(arrays)[0]
        da = ds[var_name]
        da.attrs = original.attrs.copy()
        for coord in coords + other_coords:
            if coord in original.coords:
                da.coords[coord].attrs = original.coords[coord].attrs.copy()
        return da
