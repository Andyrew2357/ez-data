import numpy as np
import pandas as pd
import xarray as xr
from typing import Callable, List, Tuple

XR_OBJ = xr.Dataset | xr.DataArray

def align_dims(obj: XR_OBJ, replace: bool = False, **alignment: dict) -> XR_OBJ:
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
        new_cd_data = og_cd.mean(dim=other_dims) if other_dims else og_cd
        
        # create new coordinate with preserved attributes
        new_cd = xr.DataArray(
            new_cd_data.values,
            dims=[dim],
            attrs=og_cd.attrs
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
                v = np.nanvar(c_vals, axis=axis)
                dim_var[d] = float(np.nanmean(v))
            cut_dim = max(dim_var, key=dim_var.get)

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
            idx = np.argmin(diff, axis=cut_axis)

            # remaining dims (in order):
            rem_dims = tuple(d for d in c_dims if d != cut_dim)
            idx_da = xr.DataArray(idx, dims=rem_dims)
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
    da = xr.DataArray(data, dims=dims, attrs=attrs)
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
        stacked = np.stack(arrays, axis=-1)  # shape (..., n_inputs)
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
