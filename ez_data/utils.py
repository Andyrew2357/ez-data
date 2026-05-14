"""General tools for manipulating xarray and pandas objects"""

import json
import sqlite3
import numpy as np
import pandas as pd
import xarray as xr
import warnings
from pathlib import Path
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

def smart_sel(obj: XR_OBJ, tolerance: float = None, drop: bool=False, **coords) -> XR_OBJ:
    """
    Intelligently select from imperfectly aligned physical coordinates.

    **coords : {coord_name: scalar or slice}
        Coordinate selections, exactly like xarray.sel, but:
        Scalars drop ONE dimension (the one the coord varies over).
        Slices keep dims but mask by range.
    """

    out = obj

    for cname, selector in coords.items():
        if cname not in out.coords:
            raise KeyError(f"Coordinate '{cname}' not found.")

        coord = out.coords[cname]
        c_dims = coord.dims
        c_vals = coord.values

        # choose the cut dimension
        if len(c_dims) == 1:
            cut_dim = c_dims[0]
        else:
            dim_var = {}
            for d in c_dims:
                axis = coord.get_axis_num(d)
                v = np.nanvar(c_vals, axis=axis)
                dim_var[d] = float(np.nanmean(v))
            cut_dim = max(dim_var, key=dim_var.get)

        cut_axis = coord.get_axis_num(cut_dim)

        # scalar selector
        if np.isscalar(selector):
            target = float(selector)
            diff = np.abs(c_vals - target)

            if tolerance is not None:
                diff = np.where(diff <= tolerance, diff, np.inf)
                if np.all(np.isinf(diff)):
                    raise ValueError(
                        f"No '{cname}' values within ±{tolerance} of {target}"
                    )

            idx = np.argmin(diff, axis=cut_axis)

            # build indexer relative to current object
            rem_dims = tuple(d for d in c_dims if d != cut_dim)
            idx_da = xr.DataArray(idx, dims=rem_dims)

            out = out.isel({cut_dim: idx_da}, drop=drop)
            continue

        # slice selector
        if isinstance(selector, slice):
            lo = -np.inf if selector.start is None else selector.start
            hi =  np.inf if selector.stop  is None else selector.stop

            mask = (c_vals >= lo) & (c_vals <= hi)
            collapsed = np.any(mask, axis=tuple(
                i for i in range(mask.ndim) if i != cut_axis
            ))
            idxs = np.where(collapsed)[0]
            if len(idxs) == 0:
                raise ValueError(
                    f"Slice {selector} selects no data on '{cname}'."
                )

            out = out.isel({cut_dim: slice(idxs.min(), idxs.max() + 1)}, drop=drop)
            continue

        raise TypeError(
            f"Selector for '{cname}' must be scalar or slice, "
            f"got {type(selector)}"
        )

    return out

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
            obj = preserve_attrs(obj, col, data, dims, xr_output_type)

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
            obj = preserve_attrs(obj, col, data, dims, xr_output_type)
            
        return obj

    else:
        raise ValueError("Input must be a pandas DataFrame or xarray.Dataset")

def bin_to_grid(data: XR_OBJ | pd.DataFrame, reduce: str | Callable = 'mean', 
                **bins) -> XR_OBJ | pd.DataFrame:
    """
    Bin irregular data to a regular N-dimensional grid using physical coords.

    Parameters
    ----------
    data: Dataset or DataArray
        input data to be binned.
    reduce: str or callable
        One of  {'mean', 'median', 'max', 'min'} or a callable.
    **bins: dict
        keyword arguments mapping coordinate names to either number of bins 
        or explicit bin edges.

    Returns
    -------
        Binned result with the same type as data passed.
    """

    if isinstance(data, (xr.DataArray, xr.Dataset)):
        df = data.to_dataframe().reset_index()
        coords = list(bins.keys())
        dims_to_strip = [d for d in data.dims if d not in data.coords]
        binned, bin_centers = _bin_dataframe(df, coords, reduce, bins)
        result = _rebuild_xarray(binned, data, coords, bin_centers)

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
        binned, bin_centers = _bin_dataframe(data, coords, reduce, bins)
        return binned
    
    else:
        raise TypeError(
            "Input must be xr.DataArray, xr.Dataset, or pd.DataFrame"
        )

def _bin_dataframe(df: pd.DataFrame, coords: List[str],  reduce: str | Callable, 
                   binspec: dict) -> Tuple[pd.DataFrame, dict]:
    
    # Always copy to avoid polluting caller
    df = df.copy()

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
        df[f"_bin_{dim}"] = pd.cut(df[dim], bins=bin_edges[dim], labels = False, 
                                   include_lowest = True, right = False)

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
    if not use_func_directly:
        # Only aggregate numeric columns
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in group_cols
        ]
        grouped = df.groupby(group_cols)[numeric_cols].agg(agg_func).reset_index()
    else:
        grouped = df.groupby(group_cols).agg(lambda g: agg_func(g)).reset_index()


    # Add physical coordinate columns back into grouped df
    center_map = {}
    for dim in coords:
        edges = bin_edges[dim]
        centers = 0.5 * (edges[:-1] + edges[1:])
        center_map[dim] = centers.copy()
        grouped[dim] = grouped[f"_bin_{dim}"].map(lambda i: centers[i])

    return grouped, center_map

def _rebuild_xarray(df: pd.DataFrame, original: XR_OBJ, coords: List[str],
                    bin_centers: dict) -> XR_OBJ:
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

    coord_values = {
        dim: np.asarray(bin_centers[dim])
        for dim in coords
    }
    shape = tuple(len(coord_values[dim]) for dim in coords)
    dims = tuple(coords)

    # Get flat index from bin indices
    multi_idx = tuple(df[f"_bin_{dim}"].values for dim in coords)
    flat_idx = np.ravel_multi_index(multi_idx, shape)

    def fill_array(values, fill_value=np.nan):
        values = np.asarray(values)
        dtype = values.dtype

        with warnings.catch_warnings(): # annoying warning involving datetimes
            warnings.simplefilter("ignore", category=RuntimeWarning)

            if np.issubdtype(dtype, np.datetime64):
                # Force consistent datetime dtype at nanosecond precision
                dtype = np.dtype("datetime64[ns]")
                fill_value = np.datetime64("NaT", "ns")
                fill = np.full(np.prod(shape), fill_value, dtype=dtype)
                valid_mask = ~pd.isnull(values)
                if np.any(valid_mask):
                    fill[flat_idx[valid_mask]] = values[valid_mask]
            else:
                fill = np.full(np.prod(shape), fill_value, dtype=dtype)
                fill[flat_idx] = values

        return fill.reshape(shape)

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
    
def where_parallelepiped(obj: XR_OBJ, origin: dict[str, float], 
                         *verts: Tuple[List[float]], **kwargs) -> XR_OBJ:
    return obj.where(parallelepiped_mask(obj, origin, *verts), **kwargs)

def parallelepiped_mask(obj: XR_OBJ | pd.DataFrame, 
                        origin: dict[str, float], 
                        *verts: Tuple[List[float]]
                        ) -> xr.DataArray | pd.DataFrame:
    
    basis = np.array(list(origin.keys()))
    N = len(basis) # dimension of the relevant superspace
    n = len(verts) # dimension of the parallelepiped
    if n == 0:
        raise ValueError("parallelepiped_mask requires at least one vertex.")
    if n > N:
        raise ValueError(
            "The dimension of a parallelepiped cannot exceed "
            "the dimension of its superspace."
        )

    vertices = np.array(verts, dtype = float).T
    N_, n = vertices.shape
    if N != N_:
        raise ValueError(
            f"Dimensions of the vertices ({N_}) do not match the origin ({N})."
        )
    v0 = np.array(list(origin.values()), dtype = float)

    # Automatic normalization (Otherwise things are numerically ill-conditioned)
    # Scale each axis by (max(|origin|, |vertices|) or 1)
    scale = np.ones(N, dtype=float)
    for i, k in enumerate(basis):
        coord_values = obj[k].values if k in obj.coords else np.array([v0[i]])
        max_val = max(
            np.nanmax(np.abs(coord_values)),
            np.nanmax(np.abs(vertices[i])),
            abs(v0[i])
        )
        scale[i] = max_val if max_val != 0 else 1.0
    
    v0 /= scale
    vertices /= scale[:, None]

    vertices -= v0.reshape(-1, 1) # translate to 0

    # dx_j = dv_ij * c_i with c_i in [0, 1] --> (dv^-1)_ij * dx_j in [0, 1]
    # Here I have used dv^-1 to denote a left inverse of dv, because dv is
    # generally not square.
    # This left inverse is calculated using least-squares formulas.
    # This will fail if dv are not linearly independent for obvious reasons, so
    # we check the rank of the matrix and return an all false mask if needed
    if np.linalg.matrix_rank(vertices) < n:
        return xr.full_like(obj, False, dtype = bool)    
    ATA = vertices.T @ vertices
    coeffs = np.linalg.inv(ATA) @ vertices.T # left inverse 
    # (gives us the relevant coefficients for constructing inequalities)

    def mask_from_coefficients(coeff):
        expression = sum([coeff[i]*(obj[basis[i]] / scale[i] - v0[i]) 
                          for i in range(N)])
        return (0 <= expression) & (expression <= 1)

    mask = mask_from_coefficients(coeffs[0, :])
    for r in range(1, n):
        mask &= mask_from_coefficients(coeffs[r, :])

    return mask

"""Dataset Load / Save Tools"""

def _to_builtin(obj):
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if hasattr(obj, "item"):  # catches numpy scalars
        return obj.item()
    return obj

def sqlite_to_xarray(path: str | Path, 
                     duplicate_mode: str = 'stack', 
                     dropbox_mode: bool = False) -> xr.Dataset:
    """
    Load data stored in SQLite using `pyPulses.core.database` into an xarray.Dataset
    """

    path = Path(path).with_suffix('.db')
    try:
        conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    except sqlite3.DatabaseError as e:
        if not dropbox_mode:
            warnings.warn(
                "DB and WAL files are inconsistent. Use `dropbox_mode`=True "
                "to ignore WAL in favor of a stale view."
            )
            raise
        warnings.warn(
            "DB and WAL files are inconsistent. Returning a stale view."
        )
        
        # Fallback: read only the main .db file bytes (ignoring WAL)
        file_conn = sqlite3.connect(str(path))
        # Create an in-memory DB
        conn = sqlite3.connect(":memory:")
        file_conn.backup(conn)
        file_conn.close()

    try:
        cur = conn.cursor()

        # load schema
        cur.execute("SELECT value FROM metadata WHERE key = '__schema__'")
        schema = json.loads(cur.fetchone()[0])

        dims = schema['dims']

        # load sweep data
        df = pd.read_sql("SELECT * FROM sweep", conn)
        if df.empty:
            # handle the empty case
            dims = schema['dims']
            coords = {dim: np.arange(0) for dim in dims}
            ds = xr.Dataset(coords=coords)
        
        else:
            # Convert binary blobs to integers if necessary
            for d in dims:
                if df[d].dtype == 'object' and isinstance(df[d].iloc[0], (bytes, bytearray)):
                    df[d] = df[d].apply(lambda x: np.frombuffer(x, dtype = '<i8')[0])


            # drop id column if present
            if 'id' in df.columns:
                df = df.drop(columns = 'id')

            # handle duplicates
            if duplicate_mode in ('mean', 'max', 'min', 'median'):
                numeric_cols = df.select_dtypes(include = np.number).columns.to_list()
                agg_funcs = {col: duplicate_mode for col in numeric_cols}
                if 'timestamp' in df.columns:
                    agg_funcs['timestamp'] = 'last'
                df = df.groupby(dims).agg(agg_funcs)

            elif duplicate_mode == 'stack' and ('repeat' not in schema['dims']):
                df = df.copy()
                df['repeat'] = df.groupby(dims).cumcount()
                df = df.set_index(dims + ['repeat'])
                dims += ['repeat']

            else:
                df = df.set_index(dims)

            ds = df.to_xarray().transpose(*dims)

            # Ensure dims are plain integer coordinates, not binary junk
            for dim in dims:
                n_actual = int(ds.sizes.get(dim, 0))
                if n_actual > 0:
                    ds = ds.assign_coords({dim: np.arange(n_actual)})
                else:
                    ds = ds.assign_coords({dim: np.arange(0)})

        # attach metadata
        cur.execute("SELECT key, value FROM metadata")
        for key, value in cur.fetchall():
            if key == '__schema__':
                continue
            try:
                ds.attrs[key] = json.loads(value)
            except Exception:
                ds.attrs[key] = value

        cur.execute("SELECT var_name, key, value FROM var_metadata")
        for var, key, value in cur.fetchall():
            if var in ds:
                try:
                    ds[var].attrs[key] = json.loads(value)
                except Exception:
                    ds[var].attrs[key] = value

        # promote physical coordinates to coords
        for cname in schema['coord_names']:
            if cname in ds:
                ds = ds.set_coords(cname)

    finally:
        conn.close()

    return ds

def xarray_to_sqlite(ds: xr.Dataset, path: str | Path, overwrite: bool = False):
    """
    Save an xarray.Dataset to SQLite in the same schema `pyPulses.core.database`
    """

    path = Path(path).with_suffix('.db')
    if path.exists():
        if overwrite: 
            path.unlink()
        else:
            raise FileExistsError(
                f"{path} already exists. Use overwrite = True to replace."
            )
        
    try:
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        # Explicitly use DELETE (rollback) journal mode so that no -wal/-shm
        # files are created alongside the saved database.
        cur.execute("PRAGMA journal_mode=DELETE")

        # create schema tables
        cur.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        cur.execute(
            "CREATE TABLE var_metadata (var_name TEXT, key TEXT, value TEXT, "
            "PRIMARY KEY (var_name, key))"
        )

        # build schema from dataset
        dims = list(ds.dims)
        shape = [ds.sizes[d] for d in dims]
        coord_names = [c for c in ds.coords if c not in dims]
        data_names = list(ds.data_vars)

        schema = {
            'dims'       : dims,
            'shape'      : shape,
            'coord_names': coord_names,
            'data_names' : data_names,
            'timestamp'  : 'timestamp' in ds,
            'version'    : 1.0,
        }

        cur.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ('__schema__', json.dumps(schema)),
        )

        # global attributes
        for k, v in ds.attrs.items():
            cur.execute(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                (k, json.dumps(_to_builtin(v))),
            )

        # variable attributes
        for var in list(ds.coords) + list(ds.data_vars):
            for k, v in ds[var].attrs.items():
                cur.execute(
                    "INSERT INTO var_metadata (var_name, key, value) VALUES (?, ?, ?)",
                    (var, k, json.dumps(_to_builtin(v))),
                )

        # flatten dataset to dataframe
        df = ds.to_dataframe().reset_index()

        # store sweep table
        df.to_sql('sweep', conn, if_exists = 'replace', index = False)
        conn.commit()

    finally:
        conn.close()
