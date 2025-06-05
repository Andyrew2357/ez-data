"""
Compatibility tools for processing data taken using smarty sweep / special 
measure.
"""

import pandas as pd
import xarray as xr
import numpy as np
import os
from typing import Callable, Dict, List, Optional, Union

def dat_to_pandas(folder: str, suffix: str, 
                  index_func: Callable[[List[float]], 
                    Union[float, List[float]]] = lambda x: x,
                  column_mapping: Optional[Dict[str, str]] = None, 
                  delimiter = r"\s*\t\s*") -> pd.DataFrame:
    """
    Read scan into a pandas DataFrame, inferring indices based on provided
    function and renaming columns based on provided dictionary
    """

    files = [os.path.join(folder, path) for path in os.listdir(folder)
             if path.endswith(suffix)]
    
    col_map = {v: k for k, v in column_mapping.items()}
    for path in files:
        df = pd.read_csv(path, delimiter=delimiter, engine = 'python')
        df = df.rename(columns=col_map)
        df = index_func(df) # typically use apply_transform
        try:
            D = pd.concat([D, df.copy()], ignore_index=True)
        except:
            D = df.copy()
    
    return D

def pandas_to_xarray(df: pd.DataFrame, coord_attr: dict, data_attr: dict, 
                     global_attr: dict = {}, ignored_attr: List[str] = []
                     ) -> xr.Dataset:
    """
    Convert pandas DataFrame and applicable metadata into an xarray Dataset
    """

    idx_cols = [col for col in df.columns if col not in coord_attr and \
                col not in data_attr and col not in ignored_attr]
    max_indices = {col: int(df[col].max()) for col in idx_cols}
    shape = tuple(max_indices[col] + 1 for col in idx_cols)
    dims = tuple(idx_cols)

    inds = {dim: np.arange(size) for dim, size in zip(dims, shape)}
    multi_idx = tuple(df[col].astype(int).values for col in idx_cols)
    flat_idx = np.ravel_multi_index(multi_idx, shape)

    def fill_array(values, shape, dtype, fill_value):
        arr = np.full(np.prod(shape), fill_value, dtype=dtype)
        arr[flat_idx] = values
        return arr.reshape(shape)

    def is_datetime_column(series: pd.Series) -> bool:
        dtype = series.dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return True
        if pd.api.types.is_object_dtype(dtype) or \
            pd.api.types.is_string_dtype(dtype):
            sample = series.dropna().astype(str).head(5)
            try:
                pd.to_datetime(sample, errors='raise')
                return True
            except Exception:
                return False
        return False

    data_vars = {}
    for name, attrs in data_attr.items():
        if name not in df.columns:
            continue
        series = df[name]
        if is_datetime_column(series):
            values = pd.to_datetime(series).astype('datetime64[ns]')
            arr = fill_array(values, shape, 'datetime64[ns]', 
                             np.datetime64('NaT'))
        else:
            values = series.astype(float).values
            arr = fill_array(values, shape, float, np.nan)
        data_vars[name] = (dims, arr, attrs)

    coord_vars = {}
    for name, attrs in coord_attr.items():
        if name not in df.columns:
            continue
        series = df[name]
        if is_datetime_column(series):
            values = pd.to_datetime(series).astype('datetime64[ns]')
            arr = fill_array(values, shape, 'datetime64[ns]', 
                             np.datetime64('NaT'))
        else:
            values = series.astype(float).values
            arr = fill_array(values, shape, float, np.nan)
        coord_vars[name] = (dims, arr, attrs)

    return xr.Dataset(coords=coord_vars, data_vars=data_vars, attrs=global_attr)
