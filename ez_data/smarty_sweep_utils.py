"""
Compatibility tools for processing data taken using smarty sweep or special 
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

def pandas_to_xarray(df: pd.DataFrame, coord_attr: dict, 
                     data_attr: dict, global_attr: dict = {}) -> xr.Dataset:
    """
    Convert pandas DataFrame and applicable metadata into an xarray Dataset
    """
    
    idx_cols = [col for col in df.columns if not (col in coord_attr or \
                                                  col in data_attr)]
    max_indices = {col: df[col].max() for col in idx_cols}
    # Get the shape from max_indices
    shape = tuple(max_indices[dim] + 1 for dim in max_indices)

    # Create logical coordinate arrays for the indices
    inds = {dim: np.arange(idx + 1) for dim, idx in max_indices.items()}

    # Create the structured dataset
    data_vars = {}
    for var_name in data_attr:
        if var_name not in df.columns:
            continue
        
        # Create empty array filled with NaN
        data_array = np.full(shape, np.nan)

        # Fill in the data at the appropriate indices
        for _, row in df.iterrows():
            indices_tuple = tuple(int((row[dim])) for dim in max_indices)
            # Only fill in the data if all indices are valid
            if all(idx >= 0 for idx in indices_tuple):
                data_array[indices_tuple] = row[var_name]

        data_vars[var_name] = (inds, data_array, data_attr)
    
    coord_vars = {}
    for var_name in coord_attr:
        if var_name not in df.columns:
            continue
        
        # Create empty array filled with NaN
        data_array = np.full(shape, np.nan)

        # Fill in the data at the appropriate indices
        for _, row in df.iterrows():
            indices_tuple = tuple(int((row[dim])) for dim in max_indices)
            # Only fill in the data if all indices are valid
            if all(idx >= 0 for idx in indices_tuple):
                data_array[indices_tuple] = row[var_name]

        coord_vars[var_name] = (inds, data_array, coord_attr)

    return xr.Dataset(coords = coord_vars, data_vars = data_vars, 
                      attrs = global_attr)
