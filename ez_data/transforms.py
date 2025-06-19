import numpy as np
import pandas as pd
import xarray as xr
from typing import Callable, List, Tuple, Union

def preserve_attrs(obj, col, data, dims, xr_output_type):
    # Get attrs if variable/coord exists
    attrs = {}
    if col in obj:
        attrs = obj[col].attrs.copy()
    # Create new DataArray with attrs
    da = xr.DataArray(data, dims=dims, attrs=attrs)
    if xr_output_type == 'coords':
        obj = obj.assign_coords({col: da})
    else:
        obj[col] = da
    return obj

def apply_transform(obj: Union[pd.DataFrame, xr.Dataset], 
                    func: Callable[[Union[float, Tuple[float]]], 
                                    Union[float, Tuple[float]]], 
                    input_cols: Union[str, List[str]], 
                    output_cols: Union[str, List[str]], 
                    xr_output_type: str = 'coords'):
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
        # All arrays are broadcastable to a common shape
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


def apply_linear_transform(obj: Union[pd.DataFrame, xr.Dataset], 
                           matrix: np.ndarray, 
                           input_cols: Union[str, List[str]], 
                           output_cols: Union[str, List[str]], 
                           xr_output_type: str = 'coords'):
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
        # Broadcast all input variables to a common shape
        arrays = [obj[var].values for var in input_cols]
        # Stack all input arrays along a new last axis
        stacked = np.stack(arrays, axis=-1)  # shape (..., n_inputs)
        # Reshape to (N, n_inputs)
        orig_shape = stacked.shape[:-1]
        n_inputs = stacked.shape[-1]
        flat = stacked.reshape(-1, n_inputs)
        Y = np.dot(flat, matrix.T)  # shape (N, n_outputs)
        # Reshape back to original shape + n_outputs
        Y = Y.reshape(*orig_shape, matrix.shape[0])
        for i, col in enumerate(output_cols):
            data = Y[..., i]
            dims = obj[input_cols[0]].dims  # Use dims from first input var
            preserve_attrs(obj, col, data, dims, xr_output_type)
            
        return obj

    else:
        raise ValueError("Input must be a pandas DataFrame or xarray.Dataset")
