from .utils import (align_dims, apply_transform, apply_linear_transform, 
                    bin_to_grid, smart_sel)

import numpy as np
import xarray as xr
from typing import Callable, List, Tuple

@xr.register_dataset_accessor("ez")
class ezDatasetAccessor():
    def __init__(self, xr_obj: xr.Dataset):
        self._obj = xr_obj

    def align_dims(self, replace: bool = False, **alignment) -> xr.Dataset:
        """Accessor for utils.align_dims"""
        return align_dims(self._obj, replace, **alignment)

    def sel(self, tolerance: float = None, **coords) -> xr.Dataset:
        """Accessor for utils.smart_sel"""
        return smart_sel(self._obj, tolerance, **coords)
    
    def transform(self, 
                  func: Callable[[float | Tuple[float]], float | Tuple[float]], 
                  input_cols: str | List[str], 
                  output_cols: str | List[str], 
                  xr_output_type: str = 'coords') -> xr.Dataset:
        """Accessor to apply_transform"""
        return apply_transform(self._obj, func, input_cols, 
                               output_cols, xr_output_type)
    
    def linear_transform(self,
                         matrix: np.ndarray, 
                         input_cols: str | List[str], 
                         output_cols: str | List[str], 
                         xr_output_type: str = 'coords') -> xr.Dataset:
        """Accessor to apply_linear_transform"""
        return apply_linear_transform(self._obj, matrix, input_cols, 
                                      output_cols, xr_output_type)
    
    def gridded(self, reduce: str | Callable, **bins) -> xr.Dataset:
        """Accessor to bin_to_grid"""
        return bin_to_grid(self._obj, reduce, **bins)

@xr.register_dataarray_accessor("ez")
class ezDataArrayAccessor():
    def __init__(self, xr_obj: xr.DataArray):
        self._obj = xr_obj

    def align_dims(self, replace: bool = False, **alignment) -> xr.DataArray:
        """Accessor for utils.align_dims"""
        return align_dims(self._obj, replace, **alignment)

    def sel(self, tolerance: float = None, **coords) -> xr.DataArray:
        """Accessor for utils.smart_sel"""
        return smart_sel(self._obj, tolerance, **coords)
    
    def gridded(self, reduce: str | Callable = 'mean', **bins) -> xr.DataArray:
        """Accessor to bin_to_grid"""
        return bin_to_grid(self._obj, reduce, **bins)
