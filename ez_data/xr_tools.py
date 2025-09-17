"""Accessors to utilities for xarray objects"""

from .fitting import curve_fit, curve_fit_peaks, fit_peaks
from .utils import (align_dims, apply_transform, apply_linear_transform, 
                    bin_to_grid, parallelepiped_mask, smart_sel, 
                    where_parallelepiped)
from .dataset_connector import sqlite_to_xarray, xarray_to_sqlite
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Callable, List, Tuple

@xr.register_dataset_accessor("ez")
class ezDatasetAccessor():
    def __init__(self, xr_obj: xr.Dataset):
        self._obj = xr_obj

    @staticmethod
    def from_sqlite(path: str | Path, 
                    duplicate_mode: str = 'stack', 
                    dropbox_mode: bool = False) -> xr.Dataset:
        """Accessor for dataset_connector.sqlite_to_xarray"""
        return sqlite_to_xarray(path, duplicate_mode = duplicate_mode, 
                                dropbox_mode = dropbox_mode)

    def to_sqlite(self, path: str | Path, overwrite: bool = False):
        """Accessor for dataset_connector.xarray_to_sqlite"""
        xarray_to_sqlite(self._obj, path, overwrite = overwrite)

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
        ds = self._obj.copy()
        return apply_transform(ds, func, input_cols, output_cols, 
                               xr_output_type)
    
    def linear_transform(self,
                         matrix: np.ndarray, 
                         input_cols: str | List[str], 
                         output_cols: str | List[str], 
                         xr_output_type: str = 'coords') -> xr.Dataset:
        """Accessor to apply_linear_transform"""
        ds = self._obj.copy()
        return apply_linear_transform(ds, matrix, input_cols, output_cols, 
                                      xr_output_type)
    
    def gridded(self, reduce: str | Callable = 'mean', **bins) -> xr.Dataset:
        """Accessor to bin_to_grid"""
        return bin_to_grid(self._obj, reduce, **bins)
    
    def where_parallelepiped(self, origin: dict[str, float], 
                             *verts: Tuple[List[float]]) -> xr.Dataset:
        """Accessor to where_parallelepiped"""
        return where_parallelepiped(self._obj, origin, *verts)
    
    def parallelepiped_mask(self, origin: dict[str, float], 
                            *verts: Tuple[List[float]]) -> xr.DataArray:
        """Accessor to parallelepiped_mask"""
        return parallelepiped_mask(self._obj, origin, *verts)

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
    
    def fit_peaks(self, x: str, y: str, dim: str = None,
                  peak_model: str = "gaussian",
                  min_peak_height: float = None,
                  background: str = 'linear',
                  initial_guesses: dict | Callable = None,
                  min_valid_points: int = 4,
                  guess_from_mask: bool = True) -> xr.Dataset:
        """Accessor to fit_peaks"""
        return fit_peaks(self._obj, x, y, dim, peak_model, 
                         min_peak_height, background, initial_guesses, 
                         min_valid_points, guess_from_mask)
    
    def curve_fit(self, x: str, yerr: xr.DataArray = None, 
                  curve_model: str | Callable = 'line', 
                  initial_guesses: dict = None, **kwargs) -> xr.Dataset:
        """Accessor to curve_fit"""
        return curve_fit(self._obj[x], self._obj, yerr, curve_model, 
                         initial_guesses, **kwargs)

    def curve_fit_peaks(self, x: str, y: str, 
                        param: str = 'peak_center',
                        curve_model: str | Callable = 'line',
                        curve_fit_kwargs: dict = None,
                        **fit_peaks_kwargs) -> xr.Dataset:
        """Accessor to curve_fit_peaks"""
        return curve_fit_peaks(self._obj, x, y, param, curve_model, 
                               curve_fit_kwargs, **fit_peaks_kwargs)

    def where_parallelepiped(self, origin: dict[str, float], 
                             *verts: Tuple[List[float]]) -> xr.DataArray:
        """Accessor to where_parallelepiped"""
        return where_parallelepiped(self._obj, origin, *verts)

    def parallelepiped_mask(self, origin: dict[str, float], 
                            *verts: Tuple[List[float]]) -> xr.DataArray:
        """Accessor to parallelepiped_mask"""
        return parallelepiped_mask(self._obj, origin, *verts)
