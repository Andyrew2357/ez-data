# STILL NEED TO WORK OUT HOW THIS ACTUALLY WORKS
import numpy as np
import xarray as xr
from typing import Callable, Dict

def apply_transforms(ds: xr.Dataset, transform_fn: Callable[[Dict[str, float]], Dict[str, float]]):
    new_coords = {name: [] for name in transform_fn({k: 0.0 for k in ds.coords}).keys()}
    for i in range(len(ds.point)):
        point = {k: float(ds.coords[k][i].values) for k in ds.coords}
        transformed = transform_fn(point)
        for name, val in transformed.items():
            new_coords[name].append(val)
    for name, values in new_coords.items():
        ds.coords[name] = ('point', np.array(values))
    return ds
