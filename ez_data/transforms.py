from typing import Callable
import numpy as np

class TransformManager:
    """
    Manages user-defined transformation logic for coordinates and measurements.
    """

    def __init__(self):
        self.transforms = {}

    def register(self, name: str, fn: Callable):
        """
        Register a transform function.
        """
        self.transforms[name] = fn

    def apply(self, name: str, coords_or_data: dict) -> dict:
        """
        Apply a registered transform.
        """
        return self.transforms[name](coords_or_data)

def apply_linear_transform(ds, x, y, matrix, offset=None):
    """
    Apply a 2D linear transformation to (x, y) coordinates.
    `matrix` should be a 2x2 numpy array.
    `offset` is an optional length-2 vector.
    """
    coords = np.vstack([ds.coords[x].values, ds.coords[y].values])
    new_coords = matrix @ coords
    if offset is not None:
        new_coords += np.array(offset).reshape(2, 1)
    
    ds = ds.copy()
    ds.coords[x + "_prime"] = ("point", new_coords[0])
    ds.coords[y + "_prime"] = ("point", new_coords[1])
    return ds
