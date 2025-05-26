import xarray as xr
import time
import warnings
from pathlib import Path
from typing import Dict, Union, List, Optional

warnings.filterwarnings("ignore", message=".*Consolidated metadata.*")

class MeasurementDataset:
    """
    Main class for collecting, saving, and merging experimental datasets
    """

    def __init__(self, path: Union[str, Path], 
                 merge_paths: Optional[List[Union[str, Path]]] = None,
                 flush_interval=5, flush_batch_size=100):
        self.path = Path(path)
        self.ds = xr.Dataset()
        self.buffer = []
        self.counter = 0
        self.last_flush_time = time.time()
        self.flush_interval = flush_interval
        self.flush_batch_size = flush_batch_size

        if merge_paths:
            self._load_and_merge(merge_paths)

    def _load_and_merge(self, paths: List[Union[str, Path]]):
        datasets = [xr.open_zarr(p) for p in paths]
        self.ds = xr.concat(datasets, dim="point")

    def add_point(self, coords: Dict[str, float], data: Dict[str, float],
                  coord_attrs: Optional[Dict[str, Dict]] = None,
                  data_attrs: Optional[Dict[str, Dict]] = None):
        """
        Add a new point to the dataset, optionally with metadata (units, labels)
        """
        point_id = self.counter
        self.counter += 1
        self.buffer.append((point_id, coords, data, coord_attrs, data_attrs))
        self._maybe_flush()

    def _maybe_flush(self):
        now = time.time()
        if len(self.buffer) >= self.flush_batch_size or \
            (now - self.last_flush_time) > self.flush_interval:
            self.flush_to_zarr()


    def flush_to_zarr(self):
        if not self.buffer:
            return

        # Collect all data from buffer first
        point_ids = []
        all_coords = {}
        all_data = {}
        _, _, _, coord_attr, data_attr = self.buffer[0]
        
        for point_id, coords, data, _, _ in self.buffer:
            point_ids.append(point_id)
            
            # Collect coordinate data
            for k, v in coords.items():
                if k not in all_coords:
                    all_coords[k] = []
                all_coords[k].append(v)
            
            # Collect data variables
            for k, v in data.items():
                if k not in all_data:
                    all_data[k] = []
                all_data[k].append(v)

        # Create new dataset with all buffered data
        new_ds = xr.Dataset()
        
        # Add coordinates
        new_ds.coords["point"] = ("point", point_ids)
        for coord_name, coord_values in all_coords.items():
            new_ds.coords[coord_name] = ("point", coord_values, 
                                         coord_attr[coord_name])
        
        # Add data variables
        for var_name, var_values in all_data.items():
            new_ds[var_name] = xr.DataArray(var_values, dims=["point"], 
                                            coords={"point": point_ids}, 
                                            attrs = data_attr[var_name])

        if not self.path.exists():
            # Set up encoding for chunking
            encoding = {
                name: {"chunks": (self.flush_batch_size,)} 
                for name in new_ds.data_vars
            }
            encoding.update({
                name: {"chunks": (self.flush_batch_size,)} 
                for name in new_ds.coords if name != "point"
            })

            new_ds.to_zarr(str(self.path), mode="w", encoding=encoding)
        else:
            new_ds.to_zarr(str(self.path), mode="a", append_dim="point")

        self.buffer = []
        self.last_flush_time = time.time()

    def save_complete(self):
        """
        Write the complete dataset to disk (used for merged or finalized data)
        """
        encoding = {
            name: {"chunks": (self.ds.sizes["point"],)} 
            for name in self.ds.data_vars
        }
        encoding.update({
            name: {"chunks": (self.ds.sizes["point"],)} 
            for name in self.ds.coords if name != "point"
        })
        self.ds.to_zarr(str(self.path), mode="w", encoding=encoding)

    def load(self):
        """Load from Zarr path into self.ds"""
        self.ds = xr.open_zarr(self.path)

    def merge(self, others: List[xr.Dataset]):
        """Merge with other datasets (must be same schema)"""
        self.ds = xr.concat([self.ds] + others, dim="point")
