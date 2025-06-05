import pandas as pd
import xarray as xr
import numpy as np
import pickle
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import sqlite3
import h5py

class Dataset:
    """
    Efficient dataset for structured experimental data using pandas backend.

    This class maintains sampling structure through index coordinates while
    efficiently storing data incrementally. Converts to xarray for analysis.
    """

    def __init__(self, path: Union[str, Path],      # path to save/load dataset
                 storage_backend: str = 'sqlite',   # 'hdf5' or 'sqlite'
                 flush_interval: float = 30.0,      # seconds
                 flush_batch_size: int = 500,       # max points before flushing
                 index_names: Optional[List[str]] = None):
        
        self.path = Path(path)
        self.storage_backend = storage_backend
        self.flush_interval = flush_interval
        self.flush_batch_size = flush_batch_size
        self.index_names = index_names

        # In-memory buffer as lists of dicts
        self.buffer = []
        self.last_flush_time = time.time()

        # Metadata storage
        self.variable_attrs = {}
        self.global_attrs = {}
        self.nind = None
        self.max_indices = {}
        self.coord_vars = None

        # Set up storage path with appropriate extension
        extensions = {'hdf5': '.h5', 'sqlite': '.db'}
        if not self.path.suffix:
            self.path = self.path.with_suffix(extensions[self.storage_backend])

    def add_variable_attrs(self, attrs: Dict[str, Any]):
        """Set attributes for variables."""

        self.variable_attrs.update(**attrs)

    def add_global_attrs(self, attrs: Dict[str, Any]):
        """Set attributes for dataset."""

        self.global_attrs.update(**attrs)

    def add_point(self, indices: List[int], coords: Dict[str, float], 
                  data: Dict[str, float]):
        """Add a new point to the dataset."""

        if not self.nind:
            self.nind = len(indices)

        if len(indices) != self.nind:
            raise RuntimeError(
                "Provided data indices are insonsistent with previous data."
            )
        
        # Update max indices
        if self.index_names:
            indices = {self.index_names[i]: indices[i]
                       for i in range(len(indices))}
        else:
            indices = {f'dim{i}': indices[i] for i in range(len(indices))}

        for dim, idx in indices.items():
            if dim in self.max_indices:
                self.max_indices[dim] = max(self.max_indices[dim], idx)
            else:
                self.max_indices[dim] = idx

        # Fill in coord_vars
        if not self.coord_vars:
            self.coord_vars = list(coords.keys())
        
        # Create the data point
        point = {**indices, **coords, **data}
        self.buffer.append(point)
        self._maybe_flush()

    def _maybe_flush(self):
        """Flush the buffer if it's getting large or enough time has passed"""

        now = time.time()
        if len(self.buffer) >= self.flush_batch_size or \
            (now - self.last_flush_time) > self.flush_interval:
            self.flush()

    def flush(self):
        """Write buffer to storage."""
        
        if not self.buffer:
            return
        
        # Convert buffer to DataFrame
        df = pd.DataFrame(self.buffer)
        
        # Write to storage
        match self.storage_backend:
            case 'hdf5':
                self._flush_hdf5(df)
            case 'sqlite':
                self._flush_sqlite(df)
            case _:
                raise RuntimeError("Unsupported backend.")
            
        # Clear buffer and update timestamp
        self.buffer = []
        self.last_flush_time = time.time()

    def _flush_hdf5(self, df: pd.DataFrame):
        """Flush to HDF5 format."""
        
        mode = 'a' if self.path.exists() else 'w'
        df.to_hdf(str(self.path), key = 'data', mode = mode, 
                  append = (mode == 'a'), format = 'table', 
                  complib = 'blosc', complevel = 9)
        
    def _flush_sqlite(self, df: pd.DataFrame):
        """Flush to SQLite format."""

        conn = sqlite3.connect(str(self.path))
        df.to_sql('data', conn, if_exists = 'append', index = False)
        conn.commit()
        conn.close()

    def write_metadata(self):
        """Write metadata to storage."""

        match self.storage_backend:
            case 'hdf5':
                self._write_metadata_hdf5()
            case 'sqlite':
                self._write_metadata_sqlite()
            case _:
                raise RuntimeError("Unsupported backend.")
            
    def _write_metadata_hdf5(self):
        """Write metadata to hdf5."""

        mode = 'a' if self.path.exists() else 'w'
        with h5py.File(self.path, mode) as f:
            f.attrs['variable_attrs'] = json.dumps(self.variable_attrs)
            f.attrs['global_attrs']   = json.dumps(self.global_attrs)
            f.attrs['max_indices']    = json.dumps(self.max_indices)
            f.attrs['coord_vars']     = json.dumps(self.coord_vars)

    def _write_metadata_sqlite(self):
        """Write the metadata to SQLite."""

        conn = sqlite3.connect(str(self.path))
        metadata = {
            'variable_attrs': self.variable_attrs,
            'global_attrs': self.global_attrs,
            'max_indices'   : self.max_indices,
            'coord_vars'    : self.coord_vars
        }
        conn.execute(
            'CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value BLOB)'
        )
        conn.execute(
            'INSERT OR REPLACE INTO metadata VALUES (?, ?)',
            ('config', sqlite3.Binary(pickle.dumps(metadata)))
        )
        conn.commit()
        conn.close()

    def load_dataframe(self) -> pd.DataFrame:
        """Load the complete dataset as a pandas DataFrame."""

        # Flush any remaining buffer data
        self.flush()

        if not self.path.exists():
            return pd.DataFrame()
        
        match self.storage_backend:
            case 'hdf5':
                return self._load_hdf5()
            case 'sqlite':
                return self._load_sqlite()
            case _:
                raise RuntimeError("Unsupported backend.")
            
    def _load_hdf5(self):
        """Load from HDF5."""

        df = pd.read_hdf(str(self.path), key = 'data')

        # Load metadata
        with h5py.File(self.path, 'r') as f:
            self.variable_attrs = json.loads(
                f.attrs.get('variable_attrs', "{}")
            )
            self.global_attrs = json.loads(
                f.attrs.get('global_attrs', "{}")
            )
            self.max_indices = json.loads(
                f.attrs.get('max_indices', "{}")
            )
            self.coord_vars = json.loads(
                f.attrs.get('coord_vars', "[]")
            )

        return df
    
    def _load_sqlite(self) -> pd.DataFrame:
        """Load from SQLite."""

        conn = sqlite3.connect(str(self.path))
        df = pd.read_sql_query('SELECT * FROM data', conn)

        try:
            cursor = conn.execute('SELECT value FROM metadata WHERE key = ?',
                                  ('config',))
            row = cursor.fetchone()
            if row:
                try:
                    metadata = pickle.loads(row[0])
                    self.variable_attrs = metadata.get('variable_attrs', {})
                    self.global_attrs = metadata.get('global_attrs', {})
                    self.max_indices = metadata.get('max_indices', {})
                    self.coord_vars = metadata.get('coord_vars', [])
                except Exception as e:
                    raise RuntimeError("Pickle load error:", e)
        
        except sqlite3.OperationalError as e:
            raise RuntimeError("OperationalError:", e)
        
        conn.close()
        return df
    
    def to_xarray(self, variables: Optional[List[str]] = None) -> xr.Dataset:
        """
        Convert to xarray Dataset with proper structure.
        
        Returns a dataset where data is arranged according to the indices.
        (These form the 'logical coordinates', wheras coords form the 
        'physical coordinates')
        """

        df = self.load_dataframe()
        if df.empty:
            return xr.Dataset()
        
        if not variables:
            variables = [col for col in df.columns 
                         if col not in self.max_indices]
        coord_v   = self.coord_vars
        data_v    = [var for var in variables if var not in coord_v]

        # Get the shape from max indices
        shape = tuple(self.max_indices[dim] + 1 for dim in self.max_indices)

        # Create logical coordinate arrays for the indices
        inds = {dim: np.arange(idx + 1) for dim, idx 
                in self.max_indices.items()}
        
        # Create the structured dataset
        data_vars = {}
        for var_name in data_v:
            if var_name not in df.columns:
                continue
            
            # Create empty array filled with NaN or NaT
            if type(df[var_name][0]) in [str, np.datetime64]:
                data_array = np.full(shape, 'NaT', dtype = 'datetime64[s]')
            else:
                data_array = np.full(shape, np.nan)

            # Fill in the data at the appropriate indices
            for _, row in df.iterrows():
                indices_tuple = tuple(int((row[dim])) for dim 
                                      in self.max_indices)
                # Only fill if all indices are valid
                if all(idx >= 0 for idx in indices_tuple):
                    data_array[indices_tuple] = row[var_name]

            attrs = self.variable_attrs.get(var_name, {})
            data_vars[var_name] = (inds, data_array, attrs)

        coord_vars = {}
        for var_name in coord_v:
            if var_name not in df.columns:
                continue
            
            # Create empty array filled with NaN or NaT
            if type(df[var_name][0]) in [str, np.datetime64]:
                data_array = np.full(shape, 'NaT', dtype = 'datetime64[s]')
            else:
                data_array = np.full(shape, np.nan)

            # Fill in the data at the appropriate indices
            for _, row in df.iterrows():
                indices_tuple = tuple(int((row[dim])) for dim 
                                      in self.max_indices)
                # Only fill if all indices are valid
                if all(idx >= 0 for idx in indices_tuple):
                    data_array[indices_tuple] = row[var_name]

            attrs = self.variable_attrs.get(var_name, {})
            coord_vars[var_name] = (inds, data_array, attrs)

        return xr.Dataset(coords = coord_vars, data_vars = data_vars, 
                          attrs = self.global_attrs)

    def get_flat_data(self) -> pd.DataFrame:
        """Get the data in flat format."""
        return self.load_dataframe()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the dataset."""
        df = self.load_dataframe()
        
        if df.empty:
            return {"empty": True}
        
        summary = {
            "shape"         : tuple(self.max_indices[dim] + 1 for dim \
                                in self.max_indices.keys()),
            "total_points"  : len(df),
            "index_dims"    : self.max_indices.keys(),
            "variables"     : [col for col in df.columns \
                                if col not in self.max_indices],
            "storage_backend": self.storage_backend,
            "file_size_mb"  : self.path.stat().st_size / (1024*1024) \
                                if self.path.exists() else 0
        }

        # Fill percentage (how much of the theoretical grid is filled)
        theoretical_points = np.prod([self.max_indices[dim] + 1 for dim \
                                      in self.max_indices.keys()])
        summary["fill_percentage"] = (len(df)/ theoretical_points) * 100

        return summary
