import json
import sqlite3
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

def _to_builtin(obj):
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if hasattr(obj, "item"):  # catches numpy scalars
        return obj.item()
    return obj

class CheckpointMode(Enum):
    NONE     = 'none'
    POINT    = 'point'
    SIZE     = 'size'
    ROWS     = 'rows'

class DsetConnector():
    """
    Lightweight connector for streaming experimental data into SQLite
    Designed for use with ParamSweepMeasure.
    """

    def __init__(
        self, 
        path       : str | Path,
        dims       : List[str] = None,
        shape      : List[int] = None,
        coord_names: List[str] = None,
        data_names : List[str] = None,
        timestamp  : bool      = None,
        use_buffer : bool      = False,
        buffer_size: int       = 100,
        checkpoint_mode: str | CheckpointMode = CheckpointMode.NONE,
    ):

        self.path = Path(path).with_suffix(".db")
        self.use_buffer = use_buffer
        self.buffer_size = buffer_size
        self.buffer: List[tuple] = []
        self.flush_count = 0
        self.points_count = 0

        if isinstance(checkpoint_mode, CheckpointMode):
            self.checkpoint_mode = checkpoint_mode
            self.checkpoint_param = None
        elif isinstance(checkpoint_mode, str):
            if checkpoint_mode in ('none', 'point'):
                self.checkpoint_mode = CheckpointMode(checkpoint_mode)
                self.checkpoint_param = None
            elif checkpoint_mode.startswith("rows:"):
                self.checkpoint_mode = CheckpointMode.ROWS
                self.checkpoint_param = int(checkpoint_mode.split(":")[1])
            elif checkpoint_mode.startswith("size:"):
                self.checkpoint_mode = CheckpointMode.SIZE
                self.checkpoint_param = int(checkpoint_mode.split(":")[1])
            else:
                raise ValueError(f"Invalid checkpoint_mode: {checkpoint_mode}")
        else:
            raise TypeError("checkpoint_mode must be str or CheckpointMode")

        self.conn = sqlite3.connect(self.path, timeout = 30)
        self.cur = self.conn.cursor()

        # safe optimizations for this use case
        self.cur.execute("PRAGMA journal_mode=WAL")
        self.cur.execute("PRAGMA synchronous=NORMAL")
        self.cur.execute("PRAGMA temp_store=MEMORY")

        if self._has_tables():
            # Existing file -> load schema info
            schema = self._load_schema()
            self.dims = dims or schema['dims']
            self.shape = shape or schema.get('shape')
            self.coord_names = coord_names or schema['coord_names']
            self.data_names = data_names or schema['data_names']
            self.timestamp = timestamp or schema.get('timestamp')

        else:
            # New file -> need full schema
            if dims is None or coord_names is None or data_names is None:
                raise ValueError(
                    "dims, coord_names, and data_names required for new database."
                )
            
            self.dims = dims
            self.shape = shape
            self.coord_names = coord_names
            self.data_names = data_names
            self.timestamp = timestamp

            self._init_tables()
            self._store_schema()

        self._insert_sql = self._make_insert_sql()

    def _has_tables(self) -> bool:
        self.cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return bool(self.cur.fetchall())
    
    def _init_tables(self):
        cols = [f"{dim} INT" for dim in self.dims]
        cols += [f"{n} REAL" for n in (self.coord_names + self.data_names)]
        if self.timestamp:
            cols.append("timestamp REAL")

        self.cur.execute(
            f"CREATE TABLE IF NOT EXISTS sweep"
            f"(id INTEGER PRIMARY KEY AUTOINCREMENT, {', '.join(cols)})"
        )
        self.cur.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)"
        )
        self.cur.execute(
            """CREATE TABLE IF NOT EXISTS var_metadata (
                var_name TEXT,
                key TEXT,
                value TEXT,
                PRIMARY KEY (var_name, key)
            )"""
        )
        self.conn.commit()

    def _store_schema(self):
        schema = {
            'dims'       : self.dims,
            'shape'      : self.shape,
            'coord_names': self.coord_names,
            'data_names' : self.data_names,
            'timestamp'  : self.timestamp,
            'version'    : 1.0,
        }
        self.cur.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("__schema__", json.dumps(_to_builtin(schema)))
        )
        self.conn.commit()

    def _load_schema(self) -> dict:
        self.cur.execute("SELECT value FROM metadata WHERE key='__schema__'")
        row = self.cur.fetchone()
        if not row:
            raise RuntimeError("Existing database has no schema metadata")
        return json.loads(row[0])
    
    def _make_insert_sql(self) -> str:
        cols = self.dims + self.coord_names + self.data_names
        if self.timestamp:
            cols.append('timestamp')
        placeholders = ','.join(['?'] * len(cols))
        return f"INSERT INTO sweep ({','.join(cols)}) VALUES ({placeholders})"
    
    @classmethod
    def from_param_sweep(
        cls,
        sweep,
        path: str | Path,
        use_buffer  : bool = False,
        buffer_size : int  = 100,
        checkpoint_mode: str = 'none',
    ):
        dims = [f"dim{i}" for i in range(sweep.dim)]
        shape = list(sweep.dimensions)
        connector = cls(
            path,
            dims        = dims,
            shape       = shape,
            coord_names = sweep.coord_name,
            data_names  = sweep.data_name,
            timestamp   = sweep.timestamp,
            use_buffer  = use_buffer,
            buffer_size = buffer_size,
            checkpoint_mode = checkpoint_mode,
        )

        # add sweep metadata
        connector.add_global_attrs({
            'time_per_point': getattr(sweep, 'time_per_point', None),
        })

        # also populate variable-specific metadata
        for coord in getattr(sweep, 'coordinates', []):
            connector.add_variable_attrs(coord['name'], {
                'long_name': coord.get('long_name'),
                'units': coord.get('units'),
            })
        for meas in getattr(sweep, 'measurements', []):
            connector.add_variable_attrs(meas['name'], {
                'long_name': meas.get('long_name'),
                'units': meas.get('units'),
                'lazy_measurement': meas.get('lazy', False),
            })

        return connector
    
    def add_point(
        self, 
        idx      : List[int], 
        coords   : List[float], 
        data     : List[float], 
        timestamp: float | None = None
    ):
        self.points_count += 1
        row = tuple(idx) + tuple(coords) + tuple(data)
        if self.timestamp:
            row += (timestamp,)

        if self.use_buffer:
            self.buffer.append(row)
            if len(self.buffer) >= self.buffer_size:
                self.flush()
        else:
            self.cur.execute(self._insert_sql, row)
            self.conn.commit()
            self.flush()

    def flush(self):
        self.flush_count += 1
        if self.buffer:
            self.cur.executemany(self._insert_sql, self.buffer)
            self.conn.commit()
            self.buffer.clear()

        if self.checkpoint_mode == CheckpointMode.NONE:
            return
        elif self.checkpoint_mode == CheckpointMode.POINT:
            self._checkpoint()
        elif self.checkpoint_mode == CheckpointMode.SIZE:
            if self.points_count >= self.checkpoint_param:
                self._checkpoint()
                self.points_count = 0
        elif self.checkpoint_mode == CheckpointMode.ROWS:
            if self.flush_count >= self.checkpoint_param:
                self._checkpoint()
                self.flush_count = 0
    
    def _checkpoint(self):
        busy, log_frames, checkpointed_frames = self.cur.execute(
            f"PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        self.conn.commit()

        if busy:
            warnings.warn(
                f"WAL checkpoint could not complete (busy). "
                f"Frames in WAL: {log_frames}, checkpointed: {checkpointed_frames}. "
                "Readers may see stale data until a later checkpoint succeeds."
            )

    def add_global_attrs(self, attrs: Dict[str, Any]):
        for k, v in attrs.items():
            self.cur.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (k, str(v)),
            )
        self.conn.commit()

    def add_variable_attrs(self, var: str, attrs: Dict[str, Any]):
        if var not in (self.coord_names + self.data_names + \
                       (['timestamp'] if self.timestamp else [])):
            raise ValueError(f"Unknown variable {var}")
        for k, v in attrs.items():
            self.cur.execute(
                "INSERT OR REPLACE INTO var_metadata (var_name, key, value) VALUES (?, ?, ?)",
                (var, k, json.dumps(_to_builtin(v))),
            )
        self.conn.commit()

    def close(self):
        self.flush()
        self.conn.close()

def sqlite_to_xarray(path: str | Path, 
                     duplicate_mode: str = 'stack', 
                     dropbox_mode: bool = False) -> xr.Dataset:
    """
    Load a sweep stored in SQLite into an xarray.Dataset
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
    Save an xarray.Dataset to SQLite in the same schema as DsetConnector
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
