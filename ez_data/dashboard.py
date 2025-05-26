import panel as pn
import xarray as xr
from pathlib import Path
from .visualize import plot_snapped_image

pn.extension()

class LiveDashboard:
    """
    Interactive dashboard for viewing datasets as they are being written.
    """

    def __init__(self, zarr_path: Path, refresh_interval=5):
        self.zarr_path = Path(zarr_path)
        self.refresh_interval = refresh_interval
        self.ds = None

        self.x = pn.widgets.Select(name='X Axis')
        self.y = pn.widgets.Select(name='Y Axis')
        self.z = pn.widgets.Select(name='Z Axis (Color)')

        self.res_x = pn.widgets.FloatInput(name='X Resolution', value=1e-3)
        self.res_y = pn.widgets.FloatInput(name='Y Resolution', value=1e-3)

        self.clim_min = pn.widgets.FloatInput(name="Color Min", value=None)
        self.clim_max = pn.widgets.FloatInput(name="Color Max", value=None)
        self.logz     = pn.widgets.Checkbox(name="Log Scale")
        self.cmap     = pn.widgets.Select(name="Color Map", 
                        options=["viridis", "inferno", "RdBu", "bwr", "plasma"])

        self.reload_button = pn.widgets.Button(name='Refresh Dataset', 
                                               button_type='primary')
        self.plot_button = pn.widgets.Button(name='Update Plot', 
                                             button_type='success')

        self.plot_pane = pn.pane.HoloViews(height=500)

        self.reload_button.on_click(self.reload)
        self.plot_button.on_click(self.plot)

        self.panel = pn.Column(
            pn.Row(self.x, self.y, self.z),
            pn.Row(self.res_x, self.res_y, self.logz),
            pn.Row(self.clim_min, self.clim_max, self.cmap),
            pn.Row(self.reload_button, self.plot_button),
            self.plot_pane
        )

        self.reload()

    def reload(self, *_):
        try:
            self.ds = xr.open_zarr(self.zarr_path)
            self.x.options = list(self.ds.coords)
            self.y.options = list(self.ds.coords)
            self.z.options = list(self.ds.data_vars)
            self.x.value = self.x.options[0] if self.x.options else None
            self.y.value = self.y.options[1] if len(self.y.options) > 1 \
                                                else self.x.value
            self.z.value = self.z.options[0] if self.z.options else None
            self.plot()
        except Exception as e:
            print("Dataset load error:", e)

    def plot(self, *_):
        if self.ds is None:
            return
        try:
            resolution = {self.x.value: self.res_x.value, 
                          self.y.value: self.res_y.value}
            clim_val = None
            if self.clim_min.value is not None and \
                self.clim_max.value is not None:
                clim_val = (self.clim_min.value, self.clim_max.value)
            plot = plot_snapped_image(
                self.ds,
                x=self.x.value,
                y=self.y.value,
                z=self.z.value,
                resolution=resolution,
                clim=clim_val,
                logz=self.logz.value,
                cmap=self.cmap.value
            )
            self.plot_pane.object = plot
        except Exception as e:
            print("Plot error: ", e)

    def show(self):
        return self.panel
