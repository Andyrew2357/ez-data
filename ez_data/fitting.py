import lmfit
from lmfit.models import (ConstantModel, GaussianModel, LorentzianModel,
                          LinearModel, VoigtModel)

import numpy as np
import xarray as xr
from typing import Callable, List

import warnings  # hyper pedantic ufloat warning
warnings.filterwarnings(
    "ignore",
    message=".*std_dev==0.*",
    category=UserWarning,
    module="uncertainties"
)

# ---------------------------------------------------------------------------
# Parameter transform registry
# ---------------------------------------------------------------------------
# Maps the suffix of a parameter name (after the last '_') to a transform
# type understood by _DataScaler.restore_result.
#
# When adding a new model whose parameters follow the naming conventions below,
# no other code needs to change.  The transform types are:
#
#   x_center    : e.g. peak_center   →  x_mean + x_std * val
#   x_scale     : e.g. peak_sigma    →  x_std * val
#   y_scale     : e.g. peak_amplitude→  y_std * val
#   xy_slope    : e.g. bg_slope      →  (y_std / x_std) * val
#   y_intercept : e.g. bg_intercept  →  y_mean + y_std * (val - x_mean/x_std * slope_norm)
#   y_const     : e.g. bg_c          →  y_mean + y_std * val
#   (anything else is left as-is, i.e. treated as dimensionless)

_PARAM_TRANSFORMS: dict[str, str] = {
    'center':    'x_center',
    'sigma':     'x_scale',
    'fwhm':      'x_scale',
    'gamma':     'x_scale',     # Voigt width parameter
    'amplitude': 'y_scale',
    'slope':     'xy_slope',
    'intercept': 'y_intercept',
    'c':         'y_const',     # ConstantModel background
}

# ---------------------------------------------------------------------------
# Data scaler
# ---------------------------------------------------------------------------

class _DataScaler:
    """
    Z-score normalisation of (x, y) data and the matching inverse transform
    for lmfit parameter objects.

    Parameters
    ----------
    x, y : np.ndarray
        Clean (NaN-free) data used to compute scale factors.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x_mean = np.nanmean(x)
        self.x_std  = np.nanstd(x)
        self.y_mean = np.nanmean(y)
        self.y_std  = np.nanstd(y)

    def norm_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def norm_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    @staticmethod
    def _transform_type(name: str) -> str:
        """Return the transform type for a parameter from its name suffix."""
        return _PARAM_TRANSFORMS.get(name.split('_')[-1], 'dimensionless')

    def restore_result(self, result: lmfit.model.ModelResult
                       ) -> lmfit.model.ModelResult:
        """
        Unscale all fitted parameters in-place from normalized to physical
        coordinates.

        The slope in *normalized* space is needed to correctly recover the
        intercept, so it is captured before the loop modifies any values.
        This avoids the ordering dependency that exists when unscaling inside
        a loop that mutates as it goes.
        """
        slope_norm = next(
            (result.params[n].value for n in result.params
             if self._transform_type(n) == 'xy_slope'),
            0.0,
        )

        for name, p in result.params.items():
            t = self._transform_type(name)
            if t == 'x_center':
                p.value = self.x_mean + self.x_std * p.value
                if p.stderr is not None:
                    p.stderr *= self.x_std
            elif t == 'x_scale':
                p.value = self.x_std * p.value
                if p.stderr is not None:
                    p.stderr *= self.x_std
            elif t == 'y_scale':
                p.value = self.y_std * p.value
                if p.stderr is not None:
                    p.stderr *= self.y_std
            elif t == 'xy_slope':
                p.value = (self.y_std / self.x_std) * p.value
                if p.stderr is not None:
                    p.stderr *= self.y_std / self.x_std
            elif t == 'y_intercept':
                # slope_norm is the pre-transform normalized slope value,
                # which is what the unscaling formula requires.
                p.value = self.y_mean + self.y_std * (
                    p.value - (self.x_mean / self.x_std) * slope_norm
                )
                if p.stderr is not None:
                    p.stderr *= self.y_std
            elif t == 'y_const':
                p.value = self.y_mean + self.y_std * p.value
                if p.stderr is not None:
                    p.stderr *= self.y_std
            # 'dimensionless': leave value and stderr unchanged

        return result

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _get_peak_model(peak_model: str = 'gaussian',
                    background: str = 'linear') -> lmfit.Model:
    """Build a composite lmfit model: peak + optional background."""
    peak_cls = {
        'gaussian':   GaussianModel,
        'lorentzian': LorentzianModel,
        'voigt':      VoigtModel,
    }.get(peak_model)
    if peak_cls is None:
        raise ValueError(
            f"Unknown peak model '{peak_model}'. "
            "Choose from 'gaussian', 'lorentzian', 'voigt'."
        )

    peak = peak_cls(prefix='peak_')
    if background == 'none':
        return peak

    bg_cls = {'constant': ConstantModel, 'linear': LinearModel}.get(background)
    if bg_cls is None:
        raise ValueError(
            f"Unknown background '{background}'. "
            "Choose from 'none', 'constant', 'linear'."
        )
    return peak + bg_cls(prefix='bg_')


def _make_initial_guesses(model: lmfit.Model,
                          xvals: np.ndarray, yvals: np.ndarray,
                          guess_from_mask: bool = True,
                          user_guesses: dict = None) -> lmfit.Parameters:
    """Build initial parameter guesses, optionally overriding with user values."""
    params = model.make_params()

    if guess_from_mask:
        peak_x    = xvals[np.nanargmax(yvals)]
        amplitude = np.nanmax(yvals) - np.nanmin(yvals)
        sigma     = (xvals.max() - xvals.min()) / 10

        if 'peak_center' in params:
            params['peak_center'].set(value=peak_x)
        if 'peak_amplitude' in params:
            params['peak_amplitude'].set(value=amplitude, min=0)
        if 'peak_sigma' in params:
            params['peak_sigma'].set(value=sigma, min=0.01)
        if 'bg_intercept' in params:
            params['bg_intercept'].set(value=np.nanmedian(yvals))
        if 'bg_slope' in params:
            params['bg_slope'].set(value=0.0)
        if 'bg_c' in params:
            params['bg_c'].set(value=np.nanmedian(yvals))

    if user_guesses:
        for key, val in user_guesses.items():
            params[key].set(**(val if isinstance(val, dict) else {'value': val}))

    return params

# ---------------------------------------------------------------------------
# Single-slice fitting
# ---------------------------------------------------------------------------

def _fit_single_slice(xvals: np.ndarray,
                      yvals: np.ndarray,
                      model: lmfit.Model,
                      initial_guesses: dict | Callable,
                      guess_from_mask: bool,
                      min_valid_points: int,
                      min_peak_height: float | None,
                      fit_kwargs: dict) -> lmfit.model.ModelResult | None:
    """
    Fit one (x, y) slice.

    Data are normalised before fitting and the returned ModelResult has all
    parameters already restored to physical units via _DataScaler.

    Returns None when the slice should be skipped (too few points, peak too
    small, or the optimiser raises).
    """
    valid = ~np.isnan(yvals)
    if np.count_nonzero(valid) < min_valid_points:
        return None

    x_clean, y_clean = xvals[valid], yvals[valid]

    if min_peak_height is not None and np.nanmax(y_clean) < min_peak_height:
        return None

    scaler = _DataScaler(x_clean, y_clean)
    x_norm = scaler.norm_x(x_clean)
    y_norm = scaler.norm_y(y_clean)

    if callable(initial_guesses):
        params = initial_guesses(x_norm, y_norm)
    else:
        params = _make_initial_guesses(
            model, x_norm, y_norm, guess_from_mask, initial_guesses
        )

    try:
        result = model.fit(y_norm, x=x_norm, params=params, **fit_kwargs)
        return scaler.restore_result(result)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def _peak_results_to_dataset(results: List[lmfit.model.ModelResult],
                             coords: xr.DataArray,
                             model: lmfit.Model,
                             include_stderr: bool = True) -> xr.Dataset:
    """Assemble a list of per-slice ModelResults into an xr.Dataset."""
    param_names = model.param_names
    coeffs  = {name: [] for name in param_names}
    stderr  = {name: [] for name in param_names}
    success = []

    for r in results:
        if r is None:
            for name in param_names:
                coeffs[name].append(np.nan)
                stderr[name].append(np.nan)
            success.append(False)
        else:
            for name in param_names:
                coeffs[name].append(r.params[name].value)
                stderr[name].append(r.params[name].stderr or np.nan)
            success.append(True)

    data_vars = {
        name: (coords.dims, np.array(vals))
        for name, vals in coeffs.items()
    }
    if include_stderr:
        data_vars.update({
            f'{name}_stderr': (coords.dims, np.array(vals))
            for name, vals in stderr.items()
        })
    data_vars['fit_success'] = (coords.dims, np.array(success))
    return xr.Dataset(data_vars, coords={coords.dims[0]: coords})

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_peaks(da: xr.DataArray,
              x: str, y: str,
              dim: str = None,
              peak_model: str = 'gaussian',
              min_peak_height: float = None,
              background: str = 'linear',
              initial_guesses: dict | Callable = None,
              min_valid_points: int = 4,
              guess_from_mask: bool = True,
              fit_kwargs: dict = None) -> xr.Dataset:
    """
    Fit a 1D model (peak + background) to each slice along `dim` in a
    2D DataArray.

    Parameters
    ----------
    da : DataArray
        The 2D data array to fit.
    x : str
        Name of the coordinate used as the fit axis. Can be 1D along the inner
        dimension or 2D (dim x inner) — either resolves to a 1D array after
        slicing.
    y : str
        Name of the coordinate used as the output axis. Two shapes are supported:

        * **1D along dim** (y depends only on the outer sweep index): after
          slicing, slice_da[y] is a scalar, recorded directly as the y value
          for that slice.
        * **2D (dim x inner)** (y varies with both axes): after slicing,
          slice_da[y] is a 1D array, and the value at the fitted peak_center
          is found by interpolation.
    dim : str, optional
        The dimension to iterate over. If None, inferred as da[y].dims[0].
    peak_model : str
        One of {'gaussian', 'lorentzian', 'voigt'}.
    min_peak_height : float, optional
        If given, fits will be skipped if max(data) < min_peak_height.
    background : str
        One of {'none', 'constant', 'linear'}.
    initial_guesses : dict or callable
        Either a dict of parameter guesses (can include bounds), or a callable
        (x_norm, y_norm) -> lmfit.Parameters operating in normalised space.
    min_valid_points : int
        Minimum number of valid (non-NaN) data points required to attempt a fit.
    guess_from_mask : bool
        If True and initial_guesses is None, infer peak center from the
        maximum of the (masked) data.
    fit_kwargs : dict
        Passed verbatim to model.fit().
    """
    fit_kwargs = fit_kwargs or {}
    dim        = dim or da[y].dims[0]
    model      = _get_peak_model(peak_model, background)
    results      = []
    y_coord_vals = []

    for index in da[dim]:
        slice_da = da.sel({dim: index})
        xvals    = slice_da[x].values
        yvals    = slice_da.values

        result = _fit_single_slice(
            xvals, yvals, model,
            initial_guesses, guess_from_mask,
            min_valid_points, min_peak_height, fit_kwargs,
        )
        results.append(result)

        if result is not None:
            peak_x     = result.params['peak_center'].value
            y_physical = slice_da[y].values
            if y_physical.ndim == 0:
                # y is a 1D coord on the outer dim; scalar after sel
                y_coord_vals.append(float(y_physical))
            else:
                # y is a 2D coord; interpolate to find y at the peak x
                sort_idx = np.argsort(xvals)
                y_coord_vals.append(
                    np.interp(peak_x, xvals[sort_idx], y_physical[sort_idx])
                )
        else:
            y_coord_vals.append(np.nan)

    ds = _peak_results_to_dataset(results, da[dim], model)
    return ds.assign_coords({y: (dim, y_coord_vals)})


def curve_fit(x: xr.DataArray, y: xr.DataArray,
              yerr: xr.DataArray = None,
              curve_model: str | Callable = 'line',
              initial_guesses: dict = None,
              **kwargs) -> xr.Dataset:
    """
    Fit a curve y(x) using lmfit, returning a Dataset of fit coefficients.

    Parameters
    ----------
    x, y : DataArray
        1D data arrays of coordinates and values.
    yerr : DataArray, optional
        1D uncertainty in y values. Used to weight the fit.
    curve_model : str or callable
        A named model ('line'), an lmfit.Model instance, or a plain callable.
    initial_guesses : dict, optional
        Initial parameter guesses for the model.
    **kwargs
        Extra kwargs passed to model.fit().

    Returns
    -------
    Dataset with one scalar variable per parameter plus matching *_stderr vars.
    """
    if isinstance(curve_model, str):
        if curve_model == 'line':
            model = LinearModel()
        else:
            raise ValueError(f"Unsupported curve model: '{curve_model}'")
    elif isinstance(curve_model, lmfit.Model):
        model = curve_model
    elif callable(curve_model):
        model = lmfit.Model(curve_model)
    else:
        raise TypeError(
            "curve_model must be a string, lmfit.Model, or callable"
        )

    valid = ~np.isnan(x) & ~np.isnan(y)
    if yerr is not None:
        valid &= ~np.isnan(yerr)

    x_clean = x.values[valid]
    y_clean = y.values[valid]

    weights = None
    if yerr is not None:
        yerr_clean = yerr.values[valid]
        weights    = 1 / yerr_clean
        if np.any(np.isnan(weights)) or np.all(weights == 0):
            weights = None  # fallback to unweighted

    if len(x_clean) == 0:
        raise ValueError("No valid data points remain for curve fitting.")

    params = model.make_params()
    if initial_guesses:
        for k, v in initial_guesses.items():
            params[k].set(**(v if isinstance(v, dict) else {'value': v}))

    result = model.fit(y_clean, x=x_clean, params=params, weights=weights,
                       **kwargs)

    data_vars = {name: result.params[name].value for name in model.param_names}
    data_vars.update({
        f'{name}_stderr': result.params[name].stderr or np.nan
        for name in model.param_names
    })
    return xr.Dataset(data_vars)


def curve_fit_peaks(da: xr.DataArray,
                    x: str,
                    y: str,
                    param: str = 'peak_center',
                    curve_model: str | Callable = 'line',
                    curve_fit_kwargs: dict = None,
                    **fit_peaks_kwargs) -> xr.Dataset:
    """
    Fit peaks across one axis, then fit a curve to one of the peak parameters.

    Parameters
    ----------
    da : DataArray
        The 2D data to fit.
    x, y : str
        Coordinate names.
    param : str
        The peak parameter to fit a curve to (default: 'peak_center').
    curve_model : str or callable
        Model for the secondary curve fit (e.g. 'line', callable).
    curve_fit_kwargs : dict
        Passed to curve_fit.
    **fit_peaks_kwargs
        Passed to fit_peaks.

    Returns
    -------
    Dataset merging peak fit parameters with curve fit parameters
    (curve_fit_* prefix).
    """
    curve_fit_kwargs = curve_fit_kwargs or {}
    peak_ds = fit_peaks(da, x=x, y=y, **fit_peaks_kwargs)

    fit_param     = peak_ds[param]
    fit_param_err = peak_ds.get(f'{param}_stderr', None)
    if fit_param_err is not None:
        # Replace NaNs in stderr so they don't produce invalid weights
        fit_param_err = fit_param_err.where(~np.isnan(fit_param_err), other=np.inf)

    y_coord = peak_ds[y]
    valid   = peak_ds['fit_success']

    curve_ds = curve_fit(
        x           = y_coord[valid],
        y           = fit_param[valid],
        yerr        = fit_param_err[valid] if fit_param_err is not None else None,
        curve_model = curve_model,
        **curve_fit_kwargs,
    )
    curve_ds = curve_ds.rename(
        {k: f'curve_fit_{k}' for k in curve_ds.data_vars}
    )
    return xr.merge([peak_ds, curve_ds])