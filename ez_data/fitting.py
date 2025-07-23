import lmfit
from lmfit.models import (ConstantModel, GaussianModel, LorentzianModel,
                          LinearModel, VoigtModel)

import numpy as np
import xarray as xr
from typing import Callable, List

import warnings # hyper pedantic ufloat warning
warnings.filterwarnings(
    "ignore",
    message = ".*std_dev==0.*",
    category = UserWarning,
    module = "uncertainties"
)

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
    x, y : str
        Names of coordinate variables in da.
    dim : str, optional
        The dimension to iterate over (e.g. rows). If None, inferred as the dim 
        not in x.
    peak_model : str
        One of {'gaussian', 'lorentzian', 'voigt'}.
    min_peak_height : float, optional
        If given, fits will be skipped if max(data) < min_peak_height.
    background : str
        One of {'none', 'constant', 'linear'}.
    initial_guesses : dict or callable
        Either a dict of parameter guesses (can include bounds),
        or a callable (x, y) -> Parameters.
    min_valid_points : int
        Minimum number of valid (non-masked) data points required to attempt 
        fit.
    guess_from_mask : bool
        If True and initial_guesses is None, infer peak center from max of 
        masked data.
    fit_kwargs : dict
        Passed to model.fit
    """

    fit_kwargs = fit_kwargs or {}
    dim = dim or da[y].dims[0]
    model = _get_peak_model(peak_model, background)
    results = []

    for index in da[dim]:
        slice_da = da.sel({dim: index})
        xvals = slice_da[x].values
        yvals = slice_da.values

        valid = ~np.isnan(yvals)
        if np.count_nonzero(valid) < min_valid_points:
            results.append(None)
            continue
        x_clean = xvals[valid]
        y_clean = yvals[valid]

        if min_peak_height is not None and np.nanmax(y_clean) < min_peak_height:
            results.append(None)
            continue

        # Normalize x and y
        x_mean = np.nanmean(x_clean)
        x_std = np.nanstd(x_clean)
        y_mean = np.nanmean(y_clean)
        y_std = np.nanstd(y_clean)
        x_norm = (x_clean - x_mean) / x_std
        y_norm = (y_clean - y_mean) / y_std

        if callable(initial_guesses):
            params = initial_guesses(x_norm, y_norm)
        else:
            params = _make_initial_guesses(
                model = model,
                xvals = x_norm,
                yvals = y_norm,
                guess_from_mask = guess_from_mask,
                user_guesses = initial_guesses,
            )

        try:
            result = model.fit(y_norm, x = x_norm, params = params, 
                                **fit_kwargs)
            # Unscale fitted parameters
            for name in model.param_names:
                param = result.params[name]
                if name == 'peak_center':
                    param.value = x_mean + x_std * param.value
                    if param.stderr is not None:
                        param.stderr *= x_std
                elif name == 'peak_sigma':
                    param.value *= x_std
                    if param.stderr is not None:
                        param.stderr *= x_std
                elif name == 'peak_amplitude':
                    param.value *= y_std
                    if param.stderr is not None:
                        param.stderr *= y_std
                elif name == 'bg_slope':
                    param.value = (y_std / x_std) * param.value
                    if param.stderr is not None:
                        param.stderr *= y_std / x_std
                elif name == 'bg_intercept':
                    slope = result.params.get('bg_slope', None)
                    if slope is None:
                        slope_val = 0.0
                    else:
                        slope_val = slope.value
                    param.value = y_mean + y_std * (param.value - \
                                                (x_mean / x_std) * slope_val)
                    if param.stderr is not None:
                        param.stderr *= y_std
                elif name == 'bg_c':
                    param.value = y_mean + y_std * param.value
                    if param.stderr is not None:
                        param.stderr *= y_std
            results.append(result)
        except Exception:
            results.append(None)

    # Build dataset from results
    ds = _peak_results_to_dataset(results, da[y], model)
    return ds


def _get_peak_model(peak_model: str = 'gaussian',
                    background: str = 'linear') -> lmfit.Model:
    """Build a model for a peak on some background"""
    
    peak = {
        'gaussian': GaussianModel,
        'lorentzian': LorentzianModel,
        'voigt': VoigtModel
    }[peak_model](prefix = 'peak_')

    if background == 'none':
        return peak

    bg = {
        'constant': ConstantModel,
        'linear': LinearModel
    }[background](prefix = 'bg_')

    return peak + bg

def _make_initial_guesses(model: lmfit.Model, 
                          xvals: np.ndarray, yvals: np.ndarray, 
                          guess_from_mask: bool = True, 
                          user_guesses: dict = None) -> lmfit.Parameters:
    params = model.make_params()

    if guess_from_mask:
        peak_x = xvals[np.nanargmax(yvals)]
        amplitude = np.nanmax(yvals) - np.nanmin(yvals)
        sigma = (xvals.max() - xvals.min()) / 10
        # min_x = xvals.min()
        # max_x = xvals.max()

        # Set peak parameters safely
        if 'peak_center' in params:
            # params['peak_center'].set(value = peak_x, min = min_x, max = max_x)
            params['peak_center'].set(value = peak_x)
        if 'peak_amplitude' in params:
            params['peak_amplitude'].set(value = amplitude, min = 0)
        if 'peak_sigma' in params:
            params['peak_sigma'].set(value = sigma, min = 0.01)

        # Set background params if present
        if 'bg_intercept' in params:
            params['bg_intercept'].set(value = np.nanmedian(yvals))
        if 'bg_slope' in params:
            params['bg_slope'].set(value = 0.0)
        if 'bg_c' in params:
            params['bg_c'].set(value = np.nanmedian(yvals))

    if user_guesses:
        for key, val in user_guesses.items():
            if isinstance(val, dict):
                params[key].set(**val)
            else:
                params[key].set(value = val)

    return params

def _peak_results_to_dataset(results: List[lmfit.model.ModelResult], 
                             coords: xr.DataArray, model: lmfit.Model,
                             include_stderr: bool = True) -> xr.Dataset:
    param_names = model.param_names
    coeffs = {name: [] for name in param_names}
    stderr = {name: [] for name in param_names}
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
        name: (coords.dims, np.array(vals)) for name, vals in coeffs.items()
    }
    if include_stderr:
        data_vars.update({
            f'{name}_stderr': (coords.dims, np.array(vals))
            for name, vals in stderr.items()
        })
    data_vars['fit_success'] = (coords.dims, np.array(success))
    return xr.Dataset(data_vars, coords = {coords.dims[0]: coords})


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
    yerr : xr.DataArray, optional
        1D uncertainty in y values. Used to weight the fit.
    curve_model : str or callable
        A named model ('line') or a callable (e.g., lmfit.Model or function).
    initial_guesses : dict, optional
        Initial parameter guesses for the model.
    kwargs : dict, optional
        Extra kwargs passed to model.fit().

    Returns
    -------
    Dataset with parameters and standard errors.
    """

    if isinstance(curve_model, str):
        if curve_model == 'line':
            model = LinearModel()
        else:
            raise ValueError(f"Unsupported curve model: {curve_model}")

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
        weights = 1 / yerr_clean
        if np.any(np.isnan(weights)) or np.all(weights == 0):
            weights = None  # fallback to unweighted

    if len(x_clean) == 0:
        raise ValueError("No valid data points remain for curve fitting.")

    params = model.make_params()
    if initial_guesses:
        for k, v in initial_guesses.items():
            params[k].set(**v if isinstance(v, dict) else {'value': v})

    result = model.fit(y_clean, x = x_clean, params = params, weights=weights,
                        **kwargs)

    param_names = model.param_names
    data_vars = {
        name: result.params[name].value for name in param_names
    }

    data_vars.update({
        f'{name}_stderr': result.params[name].stderr or np.nan
        for name in param_names
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
    Fit peaks across one axis, then fit a curve to one of the parameters.

    Parameters
    ----------
    da : DataArray
        The 2D data to fit.
    x, y : str
        Coordinate names.
    param : str
        The peak parameter to fit a curve to.
    curve_model : str or callable
        Model for the curve (e.g. 'line', callable).
    curve_fit_kwargs : dict
        Passed to curve_fit.
    fit_peaks_kwargs : dict
        Passed to fit_peaks.

    Returns
    -------
    Dataset with peak fit parameters + curve fit parameters.
    """
    curve_fit_kwargs = curve_fit_kwargs or {}
    peak_ds = fit_peaks(da, x = x, y = y, **fit_peaks_kwargs)

    # Get valid peak parameter points
    fit_param = peak_ds[param]
    fit_param_err = peak_ds.get(f'{param}_stderr', None)
    # Replace NaNs in stderr to avoid invalid weights
    if fit_param_err is not None:
        fit_param_err = fit_param_err.where(~np.isnan(fit_param_err), 
                                            other=np.inf)
    y_coord = peak_ds[y]
    valid = peak_ds['fit_success']

    # Normalize x/y if using linear model
    if curve_model == 'line':
        y_vals = y_coord[valid].values
        fit_vals = fit_param[valid].values

        y_mean = np.mean(y_vals)
        y_std = np.std(y_vals)
        f_mean = np.mean(fit_vals)
        f_std = np.std(fit_vals)

        y_norm = (y_vals - y_mean) / y_std
        f_norm = (fit_vals - f_mean) / f_std

        if fit_param_err is not None:
            err_vals = fit_param_err[valid].values
            err_norm = err_vals / f_std
        else:
            err_norm = None

        norm_ds = curve_fit(
            x = xr.DataArray(y_norm),
            y = xr.DataArray(f_norm),
            yerr = xr.DataArray(err_norm) if err_norm is not None else None,
            curve_model = curve_model,
            **curve_fit_kwargs
        )

        # Transform back to physical units
        slope = norm_ds['slope'].values * (f_std / y_std)
        intercept = f_mean + f_std * (norm_ds['intercept'].values - \
                                    (y_mean / y_std) * norm_ds['slope'].values)

        norm_ds['slope'].values = slope
        norm_ds['intercept'].values = intercept

        if 'slope_stderr' in norm_ds:
            norm_ds['slope_stderr'].values *= f_std / y_std
        if 'intercept_stderr' in norm_ds:
            norm_ds['intercept_stderr'].values *= f_std

        norm_ds = norm_ds.rename(
            {k: f'curve_fit_{k}' for k in norm_ds.data_vars}
        )
        return xr.merge([peak_ds, norm_ds])

    else:
        curve_ds = curve_fit(
            x           = y_coord[valid],
            y           = fit_param[valid],
            curve_model = curve_model,
            yerr        = fit_param_err[valid] if fit_param_err is not None \
                                                                    else None,
            **curve_fit_kwargs
        )
        curve_ds = curve_ds.rename(
            {k: f'curve_fit_{k}' for k in curve_ds.data_vars}
        )
        return xr.merge([peak_ds, curve_ds])
