"""Reduced-form vector autoregression, Cholesky identification and IRF bands.

The VAR is used as a deliberately agnostic complement to the structural
simulation.  It lets the synthetic data speak with minimal theoretical
restrictions.  Identification is recursive (Cholesky) with the ordering
``[output gap, unemployment gap, inflation, policy rate]``: real quantities and
prices respond to policy only with a lag, while the central bank reacts
contemporaneously to the macroeconomy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class VARResult:
    """Estimated stable VAR(p) with companion representation."""

    variables: tuple[str, ...]
    lags: int
    coefficients: pd.DataFrame
    sigma_u: pd.DataFrame
    residuals: pd.DataFrame
    fitted: pd.DataFrame
    companion: np.ndarray
    roots: np.ndarray
    nobs: int
    data: pd.DataFrame

    @property
    def n_variables(self) -> int:
        return len(self.variables)

    @property
    def max_root_modulus(self) -> float:
        return float(np.max(np.abs(self.roots)))

    @property
    def is_stable(self) -> bool:
        return self.max_root_modulus < 1.0

    @property
    def cholesky(self) -> np.ndarray:
        return np.linalg.cholesky(self.sigma_u.to_numpy())

    @property
    def history(self) -> pd.DataFrame:
        return self.data.iloc[-self.lags :]

    def equation_table(self) -> pd.DataFrame:
        rows = []
        for variable in self.variables:
            for regressor in self.coefficients.index:
                rows.append(
                    {
                        "dependent": variable,
                        "regressor": regressor,
                        "coefficient": float(self.coefficients.loc[regressor, variable]),
                    }
                )
        return pd.DataFrame(rows)


def _design_matrix(values: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct intercept, lagged regressors and contemporaneous targets."""
    n_obs, n_variables = values.shape
    rows = n_obs - lags
    design = np.ones((rows, 1 + n_variables * lags))
    for lag in range(1, lags + 1):
        start = 1 + (lag - 1) * n_variables
        design[:, start : start + n_variables] = values[lags - lag : n_obs - lag]
    targets = values[lags:]
    return design, targets


def _design_names(variables: tuple[str, ...], lags: int) -> list[str]:
    names = ["const"]
    for lag in range(1, lags + 1):
        names.extend(f"{variable}_lag{lag}" for variable in variables)
    return names


def fit_var(
    data: pd.DataFrame,
    variables: list[str] | tuple[str, ...],
    lags: int = 2,
) -> VARResult:
    """Estimate a VAR by equation-wise OLS."""
    variables = tuple(variables)
    if len(variables) < 2:
        raise ValueError("A VAR requires at least two variables.")
    if lags < 1:
        raise ValueError("lags must be at least one.")
    missing = set(variables) - set(data.columns)
    if missing:
        raise ValueError(f"Missing VAR variables: {sorted(missing)}")
    clean = data.loc[:, variables].dropna().astype(float)
    if len(clean) <= 1 + len(variables) * lags:
        raise ValueError("Not enough observations to estimate the VAR.")

    values = clean.to_numpy()
    design, targets = _design_matrix(values, lags)
    coefficients_array, *_ = np.linalg.lstsq(design, targets, rcond=None)
    fitted_array = design @ coefficients_array
    residuals_array = targets - fitted_array
    n_obs, n_variables = targets.shape
    n_parameters = design.shape[1]
    dof = n_obs - n_parameters
    if dof <= 0:
        raise ValueError("No residual degrees of freedom in VAR.")
    sigma = residuals_array.T @ residuals_array / dof

    coefficients = pd.DataFrame(
        coefficients_array,
        index=_design_names(variables, lags),
        columns=list(variables),
    )

    # Companion matrix for a VAR(p).  Each top block is A_l^T because the
    # coefficient matrix stores one column per equation.
    top_blocks = [
        coefficients_array[1 + lag * n_variables : 1 + (lag + 1) * n_variables, :].T
        for lag in range(lags)
    ]
    top = np.hstack(top_blocks)
    companion = np.zeros((n_variables * lags, n_variables * lags))
    companion[:n_variables, :] = top
    if lags > 1:
        companion[n_variables:, :-n_variables] = np.eye(n_variables * (lags - 1))
    roots = np.linalg.eigvals(companion)

    # Align residuals and fitted values with the target quarters.
    target_index = clean.index[lags:]
    residuals = pd.DataFrame(residuals_array, index=target_index, columns=list(variables))
    fitted = pd.DataFrame(fitted_array, index=target_index, columns=list(variables))
    sigma_frame = pd.DataFrame(sigma, index=list(variables), columns=list(variables))

    return VARResult(
        variables=variables,
        lags=lags,
        coefficients=coefficients,
        sigma_u=sigma_frame,
        residuals=residuals,
        fitted=fitted,
        companion=companion,
        roots=roots,
        nobs=n_obs,
        data=clean,
    )


@dataclass(frozen=True, slots=True)
class IRFResult:
    """Estimated impulse responses with residual-bootstrap uncertainty bands."""

    shock: str
    horizon: int
    lags: int
    point: pd.DataFrame
    median_bootstrap: pd.DataFrame
    lower_68: pd.DataFrame
    upper_68: pd.DataFrame
    lower_90: pd.DataFrame
    upper_90: pd.DataFrame
    bootstrap_reps: int
    point_stable: bool

    def tidy(self) -> pd.DataFrame:
        frames = []
        for variable in self.point.columns:
            frames.append(
                pd.DataFrame(
                    {
                        "horizon": self.point.index,
                        "variable": variable,
                        "response": self.point[variable].to_numpy(),
                        "median_bootstrap": self.median_bootstrap[variable].to_numpy(),
                        "lower_68": self.lower_68[variable].to_numpy(),
                        "upper_68": self.upper_68[variable].to_numpy(),
                        "lower_90": self.lower_90[variable].to_numpy(),
                        "upper_90": self.upper_90[variable].to_numpy(),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)


def compute_irf(
    var_result: VARResult,
    shock: str,
    horizon: int = 12,
    *,
    shock_size: float = 1.0,
    normalize_to_one_unit: bool = True,
) -> pd.DataFrame:
    """Orthogonalised impulse responses to a Cholesky shock.

    With ``normalize_to_one_unit=True`` the impact response of the shocked
    variable is exactly ``shock_size`` (one percentage point by default).
    """
    if shock not in var_result.variables:
        raise ValueError(f"Unknown shock variable: {shock}")
    if horizon < 0:
        raise ValueError("horizon must be non-negative.")
    shock_index = var_result.variables.index(shock)
    cholesky = var_result.cholesky
    scale = (
        shock_size / cholesky[shock_index, shock_index]
        if normalize_to_one_unit
        else 1.0
    )
    impact = cholesky[:, shock_index] * scale
    n_variables = var_result.n_variables
    state = np.zeros(n_variables * var_result.lags)
    state[:n_variables] = impact
    responses = np.zeros((horizon + 1, n_variables))
    responses[0] = impact
    for step in range(1, horizon + 1):
        state = var_result.companion @ state
        responses[step] = state[:n_variables]
    return pd.DataFrame(
        responses,
        columns=list(var_result.variables),
        index=pd.RangeIndex(horizon + 1, name="horizon"),
    )


def _quantile_frame(draws: np.ndarray, quantile: float, columns: list[str]) -> pd.DataFrame:
    values = np.quantile(draws, quantile, axis=0)
    return pd.DataFrame(values, columns=columns, index=pd.RangeIndex(draws.shape[1], name="horizon"))


def bootstrap_irf(
    data: pd.DataFrame,
    variables: list[str] | tuple[str, ...],
    shock: str,
    *,
    lags: int = 2,
    horizon: int = 12,
    reps: int = 300,
    seed: int = 314159,
) -> IRFResult:
    """Residual-bootstrap confidence bands for an identified IRF.

    Each replication recursively simulates the VAR from the observed initial
    history, resampling whole residual vectors to preserve cross-equation
    correlation, then re-estimates the VAR.  Unstable replications are discarded.
    """
    if reps < 20:
        raise ValueError("At least 20 bootstrap replications are required.")
    clean = data.loc[:, list(variables)].dropna().astype(float)
    base = fit_var(clean, variables, lags=lags)
    if shock not in base.variables:
        raise ValueError(f"Unknown shock variable: {shock}")
    point = compute_irf(base, shock, horizon=horizon)
    variables_tuple = tuple(variables)
    base_array = clean.to_numpy()
    initial = base_array[:lags].copy()
    innovations = base.residuals.to_numpy()
    coefficient_array = base.coefficients.to_numpy()
    rng = np.random.default_rng(seed)
    n_variables = len(variables_tuple)
    draws: list[np.ndarray] = []
    attempts = 0
    max_attempts = reps * 8
    n_obs = len(base_array)

    while len(draws) < reps and attempts < max_attempts:
        attempts += 1
        simulated = np.empty_like(base_array)
        simulated[:lags] = initial
        for t in range(lags, n_obs):
            design = np.ones(1 + n_variables * lags)
            for lag in range(1, lags + 1):
                start = 1 + (lag - 1) * n_variables
                design[start : start + n_variables] = simulated[t - lag]
            shock_draw = innovations[rng.integers(0, len(innovations))]
            simulated[t] = design @ coefficient_array + shock_draw
        simulated_frame = pd.DataFrame(
            simulated,
            index=clean.index,
            columns=list(variables_tuple),
        )
        try:
            boot_var = fit_var(simulated_frame, variables_tuple, lags=lags)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not boot_var.is_stable or not np.all(np.isfinite(boot_var.sigma_u)):
            continue
        draws.append(compute_irf(boot_var, shock, horizon=horizon).to_numpy())

    if len(draws) < max(20, reps // 2):
        raise RuntimeError(
            f"Only {len(draws)} stable bootstrap replications were generated."
        )
    stacked = np.stack(draws, axis=0)
    columns = list(variables_tuple)
    return IRFResult(
        shock=shock,
        horizon=horizon,
        lags=lags,
        point=point,
        median_bootstrap=_quantile_frame(stacked, 0.5, columns),
        lower_68=_quantile_frame(stacked, 0.16, columns),
        upper_68=_quantile_frame(stacked, 0.84, columns),
        lower_90=_quantile_frame(stacked, 0.05, columns),
        upper_90=_quantile_frame(stacked, 0.95, columns),
        bootstrap_reps=len(draws),
        point_stable=base.is_stable,
    )
