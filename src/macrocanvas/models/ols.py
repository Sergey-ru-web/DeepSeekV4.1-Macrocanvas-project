"""OLS estimation with Newey--West HAC inference and partial-regression plots.

The implementation deliberately depends only on NumPy and pandas.  It is small
enough to audit line by line, while still reporting heteroskedasticity- and
autocorrelation-robust standard errors, which are essential for quarterly
macroeconomic time series.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class OLSResult:
    """Ordinary least squares fit with HAC covariance matrix."""

    name: str
    dependent: str
    regressors: tuple[str, ...]
    params: pd.Series
    std_errors: pd.Series
    t_stats: pd.Series
    p_values: pd.Series
    covariance: pd.DataFrame
    residuals: pd.Series
    fitted: pd.Series
    nobs: int
    r_squared: float
    adj_r_squared: float
    residual_sd: float
    hac_lags: int | None

    @property
    def df_resid(self) -> int:
        return self.nobs - len(self.regressors)

    def coefficient(self, name: str) -> float:
        return float(self.params.loc[name])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coefficient": self.params,
                "std_error_hac": self.std_errors,
                "t_stat": self.t_stats,
                "p_value": self.p_values,
            }
        )


def _newey_west_meat(
    scores: np.ndarray,
    residuals: np.ndarray,
    n_lags: int,
) -> np.ndarray:
    """Bartlett-kernel long-run covariance of OLS score contributions."""
    meat = scores.T @ scores
    n_obs = len(residuals)
    for lag in range(1, n_lags + 1):
        if lag >= n_obs:
            break
        weight = 1.0 - lag / (n_lags + 1.0)
        product = residuals[lag:] * residuals[:-lag]
        cross = scores[lag:].T @ (scores[:-lag] * product[:, None])
        meat += weight * (cross + cross.T)
    return meat


def fit_ols(
    y: pd.Series,
    x: pd.DataFrame | pd.Series,
    *,
    name: str = "OLS",
    hac_lags: int | None = 4,
    add_constant: bool = True,
) -> OLSResult:
    """Fit an OLS regression on a common, non-missing sample.

    Parameters
    ----------
    y:
        Dependent variable.
    x:
        Regressor matrix.  If a Series is supplied it is converted to a
        one-column DataFrame.
    hac_lags:
        Bartlett bandwidth for Newey--West standard errors.  Set to ``None``
        for classical homoskedastic standard errors, or to zero for HC0.
    """
    if isinstance(x, pd.Series):
        x = x.to_frame()
    design = x.copy()
    if add_constant and "const" not in design.columns:
        design.insert(0, "const", 1.0)
    if not isinstance(design, pd.DataFrame):
        raise TypeError("x must be a pandas Series or DataFrame.")
    if design.shape[1] == 0:
        raise ValueError("At least one regressor is required.")
    if y.name is None:
        y = y.rename("y")

    framing = pd.concat([y.rename("__dependent__"), design], axis=1).dropna()
    if len(framing) <= design.shape[1]:
        raise ValueError(
            f"{name}: insufficient complete observations "
            f"({len(framing)}) for {design.shape[1]} regressors."
        )
    y_clean = framing["__dependent__"].astype(float)
    x_clean = framing[list(design.columns)].astype(float)
    x_matrix = x_clean.to_numpy()
    y_vector = y_clean.to_numpy()

    beta, *_ = np.linalg.lstsq(x_matrix, y_vector, rcond=None)
    fitted_array = x_matrix @ beta
    resid_array = y_vector - fitted_array
    n_obs, k_vars = x_matrix.shape
    xtx_inverse = np.linalg.pinv(x_matrix.T @ x_matrix)
    ssr = float(resid_array @ resid_array)
    tss = float(((y_vector - y_vector.mean()) ** 2).sum())
    dof = n_obs - k_vars
    sigma2 = ssr / dof if dof > 0 else np.nan

    scores = x_matrix * resid_array[:, None]
    if hac_lags is None:
        covariance_matrix = sigma2 * xtx_inverse
    else:
        if hac_lags < 0:
            raise ValueError("hac_lags must be non-negative or None.")
        meat = _newey_west_meat(scores, resid_array, hac_lags)
        covariance_matrix = xtx_inverse @ meat @ xtx_inverse
        covariance_matrix *= n_obs / max(dof, 1)
        covariance_matrix = 0.5 * (covariance_matrix + covariance_matrix.T)
    standard_errors = np.sqrt(np.clip(np.diag(covariance_matrix), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistics = np.divide(
            beta,
            standard_errors,
            out=np.full_like(beta, np.nan),
            where=standard_errors > 0,
        )
    p_values = np.array(
        [math.erfc(abs(t) / math.sqrt(2.0)) if np.isfinite(t) else np.nan for t in t_statistics]
    )
    r_squared = 1.0 - ssr / tss if tss > 0 else np.nan
    adj_r_squared = (
        1.0 - (1.0 - r_squared) * (n_obs - 1) / (n_obs - k_vars)
        if dof > 0 and np.isfinite(r_squared)
        else np.nan
    )

    names = list(x_clean.columns)
    return OLSResult(
        name=name,
        dependent=str(y.name),
        regressors=tuple(names),
        params=pd.Series(beta, index=names, name="coefficient"),
        std_errors=pd.Series(standard_errors, index=names, name="std_error_hac"),
        t_stats=pd.Series(t_statistics, index=names, name="t_stat"),
        p_values=pd.Series(p_values, index=names, name="p_value"),
        covariance=pd.DataFrame(covariance_matrix, index=names, columns=names),
        residuals=pd.Series(resid_array, index=y_clean.index, name="residual"),
        fitted=pd.Series(fitted_array, index=y_clean.index, name="fitted"),
        nobs=n_obs,
        r_squared=float(r_squared),
        adj_r_squared=float(adj_r_squared),
        residual_sd=float(np.sqrt(sigma2)) if dof > 0 else np.nan,
        hac_lags=hac_lags,
    )


@dataclass(frozen=True, slots=True)
class EquationEstimate:
    """A named behavioural equation with a distinguished focus coefficient."""

    key: str
    title: str
    dependent_label: str
    focus_label: str
    focus: str
    formula: str
    ols: OLSResult
    sample: pd.DataFrame
    expected_sign: str
    economic_note: str

    @property
    def coefficient(self) -> float:
        return self.ols.coefficient(self.focus)

    @property
    def standard_error(self) -> float:
        return float(self.ols.std_errors.loc[self.focus])

    @property
    def t_stat(self) -> float:
        return float(self.ols.t_stats.loc[self.focus])

    @property
    def p_value(self) -> float:
        return float(self.ols.p_values.loc[self.focus])

    @property
    def r_squared(self) -> float:
        return self.ols.r_squared

    def partial_data(self) -> pd.DataFrame:
        """Added-variable plot coordinates, following Frisch--Waugh--Lovell.

        Both the dependent variable and the focus regressor are purged of all
        other controls (including the intercept) by OLS.  The slope of the
        resulting scatter equals the multiple-regression coefficient exactly.
        """
        controls = [
            column
            for column in self.ols.regressors
            if column not in {"const", self.focus}
        ]
        if not controls:
            control_matrix = pd.DataFrame(
                {"const": np.ones(len(self.sample), dtype=float)},
                index=self.sample.index,
            )
        else:
            control_matrix = self.sample[controls]
        y_residual = fit_ols(
            self.sample[self.ols.dependent],
            control_matrix,
            hac_lags=None,
            name="partial-y",
        ).residuals
        x_residual = fit_ols(
            self.sample[self.focus],
            control_matrix,
            hac_lags=None,
            name="partial-x",
        ).residuals
        return pd.DataFrame(
            {"focus": x_residual, "dependent": y_residual},
            index=y_residual.index,
        )

    def to_frame(self) -> pd.DataFrame:
        frame = self.ols.to_frame()
        frame.insert(0, "equation", self.key)
        frame.insert(1, "dependent", self.ols.dependent)
        return frame


def _prepare_estimation_sample(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "output_gap",
        "inflation",
        "unemployment_gap",
        "policy_rate",
        "real_rate_ex_ante",
        "fiscal_impulse",
        "external_demand_gap",
        "financial_stress",
        "oil_price_shock",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for estimation: {sorted(missing)}")
    work = data.copy()
    work["output_gap_lag"] = work["output_gap"].shift(1)
    work["inflation_lag"] = work["inflation"].shift(1)
    work["policy_rate_lag"] = work["policy_rate"].shift(1)
    return work


def estimate_equations(
    data: pd.DataFrame,
    *,
    hac_lags: int = 4,
) -> dict[str, EquationEstimate]:
    """Estimate the four behavioural equations of the synthetic model."""
    work = _prepare_estimation_sample(data)
    estimates: dict[str, EquationEstimate] = {}

    # IS curve: x_t = rho x_{t-1} - sigma (r^e_t - r*) + ...
    is_x = work[
        [
            "output_gap_lag",
            "real_rate_ex_ante",
            "fiscal_impulse",
            "external_demand_gap",
            "financial_stress",
        ]
    ]
    is_y = work["output_gap"]
    is_fit = fit_ols(is_y, is_x, name="IS curve", hac_lags=hac_lags)
    is_sample = pd.concat([is_y.rename("output_gap"), is_x], axis=1).dropna()
    estimates["is"] = EquationEstimate(
        key="is",
        title="IS — совокупный спрос",
        dependent_label="Разрыв выпуска, п.п.",
        focus_label="Реальная ставка ex ante, п.п.",
        focus="real_rate_ex_ante",
        formula="x_t = ρx·x_{t−1} − σr·(r^e_t − r*) + δf·f_t + δx*·x*_t − δs·s_t + εx_t",
        ols=is_fit,
        sample=is_sample,
        expected_sign="negative",
        economic_note=(
            "Отрицательный коэффициент при реальной ставке отражает "
            "межвременное замещение в совокупном спросе."
        ),
    )

    # Phillips curve: pi_t = alpha pi_{t-1} + kappa x_{t-1} + oil + shock.
    pc_x = work[["inflation_lag", "output_gap_lag", "oil_price_shock"]]
    pc_y = work["inflation"]
    pc_fit = fit_ols(pc_y, pc_x, name="Phillips curve", hac_lags=hac_lags)
    pc_sample = pd.concat([pc_y.rename("inflation"), pc_x], axis=1).dropna()
    estimates["phillips"] = EquationEstimate(
        key="phillips",
        title="Кривая Филлипса",
        dependent_label="Инфляция, п.п. в год",
        focus_label="Разрыв выпуска (лаг), п.п.",
        focus="output_gap_lag",
        formula="π_t = π* + α(π_{t−1} − π*) + κx_{t−1} + γo·o_t + επ_t",
        ols=pc_fit,
        sample=pc_sample,
        expected_sign="positive",
        economic_note=(
            "Положительный наклон кривой Филлипса интерпретируется как "
            "краткосрочное давление спроса на цены."
        ),
    )

    # Okun: u_gap_t = const - beta x_t + residual.
    okun_x = work[["output_gap"]]
    okun_y = work["unemployment_gap"]
    okun_fit = fit_ols(okun_y, okun_x, name="Okun's law", hac_lags=hac_lags)
    okun_sample = pd.concat([okun_y.rename("unemployment_gap"), okun_x], axis=1).dropna()
    estimates["okun"] = EquationEstimate(
        key="okun",
        title="Закон Оукена",
        dependent_label="Разрыв безработицы, п.п.",
        focus_label="Разрыв выпуска, п.п.",
        focus="output_gap",
        formula="u_t − u* = −β·x_t + η_t",
        ols=okun_fit,
        sample=okun_sample,
        expected_sign="negative",
        economic_note=(
            "Коэффициент показывает, на сколько п.п. снижается безработица "
            "при расширении разрыва выпуска на один процентный пункт."
        ),
    )

    # Taylor-type reaction function.
    taylor_x = work[["policy_rate_lag", "inflation", "output_gap"]]
    taylor_y = work["policy_rate"]
    taylor_fit = fit_ols(taylor_y, taylor_x, name="Taylor rule", hac_lags=hac_lags)
    taylor_sample = pd.concat(
        [taylor_y.rename("policy_rate"), taylor_x], axis=1
    ).dropna()
    estimates["taylor"] = EquationEstimate(
        key="taylor",
        title="Правило Тейлора",
        dependent_label="Ключевая ставка, %",
        focus_label="Инфляция, п.п. в год",
        focus="inflation",
        formula=(
            "i_t = max{i_ELB, ρi·i_{t−1} + (1−ρi)[r* + π* + φπ(π_t − π*) "
            "+ φx·x_t] + εi_t}"
        ),
        ols=taylor_fit,
        sample=taylor_sample,
        expected_sign="positive",
        economic_note=(
            "Оценка является приведённой формой инерционного правила; "
            "она не восстанавливает структурные параметры напрямую."
        ),
    )

    return estimates


def estimates_table(estimates: dict[str, EquationEstimate]) -> pd.DataFrame:
    """Stack equation tables into one tidy CSV-ready frame."""
    frames = []
    for estimate in estimates.values():
        frame = estimate.to_frame()
        frame.insert(2, "r_squared", estimate.r_squared)
        frame.insert(3, "nobs", estimate.ols.nobs)
        frames.append(frame)
    return pd.concat(frames, axis=0)


def coefficient_table(
    estimates: dict[str, EquationEstimate],
    *,
    only_focus: bool = False,
) -> pd.DataFrame:
    """Compact table of estimated coefficients and HAC inference."""
    rows: list[dict[str, object]] = []
    for key, estimate in estimates.items():
        columns = [estimate.focus] if only_focus else list(estimate.ols.regressors)
        for regressor in columns:
            rows.append(
                {
                    "equation": key,
                    "regressor": regressor,
                    "coefficient": float(estimate.ols.params.loc[regressor]),
                    "std_error_hac": float(estimate.ols.std_errors.loc[regressor]),
                    "t_stat": float(estimate.ols.t_stats.loc[regressor]),
                    "p_value": float(estimate.ols.p_values.loc[regressor]),
                    "r_squared": estimate.r_squared,
                    "nobs": estimate.ols.nobs,
                }
            )
    return pd.DataFrame(rows)


def collect_residuals(estimates: Iterable[EquationEstimate]) -> pd.DataFrame:
    """Residuals from each equation, aligned by quarter."""
    return pd.concat(
        {item.key: item.ols.residuals for item in estimates},
        axis=1,
    ).dropna(how="all")
