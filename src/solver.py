"""
solver.py — Solveur de kd (fraction diffuse).

Deux solveurs indépendants :

    fit_kd()   : Grid Search 1D sur kd uniquement.
                 f1 et f2 sont recalculés dynamiquement par les tables de
                 Perez dans project_gti().

    fit_3d()   : Optimisation L-BFGS-B sur (kd, f1, f2) simultanément.
                 f1 et f2 sont laissés libres mais bornés physiquement.
                 Objective : min Σ |GTIsim_i(kd,f1,f2) - GTIreal_i|

Les deux solveurs consomment ModelKdSettings et retournent des scalaires.
Ils sont appelés par SolarModel, jamais directement depuis le notebook.
"""

from __future__ import annotations
import numpy as np
import scipy.optimize as opt

from src.names import ModelKdSettings
from src.constants.values import SOLVER_CFG, ATM
from src.utils import project_gti, calc_Riso


# ─── Helper : extraction de l'horizon d'un pyranomètre ───────────────────────

def _get_horizon_for_pyr(
    settings: ModelKdSettings,
    pyr:  int,
    role: str = 'fit',
) -> np.ndarray:
    """
    Extrait le profil d'horizon (360 valeurs, en radians) pour un pyranomètre.

    Gère les 3 formats de settings.horizons_rad :
      - dict {"fit-1": array, ...}
      - tableau 1D (360,)  → même horizon pour tous
      - tableau 2D (n_fit + n_dest, 360) → un horizon par pyranomètre
    """
    h = settings.horizons_rad
    if isinstance(h, dict):
        key = f"{role}-{pyr}"
        if key in h:
            return np.asarray(h[key])
        if pyr in h:
            return np.asarray(h[pyr])

    h_arr  = np.asarray(h)
    if h_arr.ndim == 1:
        return h_arr

    nrows  = h_arr.shape[0]
    n_fit  = settings.n_fit
    n_pred = settings.n_predict

    if nrows == (n_fit + n_pred):
        idx = (pyr - 1) if role == 'fit' else (n_fit + pyr - 1)
        return h_arr[idx]
    if role == 'fit' and nrows == n_fit:
        return h_arr[pyr - 1]
    if role == 'dest' and nrows == n_pred:
        return h_arr[pyr - 1]
    return h_arr[0]


def _horizon_rad_to_deg(horizon_rad: np.ndarray) -> np.ndarray:
    """Convertit un profil d'horizon de radians en degrés entiers (attendu par project_gti)."""
    return np.rad2deg(horizon_rad).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# SOLVEUR 1D — Grid Search sur kd
# ═══════════════════════════════════════════════════════════════════════════════

def get_errors_kd(
    settings:   ModelKdSettings,
    variations: list,
    kd_list:    np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """
    Matrice d'erreurs absolues (T × n_kd) entre GTI simulés et mesurés.

    Args:
        variations : liste[n_kd] de liste[n_fit] de arrays GTI simulés
        kd_list    : grille de kd testés

    Returns:
        errors        : shape (T, n_kd)
        optimal_index : liste des indices du kd optimal pour chaque instant T
    """
    errors = np.array([
        np.sum(
            np.array([
                np.abs(
                    variations[k][pyr - 1]
                    - settings.measures[f'pyrano-fit-{pyr}_value'].to_numpy()
                )
                for pyr in range(1, settings.n_fit + 1)
            ]),
            axis=0,
        )
        for k in range(len(kd_list))
    ])
    errors = errors.T
    optimal_index = [int(np.argmin(errors[i])) for i in range(len(errors))]
    return errors, optimal_index


def fit_kd(
    settings: ModelKdSettings,
    gamma_s:  np.ndarray,
    alpha_s:  np.ndarray,
    TOANI:    np.ndarray,
    Riso_fit: np.ndarray | None = None,
) -> tuple[float, np.ndarray, list[int]]:
    """
    Grid Search du paramètre kd optimal (pipeline 1D).

    Teste SOLVER_CFG.kd_steps valeurs de kd de façon exhaustive.
    Agrégation temporelle par médiane des optima.

    Returns:
        kd_opt        : fraction diffuse optimale (scalaire)
        errors        : matrice (T, n_kd)
        optimal_index : liste des indices optimaux par instant
    """
    kd_list = np.linspace(SOLVER_CFG.kd_min, SOLVER_CFG.kd_max, num=SOLVER_CFG.kd_steps)
    GHI     = settings.measures['pyrano-origin'].to_numpy()

    variations = []
    for kd in kd_list:
        BHI  = GHI * (1.0 - kd)
        DHI  = GHI * kd
        proj = []
        for pyr in range(1, settings.n_fit + 1):
            az      = np.deg2rad(settings.measures[f'pyrano-fit-{pyr}_azimuth'][0])
            tilt    = np.deg2rad(settings.measures[f'pyrano-fit-{pyr}_tilt'][0])
            Riso    = Riso_fit[pyr - 1] if (settings.use_riso and Riso_fit is not None) else None
            horizon = _get_horizon_for_pyr(settings, pyr, role='fit')
            gti, *_ = project_gti(
                alpha=az, beta=tilt,
                BHI=BHI, DHI=DHI,
                gamma_s=gamma_s, alpha_s=alpha_s, TOANI=TOANI,
                elevation=settings.elevation,
                albedo=settings.albedo,
                use_riso=settings.use_riso,
                Riso=Riso,
                horizon=_horizon_rad_to_deg(horizon),
            )
            proj.append(gti)
        variations.append(proj)

    errors, optimal_index = get_errors_kd(settings, variations, kd_list)
    median_idx = int(np.median(optimal_index))
    kd_opt     = float(kd_list[median_idx])

    print(f"[solver 1D] kd = {kd_opt:.4f}  (index médian : {median_idx})")
    return kd_opt, errors, optimal_index


# ═══════════════════════════════════════════════════════════════════════════════
# SOLVEUR 3D — L-BFGS-B sur (kd, f1, f2)
# ═══════════════════════════════════════════════════════════════════════════════

def _project_gti_with_override(
    alpha:       float,
    beta:        float,
    BHI:         np.ndarray,
    DHI:         np.ndarray,
    gamma_s:     np.ndarray,
    alpha_s:     np.ndarray,
    TOANI:       np.ndarray,
    elevation:   float,
    albedo:      float,
    use_riso:    bool,
    Riso,
    horizon:     np.ndarray,
    f1_override: float,
    f2_override: float,
) -> np.ndarray:
    """
    Variante de project_gti() avec f1 et f2 imposés (solveur 3D).

    BTI et RTI sont identiques à project_gti() car ils ne dépendent pas de f1/f2.
    Seul GTI total est retourné (le solveur n'a besoin que de l'erreur sur GTI).
    """
    DHI = DHI.copy()
    DHI[DHI == 0] = 0.001

    theta_s     = np.pi / 2.0 - gamma_s
    cos_theta_s = np.cos(theta_s)

    if horizon is None:
        daylight = gamma_s > ATM.min_solar_elevation_rad
    else:
        h_at_az  = np.deg2rad(horizon[alpha_s.astype("int32") % 360])
        daylight = gamma_s > np.maximum(ATM.min_solar_elevation_rad, h_at_az)

    Vd = Riso if use_riso else (1.0 + np.cos(beta)) / 2.0

    cos_thetaI = (
        np.cos(beta) * cos_theta_s
        + np.sin(beta) * np.cos(gamma_s) * np.cos(alpha_s - alpha)
    )
    cos_thetaI = np.clip(cos_thetaI, 0.0, None)

    Rb = np.zeros(len(daylight))
    Rb[daylight] = cos_thetaI[daylight] / np.maximum(0.087, cos_theta_s[daylight])

    f1 = np.full(len(daylight), f1_override)
    f2 = np.full(len(daylight), f2_override)

    DTI = f1 * DHI * Rb + Vd * (DHI - f1 * DHI) + f2 * np.sin(beta) * DHI

    BTI = np.zeros(len(daylight))
    BTI[daylight] = BHI[daylight] * cos_thetaI[daylight] / cos_theta_s[daylight]

    RTI = albedo * (1.0 - Vd) * (BHI + DHI)
    return BTI + DTI + RTI


def _objective_3d(
    params:   np.ndarray,
    settings: ModelKdSettings,
    gamma_s:  np.ndarray,
    alpha_s:  np.ndarray,
    TOANI:    np.ndarray,
    Riso_fit,
) -> float:
    """
    Fonction objectif du solveur 3D.

    J(kd, f1, f2) = Σ_i |GTIsim_i(kd,f1,f2) - GTIreal_i|
    """
    kd, f1, f2 = params
    GHI = settings.measures['pyrano-origin'].to_numpy()
    BHI = GHI * (1.0 - kd)
    DHI = GHI * kd

    total_error = 0.0
    for pyr in range(1, settings.n_fit + 1):
        az      = np.deg2rad(settings.measures[f'pyrano-fit-{pyr}_azimuth'][0])
        tilt    = np.deg2rad(settings.measures[f'pyrano-fit-{pyr}_tilt'][0])
        Riso    = Riso_fit[pyr - 1] if (settings.use_riso and Riso_fit is not None) else None
        horizon = _get_horizon_for_pyr(settings, pyr, role='fit')

        gti_sim = _project_gti_with_override(
            alpha=az, beta=tilt,
            BHI=BHI, DHI=DHI,
            gamma_s=gamma_s, alpha_s=alpha_s, TOANI=TOANI,
            elevation=settings.elevation,
            albedo=settings.albedo,
            use_riso=settings.use_riso,
            Riso=Riso,
            horizon=_horizon_rad_to_deg(horizon),
            f1_override=f1,
            f2_override=f2,
        )
        total_error += np.sum(np.abs(
            gti_sim - settings.measures[f'pyrano-fit-{pyr}_value'].to_numpy()
        ))
    return total_error


def fit_3d(
    settings:   ModelKdSettings,
    gamma_s:    np.ndarray,
    alpha_s:    np.ndarray,
    TOANI:      np.ndarray,
    Riso_fit=None,
    kd_init:    float | None = None,
    n_restarts: int = 6,
) -> tuple[float, float, float, float]:
    """
    Solveur 3D : optimise (kd, f1, f2) simultanément via L-BFGS-B.

    Objective :
        min_{kd, f1, f2}  Σ_i |GTIsim_i(kd,f1,f2) - GTIreal_i|

    Contraintes (bornes L-BFGS-B issues de SOLVER_CFG) :
        kd ∈ [0,            1           ]
        f1 ∈ [SOLVER_CFG.f1_min, f1_max]   (circumsolaire ≥ 0)
        f2 ∈ [SOLVER_CFG.f2_min, f2_max]   (horizon, peut être négatif)

    Multi-start : n_restarts points aléatoires + 1 point physique initial.

    Returns:
        (kd_opt, f1_opt, f2_opt, residual_error)
    """
    bounds = [
        (0.0,                   1.0),
        (SOLVER_CFG.f1_min,  SOLVER_CFG.f1_max),
        (SOLVER_CFG.f2_min,  SOLVER_CFG.f2_max),
    ]

    kd0     = kd_init if kd_init is not None else 0.3
    x0_main = np.array([kd0, 0.5, -0.05])   # f1=0.5 (médiane), f2=-0.05 (horizon typique Paris)

    rng    = np.random.default_rng(seed=42)
    starts = [x0_main] + [
        np.array([
            rng.uniform(0.05, 0.95),
            rng.uniform(SOLVER_CFG.f1_min, SOLVER_CFG.f1_max),
            rng.uniform(SOLVER_CFG.f2_min, SOLVER_CFG.f2_max),
        ])
        for _ in range(n_restarts)
    ]

    best_result, best_val = None, np.inf
    for x0 in starts:
        res = opt.minimize(
            fun=_objective_3d,
            x0=x0,
            args=(settings, gamma_s, alpha_s, TOANI, Riso_fit),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-7},
        )
        if res.fun < best_val:
            best_val, best_result = res.fun, res

    kd_opt, f1_opt, f2_opt = best_result.x
    print(
        f"[solver 3D] kd={kd_opt:.4f}  f1={f1_opt:.4f}  f2={f2_opt:.4f}  "
        f"err={best_val:.2f} W/m²"
    )
    return float(kd_opt), float(f1_opt), float(f2_opt), float(best_val)
