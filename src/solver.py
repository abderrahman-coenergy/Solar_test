"""
solver.py — Solveur de kd (fraction diffuse).

Ce module isole la logique de Grid Search qui cherche la valeur optimale du
paramètre kd (fraction diffuse de l'irradiance globale GHI) en minimisant
l'erreur entre la projection de Perez et les mesures réelles des pyranomètres
de calibration (fit).

Usage typique (appelé par SolarModel.fit_parameters) :
    from src.solver import fit_kd

    kd_opt, errors, optimal_index = fit_kd(settings, gamma_s, alpha_s, TOANI)
"""

import numpy as np
from src.names import ModelKdSettings
from src.values import KD_GRID_MIN, KD_GRID_MAX, KD_GRID_STEPS
from src.utils import project_gti, calc_Riso


# ─── Helpers horizon ──────────────────────────────────────────────────────────

def _get_horizon_for_pyr(settings: ModelKdSettings, pyr: int, role: str = 'fit') -> np.ndarray:
    """
    Retourne le profil d'horizon (360 valeurs) pour un pyranomètre donné.

    Args:
        settings : ModelKdSettings
        pyr      : index 1-based du pyranomètre
        role     : 'fit' ou 'dest'

    Returns:
        np.ndarray de forme (360,)
    """
    h = settings.horizons
    if isinstance(h, dict):
        key = f"{role}-{pyr}"
        if key in h:
            return np.asarray(h[key])
        if pyr in h:
            return np.asarray(h[pyr])

    h_arr = np.asarray(h)
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


# ─── Erreur par kd ────────────────────────────────────────────────────────────

def get_errors_kd(
    settings: ModelKdSettings,
    variations: list,
    kd_list: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """
    Calcule l'erreur absolue totale entre les GTI projetés et les mesures
    réelles, pour chaque valeur de kd dans la grille.

    Args:
        settings   : ModelKdSettings (contient n_fit et measures)
        variations : liste de longueur len(kd_list), chaque élément est
                     une liste de n_fit arrays GTI projetés
        kd_list    : grille de valeurs kd testées

    Returns:
        errors        : array (T, len(kd_list)) — erreur par instant et par kd
        optimal_index : liste de len(T) indices du kd optimal à chaque instant
    """
    errors = np.array([
        np.sum(
            np.array([
                np.abs(variations[k][pyr - 1]
                       - settings.measures[f'pyrano-fit-{pyr}_value'].to_numpy())
                for pyr in range(1, settings.n_fit + 1)
            ]),
            axis=0
        )
        for k in range(len(kd_list))
    ])
    errors = errors.T  # shape (T, n_kd)
    optimal_index = [int(np.argmin(errors[i])) for i in range(len(errors))]
    return errors, optimal_index


# ─── Solveur principal ────────────────────────────────────────────────────────

def fit_kd(
    settings: ModelKdSettings,
    gamma_s: np.ndarray,
    alpha_s: np.ndarray,
    TOANI: np.ndarray,
    Riso_fit: np.ndarray | None = None,
) -> tuple[float, np.ndarray, list[int]]:
    """
    Grid Search du paramètre kd optimal.

    Pour chaque valeur de kd dans la grille, projette le GTI sur chaque
    pyranomètre de calibration et compare aux mesures réelles.
    Retourne le kd moyen optimal (médiane des optima temporels).

    Args:
        settings  : ModelKdSettings
        gamma_s   : élévation solaire (rad), shape (T,)
        alpha_s   : azimut solaire (rad), shape (T,)
        TOANI     : irradiance extraterrestre (W/m²), shape (T,)
        Riso_fit  : array (n_fit,) de facteurs Riso pré-calculés, ou None

    Returns:
        kd_opt        : valeur optimale (scalaire)
        errors        : matrice d'erreurs (T, n_kd)
        optimal_index : liste des indices optimaux par instant
    """
    kd_list  = np.linspace(KD_GRID_MIN, KD_GRID_MAX, num=KD_GRID_STEPS)
    GHI      = settings.measures['pyrano-origin'].to_numpy()

    variations = []
    for i, kd in enumerate(kd_list):
        BHI = GHI * (1.0 - kd)
        DHI = GHI * kd
        pyr_projections = []
        for pyr in range(1, settings.n_fit + 1):
            az   = np.deg2rad(settings.measures[f'pyrano-fit-{pyr}_azimuth'][0])
            tilt = np.deg2rad(settings.measures[f'pyrano-fit-{pyr}_tilt'][0])
            Riso = (Riso_fit[pyr - 1]
                    if (settings.use_riso and Riso_fit is not None)
                    else None)
            horizon = _get_horizon_for_pyr(settings, pyr, role='fit')
            gti, *_ = project_gti(
                alpha=az, beta=tilt,
                BHI=BHI, DHI=DHI,
                gamma_s=gamma_s, alpha_s=alpha_s, TOANI=TOANI,
                elevation=settings.elevation,
                albedo=settings.albedo,
                use_riso=settings.use_riso,
                Riso=Riso,
                horizon=horizon,
            )
            pyr_projections.append(gti)
        variations.append(pyr_projections)

    errors, optimal_index = get_errors_kd(settings, variations, kd_list)

    # Agrégation : on prend la médiane des indices optimaux sur le temps
    median_idx = int(np.median(optimal_index))
    kd_opt = float(kd_list[median_idx])

    print(f"[solver] kd optimal : {kd_opt:.4f}  (index médian : {median_idx})")
    return kd_opt, errors, optimal_index
