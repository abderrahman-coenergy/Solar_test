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


# ═══════════════════════════════════════════════════════════════════════════════
# SOLVEUR 3D — fit_3d(kd, f1, f2)
# ═══════════════════════════════════════════════════════════════════════════════
 
"""
Motivations du solveur 3D
--------------------------
Le solveur 1D (fit_kd) suppose que les coefficients f1 et f2 du modele de
Perez sont entierement determines par epsilon (clarte du ciel), via les tables
de Perez. Cette hypothese est raisonnable en moyenne, mais pour un instant donne
les conditions reelles du ciel peuvent s'ecarter des valeurs tabulees.
 
Le solveur 3D etend le probleme en laissant f1 et f2 libres tout en les
contraignant dans des bornes physiques strictes :
 
    min_{kd, f1, f2}  sum_i |GTIsim_i(kd, f1, f2) - GTIreal_i|
 
Contraintes physiques :
    kd in [0, 1]               : fraction diffuse positive et <= 100%
    f1 in [0, 1]               : composante circumsolaire (energie, pas puits)
    f2 in [f2_min, f2_max]     : composante d'horizon (bornes Perez 1990)
 
L'optimisation est realisee avec scipy.optimize.minimize (L-BFGS-B),
methode a gradient borne, ce qui garantit le respect des contraintes tout en
convergant rapidement (quelques dizaines d'iterations).
 
Usage :
    from src.solver import fit_3d
    kd, f1, f2, err = fit_3d(settings, gamma_s, alpha_s, TOANI)
"""
 
import scipy.optimize as opt
 
# ── Bornes physiques de f1 et f2 (issues de Perez et al. 1990) ─────────────────
# f1 (brightening circumsolaire) : toujours >= 0 (ne peut pas absorber d'energie)
#    La valeur max tabulee dans les 8 bins est 1.1326, on majore a 1.5 pour laisser
#    de la liberte sans sortir du domaine physique.
F1_MIN =  0.0
F1_MAX =  1
 
# f2 (brightening d'horizon) : peut etre negatif (obscurcissement de l'horizon)
#    Les extremes observes dans les tables de Perez couvrent [-1.6, +0.3].
#    On prend des marges symetriques autour de ces extremes.
F2_MIN = -1.8
F2_MAX =  0.5
 
 
def _project_gti_with_override(
    alpha: float,
    beta: float,
    BHI: np.ndarray,
    DHI: np.ndarray,
    gamma_s: np.ndarray,
    alpha_s: np.ndarray,
    TOANI: np.ndarray,
    elevation: float,
    albedo: float,
    use_riso: bool,
    Riso,
    horizon: np.ndarray,
    f1_override: float,
    f2_override: float,
) -> np.ndarray:
    """
    Variante de project_gti() qui remplace les coefficients f1/f2 calcules
    par Perez par des valeurs constantes imposees par le solveur 3D.
 
    Seul GTI est retourne (le solveur n'a besoin que de l'erreur sur GTI).
    """
    from src.values import KAPPA, MIN_SOLAR_ELEVATION_RAD
    from src.utils import calc_airmass
 
    DHI = DHI.copy()
    DHI[DHI == 0] = 0.001
 
    theta_s     = np.pi / 2.0 - gamma_s
    cos_theta_s = np.cos(theta_s)
 
    if horizon is None:
        daylight = gamma_s > MIN_SOLAR_ELEVATION_RAD
    else:
        h_at_az  = np.deg2rad(horizon[alpha_s.astype("int32") % 360])
        daylight = gamma_s > np.maximum(MIN_SOLAR_ELEVATION_RAD, h_at_az)
 
    # Facteur de vue diffuse
    Vd = Riso if use_riso else (1.0 + np.cos(beta)) / 2.0
 
    # Angle d'incidence
    cos_thetaI = (np.cos(beta) * cos_theta_s
                  + np.sin(beta) * np.cos(gamma_s) * np.cos(alpha_s - alpha))
    cos_thetaI = np.clip(cos_thetaI, 0.0, None)
 
    Rb = np.zeros(len(daylight))
    Rb[daylight] = cos_thetaI[daylight] / np.maximum(0.087, cos_theta_s[daylight])
 
    # f1 et f2 imposes (scalaires -> meme valeur a chaque instant)
    f1 = np.full(len(daylight), f1_override)
    f2 = np.full(len(daylight), f2_override)
 
    DHI_cs  = f1 * DHI
    DHI_iso = DHI - DHI_cs
    DTI     = DHI_cs * Rb + Vd * DHI_iso + f2 * np.sin(beta) * DHI
 
    BTI = np.zeros(len(daylight))
    BTI[daylight] = (BHI[daylight] * cos_thetaI[daylight]
                     / cos_theta_s[daylight])
 
    RTI = albedo * (1.0 - Vd) * (BHI + DHI)
    return BTI + DTI + RTI
 
 
def _objective_3d(
    params: np.ndarray,
    settings: ModelKdSettings,
    gamma_s: np.ndarray,
    alpha_s: np.ndarray,
    TOANI:   np.ndarray,
    Riso_fit,
) -> float:
    """
    Fonction objectif pour le solveur 3D.
 
    Calcule la somme des erreurs absolues sur tous les pyranometres fit :
        J(kd, f1, f2) = sum_i |GTIsim_i - GTIreal_i|
 
    Args:
        params : [kd, f1, f2]
 
    Returns:
        float : erreur totale (scalaire)
    """
    kd, f1, f2 = params
    GHI = settings.measures['pyrano-origin'].to_numpy()
    BHI = GHI * (1.0 - kd)
    DHI = GHI * kd
 
    total_error = 0.0
    for pyr in range(1, settings.n_fit + 1):
        az      = np.deg2rad(settings.measures[f'pyrano-fit-{pyr}_azimuth'][0])
        tilt    = np.deg2rad(settings.measures[f'pyrano-fit-{pyr}_tilt'][0])
        Riso    = (Riso_fit[pyr - 1]
                   if (settings.use_riso and Riso_fit is not None) else None)
        horizon = _get_horizon_for_pyr(settings, pyr, role='fit')
 
        gti_sim = _project_gti_with_override(
            alpha=az, beta=tilt,
            BHI=BHI, DHI=DHI,
            gamma_s=gamma_s, alpha_s=alpha_s, TOANI=TOANI,
            elevation=settings.elevation,
            albedo=settings.albedo,
            use_riso=settings.use_riso,
            Riso=Riso,
            horizon=horizon,
            f1_override=f1,
            f2_override=f2,
        )
        gti_real = settings.measures[f'pyrano-fit-{pyr}_value'].to_numpy()
        total_error += np.sum(np.abs(gti_sim - gti_real))
 
    return total_error
 
 
def fit_3d(
    settings: ModelKdSettings,
    gamma_s:  np.ndarray,
    alpha_s:  np.ndarray,
    TOANI:    np.ndarray,
    Riso_fit=None,
    kd_init:  float | None = None,
    n_restarts: int = 6,
) -> tuple[float, float, float, float]:
    """
    Solveur 3D : optimise simultanement kd, f1 et f2.
 
    Objective :
        min_{kd, f1, f2}  sum_i |GTIsim_i(kd, f1, f2) - GTIreal_i|
 
    Contraintes physiques (bornes L-BFGS-B) :
        kd in [0,       1      ]
        f1 in [F1_MIN,  F1_MAX ]  (circumsolaire >= 0)
        f2 in [F2_MIN,  F2_MAX ]  (horizon, peut etre negatif)
 
    Pour eviter les minima locaux, n_restarts points de depart sont testes
    aleatoirement dans l'espace des parametres ; le meilleur est retenu.
 
    Args:
        settings   : ModelKdSettings (construit par SolarModel._build_settings)
        gamma_s    : elevation solaire (rad), shape (T,)
        alpha_s    : azimut solaire (rad), shape (T,)
        TOANI      : irradiance extraterrestre (W/m2), shape (T,)
        Riso_fit   : facteurs Riso pre-calcules (n_fit,) ou None
        kd_init    : point de depart pour kd (None = mediane kd physique)
        n_restarts : nombre de points de depart aleatoires supplementaires
 
    Returns:
        kd_opt  : fraction diffuse optimale
        f1_opt  : coefficient circumsolaire optimal
        f2_opt  : coefficient d'horizon optimal
        err_opt : valeur de la fonction objectif au minimum
    """
    bounds = [(0.0, 1.0), (F1_MIN, F1_MAX), (F2_MIN, F2_MAX)]
 
    # ── Point de depart principal ────────────────────────────────────────────
    kd0 = kd_init if kd_init is not None else 0.3
    # f1 initial : valeur mediane des tables de Perez (milieu de l'intervalle)
    f1_0 = 0.5
    # f2 initial : valeur moderement negative (horizon typique a Paris)
    f2_0 = -0.05
    x0_main = np.array([kd0, f1_0, f2_0])
 
    best_result = None
    best_val    = np.inf
 
    # ── Multi-start pour robustesse ──────────────────────────────────────────
    rng = np.random.default_rng(seed=42)
 
    starts = [x0_main]
    # Points supplementaires aleatoires dans les bornes
    for _ in range(n_restarts):
        kd_r  = rng.uniform(0.05, 0.95)
        f1_r  = rng.uniform(F1_MIN, F1_MAX)
        f2_r  = rng.uniform(F2_MIN, F2_MAX)
        starts.append(np.array([kd_r, f1_r, f2_r]))
 
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
            best_val    = res.fun
            best_result = res
 
    kd_opt, f1_opt, f2_opt = best_result.x
    print(
        f"[solver 3D] kd={kd_opt:.4f}  f1={f1_opt:.4f}  f2={f2_opt:.4f}  "
        f"err={best_val:.2f} W/m²  (restarts={n_restarts})"
    )
    return float(kd_opt), float(f1_opt), float(f2_opt), float(best_val)