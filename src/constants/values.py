"""
constants/values.py — Constantes physiques et algorithmiques du projet.

Les variables globales dispersées sont remplacées par des @dataclass(frozen=True),
regroupées par domaine logique :

    PerezCoefficients   : tables F11-F23, bins epsilon, kappa
    SolverConfig        : grille de recherche kd, bornes f1/f2 pour le 3D
    AtmosphericDefaults : constantes atmosphériques (masse d'air, élévation min)
    SiteDefaults        : coordonnées et albedo par défaut

Chaque classe expose ses données via des instances singleton MODULE-LEVEL
(PEREZ, SOLVER_CFG, ATM, SITE_DEFAULTS) prêtes à l'import direct.

Usage :
    from src.constants.values import PEREZ, SOLVER_CFG, ATM

    f11 = PEREZ.tabF11
    kd_min = SOLVER_CFG.kd_min
    elev_min = ATM.min_solar_elevation_rad
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Coefficients du modèle de Perez anisotrope
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PerezCoefficients:
    """
    Tables de coefficients du modèle de Perez (1990).

    f1(epsilon, delta, theta_s) = F11 + F12*delta + F13*theta_s
    f2(epsilon, delta, theta_s) = F21 + F22*delta + F23*theta_s

    Les 8 lignes correspondent aux 8 classes de clarté du ciel (bins d'epsilon).
    Les colonnes correspondent aux coefficients de la combinaison linéaire.

    Référence : Perez et al., "Modeling daylight availability and irradiance
    components from direct and global irradiance", Solar Energy, 1990.
    """
    # ── Composante circumsolaire (f1) ──────────────────────────────────────────
    tabF11: np.ndarray   # terme constant
    tabF12: np.ndarray   # coefficient de delta (luminosité du ciel)
    tabF13: np.ndarray   # coefficient de theta_s (angle zénithal)

    # ── Composante d'horizon (f2) ──────────────────────────────────────────────
    tabF21: np.ndarray
    tabF22: np.ndarray
    tabF23: np.ndarray

    # ── Bins de classification d'epsilon ──────────────────────────────────────
    # epsilon ∈ [1, +∞[ mesure le degré d'anisotropie du ciel
    # epsilon = 1 : ciel totalement couvert (diffus pur)
    # epsilon > 6 : ciel très clair (très directionnel)
    bin_epsilon_0: np.ndarray   # bornes inférieures des 8 classes
    bin_epsilon_1: np.ndarray   # bornes supérieures (dernière = +inf)

    # ── Constante de Perez ─────────────────────────────────────────────────────
    kappa: float   # = 1.041, correctif de l'angle zénithal dans la formule d'epsilon


# Instance singleton — importer PEREZ directement
PEREZ = PerezCoefficients(
    tabF11 = np.array([-0.0083,  0.1299,  0.3297,  0.5682,  0.8730,  1.1326,  1.0602,  0.6777]),
    tabF12 = np.array([ 0.5877,  0.6826,  0.4869,  0.1875, -0.3920, -1.2367, -1.5999, -0.3273]),
    tabF13 = np.array([-0.0621, -0.1514, -0.2211, -0.2951, -0.3616, -0.4118, -0.3589, -0.2504]),
    tabF21 = np.array([-0.060,  -0.019,   0.055,   0.109,   0.226,   0.288,   0.264,   0.156 ]),
    tabF22 = np.array([ 0.072,   0.066,  -0.064,  -0.152,  -0.462,  -0.823,  -1.127,  -1.377 ]),
    tabF23 = np.array([-0.022,  -0.029,  -0.026,  -0.014,   0.001,   0.056,   0.131,   0.251 ]),
    bin_epsilon_0 = np.array([1.000, 1.065, 1.230, 1.500, 1.950, 2.800, 4.500, 6.200]),
    bin_epsilon_1 = np.array([1.065, 1.230, 1.500, 1.950, 2.800, 4.500, 6.200, np.inf]),
    kappa = 1.041,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Configuration du solveur
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SolverConfig:
    """
    Paramètres algorithmiques des deux solveurs (1D Grid Search et 3D L-BFGS-B).

    Solveur 1D :
        kd_min, kd_max, kd_steps définissent la grille linéaire.
        Précision = (kd_max - kd_min) / kd_steps ≈ 0.01 avec 100 pas.

    Solveur 3D :
        f1_min/max et f2_min/max sont les bornes physiques strictes.
        f1 ≥ 0 : la composante circumsolaire ne peut pas être un puits d'énergie.
        f2 ∈ [-1.8, 0.5] : les extrêmes observés dans les tables Perez
                            sont ±1.6 ; on prend des marges légères.
    """
    # ── Grille kd (solveur 1D) ─────────────────────────────────────────────────
    kd_min:   float
    kd_max:   float
    kd_steps: int

    # ── Bornes physiques f1 / f2 (solveur 3D L-BFGS-B) ────────────────────────
    f1_min:   float   # brightening circumsolaire ≥ 0
    f1_max:   float
    f2_min:   float   # brightening d'horizon (peut être négatif)
    f2_max:   float


# Instance singleton
SOLVER_CFG = SolverConfig(
    kd_min   = 0.0001,
    kd_max   = 1.0,
    kd_steps = 100,
    f1_min   =  0.0,
    f1_max   =  1.5,
    f2_min   = -1.8,
    f2_max   =  0.5,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Constantes atmosphériques
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AtmosphericDefaults:
    """
    Constantes physiques de l'atmosphère standard.

    elevation_scale_height : hauteur d'échelle atmosphérique (formule barométrique).
        Utilisée dans calc_airmass() pour corriger la pression en altitude.
        m(z) = m(0) × exp(-z / H)

    min_solar_elevation_rad : angle minimal en dessous duquel le soleil est
        considéré couché. En dessous, BTI et DTI sont mis à zéro.
    """
    elevation_scale_height:   float   # mètres
    min_solar_elevation_rad:  float   # radians


# Instance singleton
ATM = AtmosphericDefaults(
    elevation_scale_height  = 8434.5,
    min_solar_elevation_rad = np.deg2rad(1.0),   # 1° au-dessus de l'horizon
)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Valeurs géographiques par défaut
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SiteDefaults:
    """
    Coordonnées et paramètres physiques par défaut.
    Utilisés comme fallback si l'utilisateur ne précise pas SolarModelOptions.
    """
    latitude:  float
    longitude: float
    elevation: float   # mètres
    albedo:    float


# Instance singleton
SITE_DEFAULTS = SiteDefaults(
    latitude  = 48.8566,   # Paris
    longitude =  2.3522,
    elevation = 35.0,
    albedo    =  0.2,
)
