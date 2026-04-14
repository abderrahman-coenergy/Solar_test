"""
names.py — Structures de données du projet solar_research.

Principes appliqués :
  - Composition stricte : ModelKdSettings contient SolarModelOptions
    au lieu de répéter ses attributs.
  - AtmosphericState est la nouvelle dataclass de sortie du fit :
    elle encapsule toute la vérité physique résolue (position solaire,
    décomposition BHI/DHI, coefficients kd/f1/f2).
  - Aucune logique complexe dans ce fichier.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Géométrie et mesures d'un pyranomètre
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PyranoInfo:
    """Caractéristiques géométriques et d'horizon d'un pyranomètre."""
    azimuth_deg:     float
    inclination_deg: float
    horizon_rad: np.ndarray = field(default_factory=lambda: np.zeros(360))
    # horizon_rad : tableau 1D de 360 valeurs, élévation de l'horizon en radians
    #               par degré d'azimut (0° = Nord, 90° = Est, ...)


@dataclass
class PyranoMeasure:
    """Série temporelle de mesures d'un pyranomètre."""
    timestamps:       np.ndarray   # chaînes "YYYY-MM-DD HH:MM:SS.sss"
    irradiation_Wm2:  np.ndarray   # W/m²


# ═══════════════════════════════════════════════════════════════════════════════
# Pyranomètres
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RealPyrano:
    """Pyranomètre physique avec mesures (utilisé pour origine et fit)."""
    info:     PyranoInfo
    measures: PyranoMeasure


@dataclass
class VirtualPyrano:
    """Pyranomètre virtuel : géométrie seule, pas de mesures (cible à projeter)."""
    info: PyranoInfo


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration du site (entrée utilisateur)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SolarModelOptions:
    """Paramètres géographiques et physiques du site de mesure."""
    latitude:        float
    longitude:       float
    elevation_meter: float
    albedo:          float = 0.2    # réflectance du sol (0.2 = sol nu classique)
    use_riso:        bool  = False  # activer Riso intégré avec horizon réel


# ═══════════════════════════════════════════════════════════════════════════════
# Settings internes consolidés (assemblés par SolarModel, consommés par solver)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelKdSettings:
    """
    Paramètres consolidés transmis au solveur kd.

    Utilise la composition : options contient SolarModelOptions
    au lieu de répéter latitude/longitude/elevation/albedo/use_riso.

    Accès aux attributs du site :
        settings.options.latitude
        settings.options.albedo
        etc.
    """
    options:       SolarModelOptions
    horizons_rad:  np.ndarray         # shape (n_fit + n_dest, 360), en radians
    measures:      pd.DataFrame
    n_fit:         int
    n_predict:     int = 0

    # ── Raccourcis de lecture (délèguent à options) ────────────────────────────
    @property
    def latitude(self)        -> float: return self.options.latitude
    @property
    def longitude(self)       -> float: return self.options.longitude
    @property
    def elevation(self)       -> float: return self.options.elevation_meter
    @property
    def albedo(self)          -> float: return self.options.albedo
    @property
    def use_riso(self)        -> bool:  return self.options.use_riso


# ═══════════════════════════════════════════════════════════════════════════════
# État atmosphérique résolu — sortie du fit, entrée de la projection
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AtmosphericState:
    """
    Encapsule toute la vérité physique résolue après le fit.

    Produite par SolarModel.fit_parameters() ou fit_parameters_3d().
    Consommée par projection.project_on_target().

    Champs temporels (shape T) :
        timestamps  : horodatages de la série
        GHI         : irradiance globale horizontale mesurée (W/m²)
        BHI         : composante directe (GHI × (1 - kd))
        DHI         : composante diffuse (GHI × kd)
        gamma_s     : élévation solaire (rad)
        alpha_s     : azimut solaire (rad)
        theta_s     : angle zénithal solaire = π/2 - gamma_s (rad)
        TOANI       : irradiance extraterrestre normale (W/m²)

    Scalaires issus du fit :
        kd          : fraction diffuse optimale
        f1          : coefficient circumsolaire (None → tables de Perez)
        f2          : coefficient d'horizon     (None → tables de Perez)
        fit_error   : valeur résiduelle de la fonction objectif (W/m²)

    Métadonnées :
        options     : SolarModelOptions du site
        pipeline    : "1D" ou "3D"
    """
    # ── Série temporelle ───────────────────────────────────────────────────────
    timestamps: np.ndarray
    GHI:        np.ndarray
    BHI:        np.ndarray
    DHI:        np.ndarray
    gamma_s:    np.ndarray
    alpha_s:    np.ndarray
    theta_s:    np.ndarray
    TOANI:      np.ndarray

    # ── Résultats du fit ───────────────────────────────────────────────────────
    kd:         float
    f1:         Optional[float]   # None si pipeline 1D (calculé dynamiquement)
    f2:         Optional[float]   # None si pipeline 1D (calculé dynamiquement)
    fit_error:  Optional[float]   # None si pipeline 1D (pas de résidu L-BFGS-B)

    # ── Métadonnées ────────────────────────────────────────────────────────────
    options:    SolarModelOptions
    pipeline:   str = "1D"        # "1D" ou "3D"
