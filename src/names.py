"""
names.py — Structures de données du projet solar_research.

Ce fichier contient uniquement les @dataclass et types de base.
Aucune logique complexe ici.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# ─── Informations physiques d'un pyranomètre ─────────────────────────────────

@dataclass
class PyranoInfo:
    """Caractéristiques géométriques et d'horizon d'un pyranomètre."""
    azimuth_deg: float          # Azimut en degrés (0 = Nord)
    inclination_deg: float      # Inclinaison en degrés (0 = horizontal, 90 = vertical)
    horizon: np.ndarray = field(default_factory=lambda: np.zeros(360))
    # horizon : tableau 1D de 360 valeurs, élévation de l'horizon en radians par degré d'azimut


@dataclass
class PyranoMeasure:
    """Mesures temporelles d'un pyranomètre (irradiance W/m²)."""
    timestamps: np.ndarray      # Tableau de chaînes "YYYY-MM-DD HH:MM:SS.sss"
    values: np.ndarray          # Tableau de valeurs en W/m²


# ─── Pyranomètres ────────────────────────────────────────────────────────────

class RealPyrano:
    """Pyranomètre réel : possède des mesures effectives."""
    def __init__(self, info: PyranoInfo, measures: PyranoMeasure):
        self.info = info
        self.measures = measures


class VirtualPyrano:
    """Pyranomètre virtuel : emplacement cible sans mesures, à projeter."""
    def __init__(self, info: PyranoInfo):
        self.info = info


# ─── Options du modèle solaire ───────────────────────────────────────────────

@dataclass
class SolarModelOptions:
    """Paramètres de configuration géographique et physique du modèle."""
    latitude: float = 0.0
    longitude: float = 0.0
    elevation_meter: float = 0.0    # Altitude du site en mètres
    albedo: float = 0.2             # Réflectance du sol (0.2 = sol classique)
    use_riso: bool = False          # Activer le calcul de Riso intégré par horizon


# ─── Settings internes (utilisés par solver.py et SolarModel.py) ─────────────

@dataclass
class ModelKdSettings:
    """
    Paramètres consolidés transmis au solveur kd.
    Construit par SolarModel._create_settings() à partir des objets Pyrano.
    """
    latitude: float = 0.0
    longitude: float = 0.0
    elevation: float = 0.0
    albedo: float = 0.2
    use_riso: bool = False

    # horizons peut être :
    #   - un tableau 1D (360,)               → même horizon pour tous les pyranomètres
    #   - un tableau 2D (n_fit+n_dest, 360)  → un horizon par pyranomètre
    #   - un dict {clé: tableau 1D}          → horizon nommé par rôle/index
    horizons: np.ndarray = field(default_factory=lambda: np.zeros(360))

    measures: pd.DataFrame = field(default_factory=pd.DataFrame)

    n_fit: int = 0        # Nombre de pyranomètres de calibration (fit)
    n_predict: int = 0    # Nombre de pyranomètres de destination (dest)
