"""
values.py — Constantes du projet solar_research.

Ce fichier regroupe tous les tableaux et valeurs par défaut utilisés
par le modèle de Perez anisotrope et le solveur kd.
"""

import numpy as np

# ─── Coefficients du modèle de Perez (interpolation par bin d'epsilon) ────────

# F11, F12, F13 : coefficients pour la composante circumsolaire (f1)
tabF11 = np.array([-0.0083, 0.1299, 0.3297, 0.5682, 0.8730, 1.1326, 1.0602, 0.6777])
tabF12 = np.array([ 0.5877, 0.6826, 0.4869, 0.1875,-0.3920,-1.2367,-1.5999,-0.3273])
tabF13 = np.array([-0.0621,-0.1514,-0.2211,-0.2951,-0.3616,-0.4118,-0.3589,-0.2504])

# F21, F22, F23 : coefficients pour la composante d'horizon (f2)
tabF21 = np.array([-0.060,-0.019, 0.055, 0.109, 0.226, 0.288, 0.264, 0.156])
tabF22 = np.array([ 0.072, 0.066,-0.064,-0.152,-0.462,-0.823,-1.127,-1.377])
tabF23 = np.array([-0.022,-0.029,-0.026,-0.014, 0.001, 0.056, 0.131, 0.251])

# ─── Bins de classification d'epsilon ─────────────────────────────────────────
# epsilon mesure le degré d'anisotropie du ciel (1 = ciel isotrope, > 6 = très direct)

bin_epsilon_0 = np.array([1.000, 1.065, 1.230, 1.500, 1.950, 2.800, 4.500, 6.200])
bin_epsilon_1 = np.array([1.065, 1.230, 1.500, 1.950, 2.800, 4.500, 6.200, np.inf])

# ─── Paramètre kappa de Perez ─────────────────────────────────────────────────

KAPPA = 1.041   # Constante empirique dans la formule d'epsilon

# ─── Constante de correction de la masse d'air ────────────────────────────────

ELEVATION_SCALE_HEIGHT = 8434.5  # mètres (hauteur d'échelle atmosphérique)

# ─── Solveur kd : grille de recherche ────────────────────────────────────────

KD_GRID_MIN   = 0.0001
KD_GRID_MAX   = 1.0
KD_GRID_STEPS = 100

# ─── Valeurs géographiques par défaut ─────────────────────────────────────────
# (utilisées comme fallback si non spécifiées dans SolarModelOptions)

DEFAULT_LATITUDE  = 48.8566   # Paris
DEFAULT_LONGITUDE =  2.3522
DEFAULT_ELEVATION =  35.0     # mètres
DEFAULT_ALBEDO    =  0.2

# ─── Angle solaire minimal pour considérer qu'il fait jour ────────────────────

MIN_SOLAR_ELEVATION_RAD = np.deg2rad(1.0)   # 1° au-dessus de l'horizon
