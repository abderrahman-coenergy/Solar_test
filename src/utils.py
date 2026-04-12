"""
utils.py — Fonctions utilitaires mathématiques et solaires.

Contient :
- Formatage des timestamps (format_hour_no_24)
- Calcul de la masse d'air (calc_airmass)
- Intégration de Riso (calc_Riso)
- Coefficients de Perez f1/f2 (calc_f1_f2)
- Divers helpers numériques
"""

from calendar import monthrange
import numpy as np
import warnings
from scipy.integrate import quad
from parse import parse

try:
    import sg2
except ImportError:
    print('[utils] ATTENTION : module sg2 introuvable.')
    print('  → Installer avec : pip install sg2 -f https://pip.oie-lab.net/python/')
    sg2 = None

from src.values import (
    tabF11, tabF12, tabF13,
    tabF21, tabF22, tabF23,
    bin_epsilon_0, bin_epsilon_1,
    KAPPA, ELEVATION_SCALE_HEIGHT,
    MIN_SOLAR_ELEVATION_RAD,
)


# ─── Timestamps ───────────────────────────────────────────────────────────────

def format_hour_no_24(hours) -> list[str]:
    """
    Normalise une série de timestamps au format "YYYY-MM-DD HH:MM:SS.sss".
    Gère le cas des heures >= 24 (passage au lendemain).

    Args:
        hours: itérable de chaînes au format "YYYY-MM-DD HH:MM:SS.sss"

    Returns:
        list[str]: timestamps corrigés et normalisés
    """
    new_hours = []
    for hour in list(hours):
        parsed = parse("{}-{}-{} {}:{}:{}", hour)
        if parsed is None:
            raise ValueError(f"Format de timestamp invalide : {hour!r}")
        year   = int(parsed[0])
        month  = int(parsed[1])
        day    = int(parsed[2])
        h      = int(parsed[3])
        mins   = int(parsed[4])
        secs   = float(parsed[5])

        if h >= 24:
            day += 1
            h  -= 24
        month_len = monthrange(year, month)[1]
        if day > month_len:
            day   -= month_len
            month += 1
        if month > 12:
            month -= 12
            year  += 1

        new_hours.append(f'{year}-{month:02d}-{day:02d}T{h:02d}:{mins:02d}:{secs:06.3f}')
    return new_hours


# ─── Position solaire via sg2 ─────────────────────────────────────────────────

def get_sun_position(longitude: float, latitude: float, elevation: float,
                     timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule la position solaire pour un site et une série de timestamps.

    Args:
        longitude, latitude, elevation: coordonnées du site
        timestamps: array de np.datetime64 ou chaînes ISO

    Returns:
        gamma_s  : élévation solaire (rad)
        alpha_s  : azimut solaire (rad)
        TOANI    : irradiance extraterrestre normale (W/m²)
    """
    if sg2 is None:
        raise RuntimeError("Le module sg2 est requis pour calculer la position solaire.")

    time_arr = np.array([np.datetime64(t) for t in format_hour_no_24(timestamps)])
    sun_info = sg2.sun_position(
        [[longitude, latitude, elevation]],
        time_arr,
        ['topoc.gamma_S0', 'topoc.alpha_S', 'topoc.toa_ni']
    )
    gamma_s = np.array(sun_info.topoc.gamma_S0[0])
    alpha_s = np.array(sun_info.topoc.alpha_S[0])
    TOANI   = np.array(sun_info.topoc.toa_ni[0])
    return gamma_s, alpha_s, TOANI


# ─── Masse d'air ──────────────────────────────────────────────────────────────

def calc_airmass(gamma_s: np.ndarray, elevation: float,
                 corr_refract: bool = True) -> np.ndarray:
    """
    Calcule la masse d'air optique relative.

    Args:
        gamma_s      : élévation solaire en radians
        elevation    : altitude du site en mètres
        corr_refract : applique la correction de réfraction atmosphérique

    Returns:
        np.ndarray : masse d'air (sans unité)
    """
    corr_h     = np.exp(-elevation / ELEVATION_SCALE_HEIGHT)
    gamma_s_deg = gamma_s * 180.0 / np.pi

    if corr_refract:
        Dgamma = (180.0 / np.pi * 0.061359
                  * (0.1594 + 1.1230 * gamma_s + 0.065656 * gamma_s ** 2)
                  / (1.0 + 28.9344 * gamma_s + 277.3971 * gamma_s ** 2))
    else:
        Dgamma = np.zeros_like(gamma_s_deg)

    gamma_c = gamma_s_deg + Dgamma
    valid   = (gamma_c + 1.79) > 0
    m       = np.full(gamma_s.shape, 57.6)
    m[valid] = corr_h / (np.sin(gamma_c[valid] * np.pi / 180.0)
                         + 0.50572 * (gamma_c[valid] + 6.07995) ** (-1.6364))
    return m


# ─── Coefficients de Perez f1 / f2 ───────────────────────────────────────────

def calc_f1_f2(epsilon: np.ndarray, delta: np.ndarray,
               theta_s: np.ndarray, nbin: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule les coefficients f1 et f2 du modèle de Perez anisotrope.

    Args:
        epsilon : paramètre de clarté du ciel
        delta   : paramètre de luminosité du ciel
        theta_s : angle zénithal solaire (rad)
        nbin    : nombre de bins d'epsilon (8 par défaut)

    Returns:
        f1, f2 : coefficients (np.ndarray de même longueur que epsilon)
    """
    epsilon = epsilon.astype("float64")
    delta   = delta.astype("float64")

    ok = (np.isfinite(epsilon) & np.isfinite(delta))
    eps_ok  = epsilon[ok]
    dlt_ok  = delta[ok]
    ths_ok  = theta_s[ok]

    kbin = np.zeros(len(eps_ok), dtype="int64")
    for kb in range(nbin):
        mask = (eps_ok >= bin_epsilon_0[kb]) & (eps_ok < bin_epsilon_1[kb])
        kbin[mask] = kb

    f1 = np.full(len(ok), np.nan)
    f2 = np.full(len(ok), np.nan)

    f1[ok] = (tabF11[kbin] + tabF12[kbin] * dlt_ok + tabF13[kbin] * ths_ok)
    f2[ok] = (tabF21[kbin] + tabF22[kbin] * dlt_ok + tabF23[kbin] * ths_ok)

    return f1, f2


# ─── Intégrale Riso ───────────────────────────────────────────────────────────

def calc_Riso(alpha: float, beta: float, horizons: np.ndarray) -> float:
    """
    Calcule le facteur de vue Riso pour un plan incliné avec horizon réel.

    Le facteur Riso remplace la formule classique (1 + cos β) / 2 en tenant
    compte de l'obstruction angulaire fournie par le profil d'horizon.

    Args:
        alpha    : azimut du plan (rad)
        beta     : inclinaison du plan (rad)
        horizons : tableau 1D de 360 valeurs (élévation de l'horizon en rad)

    Returns:
        float : facteur Riso (compris entre 0 et 1 environ)
    """
    def theta_limit(phi: float) -> float:
        c = np.cos(phi - alpha)
        if c > 0:
            return np.pi / 2.0
        b = np.tan(beta)
        if c == 0.0 or b == 0.0:
            return 1e9
        return np.arctan(-(c * b) ** (-1))

    H  = lambda phi: horizons[int(np.rad2deg(phi)) % 360]
    Z  = lambda phi: min(np.pi / 2.0 - H(phi), theta_limit(phi))
    R1 = lambda phi: np.sin(Z(phi)) ** 2
    R2 = lambda phi: np.cos(phi - alpha) * (Z(phi) - np.sin(Z(phi)) * np.cos(Z(phi)))

    Riso = (np.cos(beta) / (2.0 * np.pi) * quad(R1, 0, 2.0 * np.pi)[0]
            + np.sin(beta) / (2.0 * np.pi) * quad(R2, 0, 2.0 * np.pi)[0])

    return max(Riso, 0.0)


# ─── Projection GTI (Perez complet) ──────────────────────────────────────────

def project_gti(
    alpha: np.ndarray,
    beta: np.ndarray,
    BHI: np.ndarray,
    DHI: np.ndarray,
    gamma_s: np.ndarray,
    alpha_s: np.ndarray,
    TOANI: np.ndarray,
    elevation: float,
    albedo: float,
    use_riso: bool,
    Riso: np.ndarray | None = None,
    horizon: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Projette les composantes de l'irradiance sur un plan incliné/orienté.

    Implémentation du modèle de Perez anisotrope.

    Args:
        alpha   : azimut du plan en radians (scalaire ou array)
        beta    : inclinaison du plan en radians (scalaire ou array)
        BHI     : irradiance directe horizontale (W/m²)
        DHI     : irradiance diffuse horizontale (W/m²)
        gamma_s : élévation solaire (rad)
        alpha_s : azimut solaire (rad)
        TOANI   : irradiance extraterrestre normale (W/m²)
        elevation : altitude du site (m)
        albedo  : réflectance du sol
        use_riso : utiliser Riso intégré au lieu de (1+cos β)/2
        Riso    : facteur de vue pré-calculé (si use_riso=True)
        horizon : profil d'horizon 1D de 360 valeurs en radians

    Returns:
        GTI, DTI, BTI, RTI : composantes du rayonnement sur plan incliné (W/m²)
    """
    DHI = DHI.copy()
    DHI[DHI == 0] = 0.001  # éviter division par zéro

    theta_s     = np.pi / 2.0 - gamma_s
    cos_theta_s = np.cos(theta_s)

    if horizon is None:
        daylight = gamma_s > MIN_SOLAR_ELEVATION_RAD
    else:
        h_at_az  = np.deg2rad(horizon[alpha_s.astype("int32") % 360])
        daylight = gamma_s > np.maximum(MIN_SOLAR_ELEVATION_RAD, h_at_az)

    AM    = calc_airmass(gamma_s, elevation, corr_refract=True)
    delta = AM * DHI / TOANI

    # Epsilon (clarté du ciel)
    BNI = np.zeros_like(daylight, dtype=float)
    BNI[daylight] = BHI[daylight] / np.sin(gamma_s[daylight])
    A       = 1.0 + BNI / DHI
    B       = KAPPA * theta_s ** 3
    epsilon = (A + B) / (1.0 + B)

    f1, f2 = calc_f1_f2(epsilon, delta, theta_s)

    # Facteur de vue diffuse
    Vd = Riso if use_riso else (1.0 + np.cos(beta)) / 2.0

    # Angle d'incidence sur le plan
    cos_thetaI = (np.cos(beta) * cos_theta_s
                  + np.sin(beta) * np.cos(gamma_s) * np.cos(alpha_s - alpha))
    cos_thetaI = np.clip(cos_thetaI, 0.0, None)

    Rb = np.zeros(len(daylight))
    Rb[daylight] = cos_thetaI[daylight] / np.maximum(0.087, cos_theta_s[daylight])

    # Composantes
    DHI_cs = f1 * DHI
    DHI_iso = DHI - DHI_cs
    DTI = DHI_cs * Rb + Vd * DHI_iso + f2 * np.sin(beta) * DHI

    BTI = np.zeros(len(daylight))
    BTI[daylight] = (BHI[daylight] * cos_thetaI[daylight]
                     / cos_theta_s[daylight])

    RTI = albedo * (1.0 - Vd) * (BHI + DHI)
    GTI = BTI + DTI + RTI

    return GTI, DTI, BTI, RTI
