"""
projection.py — Projection de l'irradiance sur une surface cible.

Ce module est entièrement découplé de SolarModel. Il prend :
  - un VirtualPyrano (géométrie de la cible)
  - un AtmosphericState (état atmosphérique résolu par le fit)

Et retourne un DataFrame avec les composantes GTI, DTI, BTI, RTI.

La logique de projection diffère selon le pipeline de l'AtmosphericState :
  - pipeline="1D" : f1 et f2 recalculés par les tables de Perez (dynamique)
  - pipeline="3D" : f1 et f2 fixés aux valeurs issues du fit 3D (override)

Fonction principale :
    project_on_target(target, atmosphere) -> pd.DataFrame

Usage :
    from src.projection import project_on_target

    atm = model.fit_parameters()          # ou fit_parameters_3d()
    df  = project_on_target(target, atm)
    # df.columns = [time, GTI, DTI, BTI, RTI, kd, f1, f2, pipeline]
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.names import VirtualPyrano, AtmosphericState
from src.utils import project_gti, calc_Riso
from src.solver import _project_gti_with_override


# ─── Fonction principale ──────────────────────────────────────────────────────

def project_on_target(
    target:     VirtualPyrano,
    atmosphere: AtmosphericState,
) -> pd.DataFrame:
    """
    Projette l'énergie solaire résolue sur une surface cible virtuelle.

    Applique les équations de Perez pour calculer les composantes :
        BTI : direct sur plan incliné  (dépend de la géométrie cible et du soleil)
        DTI : diffus  sur plan incliné (circumsolaire + isotrope + horizon)
        RTI : réfléchi par le sol      (20% du GHI incident × albedo × (1 - Vd))
        GTI = BTI + DTI + RTI

    Le facteur de vue diffuse Vd est :
        - Riso intégré (avec horizon réel) si atmosphere.options.use_riso = True
        - (1 + cos β) / 2                 sinon

    Le comportement de f1/f2 dépend du pipeline :
        - "1D" : f1/f2 recalculés à chaque instant par les tables de Perez
        - "3D" : f1/f2 constants = atmosphere.f1 et atmosphere.f2

    Args:
        target     : VirtualPyrano avec azimuth_deg, inclination_deg, horizon_rad
        atmosphere : AtmosphericState produit par SolarModel.fit_parameters[_3d]()

    Returns:
        pd.DataFrame avec colonnes :
            time    : timestamps
            GTI     : irradiance globale sur plan incliné (W/m²)
            DTI     : composante diffuse (W/m²)
            BTI     : composante directe (W/m²)
            RTI     : composante réfléchie (W/m²)
            kd      : fraction diffuse utilisée
            f1      : coefficient f1 (None si pipeline 1D)
            f2      : coefficient f2 (None si pipeline 1D)
            pipeline: "1D" ou "3D"
    """
    az   = np.deg2rad(target.info.azimuth_deg)
    tilt = np.deg2rad(target.info.inclination_deg)

    # Horizon en degrés entiers pour project_gti
    horizon_deg = np.rad2deg(np.asarray(target.info.horizon_rad)).astype(int)

    # Facteur de vue Riso (si activé)
    Riso = None
    if atmosphere.options.use_riso:
        Riso = calc_Riso(az, tilt, horizon_deg)

    if atmosphere.pipeline == "3D":
        gti, dti, bti, rti = _project_3d(
            az=az, tilt=tilt,
            atmosphere=atmosphere,
            horizon_deg=horizon_deg,
            Riso=Riso,
        )
    else:
        gti, dti, bti, rti = _project_1d(
            az=az, tilt=tilt,
            atmosphere=atmosphere,
            horizon_deg=horizon_deg,
            Riso=Riso,
        )

    return pd.DataFrame({
        'time':     atmosphere.timestamps,
        'GTI':      gti,
        'DTI':      dti,
        'BTI':      bti,
        'RTI':      rti,
        'kd':       atmosphere.kd,
        'f1':       atmosphere.f1,
        'f2':       atmosphere.f2,
        'pipeline': atmosphere.pipeline,
    })


# ─── Moteurs internes ─────────────────────────────────────────────────────────

def _project_1d(
    az:          float,
    tilt:        float,
    atmosphere:  AtmosphericState,
    horizon_deg: np.ndarray,
    Riso,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pipeline 1D : f1 et f2 recalculés par les tables de Perez.
    Délègue entièrement à utils.project_gti().
    """
    return project_gti(
        alpha=az, beta=tilt,
        BHI=atmosphere.BHI,
        DHI=atmosphere.DHI,
        gamma_s=atmosphere.gamma_s,
        alpha_s=atmosphere.alpha_s,
        TOANI=atmosphere.TOANI,
        elevation=atmosphere.options.elevation_meter,
        albedo=atmosphere.options.albedo,
        use_riso=atmosphere.options.use_riso,
        Riso=Riso,
        horizon=horizon_deg,
    )


def _project_3d(
    az:          float,
    tilt:        float,
    atmosphere:  AtmosphericState,
    horizon_deg: np.ndarray,
    Riso,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pipeline 3D : f1 et f2 fixés (résultats du fit 3D).

    GTI est calculé via _project_gti_with_override (f1/f2 constants).
    BTI et RTI sont récupérés séparément via project_gti (ils ne dépendent
    pas de f1/f2 et sont identiques dans les deux pipelines).
    DTI = GTI - BTI - RTI.
    """
    gti = _project_gti_with_override(
        alpha=az, beta=tilt,
        BHI=atmosphere.BHI,
        DHI=atmosphere.DHI,
        gamma_s=atmosphere.gamma_s,
        alpha_s=atmosphere.alpha_s,
        TOANI=atmosphere.TOANI,
        elevation=atmosphere.options.elevation_meter,
        albedo=atmosphere.options.albedo,
        use_riso=atmosphere.options.use_riso,
        Riso=Riso,
        horizon=horizon_deg,
        f1_override=atmosphere.f1,
        f2_override=atmosphere.f2,
    )
    _, _, bti, rti = project_gti(
        alpha=az, beta=tilt,
        BHI=atmosphere.BHI,
        DHI=atmosphere.DHI,
        gamma_s=atmosphere.gamma_s,
        alpha_s=atmosphere.alpha_s,
        TOANI=atmosphere.TOANI,
        elevation=atmosphere.options.elevation_meter,
        albedo=atmosphere.options.albedo,
        use_riso=atmosphere.options.use_riso,
        Riso=Riso,
        horizon=horizon_deg,
    )
    dti = gti - bti - rti
    return gti, dti, bti, rti


# ─── Utilitaire : projection sur plusieurs cibles ────────────────────────────

def project_on_targets(
    targets:    list[VirtualPyrano],
    atmosphere: AtmosphericState,
) -> list[pd.DataFrame]:
    """
    Projette sur une liste de cibles. Retourne une liste de DataFrames.

    Usage :
        dfs = project_on_targets([facade_sud, facade_est], atm)
    """
    return [project_on_target(t, atmosphere) for t in targets]
