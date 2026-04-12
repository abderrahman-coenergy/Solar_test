"""
SolarModel.py — Orchestrateur principal du pipeline d'irradiance solaire.

Ce module expose une API simple pour :
    1. Configurer le site (latitude, longitude, altitude, albédo)
    2. Enregistrer les pyranomètres (origine, fit, cibles)
    3. Calibrer le paramètre kd
    4. Projeter l'irradiance sur les cibles

Usage typique (depuis un notebook) :
    from src.SolarModel import SolarModel, SolarModelOptions
    from src.names import PyranoInfo, PyranoMeasure, RealPyrano, VirtualPyrano

    options = SolarModelOptions(latitude=48.85, longitude=2.35,
                                elevation_meter=35, use_riso=True)
    model = SolarModel(options)

    model.set_origin(RealPyrano(info_origin, measures_origin))
    model.add_fit(RealPyrano(info_fit, measures_fit))
    model.add_target(VirtualPyrano(info_target))

    model.fit_parameters()
    df = model.project()
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.names import (
    PyranoInfo, PyranoMeasure,
    RealPyrano, VirtualPyrano,
    SolarModelOptions, ModelKdSettings,
)
from src.utils import (
    format_hour_no_24, get_sun_position, calc_Riso, project_gti,
)
from src.solver import fit_kd, _get_horizon_for_pyr


class SolarModel:
    """
    Orchestrateur du modèle d'irradiance solaire de Perez anisotrope.

    Séquence d'utilisation :
        set_origin → add_fit (1-N fois) → add_target (1-M fois)
        → fit_parameters() → project()
    """

    def __init__(self, options: SolarModelOptions):
        self.options          = options
        self._origin_pyr: RealPyrano | None    = None
        self._fit_pyr:    list[RealPyrano]     = []
        self._target_pyr: list[VirtualPyrano]  = []
        self.fitted           = False
        self._settings: ModelKdSettings | None = None
        self._gamma_s:  np.ndarray | None      = None
        self._alpha_s:  np.ndarray | None      = None
        self._TOANI:    np.ndarray | None      = None
        self.kd: float | None                  = None
        self.errors: np.ndarray | None         = None
        self.optimal_index: list[int] | None   = None

    # ─── API Publique ──────────────────────────────────────────────────────────

    def set_origin(self, pyr: RealPyrano):
        """Enregistre le pyranomètre d'origine (GHI horizontal)."""
        self._origin_pyr = pyr

    def add_fit(self, pyr: RealPyrano):
        """Ajoute un pyranomètre de calibration (mesure réelle inclinée)."""
        self._fit_pyr.append(pyr)

    def add_target(self, pyr: VirtualPyrano):
        """Ajoute une cible virtuelle dont on veut prédire l'irradiance."""
        self._target_pyr.append(pyr)

    def fit_parameters(self):
        """
        Calibre le paramètre kd en comparant la projection de Perez
        aux mesures réelles des pyranomètres de calibration.

        Stocke self.kd, self.errors, self.optimal_index.
        """
        self._validate_inputs()
        self._assert_timestamps_match()
        self._build_settings()
        self._compute_sun_position()
        self._compute_riso()

        self.kd, self.errors, self.optimal_index = fit_kd(
            settings=self._settings,
            gamma_s=self._gamma_s,
            alpha_s=self._alpha_s,
            TOANI=self._TOANI,
            Riso_fit=self._Riso_fit,
        )
        self.fitted = True

    def project(self) -> pd.DataFrame:
        """
        Projette l'irradiance sur tous les pyranomètres cibles.

        Returns:
            DataFrame avec colonnes :
                time, pyrano-origin,
                pyrano-fit-i_value / _azimuth / _tilt,
                pyrano-dest-i_value / _azimuth / _tilt
        """
        if not self.fitted:
            raise RuntimeError("Appelez fit_parameters() avant project().")

        df = self._settings.measures.copy()
        GHI = df['pyrano-origin'].to_numpy()
        BHI = GHI * (1.0 - self.kd)
        DHI = GHI * self.kd

        for i, pyr in enumerate(self._target_pyr, start=1):
            az   = np.deg2rad(pyr.info.azimuth_deg)
            tilt = np.deg2rad(pyr.info.inclination_deg)
            Riso = self._Riso_dest[i - 1] if (self.options.use_riso
                                                and self._Riso_dest is not None) else None
            horizon = np.deg2rad(np.asarray(pyr.info.horizon))

            gti, dti, bti, rti = project_gti(
                alpha=az, beta=tilt,
                BHI=BHI, DHI=DHI,
                gamma_s=self._gamma_s, alpha_s=self._alpha_s, TOANI=self._TOANI,
                elevation=self.options.elevation_meter,
                albedo=self.options.albedo,
                use_riso=self.options.use_riso,
                Riso=Riso,
                horizon=np.rad2deg(horizon).astype(int),  # horizon en degrés entiers
            )
            df[f'pyrano-dest-{i}_value'] = gti
            df[f'pyrano-dest-{i}_dti']   = dti
            df[f'pyrano-dest-{i}_bti']   = bti
            df[f'pyrano-dest-{i}_rti']   = rti

        return df

    # ─── Méthodes internes ────────────────────────────────────────────────────

    def _validate_inputs(self):
        if self._origin_pyr is None:
            raise ValueError("Pyranomètre d'origine non défini (set_origin).")
        if len(self._fit_pyr) == 0:
            raise ValueError("Aucun pyranomètre de calibration (add_fit).")

    def _assert_timestamps_match(self):
        """Vérifie que les timestamps de l'origine et des fit coïncident."""
        origin_ts = _normalize_timestamps(self._origin_pyr.measures.timestamps)
        for i, fit_pyr in enumerate(self._fit_pyr, start=1):
            fit_ts = _normalize_timestamps(fit_pyr.measures.timestamps)
            if not np.array_equal(origin_ts, fit_ts):
                raise ValueError(
                    f"Les timestamps du pyranomètre fit-{i} "
                    f"ne correspondent pas à ceux de l'origine."
                )

    def _build_settings(self):
        """Construit le ModelKdSettings consolidé."""
        opts = self.options
        length = len(self._origin_pyr.measures.timestamps)

        data = {
            'time':          _normalize_timestamps(self._origin_pyr.measures.timestamps),
            'pyrano-origin': np.asarray(self._origin_pyr.measures.values, dtype=float),
        }
        for i, fp in enumerate(self._fit_pyr, start=1):
            data[f'pyrano-fit-{i}_value']   = np.asarray(fp.measures.values, dtype=float)
            data[f'pyrano-fit-{i}_azimuth'] = np.full(length, fp.info.azimuth_deg)
            data[f'pyrano-fit-{i}_tilt']    = np.full(length, fp.info.inclination_deg)

        for i, tp in enumerate(self._target_pyr, start=1):
            data[f'pyrano-dest-{i}_azimuth'] = np.full(length, tp.info.azimuth_deg)
            data[f'pyrano-dest-{i}_tilt']    = np.full(length, tp.info.inclination_deg)

        # Construction du tableau d'horizons (fit d'abord, dest ensuite)
        n_fit  = len(self._fit_pyr)
        n_dest = len(self._target_pyr)
        horizons = np.zeros((n_fit + n_dest, 360))
        for i, fp in enumerate(self._fit_pyr):
            horizons[i] = np.asarray(fp.info.horizon)[:360]
        for i, tp in enumerate(self._target_pyr):
            horizons[n_fit + i] = np.asarray(tp.info.horizon)[:360]

        self._settings = ModelKdSettings(
            latitude=opts.latitude,
            longitude=opts.longitude,
            elevation=opts.elevation_meter,
            albedo=opts.albedo,
            use_riso=opts.use_riso,
            horizons=horizons,
            measures=pd.DataFrame(data),
            n_fit=n_fit,
            n_predict=n_dest,
        )

    def _compute_sun_position(self):
        """Calcule gamma_s, alpha_s, TOANI via sg2."""
        self._gamma_s, self._alpha_s, self._TOANI = get_sun_position(
            longitude=self.options.longitude,
            latitude=self.options.latitude,
            elevation=self.options.elevation_meter,
            timestamps=self._settings.measures['time'].to_numpy(),
        )

    def _compute_riso(self):
        """Pré-calcule les facteurs Riso pour fit et dest si use_riso=True."""
        if not self.options.use_riso:
            self._Riso_fit  = None
            self._Riso_dest = None
            return

        self._Riso_fit = []
        for i, fp in enumerate(self._fit_pyr, start=1):
            az      = np.deg2rad(fp.info.azimuth_deg)
            tilt    = np.deg2rad(fp.info.inclination_deg)
            horizon = np.deg2rad(np.asarray(fp.info.horizon))
            self._Riso_fit.append(calc_Riso(az, tilt, np.rad2deg(horizon).astype(int)))
        self._Riso_fit = np.array(self._Riso_fit)

        self._Riso_dest = []
        for i, tp in enumerate(self._target_pyr, start=1):
            az      = np.deg2rad(tp.info.azimuth_deg)
            tilt    = np.deg2rad(tp.info.inclination_deg)
            horizon = np.deg2rad(np.asarray(tp.info.horizon))
            self._Riso_dest.append(calc_Riso(az, tilt, np.rad2deg(horizon).astype(int)))
        self._Riso_dest = np.array(self._Riso_dest)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _normalize_timestamps(arr) -> np.ndarray:
    """Convertit n'importe quel array de timestamps en chaînes ISO normalisées."""
    out = []
    for v in arr:
        if isinstance(v, np.datetime64):
            out.append(np.datetime_as_string(v, unit='ms'))
        else:
            out.append(str(v))
    return np.array(out)
