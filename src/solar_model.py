"""
SolarModel.py — Résoluteur atmosphérique (fit uniquement).

SolarModel résout l'atmosphère : il prend des mesures réelles et retourne
un AtmosphericState contenant toute la vérité physique résolue.

Il ne connaît PAS les cibles (VirtualPyrano). La projection vers les
façades est entièrement déléguée à projection.project_on_target().

    ┌─────────────────────────────────────────────────────────────────┐
    │  Entrées                                                        │
    │    RealPyrano (origine GHI + fits GTI mesurés)                  │
    │    SolarModelOptions (lat, lon, alt, albedo, use_riso)          │
    ├─────────────────────────────────────────────────────────────────┤
    │  Sortie : AtmosphericState                                      │
    │    timestamps, GHI, BHI, DHI                                    │
    │    gamma_s, alpha_s, theta_s, TOANI                             │
    │    kd  (+ f1, f2 si pipeline 3D)                                │
    └─────────────────────────────────────────────────────────────────┘

Usage :
    model = SolarModel(options)
    model.set_origin(origin_pyr)
    model.add_fit(fit_pyr_1)
    model.add_fit(fit_pyr_2)

    atm_1d = model.fit_parameters()        # AtmosphericState pipeline 1D
    atm_3d = model.fit_parameters_3d()     # AtmosphericState pipeline 3D
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.names import (
    RealPyrano, VirtualPyrano,
    SolarModelOptions, ModelKdSettings,
    AtmosphericState,
)
from src.utils import get_sun_position, calc_Riso
from src.solver import fit_kd, fit_3d, _get_horizon_for_pyr, _horizon_rad_to_deg


class SolarModel:
    """
    Résoluteur atmosphérique : transforme des mesures en AtmosphericState.

    Méthodes publiques :
        set_origin(pyr)          : enregistre le pyranomètre GHI
        add_fit(pyr)             : ajoute un pyranomètre de calibration
        fit_parameters()         : résout kd    → AtmosphericState (pipeline 1D)
        fit_parameters_3d()      : résout kd+f1+f2 → AtmosphericState (pipeline 3D)

    Pas de add_target(), pas de project() — voir projection.py.
    """

    def __init__(self, options: SolarModelOptions):
        self.options = options
        self._origin_pyr: RealPyrano | None  = None
        self._fit_pyr:    list[RealPyrano]   = []

        # État interne (construit lazily à la première invocation de fit)
        self._settings:  ModelKdSettings | None = None
        self._gamma_s:   np.ndarray | None      = None
        self._alpha_s:   np.ndarray | None      = None
        self._TOANI:     np.ndarray | None      = None
        self._Riso_fit:  np.ndarray | None      = None

    # ─── Enregistrement des pyranomètres ──────────────────────────────────────

    def set_origin(self, pyr: RealPyrano):
        """Enregistre le pyranomètre d'origine (GHI horizontal, inclinaison 0°)."""
        self._origin_pyr = pyr
        self._settings   = None   # invalide le cache

    def add_fit(self, pyr: RealPyrano):
        """Ajoute un pyranomètre de calibration (mesure GTI réelle sur plan incliné)."""
        self._fit_pyr.append(pyr)
        self._settings = None

    # ─── Pipeline 1D ──────────────────────────────────────────────────────────

    def fit_parameters(self) -> AtmosphericState:
        """
        Pipeline 1D : calibre kd uniquement via Grid Search.

        f1 et f2 restent None dans l'AtmosphericState retourné —
        ils seront recalculés dynamiquement par les tables de Perez
        lors de la projection (project_on_target avec pipeline="1D").

        Returns:
            AtmosphericState avec pipeline="1D", f1=None, f2=None.
        """
        self._prepare()
        kd, errors, optimal_index = fit_kd(
            settings=self._settings,
            gamma_s=self._gamma_s,
            alpha_s=self._alpha_s,
            TOANI=self._TOANI,
            Riso_fit=self._Riso_fit,
        )
        return self._build_atmospheric_state(
            kd=kd, f1=None, f2=None,
            fit_error=None,
            pipeline="1D",
        )

    # ─── Pipeline 3D ──────────────────────────────────────────────────────────

    def fit_parameters_3d(
        self,
        kd_init:    float | None = None,
        n_restarts: int = 6,
    ) -> AtmosphericState:
        """
        Pipeline 3D : optimise (kd, f1, f2) simultanément via L-BFGS-B.

        Si fit_parameters() a déjà été appelé sur ce modèle, son kd est
        récupéré automatiquement comme point de départ (plus rapide).

        Args:
            kd_init    : point de départ pour kd (None = auto depuis pipeline 1D)
            n_restarts : nombre de restarts aléatoires (défaut 6)

        Returns:
            AtmosphericState avec pipeline="3D", f1 et f2 optimisés.
        """
        self._prepare()

        # Récupérer le kd_1D si disponible depuis les settings (pas stocké ici)
        # → l'utilisateur peut passer kd_init explicitement depuis l'extérieur
        kd_opt, f1_opt, f2_opt, err = fit_3d(
            settings=self._settings,
            gamma_s=self._gamma_s,
            alpha_s=self._alpha_s,
            TOANI=self._TOANI,
            Riso_fit=self._Riso_fit,
            kd_init=kd_init,
            n_restarts=n_restarts,
        )
        return self._build_atmospheric_state(
            kd=kd_opt, f1=f1_opt, f2=f2_opt,
            fit_error=err,
            pipeline="3D",
        )

    # ─── Méthodes internes ────────────────────────────────────────────────────

    def _prepare(self):
        """
        Construction lazy des settings et de la position solaire.
        Exécutée une seule fois par configuration de pyranomètres.
        """
        if self._settings is not None:
            return
        self._validate_inputs()
        self._assert_timestamps_match()
        self._build_settings()
        self._compute_sun_position()
        self._compute_riso()

    def _validate_inputs(self):
        if self._origin_pyr is None:
            raise ValueError("Pyranomètre d'origine non défini — appelez set_origin().")
        if len(self._fit_pyr) == 0:
            raise ValueError("Aucun pyranomètre de calibration — appelez add_fit().")

    def _assert_timestamps_match(self):
        origin_ts = _normalize_timestamps(self._origin_pyr.measures.timestamps)
        for i, fp in enumerate(self._fit_pyr, start=1):
            fit_ts = _normalize_timestamps(fp.measures.timestamps)
            if not np.array_equal(origin_ts, fit_ts):
                raise ValueError(
                    f"Timestamps du fit-{i} ≠ origine.\n"
                    f"  origine : {origin_ts}\n  fit-{i} : {fit_ts}"
                )

    def _build_settings(self):
        """Construit ModelKdSettings par composition avec SolarModelOptions."""
        opts   = self.options
        length = len(self._origin_pyr.measures.timestamps)

        data = {
            'time':          _normalize_timestamps(self._origin_pyr.measures.timestamps),
            'pyrano-origin': np.asarray(
                self._origin_pyr.measures.irradiation_Wm2, dtype=float
            ),
        }
        for i, fp in enumerate(self._fit_pyr, start=1):
            data[f'pyrano-fit-{i}_value']   = np.asarray(fp.measures.irradiation_Wm2, dtype=float)
            data[f'pyrano-fit-{i}_azimuth'] = np.full(length, fp.info.azimuth_deg)
            data[f'pyrano-fit-{i}_tilt']    = np.full(length, fp.info.inclination_deg)

        # Horizons en radians — shape (n_fit, 360)
        n_fit = len(self._fit_pyr)
        horizons_rad = np.zeros((n_fit, 360))
        for i, fp in enumerate(self._fit_pyr):
            horizons_rad[i] = np.asarray(fp.info.horizon_rad)[:360]

        # Composition : on passe options directement (pas de répétition d'attributs)
        self._settings = ModelKdSettings(
            options=opts,
            horizons_rad=horizons_rad,
            measures=pd.DataFrame(data),
            n_fit=n_fit,
            n_predict=0,   # SolarModel ne connaît pas les cibles
        )

    def _compute_sun_position(self):
        self._gamma_s, self._alpha_s, self._TOANI = get_sun_position(
            longitude=self.options.longitude,
            latitude=self.options.latitude,
            elevation=self.options.elevation_meter,
            timestamps=self._settings.measures['time'].to_numpy(),
        )

    def _compute_riso(self):
        if not self.options.use_riso:
            self._Riso_fit = None
            return
        self._Riso_fit = np.array([
            calc_Riso(
                np.deg2rad(fp.info.azimuth_deg),
                np.deg2rad(fp.info.inclination_deg),
                np.rad2deg(np.asarray(fp.info.horizon_rad)).astype(int),
            )
            for fp in self._fit_pyr
        ])

    def _build_atmospheric_state(
        self,
        kd:        float,
        f1:        float | None,
        f2:        float | None,
        fit_error: float | None,
        pipeline:  str,
    ) -> AtmosphericState:
        """Construit l'AtmosphericState à partir de l'état interne résolu."""
        GHI = self._settings.measures['pyrano-origin'].to_numpy()
        BHI = GHI * (1.0 - kd)
        DHI = GHI * kd
        return AtmosphericState(
            timestamps = self._settings.measures['time'].to_numpy(),
            GHI        = GHI,
            BHI        = BHI,
            DHI        = DHI,
            gamma_s    = self._gamma_s,
            alpha_s    = self._alpha_s,
            theta_s    = np.pi / 2.0 - self._gamma_s,
            TOANI      = self._TOANI,
            kd         = kd,
            f1         = f1,
            f2         = f2,
            fit_error  = fit_error,
            options    = self.options,
            pipeline   = pipeline,
        )


# ─── Helpers module-level ──────────────────────────────────────────────────────

def _normalize_timestamps(arr) -> np.ndarray:
    out = []
    for v in arr:
        if isinstance(v, np.datetime64):
            out.append(np.datetime_as_string(v, unit='ms'))
        else:
            out.append(str(v))
    return np.array(out)
