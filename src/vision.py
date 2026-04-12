"""
vision.py — Pipeline de vision fisheye pour extraction du profil d'horizon.

Format d'image supporté
-----------------------
Deux formats sont détectés automatiquement :

  • DUAL fisheye  : image large (ex 4096×2048) avec DEUX disques côte à côte.
                   Chaque moitié est un fisheye complet. Ex : format original
                   du projet (sky_00001.jpg).

  • SINGLE fisheye: image avec UN SEUL disque fisheye visible, l'autre moitié
                   étant noire. Ex : sample_image.jpg (4096×2048, disque dans
                   la moitié droite uniquement).

Corrections appliquées vs version précédente
--------------------------------------------
Problème #1 — Masque MiDaS inversé :
  L'ancienne version prenait depth > threshold (profondeur élevée = ciel).
  Or dans MiDaS, la sortie est une PROFONDEUR INVERSE : grande valeur = objet
  PROCHE, petite valeur = objet LOINTAIN (ciel, infini).
  → Fix : on prend depth < threshold (petite valeur = lointain = ciel).

Problème #2 — Noir autour du disque détecté comme ciel :
  MiDaS voit les pixels noirs du fond comme "très lointains" → ciel.
  → Fix : on détecte le disque circulaire via seuillage, on applique
    MiDaS uniquement dans le ROI du disque, et on force 0 hors du disque.

Problème #3 — Format single fisheye non géré :
  L'image réelle n'est pas dual-fisheye mais single fisheye dans la moitié
  droite.
  → Fix : détection automatique du format par comptage de pixels non noirs
    dans chaque moitié.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# ─── Imports optionnels ────────────────────────────────────────────────────────

try:
    import cv2
except ImportError:
    cv2 = None
    log.warning("OpenCV (cv2) non disponible.")

try:
    import torch
except ImportError:
    torch = None
    log.warning("PyTorch non disponible — masquage MiDaS désactivé.")

# ─── Constantes ────────────────────────────────────────────────────────────────

SINGLE_HALF_LEFT  = 'left'
SINGLE_HALF_RIGHT = 'right'
DUAL              = None

# Seuil de pixels non-noirs pour décider si une moitié contient un disque
_DISK_PRESENCE_RATIO = 0.15   # au moins 15 % de pixels non-noirs

# ─── Singleton MiDaS ──────────────────────────────────────────────────────────

_MIDAS_MODEL      = None
_MIDAS_TRANSFORMS = None


def _get_midas(device: str):
    global _MIDAS_MODEL, _MIDAS_TRANSFORMS
    if torch is None:
        raise RuntimeError("PyTorch est requis pour le masquage MiDaS.")
    if _MIDAS_MODEL is None:
        log.info("Chargement du modèle MiDaS (premier appel)...")
        _MIDAS_MODEL = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        _MIDAS_MODEL.to(device).eval()
        _MIDAS_TRANSFORMS = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        log.info("MiDaS chargé.")
    return _MIDAS_MODEL, _MIDAS_TRANSFORMS


def _select_device(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Détection du disque fisheye
# ═══════════════════════════════════════════════════════════════════════════════

def detect_fisheye_disk(img: np.ndarray,
                        black_threshold: int = 15,
                        border_inset: float = 0.02) -> Tuple[int, int, int]:
    """
    Détecte le centre et le rayon du disque fisheye dans une image.

    Stratégie : les pixels appartenant au disque sont non-noirs.
    On calcule la bounding box des pixels valides, puis on estime
    un cercle inscrit.

    Args:
        img             : array HxWx3 ou HxW (une moitié de l'image originale)
        black_threshold : valeur max des 3 canaux pour qu'un pixel soit "noir"
        border_inset    : fraction du rayon à retrancher pour exclure le bord
                          métallique/plastique de la lentille

    Returns:
        (cx, cy, radius) en pixels, dans le repère de img
    """
    if img.ndim == 3:
        nonblack = np.any(img > black_threshold, axis=2)
    else:
        nonblack = img > black_threshold

    if nonblack.sum() == 0:
        h, w = img.shape[:2]
        return w // 2, h // 2, min(h, w) // 2

    rows = np.where(np.any(nonblack, axis=1))[0]
    cols = np.where(np.any(nonblack, axis=0))[0]

    rmin, rmax = rows[0], rows[-1]
    cmin, cmax = cols[0], cols[-1]

    cy = int((rmin + rmax) / 2)
    cx = int((cmin + cmax) / 2)
    r  = int(min(rmax - rmin, cmax - cmin) / 2)

    # Réduction légère pour exclure le bord de la lentille
    r = int(r * (1.0 - border_inset))
    return cx, cy, r


def make_disk_mask(shape: Tuple[int, int],
                   cx: int, cy: int, r: int) -> np.ndarray:
    """Crée un masque binaire (bool) True à l'intérieur du disque."""
    H, W = shape[:2]
    Y, X = np.ogrid[:H, :W]
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2


def detect_image_format(img: np.ndarray,
                        black_threshold: int = 15) -> str:
    """
    Détecte automatiquement le format de l'image fisheye.

    Returns:
        'dual'  : deux disques côte à côte (format original du projet)
        'left'  : disque unique dans la moitié gauche
        'right' : disque unique dans la moitié droite
        'full'  : disque unique plein format (image carrée)
    """
    h, w = img.shape[:2]
    w_half = w // 2

    left_half  = img[:, :w_half]
    right_half = img[:, w_half:]

    total_half = h * w_half
    left_ratio  = np.any(left_half  > black_threshold, axis=2).sum() / total_half
    right_ratio = np.any(right_half > black_threshold, axis=2).sum() / total_half

    both_present = left_ratio > _DISK_PRESENCE_RATIO and right_ratio > _DISK_PRESENCE_RATIO

    if both_present:
        # Si les deux moitiés ont des pixels : dual ou image non-fisheye
        return 'dual'
    elif right_ratio > _DISK_PRESENCE_RATIO:
        return 'right'
    elif left_ratio > _DISK_PRESENCE_RATIO:
        return 'left'
    else:
        return 'full'


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Masquage du ciel (MiDaS + couleur)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_midas(roi: np.ndarray, device: str) -> np.ndarray:
    """
    Applique MiDaS sur une ROI et retourne la carte de profondeur inverse
    normalisée [0, 1].

    IMPORTANT : dans MiDaS, grande valeur = objet PROCHE, petite valeur = lointain.
    Le ciel (à l'infini) a donc de PETITES valeurs de profondeur inverse.
    """
    midas, transforms = _get_midas(device)
    transform = transforms.small_transform
    inp = transform(roi)
    if isinstance(inp, np.ndarray):
        inp = torch.from_numpy(inp)
    if inp.ndim == 3:
        inp = inp.unsqueeze(0)
    inp = inp.to(device)
    with torch.no_grad():
        depth = midas(inp)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=roi.shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze().cpu().numpy()

    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        depth = (depth - d_min) / (d_max - d_min)
    return depth


def _sky_by_color(img_rgb: np.ndarray,
                  disk_mask: np.ndarray,
                  value_min: int = 100,
                  saturation_max: int = 140,
                  hue_min: int = 70,
                  hue_max: int = 140) -> np.ndarray:
    """
    Détecte le ciel par critère HSV (Value + Saturation + Hue).
    Beaucoup plus robuste que le ratio B/R en RGB :
      - Value elevee  : ciel lumineux vs batiment/arbres sombres
      - Saturation moderee : batiment souvent plus sature que le ciel
      - Hue bleu/cyan : H OpenCV 70-140 = cyan->bleu clair
    Returns:
        Masque binaire uint8 (0/255) de meme taille que img_rgb.
    """
    if cv2 is None:
        r = img_rgb[:, :, 0].astype(np.float32)
        b = img_rgb[:, :, 2].astype(np.float32)
        bright = img_rgb.astype(np.float32).sum(axis=2) / 3
        sky = disk_mask & (b > r * 1.05) & (bright > 80)
        return sky.astype(np.uint8) * 255
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    sky = (disk_mask
           & (v_ch > value_min)
           & (s_ch < saturation_max)
           & (h_ch > hue_min)
           & (h_ch < hue_max))
    return sky.astype(np.uint8) * 255


def mask_sky(
    image: np.ndarray,
    disk_cx: int | None = None,
    disk_cy: int | None = None,
    disk_r:  int | None = None,
    device: Optional[str] = None,
    midas_depth_threshold: float = 0.35,
    use_color_fallback: bool = True,
    morph_ksize: int = 15,
) -> np.ndarray:
    """
    Segmente le ciel depuis un patch fisheye (une moitié de l'image originale).

    Pipeline :
        1. Détection du disque circulaire (si coordonnées non fournies)
        2. Masque hors-disque → forcé à 0 (non-ciel)
        3. MiDaS sur le ROI du disque :
               ciel = profondeur inverse FAIBLE (valeur normalisée < threshold)
               car MiDaS: grande valeur = proche, petite valeur = lointain/ciel
        4. Fusion avec critère couleur (fallback robuste)
        5. Nettoyage morphologique

    Args:
        image                  : array HxWx3 RGB (une moitié de l'image)
        disk_cx/cy/r           : centre et rayon du disque (auto-détectés si None)
        device                 : 'cpu', 'cuda' ou None (auto)
        midas_depth_threshold  : seuil de profondeur inverse pour "ciel" (0-1).
                                 Valeur basse → critère plus strict.
        use_color_fallback     : compléter MiDaS avec détection couleur
        morph_ksize            : taille du noyau pour morphologie

    Returns:
        Masque uint8 (0/255), même taille que image.
    """
    h, w = image.shape[:2]

    # ── 1. Détection du disque ────────────────────────────────────────────────
    if disk_cx is None or disk_cy is None or disk_r is None:
        disk_cx, disk_cy, disk_r = detect_fisheye_disk(image)
    disk_mask = make_disk_mask((h, w), disk_cx, disk_cy, disk_r)

    # ── 2. Extraction du ROI du disque pour MiDaS ─────────────────────────────
    x0 = max(0, disk_cx - disk_r)
    y0 = max(0, disk_cy - disk_r)
    x1 = min(w, disk_cx + disk_r)
    y1 = min(h, disk_cy + disk_r)
    roi = image[y0:y1, x0:x1]

    full_sky_mask = np.zeros((h, w), dtype=np.uint8)

    # ── 3. MiDaS (si disponible) ──────────────────────────────────────────────
    if torch is not None and cv2 is not None:
        try:
            dev   = _select_device(device)
            depth = _run_midas(roi, dev)

            # FIX : ciel = profondeur inverse FAIBLE (objet lointain)
            # On prend les pixels avec une profondeur normalisée < threshold
            sky_roi = (depth < midas_depth_threshold).astype(np.uint8) * 255

            # Remettre dans les dimensions de l'image complète
            sky_full = np.zeros((h, w), dtype=np.uint8)
            resized  = cv2.resize(sky_roi, (x1 - x0, y1 - y0),
                                  interpolation=cv2.INTER_NEAREST)
            sky_full[y0:y1, x0:x1] = resized

            # Forcer 0 hors du disque
            sky_full[~disk_mask] = 0
            full_sky_mask = sky_full

        except Exception as e:
            log.warning(f"MiDaS échoué ({e}) — passage au fallback couleur.")
            full_sky_mask = np.zeros((h, w), dtype=np.uint8)

    # ── 4. Fusion avec critère couleur ────────────────────────────────────────
    if use_color_fallback:
        color_mask = _sky_by_color(image, disk_mask)
        # Union : un pixel est ciel s'il est détecté par l'un OU l'autre
        full_sky_mask = np.where(
            (full_sky_mask > 0) | (color_mask > 0), 255, 0
        ).astype(np.uint8)

    # ── 5. Nettoyage morphologique ────────────────────────────────────────────
    if cv2 is not None and morph_ksize > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize)
        )
        full_sky_mask = cv2.morphologyEx(full_sky_mask, cv2.MORPH_CLOSE, kernel)
        full_sky_mask = cv2.morphologyEx(full_sky_mask, cv2.MORPH_OPEN, kernel // 2 + 1)

    # Garantir que le hors-disque reste à 0 après morphologie
    full_sky_mask[~disk_mask] = 0

    return full_sky_mask


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Projection fisheye → équirectangulaire
# ═══════════════════════════════════════════════════════════════════════════════

class Equirectangular:
    """Projection équirectangulaire d'une sphère (theta × phi)."""

    def __init__(self, array=None, theta=None, phi=None):
        if array is None:
            array = np.zeros((180, 360, 3), dtype=np.uint8)
        arr = np.asarray(array).astype(np.uint8)
        # Convertir un masque 2D (H, W) en (H, W, 3) pour imshow matplotlib
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        self.array = arr
        self.h, self.w = self.array.shape[:2]
        if theta is None:
            self.theta = (np.arange(self.h) + 0.5) / self.h * np.pi
        else:
            self.theta = np.asarray(theta)
        if phi is None:
            self.phi = (np.arange(self.w) + 0.5) / self.w * 2.0 * np.pi - np.pi
        else:
            self.phi = np.asarray(phi)

    def luminance(self) -> np.ndarray:
        """Retourne une image de luminance 2D (H, W) toujours, meme si array est 2D."""
        if self.array.ndim == 2:
            return self.array.astype(np.float32) / 255.0
        rgb = self.array.astype(np.float32) / 255.0
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    def flipped(self) -> "Equirectangular":
        return Equirectangular(np.flipud(self.array),
                               theta=np.flipud(self.theta), phi=self.phi)

    def rotate(self, delta_azimuth: float = 0.0,
               delta_inclination: float = 0.0,
               delta_roll: float = 0.0):
        """Rotation en place (yaw/pitch/roll) de la carte équirectangulaire."""
        eps = 1e-12
        if (abs(delta_azimuth) < eps and abs(delta_inclination) < eps
                and abs(delta_roll) < eps):
            return

        if cv2 is None:
            log.warning("cv2 absent — rotation ignorée.")
            return

        h, w = self.h, self.w
        yaw   = float(delta_azimuth)
        pitch = float(delta_inclination)
        roll  = float(delta_roll)

        if abs(pitch) < eps and abs(roll) < eps:
            col_shift = int(round((yaw / (2.0 * np.pi)) * w))
            if col_shift != 0:
                self.array = np.roll(self.array, col_shift, axis=1)
            return

        cy_ = np.cos(yaw);  sy_ = np.sin(yaw)
        cp  = np.cos(pitch); sp = np.sin(pitch)
        cr  = np.cos(roll);  sr = np.sin(roll)
        Ry = np.array([[cy_, -sy_, 0.], [sy_, cy_, 0.], [0., 0., 1.]])
        Rp = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
        Rr = np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
        R  = Ry @ Rp @ Rr

        phi_g, theta_g = np.meshgrid(self.phi, self.theta)
        s  = np.sin(theta_g)
        V  = np.stack([s * np.cos(phi_g), s * np.sin(phi_g),
                       np.cos(theta_g)], axis=-1).reshape(-1, 3)
        Vr = (R @ V.T).T

        theta_src = np.arccos(np.clip(Vr[:, 2], -1., 1.)).reshape(h, w)
        phi_src   = np.arctan2(Vr[:, 1], Vr[:, 0]).reshape(h, w)

        src_row = (theta_src / np.pi * h - 0.5).astype(np.float32)
        src_col = np.mod((phi_src + np.pi) / (2. * np.pi) * w - 0.5, w).astype(np.float32)

        self.array = cv2.remap(self.array, src_col, src_row,
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_WRAP)

    def preview(self, show_2d: bool = True):
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.imshow(self.array)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except ImportError:
            pass


def _single_fisheye_to_equirectangular(
    disk_img: np.ndarray,
    disk_cx: int, disk_cy: int, disk_r: int,
    out_h: int = 180, out_w: int = 360,
    fov_deg: float = 180.0,
) -> Equirectangular:
    """
    Projette un UNIQUE disque fisheye centré sur (disk_cx, disk_cy) avec
    rayon disk_r vers une carte équirectangulaire.

    Modèle : projection équidistante   r = f * theta
    avec f = disk_r / (fov_rad / 2)
    """
    h_img, w_img = disk_img.shape[:2]
    fov_rad = np.deg2rad(fov_deg)
    f = disk_r / (fov_rad / 2.0)

    # Grilles sphériques
    theta = (np.arange(out_h) + 0.5) / out_h * np.pi           # [0, pi]
    phi   = (np.arange(out_w) + 0.5) / out_w * 2.0 * np.pi - np.pi  # [-pi, pi]
    phi_g, theta_g = np.meshgrid(phi, theta)

    # Hémisphère supérieur (theta < pi/2) → zénith = centre du disque
    ux = np.cos(phi_g)
    uy = np.sin(phi_g)

    map_x = np.full((out_h, out_w), -1.0, dtype=np.float32)
    map_y = np.full((out_h, out_w), -1.0, dtype=np.float32)

    mask_upper = theta_g <= (np.pi / 2.0)
    if np.any(mask_upper):
        tp  = theta_g[mask_upper]
        r   = f * tp
        xi  = disk_cx + r * ux[mask_upper]
        yi  = disk_cy - r * uy[mask_upper]
        vis = tp <= (fov_rad / 2.0)
        map_x[mask_upper] = np.where(vis, xi, -1.0)
        map_y[mask_upper] = np.where(vis, yi, -1.0)

    # Hémisphère inférieur → nadir (pas de données pour un seul fisheye → noir)

    if cv2 is not None:
        out = cv2.remap(disk_img.astype(np.uint8), map_x, map_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        out  = np.zeros((out_h, out_w) + disk_img.shape[2:], dtype=np.uint8)
        xs   = np.round(map_x).astype(int)
        ys   = np.round(map_y).astype(int)
        valid = (xs >= 0) & (xs < w_img) & (ys >= 0) & (ys < h_img)
        if disk_img.ndim == 3:
            out[valid] = disk_img[ys[valid], xs[valid]]
        else:
            out[valid] = disk_img[ys[valid], xs[valid]]

    return Equirectangular(out, theta=theta, phi=phi)


def dual_fisheye_to_equirectangular(
    img,
    out_h: int = 180,
    out_w: int = 360,
    fov_deg: float = 180.0,
    single_half: Optional[str] = None,
) -> Equirectangular:
    """
    Projette une image fisheye (dual ou single) en carte équirectangulaire.

    Détecte automatiquement le format si single_half n'est pas spécifié.

    Args:
        img        : chemin, PIL Image ou array HxWx3
        out_h/out_w: dimensions de la carte de sortie
        fov_deg    : champ de vision d'un fisheye (degrés)
        single_half: forcer 'left', 'right' ou None (auto-détection)

    Returns:
        Equirectangular
    """
    if isinstance(img, str):
        img = np.array(Image.open(img).convert('RGB'))
    elif isinstance(img, Image.Image):
        img = np.array(img.convert('RGB'))
    img = np.asarray(img)

    h, w = img.shape[:2]
    w_half = w // 2

    # Auto-détection du format si non spécifié
    fmt = single_half if single_half is not None else detect_image_format(img)

    theta = (np.arange(out_h) + 0.5) / out_h * np.pi
    phi   = (np.arange(out_w) + 0.5) / out_w * 2.0 * np.pi - np.pi

    if fmt in ('left', 'right'):
        # Single fisheye : on extrait la moitié concernée
        if fmt == 'right':
            half = img[:, w_half:]
        else:
            half = img[:, :w_half]

        cx, cy, r = detect_fisheye_disk(half)
        log.info(f"Single fisheye détecté ({fmt}): centre=({cx},{cy}), r={r}")
        return _single_fisheye_to_equirectangular(
            half, cx, cy, r, out_h=out_h, out_w=out_w, fov_deg=fov_deg
        )

    else:
        # Dual fisheye : projection côte à côte classique
        if single_half == SINGLE_HALF_LEFT:
            img_work = img.copy(); img_work[:, w_half:] = 0
        elif single_half == SINGLE_HALF_RIGHT:
            img_work = img.copy(); img_work[:, :w_half] = 0
        else:
            img_work = img

        cx_left  = float(w_half) / 2.0
        cx_right = float(w_half) + cx_left
        cy_c     = float(h) / 2.0
        R_       = min(cx_left, cy_c)
        fov_rad  = np.deg2rad(fov_deg)
        f_       = R_ / (fov_rad / 2.0)

        phi_g, theta_g = np.meshgrid(phi, theta)
        ux = np.cos(phi_g); uy = np.sin(phi_g)

        map_x = np.full((out_h, out_w), -1.0, dtype=np.float32)
        map_y = np.full((out_h, out_w), -1.0, dtype=np.float32)

        mask_upper = theta_g <= (np.pi / 2.0)
        if np.any(mask_upper):
            tp = theta_g[mask_upper]; r = f_ * tp
            vis = tp <= (fov_rad / 2.0)
            map_x[mask_upper] = np.where(vis, cx_right + r * ux[mask_upper], -1.)
            map_y[mask_upper] = np.where(vis, cy_c    - r * uy[mask_upper], -1.)

        mask_lower = ~mask_upper
        if np.any(mask_lower):
            tp = np.pi - theta_g[mask_lower]; r = f_ * tp
            vis = tp <= (fov_rad / 2.0)
            map_x[mask_lower] = np.where(vis, cx_left + r * ux[mask_lower], -1.)
            map_y[mask_lower] = np.where(vis, cy_c    - r * uy[mask_lower], -1.)

        if cv2 is not None:
            out = cv2.remap(img_work.astype(np.uint8), map_x, map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            out   = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            xs    = np.round(map_x).astype(int)
            ys    = np.round(map_y).astype(int)
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            out[valid] = img_work[ys[valid], xs[valid]]

        return Equirectangular(out, theta=theta, phi=phi)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Extraction du profil d'horizon
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_uint8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if set(np.unique(mask).tolist()).issubset({0, 1}):
        mask = mask * 255
    if mask.ndim == 3 and cv2 is not None:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def get_horizon_from_sphere(eq: Equirectangular) -> np.ndarray:
    """
    Extrait le profil d'horizon depuis une Equirectangular (masque binaire).

    Pour chaque colonne (= azimut), cherche la frontière ciel/sol dans la
    demi-sphère supérieure et la convertit en élévation angulaire (radians).

    Returns:
        np.ndarray de 360 valeurs — élévation de l'horizon en radians.
        Valeur positive = horizon au-dessus de l'horizontale.
    """
    # array est toujours (H,W,3) depuis Equirectangular.__init__
    eq_mask = eq.array[:, :, 0]   # canal R = niveau de gris du masque

    h, w = eq_mask.shape[:2]
    h_top = h // 2   # on cherche uniquement dans la moitie superieure

    horizon_rows = []
    for col in range(w):
        col_data = eq_mask[:h_top, col]
        # Chercher la dernière ligne de ciel (non-noire) depuis le bas
        sky_rows = np.where(col_data > 0)[0]
        if sky_rows.size > 0:
            horizon_rows.append(int(sky_rows[-1]))   # dernière ligne de ciel
        else:
            horizon_rows.append(0)  # pas de ciel → horizon au zénith (masqué)

    # Conversion ligne → élévation en radians
    # Ligne 0 = zénith (pi/2), ligne h_top = horizon (0 rad)
    horizon_rad = np.array([
        np.pi / 2.0 * (1.0 - row / h_top)
        for row in horizon_rows
    ])

    # Ré-échantillonner à exactement 360 valeurs
    if len(horizon_rad) != 360:
        idx = np.round(np.linspace(0, len(horizon_rad) - 1, 360)).astype(int)
        horizon_rad = horizon_rad[idx]

    return np.clip(horizon_rad, 0.0, np.pi / 2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Point d'entrée principal
# ═══════════════════════════════════════════════════════════════════════════════

def compute_horizon_from_image(
    image_path: str,
    fov_deg: int = 180,
    single_half: Optional[str] = None,
    azimuth_deg: float = 0.0,
    inclination_deg: float = 90.0,
    preview: bool = False,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Pipeline complet : image fisheye → profil d'horizon 360° (en radians).

    Détecte automatiquement le format de l'image (single / dual fisheye).

    Étapes :
        1. Chargement + détection du format
        2. Détection du/des disque(s) circulaire(s)
        3. Masquage du ciel (MiDaS + couleur, corrigé)
        4. Projection équirectangulaire
        5. Rotation selon azimut et inclinaison de la caméra
        6. Extraction du profil d'horizon

    Args:
        image_path      : chemin vers l'image fisheye
        fov_deg         : champ de vision d'un fisheye (degrés)
        single_half     : forcer 'left', 'right' ou None (auto-détection)
        azimuth_deg     : azimut de la caméra (degrés, 0 = Nord)
        inclination_deg : inclinaison de la caméra (90 = pointée vers le zénith)
        preview         : affiche la projection équirectangulaire du masque
        device          : appareil PyTorch ('cpu', 'cuda', ou None=auto)

    Returns:
        np.ndarray de forme (360,) — élévation de l'horizon en radians ∈ [0, π/2]
    """
    if cv2 is None:
        log.error("OpenCV est requis pour compute_horizon_from_image.")
        return np.zeros(360)

    # ── Chargement ─────────────────────────────────────────────────────────────
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Impossible de charger : {image_path!r}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]
    w_half = w // 2

    # ── Détection du format ────────────────────────────────────────────────────
    fmt = single_half if single_half is not None else detect_image_format(image)
    log.info(f"Format détecté : {fmt}")

    # ── Masquage selon le format ───────────────────────────────────────────────
    def _safe_mask(half_img: np.ndarray) -> np.ndarray:
        cx, cy, r = detect_fisheye_disk(half_img)
        log.info(f"  Disque détecté : centre=({cx},{cy}), r={r}")
        try:
            return mask_sky(half_img, disk_cx=cx, disk_cy=cy, disk_r=r,
                            device=device)
        except Exception as e:
            log.warning(f"  mask_sky échoué ({e}) — fallback couleur.")
            dm = make_disk_mask(half_img.shape[:2], cx, cy, r)
            return _sky_by_color(half_img, dm)

    if fmt == 'right':
        right_half = image[:, w_half:]
        sky_right  = _safe_mask(right_half)
        sky_left   = np.zeros((h, w_half), dtype=np.uint8)
        image_mask = np.concatenate([sky_left, sky_right], axis=1)

    elif fmt == 'left':
        left_half  = image[:, :w_half]
        sky_left   = _safe_mask(left_half)
        sky_right  = np.zeros((h, w_half), dtype=np.uint8)
        image_mask = np.concatenate([sky_left, sky_right], axis=1)

    else:  # dual
        left_half  = image[:, :w_half]
        right_half = image[:, w_half:]
        sky_left   = _safe_mask(left_half)
        sky_right  = _safe_mask(right_half)
        image_mask = np.concatenate([sky_left, sky_right], axis=1)

    image_mask = _ensure_uint8_mask(image_mask)

    # ── Projection équirectangulaire ────────────────────────────────────────────
    eq = dual_fisheye_to_equirectangular(
        image_mask,
        out_h=180, out_w=360,
        fov_deg=fov_deg,
        single_half=fmt if fmt in ('left', 'right') else None,
    )

    eq.rotate(
        delta_azimuth=np.deg2rad(float(azimuth_deg)),
        delta_inclination=np.deg2rad(float(inclination_deg)),
    )

    if preview:
        eq.preview()

    # ── Extraction du profil d'horizon ─────────────────────────────────────────
    horizon = get_horizon_from_sphere(eq)

    if horizon is None or horizon.shape[0] != 360:
        log.warning("Extraction d'horizon échouée → fallback horizon plat.")
        return np.zeros(360)

    return horizon
