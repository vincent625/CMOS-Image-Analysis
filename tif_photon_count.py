#!/usr/bin/env python3
"""
tif_photon_count.py

Quantify photons per frame from a TIFF sequence (chemiluminescence / low-light microscopy).

What it computes
----------------
For each frame it computes an *integrated net signal* in a region of interest (ROI),
then converts camera digital numbers (ADU/DN) -> photoelectrons -> photons at the sensor.

Core formula (per frame):
    corrected_ADU  = raw_ADU - dark_frame_ADU (optional) - offset_ADU (optional)
    bg_ADU         = estimated background level (percentile, border, or background ROI)
    net_ADU        = max(corrected_ADU - bg_ADU, 0)
    sum_ADU        = sum(net_ADU over signal ROI)

    photoelectrons = sum_ADU * (gain_e_per_adu / em_gain)
    photons_sensor = photoelectrons / qe

Optionally estimate photons emitted by the sample:
    photons_emitted = photons_sensor / system_efficiency

Dependencies:
  pip install numpy pillow tifffile
Optional for faster decoding:
  pip install imagecodecs
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageSequence

try:
    import tifffile
except Exception:
    tifffile = None  # We'll still work using Pillow for I/O.


def natural_key(s: str) -> List[Union[int, str]]:
    parts = re.split(r"(\d+)", s)
    out: List[Union[int, str]] = []
    for p in parts:
        out.append(int(p) if p.isdigit() else p.lower())
    return out


def expand_inputs(input_arg: str) -> List[Path]:
    p = Path(input_arg)
    if p.exists() and p.is_dir():
        files: List[Path] = []
        for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
            files.extend(p.glob(ext))
        return sorted(files, key=lambda x: natural_key(x.name))

    if any(ch in input_arg for ch in ["*", "?", "["]):
        import glob as _glob
        matches = [Path(m) for m in _glob.glob(input_arg)]
        matches = [m for m in matches if m.suffix.lower() in (".tif", ".tiff")]
        return sorted(matches, key=lambda x: natural_key(x.name))

    return [p]


@dataclass(frozen=True)
class FrameInfo:
    frame_index: int
    file_path: Path
    page_index: int


def _get_description_pil(im: Image.Image) -> Optional[str]:
    """Try to read TIFF ImageDescription (tag 270) via Pillow."""
    try:
        if hasattr(im, "tag_v2") and im.tag_v2 is not None:
            desc = im.tag_v2.get(270)
            if desc:
                return str(desc)
        if hasattr(im, "tag") and im.tag is not None:
            desc = im.tag.get(270)
            if isinstance(desc, (list, tuple)) and desc:
                return str(desc[0])
            if desc:
                return str(desc)
    except Exception:
        pass
    return None


def _exposure_seconds_from_ome(description: Optional[str]) -> Optional[float]:
    """
    Parse exposure time (seconds) from OME-XML if present.
    Looks for <Plane ... ExposureTime="0.5" ExposureTimeUnit="s" ... />
    """
    if not description or "<OME" not in description:
        return None

    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(description)
        for elem in root.iter():
            if elem.tag.endswith("Plane"):
                et = elem.attrib.get("ExposureTime")
                if et is None:
                    continue
                unit = elem.attrib.get("ExposureTimeUnit", "s").lower()
                val = float(et)
                if unit in ("s", "sec", "second", "seconds"):
                    return val
                if unit in ("ms", "millisecond", "milliseconds"):
                    return val / 1000.0
                if unit in ("us", "µs", "microsecond", "microseconds"):
                    return val / 1_000_000.0
                return val
    except Exception:
        return None
    return None


def _iter_pages_tifffile(path: Path) -> Iterator[np.ndarray]:
    assert tifffile is not None
    with tifffile.TiffFile(path) as tf:
        for page in tf.pages:
            yield page.asarray()


def _iter_pages_pillow(path: Path) -> Iterator[np.ndarray]:
    with Image.open(path) as im:
        for page in ImageSequence.Iterator(im):
            yield np.asarray(page)


def _read_exposure_from_first_page(path: Path) -> Optional[float]:
    if tifffile is not None:
        try:
            with tifffile.TiffFile(path) as tf:
                desc = tf.pages[0].description
            exp = _exposure_seconds_from_ome(desc)
            if exp is not None:
                return exp
        except Exception:
            pass

    try:
        with Image.open(path) as im:
            desc = _get_description_pil(im)
        return _exposure_seconds_from_ome(desc)
    except Exception:
        return None


def iter_frames(paths: Sequence[Path]) -> Iterator[Tuple[FrameInfo, np.ndarray, Optional[float]]]:
    frame_i = 0
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing TIFF: {path}")

        exposure_s = _read_exposure_from_first_page(path)

        used_pillow = False
        if tifffile is not None:
            try:
                pages_iter = _iter_pages_tifffile(path)
                first = next(pages_iter)

                def chain_first():
                    yield first
                    for x in pages_iter:
                        yield x

                pages = chain_first()
            except Exception:
                used_pillow = True
        else:
            used_pillow = True

        if used_pillow:
            pages = _iter_pages_pillow(path)

        for page_i, arr in enumerate(pages):
            yield FrameInfo(frame_i, path, page_i), arr, exposure_s
            frame_i += 1


def pick_channel(arr: np.ndarray, channel: str) -> np.ndarray:
    """
    Return a 2D float32 image.

    Handles:
      - (Y, X)
      - (Y, X, 1)
      - (Y, X, 3/4) RGB/RGBA
      - (C, Y, X)   multi-channel planes
    """
    if arr.ndim == 2:
        return arr.astype(np.float32)

    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[..., 0].astype(np.float32)

    ch = channel.lower()

    # RGB/RGBA
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr_f = arr.astype(np.float32)
        if ch == "auto":
            means = [float(np.mean(arr_f[..., i])) for i in range(3)]
            return arr_f[..., int(np.argmax(means))]
        if ch in ("r", "g", "b"):
            return arr_f[..., {"r": 0, "g": 1, "b": 2}[ch]]
        if ch == "gray":
            return 0.299 * arr_f[..., 0] + 0.587 * arr_f[..., 1] + 0.114 * arr_f[..., 2]

    # (C, Y, X)
    if arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16:
        arr_f = arr.astype(np.float32)
        if ch == "auto":
            means = [float(np.mean(arr_f[i, ...])) for i in range(arr_f.shape[0])]
            return arr_f[int(np.argmax(means)), ...]
        if ch in ("r", "g", "b") and arr_f.shape[0] >= 3:
            return arr_f[{"r": 0, "g": 1, "b": 2}[ch], ...]
        if ch == "gray":
            return np.mean(arr_f, axis=0)

    raise ValueError(f"Unsupported image shape {arr.shape}.")


def crop(img: np.ndarray, roi: Optional[Sequence[int]]) -> np.ndarray:
    if roi is None:
        return img
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        raise ValueError("ROI width/height must be > 0")
    return img[y:y + h, x:x + w]


def border_region(img: np.ndarray, border: int) -> np.ndarray:
    b = int(border)
    h, w = img.shape
    b = min(b, h // 2, w // 2)
    top = img[:b, :]
    bot = img[h - b:, :]
    left = img[b:h - b, :b]
    right = img[b:h - b, w - b:]
    return np.concatenate([top.ravel(), bot.ravel(), left.ravel(), right.ravel()])


def estimate_background(
    img: np.ndarray,
    bg_mode: str,
    bg_percentile: float,
    bg_roi: Optional[Sequence[int]],
    border: int,
) -> float:
    mode = bg_mode.lower()
    if mode == "none":
        return 0.0
    if mode == "roi":
        if bg_roi is None:
            raise ValueError("--bg-mode roi requires --bg-roi x y w h")
        region = crop(img, bg_roi).ravel()
        return float(np.percentile(region, bg_percentile))
    if mode == "border":
        region = border_region(img, border)
        return float(np.percentile(region, bg_percentile))
    if mode == "percentile":
        return float(np.percentile(img, bg_percentile))
    raise ValueError(f"Unknown --bg-mode: {bg_mode}")


def load_dark_frame(dark_path: str, channel: str) -> np.ndarray:
    p = Path(dark_path)
    frames = []
    for _, arr, _ in iter_frames([p]):
        frames.append(pick_channel(arr, channel))
    return np.mean(np.stack(frames, axis=0), axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute photons per frame from TIFF sequence.")
    ap.add_argument("--input", required=True, help="Directory, glob, or TIFF file.")
    ap.add_argument("--out", default="photons_by_frame.csv", help="Output CSV.")
    ap.add_argument("--channel", default="auto", choices=["auto", "r", "g", "b", "gray"])
    ap.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                    help="Signal ROI. If omitted, uses whole image.")
    ap.add_argument("--bg-mode", default="percentile", choices=["none", "percentile", "border", "roi"])
    ap.add_argument("--bg-percentile", type=float, default=10.0)
    ap.add_argument("--bg-roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    ap.add_argument("--border", type=int, default=32)

    # Calibration
    ap.add_argument("--offset-adu", type=float, default=0.0,
                    help="Bias/offset in ADU to subtract (default 0).")
    ap.add_argument("--dark-frame", type=str, default=None,
                    help="Dark frame TIFF (same exposure/settings). Recommended.")
    ap.add_argument("--gain-e-per-adu", type=float, default=1.0,
                    help="Conversion gain: electrons per ADU.")
    ap.add_argument("--em-gain", type=float, default=1.0,
                    help="EM multiplication gain (EMCCD). Use 1 if not applicable.")
    ap.add_argument("--qe", type=float, default=1.0,
                    help="Quantum efficiency (0..1).")
    ap.add_argument("--system-efficiency", type=float, default=1.0,
                    help="Fraction of emitted photons that reach the sensor (0..1).")
    ap.add_argument("--exposure-s", type=float, default=None,
                    help="Exposure time in seconds. If omitted, tries to parse OME metadata.")

    args = ap.parse_args()

    paths = expand_inputs(args.input)
    if not paths:
        raise SystemExit(f"No TIFFs found for input: {args.input}")

    if not (0 < args.qe <= 1):
        raise SystemExit("--qe must be in (0, 1].")
    if not (0 < args.system_efficiency <= 1):
        raise SystemExit("--system-efficiency must be in (0, 1].")
    if args.gain_e_per_adu <= 0:
        raise SystemExit("--gain-e-per-adu must be > 0.")
    if args.em_gain <= 0:
        raise SystemExit("--em-gain must be > 0.")

    dark = load_dark_frame(args.dark_frame, args.channel) if args.dark_frame else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for info, arr, exposure_from_meta in iter_frames(paths):
        img = pick_channel(arr, args.channel)

        if dark is not None:
            if dark.shape != img.shape:
                raise SystemExit(f"Dark frame shape {dark.shape} != image shape {img.shape}")
            img = img - dark

        img = img - float(args.offset_adu)

        bg = estimate_background(img, args.bg_mode, args.bg_percentile, args.bg_roi, args.border)

        sig_region = crop(img, args.roi)
        net = np.clip(sig_region - bg, 0, None)

        sum_adu = float(np.sum(net))
        mean_adu = float(np.mean(net))
        n_pix = int(net.size)

        photoelectrons = sum_adu * (float(args.gain_e_per_adu) / float(args.em_gain))
        photons_sensor = photoelectrons / float(args.qe)
        photons_emitted = photons_sensor / float(args.system_efficiency)

        exposure_s = args.exposure_s if args.exposure_s is not None else exposure_from_meta
        photons_per_s = (photons_sensor / exposure_s) if exposure_s and exposure_s > 0 else ""

        rows.append({
            "frame": info.frame_index,
            "file": info.file_path.name,
            "page_in_file": info.page_index,
            "n_pixels": n_pix,
            "bg_level_adu": bg,
            "sum_net_adu": sum_adu,
            "photoelectrons": photoelectrons,
            "photons_sensor": photons_sensor,
            "photons_emitted": photons_emitted,
            "exposure_s": exposure_s if exposure_s is not None else "",
            "photons_per_s": photons_per_s,
        })

    fieldnames = [
        "frame", "file", "page_in_file", "n_pixels",
        "bg_level_adu", "sum_net_adu",
        "photoelectrons", "photons_sensor", "photons_emitted",
        "exposure_s", "photons_per_s",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} frame(s) to {out_path.resolve()}")


if __name__ == "__main__":
    main()

