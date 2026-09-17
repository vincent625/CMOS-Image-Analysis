#!/usr/bin/env python3
"""
raw_photon_count_anysize.py

Integrate chemiluminescence RAW frames and output per-frame integrated intensity
(DN / electrons / photons). Compatible with "any size" RAW in the sense that it can:
  - Use explicit --width/--height (always reliable), OR
  - Auto-infer dimensions from file size via heuristics (best effort).

Because .raw has no metadata, inference can be ambiguous. This script prints what it
chose and a few top candidates. If it guesses wrong, pass --width/--height.

Dependencies:
  pip install numpy

Example:
  python raw_photon_count_anysize.py --input "seq/*.raw" --out_csv out.csv --dark dark.raw
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------
# Helpers: natural sorting
# -----------------------------

_num_re = re.compile(r"(\d+)")

def natural_key(s: str):
    parts = _num_re.split(s)
    key = []
    for i, p in enumerate(parts):
        if i % 2 == 1:
            try:
                key.append(int(p))
            except ValueError:
                key.append(p)
        else:
            key.append(p.lower())
    return tuple(key)

def expand_inputs(inputs: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for item in inputs:
        matches = glob.glob(item)
        if matches:
            out.extend(Path(m) for m in matches)
        else:
            out.append(Path(item))

    uniq: List[Path] = []
    seen = set()
    for p in out:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in seen:
            continue
        seen.add(rp)
        if p.exists():
            uniq.append(p)

    return sorted(uniq, key=lambda p: natural_key(p.name))


# -----------------------------
# RAW reading / unpacking
# -----------------------------

def unpack_mipi_raw10(buf: np.ndarray, width: int, height: int) -> np.ndarray:
    """Unpack MIPI RAW10 packed (5 bytes for 4 pixels) into uint16 image (H, W)."""
    expected = (width * height // 4) * 5
    if buf.size != expected:
        raise ValueError(f"RAW10 packed size mismatch: got {buf.size} bytes, expected {expected} bytes.")
    b = buf.reshape(-1, 5).astype(np.uint16)
    p0 = b[:, 0] | ((b[:, 4] & 0x03) << 8)
    p1 = b[:, 1] | ((b[:, 4] & 0x0C) << 6)
    p2 = b[:, 2] | ((b[:, 4] & 0x30) << 4)
    p3 = b[:, 3] | ((b[:, 4] & 0xC0) << 2)
    out = np.empty((b.shape[0], 4), dtype=np.uint16)
    out[:, 0] = p0
    out[:, 1] = p1
    out[:, 2] = p2
    out[:, 3] = p3
    return out.reshape(height, width)

def unpack_mipi_raw12(buf: np.ndarray, width: int, height: int) -> np.ndarray:
    """Unpack MIPI RAW12 packed (3 bytes for 2 pixels) into uint16 image (H, W)."""
    expected = (width * height // 2) * 3
    if buf.size != expected:
        raise ValueError(f"RAW12 packed size mismatch: got {buf.size} bytes, expected {expected} bytes.")
    b = buf.reshape(-1, 3).astype(np.uint16)
    p0 = b[:, 0] | ((b[:, 2] & 0x0F) << 8)
    p1 = b[:, 1] | ((b[:, 2] & 0xF0) << 4)
    out = np.empty((b.shape[0], 2), dtype=np.uint16)
    out[:, 0] = p0
    out[:, 1] = p1
    return out.reshape(height, width)

@dataclass(frozen=True)
class RawParams:
    width: int
    height: int
    bit_depth: int          # 8/10/12/16 (used for saturation logic)
    pack: str               # none | mipi10 | mipi12
    offset: int = 0
    stride_bytes: Optional[int] = None  # bytes per row (padded). None = contiguous.

def _dtype_for_bit_depth(bit_depth: int) -> np.dtype:
    return np.uint8 if bit_depth <= 8 else np.uint16

def load_raw_mosaic(path: Path, params: RawParams) -> np.ndarray:
    """
    Load raw mosaic.
    - pack=none: memmap if contiguous, otherwise row-by-row into RAM.
    - pack=mipi10/12: unpack into uint16 array.
    """
    pack = params.pack.lower()
    if pack == "none":
        dtype = _dtype_for_bit_depth(params.bit_depth)
        bpp = np.dtype(dtype).itemsize
        row_bytes = params.width * bpp
        if params.stride_bytes is None:
            expected = params.offset + row_bytes * params.height
            actual = path.stat().st_size
            if actual < expected:
                raise ValueError(f"{path.name}: file too small; got {actual}, need >= {expected}.")
            return np.memmap(path, dtype=dtype, mode="r", offset=params.offset, shape=(params.height, params.width))
        else:
            out = np.empty((params.height, params.width), dtype=dtype)
            with path.open("rb") as f:
                f.seek(params.offset, 0)
                for y in range(params.height):
                    row = f.read(row_bytes)
                    if len(row) != row_bytes:
                        raise ValueError(f"{path.name}: short read on row {y}")
                    out[y] = np.frombuffer(row, dtype=dtype, count=params.width)
                    skip = params.stride_bytes - row_bytes
                    if skip < 0:
                        raise ValueError("--stride_bytes must be >= width*bytes_per_pixel")
                    if skip:
                        f.seek(skip, 1)
            return out

    buf = np.fromfile(path, dtype=np.uint8)
    if pack == "mipi10":
        return unpack_mipi_raw10(buf, params.width, params.height)
    if pack == "mipi12":
        return unpack_mipi_raw12(buf, params.width, params.height)
    raise ValueError(f"Unknown pack '{params.pack}'. Use: none, mipi10, mipi12.")


# -----------------------------
# Auto-inference of width/height (best effort)
# -----------------------------

ASPECT_TARGETS = [
    (4, 3),
    (16, 9),
    (3, 2),
    (1, 1),
    (5, 4),
    (17, 9),
    (9, 16),
    (3, 4),
    (2, 3),
]

def _aspect_score(w: int, h: int) -> float:
    r = w / h
    best = 1e9
    for a, b in ASPECT_TARGETS:
        t = a / b
        best = min(best, abs(math.log(r / t)))
    return best

def _dim_score(w: int, h: int) -> float:
    """
    Lower is better. Preferences:
      - even dims (Bayer friendly)
      - multiples of 16
      - common aspect ratios
      - "reasonable" width range
    """
    score = 0.0
    if (w & 1) or (h & 1):
        score += 2.0  # penalize odd dims
    if (w % 16) != 0:
        score += 0.3
    if (h % 16) != 0:
        score += 0.3
    if w < 64 or h < 64:
        score += 5.0
    score += 2.0 * _aspect_score(w, h)
    return score

def _candidate_shapes_from_pixels(pixels: int) -> List[Tuple[int, int, float]]:
    """
    Return candidate (w,h,score) pairs such that w*h=pixels.
    Only enumerates divisors up to sqrt(pixels).
    """
    cand: List[Tuple[int, int, float]] = []
    root = int(math.sqrt(pixels))
    for h in range(1, root + 1):
        if pixels % h != 0:
            continue
        w = pixels // h
        # prefer landscape-ish by default: w>=h; still allow both by swapping
        s1 = _dim_score(w, h)
        cand.append((w, h, s1))
        if w != h:
            s2 = _dim_score(h, w)
            cand.append((h, w, s2))
    cand.sort(key=lambda x: x[2])
    return cand

def infer_raw_params_generic(
    path: Path,
    user_width: Optional[int],
    user_height: Optional[int],
    user_bit_depth: Optional[int],
    user_pack: str,
    offset: int,
    stride_bytes: Optional[int],
    verbose: bool = True,
) -> RawParams:
    """
    If width/height provided -> use them.
    Else attempt to infer from file size using selected pack + bit_depth or by trying:
      - none + 8-bit
      - none + 16-bit container
      - mipi10
      - mipi12
    """
    if user_width is not None and user_height is not None and user_bit_depth is not None and user_pack.lower() != "auto":
        return RawParams(int(user_width), int(user_height), int(user_bit_depth), str(user_pack), offset=int(offset), stride_bytes=stride_bytes)

    file_bytes = path.stat().st_size
    usable = file_bytes - int(offset)
    if usable <= 0:
        raise ValueError(f"{path.name}: invalid --offset ({offset}) for file size ({file_bytes}).")

    pack_options = [user_pack.lower()] if user_pack.lower() != "auto" else ["none", "mipi10", "mipi12"]
    bit_options = [int(user_bit_depth)] if user_bit_depth is not None else [8, 16, 10, 12]

    # Build candidate interpretations: (pack, bit_depth, pixels)
    interpretations: List[Tuple[str, int, int]] = []
    for pack in pack_options:
        if pack == "none":
            for bd in bit_options:
                if bd <= 8:
                    # RAW8: bytes = w*h (if contiguous). If stride_bytes is set, cannot infer safely.
                    if stride_bytes is not None:
                        continue
                    pixels = usable
                    interpretations.append(("none", 8, pixels))
                else:
                    if stride_bytes is not None:
                        continue
                    if usable % 2 != 0:
                        continue
                    pixels = usable // 2
                    # treat as 16-bit container; user can specify real bd for saturation
                    interpretations.append(("none", max(bd, 10), pixels))
        elif pack == "mipi10":
            # bytes = (pixels/4)*5 => pixels = bytes*4/5, must be divisible
            if usable % 5 != 0:
                continue
            pixels = (usable // 5) * 4
            interpretations.append(("mipi10", 10, pixels))
        elif pack == "mipi12":
            # bytes = (pixels/2)*3 => pixels = bytes*2/3
            if usable % 3 != 0:
                continue
            pixels = (usable // 3) * 2
            interpretations.append(("mipi12", 12, pixels))
        else:
            raise ValueError(f"Unknown --pack '{user_pack}'. Use: none, mipi10, mipi12, auto.")

    # If user provided width or height, constrain candidates.
    best: Optional[Tuple[RawParams, float, List[Tuple[int,int,float]]]] = None

    for pack, bd, pixels in interpretations:
        if pixels <= 0:
            continue

        # Constrain by partial user dims
        if user_width is not None and user_height is None:
            w = int(user_width)
            if pixels % w != 0:
                continue
            h = pixels // w
            cands = [(w, h, _dim_score(w, h))]
        elif user_height is not None and user_width is None:
            h = int(user_height)
            if pixels % h != 0:
                continue
            w = pixels // h
            cands = [(w, h, _dim_score(w, h))]
        else:
            cands = _candidate_shapes_from_pixels(pixels)
            if not cands:
                continue

        w0, h0, s0 = cands[0]
        params = RawParams(width=w0, height=h0, bit_depth=bd, pack=pack, offset=int(offset), stride_bytes=stride_bytes)

        if best is None or s0 < best[1]:
            best = (params, s0, cands[:8])

    if best is None:
        raise ValueError(
            "Could not infer width/height from file size.\n"
            "Try providing --width and --height (recommended), and set --bit_depth/--pack if known.\n"
            "Note: if your dump has per-row padding, inference is impossible without --stride_bytes."
        )

    params, score, top = best
    if verbose:
        print(f"[infer] {path.name}: chose pack={params.pack}, bit_depth={params.bit_depth}, "
              f"width={params.width}, height={params.height} (score={score:.3f})")
        print("[infer] top candidates (w x h):")
        for w, h, sc in top:
            print(f"         {w} x {h}  (score={sc:.3f})")

    return params


# -----------------------------
# CFA-aware photon integration (unchanged)
# -----------------------------

BAYER_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")

def _sum_cfa_planes(block: np.ndarray, y0: int, x0: int, bayer: str) -> Tuple[float, float, float]:
    b = bayer.upper()
    if b not in BAYER_PATTERNS:
        raise ValueError(f"Unsupported bayer '{bayer}'. Use RGGB/BGGR/GRBG/GBRG.")
    tile = {
        "RGGB": (("R", "G"), ("G", "B")),
        "BGGR": (("B", "G"), ("G", "R")),
        "GRBG": (("G", "R"), ("B", "G")),
        "GBRG": (("G", "B"), ("R", "G")),
    }[b]
    rp0 = y0 & 1
    cp0 = x0 & 1
    sums = {"R": 0.0, "G": 0.0, "B": 0.0}
    for rp in (0, 1):
        for cp in (0, 1):
            color = tile[(rp0 + rp) % 2][(cp0 + cp) % 2]
            plane = block[rp::2, cp::2]
            sums[color] += float(plane.sum(dtype=np.float64))
    return sums["R"], sums["G"], sums["B"]

@dataclass
class PhotonModel:
    e_per_dn: float = 1.0
    qe: Optional[float] = None
    qe_r: Optional[float] = None
    qe_g: Optional[float] = None
    qe_b: Optional[float] = None
    bayer: str = "RGGB"

    def has_per_cfa_qe(self) -> bool:
        return (self.qe_r is not None) and (self.qe_g is not None) and (self.qe_b is not None)

    def validate(self) -> None:
        if self.e_per_dn <= 0:
            raise ValueError("--e_per_dn must be > 0")
        if self.qe is not None and not (0 < self.qe <= 1.0):
            raise ValueError("--qe must be in (0,1]")
        if self.has_per_cfa_qe():
            for name, val in (("qe_r", self.qe_r), ("qe_g", self.qe_g), ("qe_b", self.qe_b)):
                if val is None or not (0 < val <= 1.0):
                    raise ValueError(f"--{name} must be in (0,1]")
        if self.bayer.upper() not in BAYER_PATTERNS:
            raise ValueError("--bayer must be RGGB/BGGR/GRBG/GBRG")

@dataclass
class IntegrateResult:
    sum_dn: float
    sum_e: float
    sum_photons: float
    sat_pixels: int
    total_pixels: int
    black_est: float

def integrate_frame(
    frame: np.ndarray,
    *,
    dark: Optional[np.ndarray],
    roi: Optional[Tuple[int, int, int, int]],
    black_level: float,
    auto_black_percentile: Optional[float],
    dn_scale: float,
    bit_depth: int,
    photon_model: PhotonModel,
    block_rows: int = 256,
    frame_origin: Tuple[int, int] = (0, 0),
) -> IntegrateResult:
    photon_model.validate()

    H, W = frame.shape
    x0, y0, rw, rh = (0, 0, W, H) if roi is None else roi
    x1 = min(W, x0 + rw)
    y1 = min(H, y0 + rh)
    if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
        raise ValueError("ROI is out of bounds.")

    black_est = float(black_level)
    if dark is None and auto_black_percentile is not None:
        p = float(auto_black_percentile)
        if not (0.0 <= p <= 50.0):
            raise ValueError("--auto_black_percentile should be between 0 and 50 (e.g., 1.0).")
        sample = np.array(frame[y0:y1:32, x0:x1:32], dtype=np.float32) * float(dn_scale)
        black_est = float(np.percentile(sample, p))

    sat_dn = (2 ** int(bit_depth)) - 1

    sum_dn = 0.0
    sat_pixels = 0
    total_pixels = (y1 - y0) * (x1 - x0)

    use_cfa = photon_model.has_per_cfa_qe()
    sum_dn_r = sum_dn_g = sum_dn_b = 0.0

    for yy in range(y0, y1, block_rows):
        y_end = min(y1, yy + block_rows)
        block = frame[yy:y_end, x0:x1]

        block_f = block.astype(np.float32) * float(dn_scale)

        if dark is not None:
            dark_block = dark[yy:y_end, x0:x1].astype(np.float32) * float(dn_scale)
            corr = block_f - dark_block
        else:
            corr = block_f - black_est

        sat_pixels += int(np.count_nonzero(block_f >= sat_dn))
        np.maximum(corr, 0.0, out=corr)

        if use_cfa:
            global_y0 = frame_origin[0] + yy
            global_x0 = frame_origin[1] + x0
            r, g, b = _sum_cfa_planes(corr, global_y0, global_x0, photon_model.bayer)
            sum_dn_r += r
            sum_dn_g += g
            sum_dn_b += b
        else:
            sum_dn += float(corr.sum(dtype=np.float64))

    if use_cfa:
        sum_e_r = sum_dn_r * photon_model.e_per_dn
        sum_e_g = sum_dn_g * photon_model.e_per_dn
        sum_e_b = sum_dn_b * photon_model.e_per_dn
        sum_e = sum_e_r + sum_e_g + sum_e_b
        sum_ph = (sum_e_r / photon_model.qe_r) + (sum_e_g / photon_model.qe_g) + (sum_e_b / photon_model.qe_b)
        return IntegrateResult(sum_dn=sum_dn_r + sum_dn_g + sum_dn_b,
                               sum_e=sum_e, sum_photons=float(sum_ph),
                               sat_pixels=sat_pixels, total_pixels=total_pixels,
                               black_est=black_est)

    sum_e = sum_dn * photon_model.e_per_dn
    sum_ph = float("nan") if photon_model.qe is None else (sum_e / photon_model.qe)

    return IntegrateResult(sum_dn=sum_dn, sum_e=sum_e, sum_photons=float(sum_ph),
                           sat_pixels=sat_pixels, total_pixels=total_pixels,
                           black_est=black_est)


# -----------------------------
# Dark handling
# -----------------------------

def build_master_dark_mean(
    dark_paths: Sequence[Path],
    params: RawParams,
    cache_path: Path,
    block_rows: int = 256,
) -> np.ndarray:
    if len(dark_paths) == 0:
        raise ValueError("No dark files provided.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    acc = np.memmap(cache_path, dtype=np.float32, mode="w+", shape=(params.height, params.width))
    acc[:] = 0.0
    for i, p in enumerate(dark_paths, 1):
        dark = load_raw_mosaic(p, params)
        for yy in range(0, params.height, block_rows):
            y_end = min(params.height, yy + block_rows)
            acc[yy:y_end, :] += dark[yy:y_end, :].astype(np.float32)
        print(f"[dark] accumulated {i}/{len(dark_paths)}: {p.name}")
    acc[:] /= float(len(dark_paths))
    acc.flush()
    return acc


# -----------------------------
# CLI
# -----------------------------

def parse_roi(vals: Optional[Sequence[int]]) -> Optional[Tuple[int, int, int, int]]:
    if vals is None:
        return None
    if len(vals) != 4:
        raise ValueError("--roi must be 4 integers: x y w h")
    x, y, w, h = (int(v) for v in vals)
    if w <= 0 or h <= 0:
        raise ValueError("ROI w/h must be > 0")
    return (x, y, w, h)

def main() -> None:
    ap = argparse.ArgumentParser(description="Integrate RAW chemiluminescence frames and estimate photon counts.")
    ap.add_argument("--input", nargs="+", required=True, help="Input .raw file(s) or glob(s) like 'dir/*.raw'")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")

    ap.add_argument("--width", type=int, default=None, help="Image width (optional; if omitted script tries to infer)")
    ap.add_argument("--height", type=int, default=None, help="Image height (optional; if omitted script tries to infer)")
    ap.add_argument("--bit_depth", type=int, default=None, help="Bit depth (8/10/12/16). If omitted tries to infer.")
    ap.add_argument("--pack", type=str, default="auto", help="Packing: none, mipi10, mipi12, auto (default auto)")
    ap.add_argument("--offset", type=int, default=0, help="Byte offset to skip header (default 0)")
    ap.add_argument("--stride_bytes", type=int, default=None, help="Bytes per row (padded dumps). If set, inference is hard.")
    ap.add_argument("--quiet_infer", action="store_true", help="Suppress printing inference candidates.")

    ap.add_argument("--dark", nargs="*", default=None,
                    help="Dark frame file(s) or glob(s). If one file -> per-pixel subtract. "
                         "If multiple -> build mean dark (disk cached).")
    ap.add_argument("--dark_cache", type=str, default=None,
                    help="Optional path to cache master dark (default: next to out_csv as master_dark_float32.dat)")

    ap.add_argument("--roi", nargs=4, type=int, default=None, metavar=("X", "Y", "W", "H"),
                    help="ROI (x y w h). If omitted, uses full frame.")

    ap.add_argument("--black_level", type=float, default=0.0,
                    help="Black/bias level to subtract ONLY if no dark frame is given (in DN after scaling).")
    ap.add_argument("--auto_black_percentile", type=float, default=None,
                    help="If set and no dark provided, estimate black level per frame from ROI percentile (e.g. 1.0).")

    ap.add_argument("--dn_scale", type=float, default=1.0,
                    help="Multiply stored DN by this factor before processing.")

    ap.add_argument("--e_per_dn", type=float, default=1.0,
                    help="Electrons per DN. For relative counts, keep 1.0.")
    ap.add_argument("--qe", type=float, default=None, help="Effective QE (0..1) for all pixels.")
    ap.add_argument("--qe_r", type=float, default=None, help="QE for R pixels (0..1). Requires --qe_g and --qe_b.")
    ap.add_argument("--qe_g", type=float, default=None, help="QE for G pixels (0..1). Requires --qe_r and --qe_b.")
    ap.add_argument("--qe_b", type=float, default=None, help="QE for B pixels (0..1). Requires --qe_r and --qe_g.")
    ap.add_argument("--bayer", type=str, default="RGGB", help="Bayer order for per-CFA QE: RGGB/BGGR/GRBG/GBRG")

    ap.add_argument("--exposure", type=float, default=None, help="Exposure time (s). If set, outputs photons_per_s.")
    ap.add_argument("--block_rows", type=int, default=256, help="Processing block height (rows) (default 256)")

    args = ap.parse_args()

    in_paths = expand_inputs(args.input)
    if not in_paths:
        raise SystemExit("No input files found.")

    # Infer RAW params from the first frame (assumes all frames share same format)
    params = infer_raw_params_generic(
        in_paths[0],
        user_width=args.width,
        user_height=args.height,
        user_bit_depth=args.bit_depth,
        user_pack=args.pack,
        offset=args.offset,
        stride_bytes=args.stride_bytes,
        verbose=(not args.quiet_infer),
    )

    roi = parse_roi(args.roi)

    photon_model = PhotonModel(
        e_per_dn=float(args.e_per_dn),
        qe=(None if args.qe is None else float(args.qe)),
        qe_r=(None if args.qe_r is None else float(args.qe_r)),
        qe_g=(None if args.qe_g is None else float(args.qe_g)),
        qe_b=(None if args.qe_b is None else float(args.qe_b)),
        bayer=str(args.bayer),
    )
    photon_model.validate()

    # Dark handling
    dark_arr: Optional[np.ndarray] = None
    if args.dark:
        dark_paths = expand_inputs(args.dark)
        if len(dark_paths) == 0:
            raise SystemExit("No dark files found from --dark inputs.")
        if len(dark_paths) == 1:
            dark_arr = load_raw_mosaic(dark_paths[0], params)
            print(f"[dark] Using single dark frame: {dark_paths[0].name}")
        else:
            out_csv_path = Path(args.out_csv)
            cache_path = Path(args.dark_cache) if args.dark_cache else (out_csv_path.parent / "master_dark_float32.dat")
            if cache_path.exists():
                dark_arr = np.memmap(cache_path, dtype=np.float32, mode="r", shape=(params.height, params.width))
                print(f"[dark] Loaded cached master dark: {cache_path}")
            else:
                print(f"[dark] Building mean dark from {len(dark_paths)} frames -> {cache_path}")
                dark_arr = build_master_dark_mean(dark_paths, params, cache_path, block_rows=args.block_rows)
                print(f"[dark] Built and cached master dark: {cache_path}")

    # Output CSV
    out_csv_path = Path(args.out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame_idx",
        "filename",
        "sum_dn_corrected",
        "sum_electrons",
        "sum_photons_est",
        "photons_per_s",
        "black_level_used",
        "sat_pixels",
        "sat_fraction",
        "roi_x",
        "roi_y",
        "roi_w",
        "roi_h",
        "width",
        "height",
        "bit_depth",
        "pack",
    ]

    with out_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, p in enumerate(in_paths):
            frame = load_raw_mosaic(p, params)

            res = integrate_frame(
                frame,
                dark=dark_arr,
                roi=roi,
                black_level=float(args.black_level),
                auto_black_percentile=args.auto_black_percentile,
                dn_scale=float(args.dn_scale),
                bit_depth=int(params.bit_depth),
                photon_model=photon_model,
                block_rows=int(args.block_rows),
                frame_origin=(0, 0),
            )

            if args.exposure is None or math.isnan(res.sum_photons):
                photons_per_s = float("nan")
            else:
                photons_per_s = res.sum_photons / float(args.exposure)

            rx, ry, rw, rh = (0, 0, params.width, params.height) if roi is None else roi
            sat_frac = res.sat_pixels / float(res.total_pixels) if res.total_pixels else 0.0

            writer.writerow({
                "frame_idx": idx,
                "filename": p.name,
                "sum_dn_corrected": f"{res.sum_dn:.6f}",
                "sum_electrons": f"{res.sum_e:.6f}",
                "sum_photons_est": f"{res.sum_photons:.6f}" if not math.isnan(res.sum_photons) else "",
                "photons_per_s": f"{photons_per_s:.6f}" if not math.isnan(photons_per_s) else "",
                "black_level_used": f"{res.black_est:.6f}",
                "sat_pixels": res.sat_pixels,
                "sat_fraction": f"{sat_frac:.8f}",
                "roi_x": rx,
                "roi_y": ry,
                "roi_w": rw,
                "roi_h": rh,
                "width": params.width,
                "height": params.height,
                "bit_depth": params.bit_depth,
                "pack": params.pack,
            })

            print(f"[OK] {idx:04d} {p.name}: DN={res.sum_dn:.3e}, e-={res.sum_e:.3e}, photons={res.sum_photons:.3e}")

    print(f"[done] Wrote: {out_csv_path}")


if __name__ == "__main__":
    main()

