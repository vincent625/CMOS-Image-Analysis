#!/usr/bin/env python3
"""
raw2tiff_ov200mp.py

Convert "headerless" sensor Bayer .raw dumps into viewable TIFFs.

Originally tuned for OmniVision 200MP dumps that are exactly:
    16384 x 12288 pixels, 8-bit Bayer (file size 201,326,592 bytes)

Updated to also auto-infer a common 50MP format:
    8192 x 6144 pixels, 16-bit container (file size 100,663,296 bytes = 96 MiB)

…but the script is configurable for other sizes/bit-depths and can also
unpack MIPI RAW10/RAW12 packed formats.

Dependencies:
  pip install numpy opencv-python tifffile

Examples:

1) Convert one file (default assumes OV 200MP raw8 and outputs a *binned* TIFF):
  python raw2tiff_ov200mp.py --input frame0.raw --out_dir out

2) Convert all .raw in a folder:
  python raw2tiff_ov200mp.py --input "captures/*.raw" --out_dir out

3) Full-resolution output (HUGE files; needs lots of disk):
  python raw2tiff_ov200mp.py --input frame0.raw --out_dir out --bin 1

4) Try a different Bayer pattern:
  python raw2tiff_ov200mp.py --input frame0.raw --out_dir out --bayer BGGR

5) If your dump is MIPI RAW10 packed:
  python raw2tiff_ov200mp.py --input frame0.raw --out_dir out --width 16384 --height 12288 --bit_depth 10 --pack mipi10

Notes:
- Because .raw has no metadata, you MUST know (or guess) width/height, bit depth, packing, and Bayer order.
- Default --bin=4 makes the output ~12.5MP from 200MP, which is much easier to view/process.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import cv2
import tifffile


BAYER2CV_RGB = {
    "RGGB": cv2.COLOR_BayerRG2RGB,
    "BGGR": cv2.COLOR_BayerBG2RGB,
    "GRBG": cv2.COLOR_BayerGR2RGB,
    "GBRG": cv2.COLOR_BayerGB2RGB,
}


def infer_shape_from_size(byte_size: int) -> Optional[Tuple[int, int, int, str]]:
    """
    Try to infer common headerless Bayer dump formats from file size.
    Returns (width, height, bit_depth, pack) or None.

    Known presets:
      - 200MP: 16384 x 12288
      - 50MP :  8192 x  6144  (96 MiB if 16-bit container)
    """
    candidates = [
        (16384, 12288),  # ~200MP
        (8192, 6144),    # ~50MP
    ]

    for w, h in candidates:
        px = w * h

        # Unpacked RAW8
        if byte_size == px:
            return (w, h, 8, "none")

        # Unpacked 16-bit container (often RAW10/12 stored in uint16)
        if byte_size == px * 2:
            # Default to 12-bit effective if user doesn't override later
            return (w, h, 12, "none")

        # RAW10 packed: 4 pixels -> 5 bytes => bytes = (px/4)*5
        if byte_size == (px // 4) * 5:
            return (w, h, 10, "mipi10")

        # RAW12 packed: 2 pixels -> 3 bytes => bytes = (px/2)*3
        if byte_size == (px // 2) * 3:
            return (w, h, 12, "mipi12")

    return None


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


def read_raw_as_mosaic(
    path: Path,
    width: Optional[int],
    height: Optional[int],
    bit_depth: Optional[int],
    pack: str,
) -> Tuple[np.ndarray, int, int, int]:
    """
    Read raw file and return (mosaic, width, height, bit_depth_effective).

    - If width/height/bit_depth not provided, attempts to infer for common sizes (200MP/50MP).
    - Returns mosaic as uint8 for <=8-bit, else uint16.
    """
    size = path.stat().st_size

    if width is None or height is None or bit_depth is None:
        inferred = infer_shape_from_size(size)
        if inferred is not None:
            iw, ih, ib, ipack = inferred
            width = width or iw
            height = height or ih
            bit_depth = bit_depth or ib
            if pack == "auto":
                pack = ipack

    if width is None or height is None or bit_depth is None:
        raise ValueError(
            "Could not infer width/height/bit_depth from file size. "
            "Please pass --width, --height, and --bit_depth."
        )

    pack = pack.lower()
    if pack == "auto":
        if size == width * height:
            pack = "none"
            bit_depth = 8
        elif size == width * height * 2:
            pack = "none"
            # If user didn't specify bit depth, assume typical 12-bit in uint16 container.
            # If they did specify (10/12/16), keep it.
            bit_depth = 12 if bit_depth is None else bit_depth
        elif size == (width * height // 4) * 5:
            pack = "mipi10"
            bit_depth = 10
        elif size == (width * height // 2) * 3:
            pack = "mipi12"
            bit_depth = 12
        else:
            raise ValueError("pack=auto could not infer packing from file size; specify --pack.")

    if pack == "none":
        if bit_depth <= 8:
            expected = width * height
            if size != expected:
                raise ValueError(f"Size mismatch: got {size} bytes, expected {expected} for RAW8.")
            mosaic = np.memmap(path, dtype=np.uint8, mode="r", shape=(height, width))
            return mosaic, width, height, 8
        else:
            expected = width * height * 2
            if size != expected:
                raise ValueError(f"Size mismatch: got {size} bytes, expected {expected} for RAW16 container.")
            mosaic = np.memmap(path, dtype=np.uint16, mode="r", shape=(height, width))
            return mosaic, width, height, int(bit_depth)

    buf = np.fromfile(path, dtype=np.uint8)

    if pack == "mipi10":
        mosaic16 = unpack_mipi_raw10(buf, width, height)
        return mosaic16, width, height, 10
    if pack == "mipi12":
        mosaic16 = unpack_mipi_raw12(buf, width, height)
        return mosaic16, width, height, 12

    raise ValueError(f"Unknown --pack '{pack}'. Use: none, mipi10, mipi12, auto")


def bin_lattice(lattice: np.ndarray, factor: int) -> np.ndarray:
    """Average a 2D lattice by 'factor' in each dimension."""
    if factor == 1:
        return np.array(lattice)

    h, w = lattice.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    lattice = lattice[:h2, :w2]

    acc_dtype = np.uint32 if lattice.dtype == np.uint16 else np.uint16
    reshaped = lattice.reshape(h2 // factor, factor, w2 // factor, factor).astype(acc_dtype)
    summed = reshaped.sum(axis=(1, 3))
    denom = factor * factor
    avg = (summed + denom // 2) // denom
    return avg.astype(lattice.dtype)


def downscale_bayer_mosaic(mosaic: np.ndarray, factor: int) -> np.ndarray:
    """
    Downscale a Bayer mosaic by 'factor' while keeping the same 2x2 Bayer layout.
    Output shape will be (H//factor, W//factor).
    """
    if factor == 1:
        return np.array(mosaic)
    if factor < 1:
        raise ValueError("bin factor must be >= 1")

    latt00 = mosaic[0::2, 0::2]
    latt01 = mosaic[0::2, 1::2]
    latt10 = mosaic[1::2, 0::2]
    latt11 = mosaic[1::2, 1::2]

    s00 = bin_lattice(latt00, factor)
    s01 = bin_lattice(latt01, factor)
    s10 = bin_lattice(latt10, factor)
    s11 = bin_lattice(latt11, factor)

    out_h = s00.shape[0] * 2
    out_w = s00.shape[1] * 2
    out = np.empty((out_h, out_w), dtype=mosaic.dtype)
    out[0::2, 0::2] = s00
    out[0::2, 1::2] = s01
    out[1::2, 0::2] = s10
    out[1::2, 1::2] = s11
    return out


def auto_bayer_guess(mosaic: np.ndarray) -> str:
    """
    Guess Bayer pattern using a simple artifact metric on a central ROI.
    If ambiguous, defaults to RGGB.
    """
    h, w = mosaic.shape
    roi_size = min(1024, h, w)
    y0 = (h - roi_size) // 2
    x0 = (w - roi_size) // 2
    roi = np.array(mosaic[y0:y0 + roi_size, x0:x0 + roi_size])

    def score(pattern: str) -> float:
        rgb = cv2.cvtColor(roi, BAYER2CV_RGB[pattern]).astype(np.float32)
        means = rgb.reshape(-1, 3).mean(axis=0) + 1e-6
        gain = means.mean() / means
        rgb *= gain
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        rg = r - g
        bg = b - g
        lap_rg = cv2.Laplacian(rg, cv2.CV_32F, ksize=3)
        lap_bg = cv2.Laplacian(bg, cv2.CV_32F, ksize=3)
        return float(np.mean(lap_rg ** 2) + np.mean(lap_bg ** 2))

    scores = {p: score(p) for p in ("RGGB", "BGGR", "GRBG", "GBRG")}
    best = min(scores, key=scores.get)
    return best


def apply_white_balance(rgb: np.ndarray, wb: str) -> np.ndarray:
    """
    wb:
      - "auto": gray-world
      - "r,g,b": comma-separated gains
    """
    if wb.lower() == "auto":
        rgb_f = rgb.astype(np.float32)
        means = rgb_f.reshape(-1, 3).mean(axis=0) + 1e-6
        gain = means.mean() / means
        rgb_f *= gain
        return rgb_f

    parts = wb.split(",")
    if len(parts) != 3:
        raise ValueError("wb must be 'auto' or 'r,g,b' (e.g. 2.0,1.0,1.6)")
    g = np.array([float(p) for p in parts], dtype=np.float32)
    return rgb.astype(np.float32) * g[None, None, :]


def normalize_and_encode(
    rgb_f: np.ndarray,
    out_depth: int,
    gamma: float,
    black: Optional[float] = None,
    white: Optional[float] = None,
    clip_percent: float = 0.1,
) -> np.ndarray:
    """
    Percentile normalize to [0,1], optional gamma, then quantize to 8/16-bit.
    """
    x = rgb_f
    if black is None or white is None:
        lo = np.percentile(x, clip_percent)
        hi = np.percentile(x, 100.0 - clip_percent)
        black = lo if black is None else black
        white = hi if white is None else white

    x = (x - float(black)) / (float(white) - float(black) + 1e-6)
    x = np.clip(x, 0.0, 1.0)

    if gamma and gamma != 1.0:
        x = np.power(x, 1.0 / float(gamma))

    if out_depth == 8:
        return (x * 255.0 + 0.5).astype(np.uint8)
    if out_depth == 16:
        return (x * 65535.0 + 0.5).astype(np.uint16)
    raise ValueError("out_depth must be 8 or 16")


def convert_one(
    in_path: Path,
    out_path: Path,
    *,
    width: Optional[int],
    height: Optional[int],
    bit_depth: Optional[int],
    pack: str,
    bayer: str,
    bin_factor: int,
    wb: str,
    out_depth: int,
    gamma: float,
    black_level: int,
    compression: str,
) -> None:
    mosaic, _, _, _ = read_raw_as_mosaic(in_path, width, height, bit_depth, pack)

    if bayer.lower() == "auto":
        bayer_used = auto_bayer_guess(mosaic)
    else:
        bayer_used = bayer.upper()
        if bayer_used not in BAYER2CV_RGB:
            raise ValueError(f"Unknown bayer '{bayer}'. Choose RGGB/BGGR/GRBG/GBRG/auto")

    if bin_factor > 1:
        mosaic_proc = downscale_bayer_mosaic(mosaic, bin_factor)
    else:
        mosaic_proc = np.array(mosaic)

    if black_level:
        if mosaic_proc.dtype == np.uint16:
            mosaic_proc = np.clip(mosaic_proc.astype(np.int32) - int(black_level), 0, None).astype(np.uint16)
        else:
            mosaic_proc = np.clip(mosaic_proc.astype(np.int32) - int(black_level), 0, 255).astype(np.uint8)

    rgb = cv2.cvtColor(mosaic_proc, BAYER2CV_RGB[bayer_used])
    rgb_f = apply_white_balance(rgb, wb)
    out_img = normalize_and_encode(rgb_f, out_depth=out_depth, gamma=gamma)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    comp = None if compression.lower() in ("none", "off", "false", "0") else compression.lower()

    tifffile.imwrite(
        out_path,
        out_img,
        photometric="rgb",
        compression=comp,
        bigtiff=True,
        metadata=None,
    )
    print(f"[OK] {in_path.name} -> {out_path.name} (bayer={bayer_used}, bin={bin_factor}, out_depth={out_depth})")


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
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if p.exists():
            uniq.append(p)
    return sorted(uniq)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Bayer .raw dumps to TIFF (batch).")
    ap.add_argument("--input", nargs="+", required=True, help="Input .raw file(s) or glob(s) like 'dir/*.raw'")
    ap.add_argument("--out_dir", required=True, help="Output directory for TIFFs")

    ap.add_argument("--width", type=int, default=None, help="Image width in pixels (if omitted, tries to infer)")
    ap.add_argument("--height", type=int, default=None, help="Image height in pixels (if omitted, tries to infer)")
    ap.add_argument("--bit_depth", type=int, default=None, help="Sensor bit depth: 8/10/12/16 (if omitted, tries to infer)")
    ap.add_argument("--pack", type=str, default="auto", help="Packing: none, mipi10, mipi12, auto (default auto)")

    ap.add_argument("--bayer", type=str, default="RGGB", help="Bayer order: RGGB/BGGR/GRBG/GBRG/auto (default RGGB)")
    ap.add_argument("--bin", type=int, default=4, help="Downscale factor in Bayer domain (default 4 -> ~12.5MP from 200MP)")
    ap.add_argument("--wb", type=str, default="auto", help="White balance: auto or 'r,g,b' (e.g. 2.0,1.0,1.7)")
    ap.add_argument("--out_depth", type=int, default=8, choices=[8, 16], help="Output bit depth per channel (8 or 16)")
    ap.add_argument("--gamma", type=float, default=2.2, help="Gamma for viewable output (2.2 typical). Use 1.0 for linear.")
    ap.add_argument("--black_level", type=int, default=0, help="Subtract black level before demosaic (default 0).")
    ap.add_argument("--compression", type=str, default="none", help="TIFF compression: none/deflate")

    args = ap.parse_args()

    in_paths = expand_inputs(args.input)
    if not in_paths:
        raise SystemExit("No input files found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for in_path in in_paths:
        out_path = out_dir / (in_path.stem + ".tif")
        convert_one(
            in_path,
            out_path,
            width=args.width,
            height=args.height,
            bit_depth=args.bit_depth,
            pack=args.pack,
            bayer=args.bayer,
            bin_factor=args.bin,
            wb=args.wb,
            out_depth=args.out_depth,
            gamma=args.gamma,
            black_level=args.black_level,
            compression=args.compression,
        )


if __name__ == "__main__":
    main()

