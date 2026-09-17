#!/usr/bin/env python3
"""
Interactive ROI picker for large raw/scientific images.

Supports:
- custom binary .raw files via numpy.fromfile
- camera RAW formats via rawpy
- regular images via imageio

Example:
python interactive_roi_viewer.py A2_E1_S1.raw \
    --roi-w 10 --roi-h 10 --pad 2 \
    --raw-width 8192 --raw-height 6144 --raw-dtype uint16
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

RAW_EXTS = {
    ".dng", ".nef", ".cr2", ".cr3", ".arw", ".rw2", ".orf", ".raf", ".pef"
}


def load_image(
    path: str,
    grayscale: bool = True,
    use_raw_visible: bool = False,
    raw_width: int = None,
    raw_height: int = None,
    raw_dtype: str = "uint16",
):
    ext = Path(path).suffix.lower()

    # Case 1: custom scientific binary RAW
    if ext == ".raw":
        if raw_width is None or raw_height is None:
            raise ValueError(
                "For .raw files, you must provide --raw-width and --raw-height."
            )

        dtype_map = {
            "uint8": np.uint8,
            "uint16": np.uint16,
            "uint32": np.uint32,
            "int16": np.int16,
            "float32": np.float32,
        }
        if raw_dtype not in dtype_map:
            raise ValueError(
                f"Unsupported --raw-dtype {raw_dtype}. "
                f"Use one of: {', '.join(dtype_map.keys())}"
            )

        dtype = dtype_map[raw_dtype]
        data = np.fromfile(path, dtype=dtype)

        expected = raw_width * raw_height
        if data.size != expected:
            raise ValueError(
                f"RAW size mismatch.\n"
                f"Expected pixels: {expected} = {raw_width} x {raw_height}\n"
                f"Read values:     {data.size}\n"
                f"This usually means width/height/dtype are wrong."
            )

        img = data.reshape((raw_height, raw_width)).astype(np.float32)
        return img

    # Case 2: standard camera RAW
    if ext in RAW_EXTS:
        try:
            import rawpy
        except ImportError as e:
            raise ImportError("Install rawpy for camera RAW files: pip install rawpy") from e

        with rawpy.imread(path) as raw:
            if use_raw_visible:
                img = raw.raw_image_visible.copy().astype(np.float32)
            else:
                img = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=16
                ).astype(np.float32)

        if grayscale and img.ndim == 3:
            img = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
        return img

    # Case 3: regular images
    try:
        import imageio.v3 as iio
    except ImportError as e:
        raise ImportError("Install imageio for normal image formats: pip install imageio") from e

    img = iio.imread(path).astype(np.float32)

    if grayscale and img.ndim == 3:
        img = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]

    return img


def make_preview(img: np.ndarray, max_size: int = 1600):
    h, w = img.shape[:2]

    arr = img
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max <= arr_min:
        arr_max = arr_min + 1.0

    arr_norm = (arr - arr_min) / (arr_max - arr_min)
    arr_uint8 = (arr_norm * 255).clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(arr_uint8, mode="L")

    scale = min(max_size / w, max_size / h, 1.0)
    preview_w = max(1, int(round(w * scale)))
    preview_h = max(1, int(round(h * scale)))

    pil_preview = pil_img.resize((preview_w, preview_h), Image.Resampling.BILINEAR)
    preview = np.array(pil_preview)

    scale_x = w / preview_w
    scale_y = h / preview_h
    return preview, scale_x, scale_y


def clamp_roi(x0, y0, roi_w, roi_h, img_w, img_h):
    x0 = max(0, min(x0, img_w - roi_w))
    y0 = max(0, min(y0, img_h - roi_h))
    return x0, y0


def extract_regions(img: np.ndarray, x0: int, y0: int, roi_w: int, roi_h: int, pad: int):
    img_h, img_w = img.shape[:2]

    x1 = x0 + roi_w
    y1 = y0 + roi_h

    roi = img[y0:y1, x0:x1]

    cx0 = max(0, x0 - pad)
    cy0 = max(0, y0 - pad)
    cx1 = min(img_w, x1 + pad)
    cy1 = min(img_h, y1 + pad)

    context = img[cy0:cy1, cx0:cx1]
    return roi, context, (cx0, cy0, cx1, cy1)


def save_csv(arr: np.ndarray, path: str):
    np.savetxt(path, arr, delimiter=",", fmt="%.6f")


def draw_pixel_grid_image(arr: np.ndarray, save_path: str, title: str, highlight_roi=None):
    h, w = arr.shape
    fig_w = max(6, 0.9 * w)
    fig_h = max(6, 0.9 * h)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax == vmin:
        vmax = vmin + 1.0

    ax.imshow(arr, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=1)

    ax.set_xticks(np.arange(w))
    ax.set_yticks(np.arange(h))
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    ax.set_title(title)

    threshold = (vmin + vmax) / 2.0
    for yy in range(h):
        for xx in range(w):
            val = arr[yy, xx]
            txt_color = "white" if val < threshold else "black"
            ax.text(xx, yy, f"{val:.0f}", ha="center", va="center", fontsize=8, color=txt_color)

    if highlight_roi is not None:
        from matplotlib.patches import Rectangle
        rx, ry, rw, rh = highlight_roi
        rect = Rectangle((rx - 0.5, ry - 0.5), rw, rh, fill=False, linewidth=2.5, edgecolor="red")
        ax.add_patch(rect)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_line_profile(values, indices, title, xlabel, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(indices, values, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


class InteractiveROIPicker:
    def __init__(self, preview_img, scale_x, scale_y, full_img, roi_w, roi_h, pad, outdir):
        self.preview_img = preview_img
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.full_img = full_img
        self.roi_w = roi_w
        self.roi_h = roi_h
        self.pad = pad
        self.outdir = outdir

        self.fig = None
        self.ax = None
        self.rect = None

    def onclick(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        preview_x = int(round(event.xdata))
        preview_y = int(round(event.ydata))

        full_x_center = int(round(preview_x * self.scale_x))
        full_y_center = int(round(preview_y * self.scale_y))

        x0 = full_x_center - self.roi_w // 2
        y0 = full_y_center - self.roi_h // 2

        img_h, img_w = self.full_img.shape[:2]
        x0, y0 = clamp_roi(x0, y0, self.roi_w, self.roi_h, img_w, img_h)

        print(f"Selected ROI: x={x0}, y={y0}, w={self.roi_w}, h={self.roi_h}")
        self.process_roi(x0, y0)

        preview_rect_x = x0 / self.scale_x
        preview_rect_y = y0 / self.scale_y
        preview_rect_w = self.roi_w / self.scale_x
        preview_rect_h = self.roi_h / self.scale_y

        if self.rect is not None:
            self.rect.remove()

        from matplotlib.patches import Rectangle
        self.rect = Rectangle(
            (preview_rect_x, preview_rect_y),
            preview_rect_w,
            preview_rect_h,
            fill=False,
            linewidth=2,
            edgecolor="red"
        )
        self.ax.add_patch(self.rect)
        self.fig.canvas.draw_idle()

    def process_roi(self, x0, y0):
        roi, context, (cx0, cy0, cx1, cy1) = extract_regions(
            self.full_img, x0, y0, self.roi_w, self.roi_h, self.pad
        )

        os.makedirs(self.outdir, exist_ok=True)

        roi_csv = os.path.join(self.outdir, "roi_values.csv")
        context_csv = os.path.join(self.outdir, "roi_context_values.csv")
        roi_png = os.path.join(self.outdir, "roi_visual.png")
        context_png = os.path.join(self.outdir, "roi_with_context_visual.png")
        horiz_png = os.path.join(self.outdir, "horizontal_profile.png")
        vert_png = os.path.join(self.outdir, "vertical_profile.png")

        save_csv(roi, roi_csv)
        save_csv(context, context_csv)

        draw_pixel_grid_image(
            roi,
            roi_png,
            title=f"ROI only: x={x0}, y={y0}, w={self.roi_w}, h={self.roi_h}"
        )

        roi_in_context_x = x0 - cx0
        roi_in_context_y = y0 - cy0

        draw_pixel_grid_image(
            context,
            context_png,
            title=f"ROI + adjacent pixels (pad={self.pad})",
            highlight_roi=(roi_in_context_x, roi_in_context_y, self.roi_w, self.roi_h)
        )

        center_row = self.roi_h // 2
        center_col = self.roi_w // 2

        horiz_values = roi[center_row, :]
        vert_values = roi[:, center_col]

        horiz_indices = np.arange(x0, x0 + self.roi_w)
        vert_indices = np.arange(y0, y0 + self.roi_h)

        plot_line_profile(
            horiz_values,
            horiz_indices,
            title=f"Horizontal line profile at y={y0 + center_row}",
            xlabel="X pixel coordinate",
            ylabel="Signal",
            save_path=horiz_png
        )

        plot_line_profile(
            vert_values,
            vert_indices,
            title=f"Vertical line profile at x={x0 + center_col}",
            xlabel="Y pixel coordinate",
            ylabel="Signal",
            save_path=vert_png
        )

        print("Saved outputs:")
        print(f"  {roi_csv}")
        print(f"  {context_csv}")
        print(f"  {roi_png}")
        print(f"  {context_png}")
        print(f"  {horiz_png}")
        print(f"  {vert_png}")

    def show(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.ax.imshow(self.preview_img, cmap="gray")
        self.ax.set_title(
            "Click to select ROI center\n"
            f"ROI size = {self.roi_w} x {self.roi_h}, pad = {self.pad}"
        )
        self.ax.set_xlabel("Preview X")
        self.ax.set_ylabel("Preview Y")
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive ROI picker with line profiles.")
    parser.add_argument("image", help="Path to RAW or regular image")
    parser.add_argument("--roi-w", type=int, default=10, help="ROI width")
    parser.add_argument("--roi-h", type=int, default=10, help="ROI height")
    parser.add_argument("--pad", type=int, default=1, help="Adjacent border around ROI")
    parser.add_argument("--outdir", default="interactive_roi_output", help="Output directory")
    parser.add_argument("--preview-max", type=int, default=1600, help="Max preview size")
    parser.add_argument("--use-raw-visible", action="store_true", help="Use raw sensor plane for camera RAW")
    parser.add_argument("--raw-width", type=int, default=None, help="Width for custom .raw")
    parser.add_argument("--raw-height", type=int, default=None, help="Height for custom .raw")
    parser.add_argument(
        "--raw-dtype",
        type=str,
        default="uint16",
        choices=["uint8", "uint16", "uint32", "int16", "float32"],
        help="Data type for custom .raw"
    )

    args = parser.parse_args()

    full_img = load_image(
        args.image,
        grayscale=True,
        use_raw_visible=args.use_raw_visible,
        raw_width=args.raw_width,
        raw_height=args.raw_height,
        raw_dtype=args.raw_dtype,
    )

    if full_img.ndim != 2:
        raise RuntimeError("Expected grayscale 2D image after loading.")

    print(f"Loaded image shape: {full_img.shape}")
    print(f"Min={full_img.min()}, Max={full_img.max()}, Mean={full_img.mean():.2f}")

    preview_img, scale_x, scale_y = make_preview(full_img, max_size=args.preview_max)
    print(f"Preview shape: {preview_img.shape}, scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")

    picker = InteractiveROIPicker(
        preview_img=preview_img,
        scale_x=scale_x,
        scale_y=scale_y,
        full_img=full_img,
        roi_w=args.roi_w,
        roi_h=args.roi_h,
        pad=args.pad,
        outdir=args.outdir,
    )
    picker.show()


if __name__ == "__main__":
    main()
