# OmniVision CMOS Image Analysis

A Python toolkit for processing and quantitatively analyzing **OmniVision CMOS image-sensor data**, with a focus on high-resolution RAW Bayer images and low-light / chemiluminescence imaging.

The project was developed primarily for **OmniVision CMOS sensor/chip RAW data**, including high-resolution formats such as 200 MP and 50 MP sensor captures. The tools support RAW decoding, Bayer processing, image conversion, region-of-interest analysis, dark correction, and calibrated photon-count estimation.

Although originally designed around OmniVision sensor data, many components are configurable for other CMOS sensors when the image dimensions, bit depth, packing format, and Bayer pattern are known.

## Overview

Raw image output from OmniVision CMOS sensors may be stored as headerless binary data rather than a standard image format. These files generally do not contain metadata describing image dimensions, bit depth, Bayer pattern, or pixel packing, so the sensor format must be known or inferred before the images can be analyzed.

Quantitative low-light measurements additionally require processing such as dark subtraction, background correction, camera-gain calibration, quantum-efficiency correction, and region-of-interest integration.

This repository provides tools for:

* Processing OmniVision CMOS sensor RAW data
* Decoding headerless CMOS RAW files
* Handling Bayer sensor mosaics
* Unpacking MIPI RAW10 and RAW12 data
* Converting RAW sensor data to viewable TIFF images
* Inspecting individual pixels and selecting regions interactively
* Integrating low-light signals over an ROI
* Dark-frame and black-level correction
* Converting digital numbers to electrons and estimated photons
* Processing both RAW and TIFF image sequences
* Exporting quantitative measurements to CSV

The overall workflow is:

```text
OmniVision CMOS Sensor
        │
        ▼
   RAW Sensor Data
        │
        ├──────────── Image Processing ────────────┐
        │                                          │
        ▼                                          ▼
RAW decoding                               Quantitative Analysis
        │                                          │
        ├── RAW8                                   ├── Dark subtraction
        ├── RAW10                                  ├── ROI integration
        ├── RAW12                                  ├── Saturation detection
        └── 16-bit container                       ├── DN → electrons
        │                                          └── electrons → photons
        ▼
Bayer processing
        │
        ├── Demosaicing
        ├── White balance
        ├── Black-level correction
        └── Bayer-aware binning
        │
        ▼
      TIFF
        │
        ├────────────────┐
        ▼                ▼
ROI inspection     TIFF photon analysis
```

---

## Repository Structure

```text
CMOS-Image-Analysis/
│
├── raw2tiff.py
│   Decode OmniVision Bayer RAW sensor dumps and convert them to TIFF
│
├── raw_photon_count.py
│   Integrate RAW low-light frames and estimate electrons / photons
│
├── tif_photon_count.py
│   Quantify signal and photon counts from TIFF sequences
│
├── interactive_roi_viewer.py
│   Interactive ROI selection, pixel inspection, and line profiles
│
└── README.md
```

---

# 1. OmniVision RAW to TIFF Conversion

`raw2tiff.py` converts headerless Bayer RAW sensor dumps into standard TIFF images.

The converter was originally developed for high-resolution **OmniVision CMOS image sensor data**.

Known formats used during development include:

```text
200 MP OmniVision:
16384 × 12288 pixels

50 MP:
8192 × 6144 pixels
```

The script can also be configured manually for other sensor dimensions and bit depths.

## Supported RAW Formats

The converter supports:

* Unpacked RAW8
* RAW data stored in a 16-bit container
* MIPI RAW10
* MIPI RAW12

For known OmniVision capture formats, the program can attempt to infer sensor dimensions from file size.

For other formats, parameters can be supplied explicitly.

---

## MIPI RAW Unpacking

OmniVision sensors commonly use MIPI-compatible RAW pixel formats.

The converter includes explicit decoding for:

```text
RAW10
4 pixels → 5 bytes
```

and:

```text
RAW12
2 pixels → 3 bytes
```

The decoded sensor values are represented internally as 16-bit numerical arrays for subsequent processing.

---

## Bayer Pattern Support

OmniVision color image sensors use a Bayer color-filter array.

The following layouts are supported:

```text
RGGB
BGGR
GRBG
GBRG
```

The Bayer pattern can be specified explicitly:

```bash
--bayer RGGB
```

or estimated automatically:

```bash
--bayer auto
```

For quantitative or production analysis, using the Bayer layout specified for the particular OmniVision sensor is recommended.

---

## Bayer-Aware Binning

High-resolution OmniVision sensors can generate extremely large image files.

For example:

```text
16384 × 12288
≈ 201 million pixels
```

Processing those images at full resolution can require substantial memory and disk space.

The converter therefore supports spatial downsampling directly in the Bayer domain while preserving the CFA structure.

Default:

```bash
--bin 4
```

For full-resolution processing:

```bash
--bin 1
```

---

## Image Processing Pipeline

```text
OmniVision RAW bytes
        │
        ▼
Decode / unpack
        │
        ▼
Bayer mosaic
        │
        ├── optional black-level subtraction
        │
        ├── optional Bayer-domain binning
        │
        ▼
Demosaicing
        │
        ▼
White balance
        │
        ▼
Intensity normalization
        │
        ▼
Gamma correction
        │
        ▼
8-bit or 16-bit RGB TIFF
```

Automatic gray-world white balance is available:

```bash
--wb auto
```

or gains can be supplied manually:

```bash
--wb 2.0,1.0,1.7
```

---

## RAW-to-TIFF Examples

Convert an OmniVision RAW image:

```bash
python raw2tiff.py \
    --input frame.raw \
    --out_dir output
```

Convert a directory of sensor captures:

```bash
python raw2tiff.py \
    --input "captures/*.raw" \
    --out_dir output
```

Process at full resolution:

```bash
python raw2tiff.py \
    --input frame.raw \
    --out_dir output \
    --bin 1
```

Explicitly define a 50 MP sensor image:

```bash
python raw2tiff.py \
    --input frame.raw \
    --out_dir output \
    --width 8192 \
    --height 6144 \
    --bit_depth 12 \
    --pack none
```

Decode MIPI RAW10:

```bash
python raw2tiff.py \
    --input frame.raw \
    --out_dir output \
    --width 16384 \
    --height 12288 \
    --bit_depth 10 \
    --pack mipi10
```

Generate a 16-bit TIFF:

```bash
python raw2tiff.py \
    --input frame.raw \
    --out_dir output \
    --out_depth 16
```

---

# 2. RAW Photon Counting

`raw_photon_count.py` performs quantitative analysis directly on OmniVision CMOS RAW image sequences.

It is intended for low-light applications such as:

* Chemiluminescence imaging
* Low-light microscopy
* Sensor characterization
* Luminescence assays
* Photon-response experiments
* Exposure optimization
* CMOS sensitivity evaluation

Unlike the TIFF converter, this script works directly with sensor digital values.

---

## Quantitative Signal Model

Each RAW frame is corrected using either a dark image or an estimated black level.

```text
Corrected DN
    =
Raw DN
- Dark / Black Level
```

Negative corrected values are clipped to zero.

The signal is integrated over a selected region:

```text
Integrated DN
    =
Σ Corrected DN
```

If the OmniVision sensor's conversion gain is known:

```text
Electrons
    =
Integrated DN × electrons_per_DN
```

If quantum efficiency is known:

```text
Photons at Sensor
    =
Electrons / Quantum Efficiency
```

If exposure time is supplied:

```text
Photon Rate
    =
Photons / Exposure Time
```

---

## ROI Integration

Analysis can use either the entire CMOS frame or a defined ROI.

Example:

```bash
--roi 1000 800 500 500
```

This represents:

```text
x      = 1000
y      = 800
width  = 500
height = 500
```

ROI analysis is especially useful for localized chemiluminescence or optical signals.

---

## Dark-Frame Correction

CMOS sensors contain readout offset, dark current, fixed-pattern noise, and other background components.

A dark image can therefore be supplied:

```bash
--dark dark.raw
```

The dark image is subtracted pixel-by-pixel.

Multiple dark images can also be combined:

```bash
--dark "dark/*.raw"
```

The program calculates a mean **master dark frame** and can cache it for subsequent analyses.

This is particularly useful for very large OmniVision sensor images where repeatedly calculating the master dark would be computationally expensive.

---

## Automatic Black-Level Estimation

If a dark image is unavailable, a fixed sensor black level can be supplied:

```bash
--black_level 64
```

or estimated independently for every frame:

```bash
--auto_black_percentile 1
```

For quantitative imaging, dark frames acquired using the same sensor settings and exposure conditions are generally preferable.

---

## Quantum Efficiency and Bayer CFA

If a single effective quantum efficiency is known:

```bash
--qe 0.80
```

it can be applied to the integrated signal.

The program also supports separate quantum efficiencies for individual Bayer channels:

```bash
--qe_r 0.60 \
--qe_g 0.82 \
--qe_b 0.55 \
--bayer RGGB
```

When separate quantum efficiencies are supplied, the RAW Bayer planes are integrated independently before photon conversion.

This can be useful when characterizing OmniVision sensor response at different wavelengths.

---

## Memory-Efficient Processing

Large OmniVision CMOS frames may contain hundreds of millions of pixels.

To avoid unnecessary memory consumption, image integration is performed in blocks:

```bash
--block_rows 256
```

Unpacked RAW files can also be accessed using NumPy memory mapping rather than loading the entire file into RAM.

---

## Saturation Detection

The script tracks pixels reaching the maximum value associated with the sensor bit depth.

Output fields include:

```text
sat_pixels
sat_fraction
```

This is useful for determining whether an exposure exceeds the useful dynamic range of the sensor.

---

## RAW Photon Analysis Example

Relative CMOS signal measurement:

```bash
python raw_photon_count.py \
    --input "sequence/*.raw" \
    --out_csv results.csv \
    --width 8192 \
    --height 6144 \
    --bit_depth 12 \
    --pack none
```

With dark correction:

```bash
python raw_photon_count.py \
    --input "sequence/*.raw" \
    --dark dark.raw \
    --out_csv results.csv \
    --width 8192 \
    --height 6144 \
    --bit_depth 12
```

Analyze an ROI:

```bash
python raw_photon_count.py \
    --input "sequence/*.raw" \
    --out_csv results.csv \
    --width 8192 \
    --height 6144 \
    --bit_depth 12 \
    --roi 3000 2000 500 500
```

Estimate photons using sensor calibration:

```bash
python raw_photon_count.py \
    --input "sequence/*.raw" \
    --dark "dark/*.raw" \
    --out_csv photon_counts.csv \
    --width 8192 \
    --height 6144 \
    --bit_depth 12 \
    --e_per_dn 0.8 \
    --qe 0.75 \
    --exposure 10
```

---

## RAW Analysis Output

The output CSV includes:

```text
frame_idx
filename
sum_dn_corrected
sum_electrons
sum_photons_est
photons_per_s
black_level_used
sat_pixels
sat_fraction
roi_x
roi_y
roi_w
roi_h
width
height
bit_depth
pack
```

The resulting data can be analyzed using Python, R, Excel, or other scientific-analysis tools.

---

# 3. TIFF Photon Counting

`tif_photon_count.py` performs quantitative analysis on TIFF images derived from CMOS data or other scientific imaging systems.

It supports:

* Individual TIFF files
* TIFF directories
* Filename globs
* Multi-page TIFFs
* Grayscale images
* RGB/RGBA images
* Multi-channel image arrays

---

## Channel Selection

Available modes include:

```text
auto
r
g
b
gray
```

Example:

```bash
--channel g
```

In automatic mode, the program selects the channel with the highest mean intensity.

---

## Background Estimation

Available background models include:

### Global Percentile

```bash
--bg-mode percentile
```

### Border Pixels

```bash
--bg-mode border
```

### Separate Background ROI

```bash
--bg-mode roi \
--bg-roi 100 100 200 200
```

### No Additional Background Correction

```bash
--bg-mode none
```

---

## TIFF Signal Model

```text
Corrected ADU
    =
Raw ADU
- Dark Frame
- Camera Offset
```

Then:

```text
Net ADU
    =
max(Corrected ADU - Background, 0)
```

Integrated signal:

```text
Sum ADU
    =
Σ Net ADU
```

Conversion to electrons:

```text
Photoelectrons
    =
Sum ADU ×
(gain_e_per_adu / EM_gain)
```

Photon estimate:

```text
Photons Sensor
    =
Photoelectrons / QE
```

If optical collection efficiency is known:

```text
Photons Emitted
    =
Photons Sensor / System Efficiency
```

---

## TIFF Photon Analysis Example

```bash
python tif_photon_count.py \
    --input "images/*.tif" \
    --out photons_by_frame.csv
```

With ROI:

```bash
python tif_photon_count.py \
    --input "images/*.tif" \
    --roi 500 400 300 300 \
    --out photons_by_frame.csv
```

With calibration:

```bash
python tif_photon_count.py \
    --input "images/*.tif" \
    --dark-frame dark.tif \
    --gain-e-per-adu 0.8 \
    --qe 0.75 \
    --system-efficiency 0.15 \
    --exposure-s 10 \
    --out photons_by_frame.csv
```

---

# 4. Interactive ROI Viewer

`interactive_roi_viewer.py` provides interactive inspection of large CMOS images.

This is useful for identifying signal regions in high-resolution OmniVision sensor images before quantitative integration.

Instead of manually determining sensor coordinates, the user clicks directly on a downscaled preview.

The corresponding coordinates in the original full-resolution image are calculated automatically.

---

## Supported Image Types

### Headerless CMOS RAW

```text
.raw
```

Supported data types include:

```text
uint8
uint16
uint32
int16
float32
```

### Camera RAW

When `rawpy` is installed:

```text
DNG
NEF
CR2
CR3
ARW
RW2
ORF
RAF
PEF
```

### Standard Images

Other image formats can be loaded using `imageio`.

---

## Interactive Selection

Example for an OmniVision RAW frame:

```bash
python interactive_roi_viewer.py frame.raw \
    --raw-width 8192 \
    --raw-height 6144 \
    --raw-dtype uint16 \
    --roi-w 10 \
    --roi-h 10 \
    --pad 2
```

The selected location is reported as:

```text
Selected ROI:
x = 4251
y = 3178
w = 10
h = 10
```

Those coordinates can then be used directly with:

```bash
python raw_photon_count.py \
    --input "sequence/*.raw" \
    --width 8192 \
    --height 6144 \
    --bit_depth 12 \
    --roi 4251 3178 10 10 \
    --out_csv roi_signal.csv
```

---

## ROI Outputs

The interactive tool generates:

```text
roi_values.csv
roi_context_values.csv
roi_visual.png
roi_with_context_visual.png
horizontal_profile.png
vertical_profile.png
```

These outputs provide:

* Individual sensor pixel values
* ROI coordinates
* Neighboring-pixel context
* Horizontal signal profiles
* Vertical signal profiles

They are useful for examining:

* Point-source signals
* Hot pixels
* Sensor uniformity
* Signal localization
* Local background
* Optical spot profiles

---

# Suggested OmniVision Analysis Workflow

## Step 1 — Inspect the RAW Sensor Image

```bash
python interactive_roi_viewer.py frame.raw \
    --raw-width 8192 \
    --raw-height 6144 \
    --raw-dtype uint16
```

Select the relevant signal region.

## Step 2 — Convert RAW Data to TIFF

```bash
python raw2tiff.py \
    --input frame.raw \
    --out_dir converted \
    --width 8192 \
    --height 6144 \
    --bit_depth 12
```

## Step 3 — Quantify the RAW Sequence

```bash
python raw_photon_count.py \
    --input "experiment/*.raw" \
    --dark "dark/*.raw" \
    --width 8192 \
    --height 6144 \
    --bit_depth 12 \
    --roi X Y W H \
    --e_per_dn CAMERA_GAIN \
    --qe SENSOR_QE \
    --exposure EXPOSURE_SECONDS \
    --out_csv results.csv
```

## Step 4 — Analyze the Time Series

The CSV output can be used to evaluate:

* Signal intensity over time
* Photon flux
* Signal decay
* Replicate measurements
* Exposure dependence
* Sensor saturation
* Dark-current effects
* Signal-to-background behavior

---

# Installation

Clone the repository:

```bash
git clone https://github.com/vincent625/CMOS-Image-Analysis.git
cd CMOS-Image-Analysis
```

Install core dependencies:

```bash
pip install numpy opencv-python tifffile pillow matplotlib imageio
```

Optional dependencies:

```bash
pip install rawpy imagecodecs
```

---

# Important OmniVision Calibration Notes

Absolute photon quantification requires calibration parameters specific to the **OmniVision sensor and acquisition configuration**.

Important parameters include:

```text
Electrons per DN
Quantum efficiency
Black level
Sensor bit depth
Dark current
Exposure time
Analog / digital gain
Optical system efficiency
Wavelength
Sensor temperature
```

These values can depend strongly on:

* OmniVision sensor model
* Sensor operating mode
* Analog gain
* Digital gain
* Exposure settings
* Wavelength
* Temperature
* Optical system

Therefore:

```text
Raw Digital Counts ≠ Absolute Photon Counts
```

unless appropriate sensor calibration parameters are known.

The software supports applying these calibration parameters but does not determine them automatically.

For uncalibrated experiments:

```text
sum_dn_corrected
```

is generally the most appropriate relative signal measurement.

---

# RAW Format Considerations

Headerless OmniVision RAW dumps may not contain metadata describing:

* Width
* Height
* Bit depth
* MIPI packing
* Bayer pattern
* Row stride
* Header offset
* Byte order

The software includes best-effort inference, but automatic RAW interpretation can be ambiguous.

For quantitative sensor analysis, explicitly specifying known sensor parameters is recommended.

Example:

```bash
--width 8192 \
--height 6144 \
--bit_depth 12 \
--pack none \
--bayer RGGB
```

These parameters should be matched to the specific **OmniVision chip configuration and acquisition mode**.

---

# Technologies

The project uses:

* Python
* NumPy
* OpenCV
* tifffile
* Pillow
* Matplotlib
* imageio
* rawpy
* Memory-mapped arrays
* MIPI RAW decoding
* Bayer CFA processing

---

# Skills Demonstrated

This project demonstrates experience with:

* OmniVision CMOS image sensors
* Scientific image processing
* CMOS sensor characterization
* RAW sensor-data decoding
* Bayer color-filter arrays
* MIPI RAW10 / RAW12 formats
* High-resolution image processing
* Memory-efficient numerical computing
* Region-of-interest analysis
* Sensor calibration
* Dark-frame correction
* Background estimation
* Photon-count estimation
* Low-light imaging
* Chemiluminescence imaging
* Scientific visualization
* Command-line application development

---

## Author

**Yan Zhu**

Python · OmniVision CMOS Sensors · Scientific Imaging · Quantitative Image Analysis
