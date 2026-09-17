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
