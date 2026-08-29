# Dataset Information

## Status

The dataset is **not yet publicly available** in this repository. It is being prepared for release alongside the accompanying research paper.

## Dataset Description

| Property | Value |
|----------|-------|
| **Total samples** | 6,967 audio recordings |
| **Format** | `.wav` files |
| **Sample rate** | 22,050 Hz |
| **Classes** | 5 (see table below) |
| **Recording type** | Vehicle pass-by events |

## Class Distribution

| Class | Description | Samples | Percentage |
|:-----:|:------------|:-------:|:----------:|
| 0 | Far - Approaching (~50m -> 20m) | 1,156 | 16.59% |
| 1 | **DANGER** - Approaching (~20m -> 0m) | 1,567 | 22.49% |
| 2 | Near - Receding (0m -> ~20m) | 1,255 | 18.01% |
| 3 | Far - Receding (20m -> 50m) | 1,026 | 14.73% |
| 4 | Safe - No vehicle | 1,963 | 28.18% |

## File Naming Convention

Files are named using the pattern: `number_class.wav` or `number_class-extra.wav`

Examples:
- `12_3.wav` — file #12, class 3
- `8_1-4.wav` — file #8, class 1 (variant 4)

## Access

For research collaboration or early access to the dataset, please contact the authors or open an issue in this repository.

## Planned Release

Upon publication of the research paper, the dataset will be released under a **Creative Commons Attribution 4.0 (CC BY 4.0)** license via a dedicated data repository (e.g., Zenodo or Hugging Face Datasets).
