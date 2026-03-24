# Image Feature Detection and Matching with OpenCV

This project is a practical study of local image features using Python and OpenCV. It covers the full pipeline from basic convolutions and gradient analysis to Harris corner detection, ORB and KAZE feature extraction, descriptor matching, and quantitative evaluation under controlled geometric transformations.

The repository was developed as part of an ENSTA Paris image recognition assignment focused on understanding how local features are detected, described, matched, and evaluated in realistic image-matching scenarios.

## What was done

The project explores four main topics:

1. Image filtering and convolutions
   Direct pixel-wise convolution was compared against OpenCV's optimized `filter2D`, and the effect of a sharpening kernel was analyzed. The project also computes image gradients and discusses correct visualization of signed responses.

2. Harris corner detection
   A Harris detector was implemented and benchmarked, including local maxima extraction with morphological dilation and parameter sweeps to study the effect of window size, Harris coefficient, and neighborhood suppression.

3. ORB and KAZE feature detection
   The project compares two widely used local feature methods:
   `ORB`, a fast binary approach based on oriented FAST keypoints and rotated BRIEF descriptors.
   `KAZE`, a nonlinear scale-space method that is generally more robust but computationally heavier.

4. Descriptor matching and evaluation
   Matching was studied with three strategies:
   brute-force cross-check,
   Lowe ratio test,
   FLANN-based nearest-neighbor search.
   The quality of matches was also evaluated quantitatively by applying known geometric transforms and measuring reprojection consistency.

## Repository structure

- `Convolutions.py`
  Entry point for convolution experiments and gradient analysis.

- `Harris.py`
  Harris corner detector implementation, benchmarking, and parameter studies.

- `Features_Detect.py`
  Detects and visualizes ORB or KAZE keypoints on a test image pair.

- `Features_Match_CrossCheck.py`
  Matches ORB or KAZE descriptors using brute-force cross-checking.

- `Features_Match_RatioTest.py`
  Matches descriptors using `k`-NN plus Lowe's ratio test.

- `Features_Match_FLANN.py`
  Matches descriptors with FLANN, using LSH for ORB and KD-tree for KAZE.

- `pair_eval.py`
  Core quantitative evaluation utility for checking matches against a known affine transform or homography.

- `graph generators/`
  Scripts that generate precision plots under scale, rotation, and viewpoint changes.

- `image_formats_and_convolutions/`
  Support code for convolution analysis, visualization, and timing plots.

- `Image_Pairs/`
  Test images used throughout the experiments.

- `results/`
  Generated outputs such as Harris visualizations and keypoint figures.

- `docs/rappport/`
  The LaTeX report and the compiled PDF summarizing the methodology and results.

## Main results and observations

- OpenCV's optimized convolution is much faster than the naive direct implementation.
- The sharpening kernel acts as a contrast-enhancing filter related to an unsharp-mask or Laplacian-style boost.
- Harris corner detection is sensitive to the summation window, suppression neighborhood, and Harris coefficient.
- ORB is significantly faster than KAZE.
- KAZE is typically more computationally expensive but tends to produce stronger robustness under stronger geometric changes.
- ORB and KAZE require different matching distances:
  ORB uses Hamming distance for binary descriptors,
  KAZE uses Euclidean (`L2`) distance for floating-point descriptors.
- Ratio test and FLANN-based matching improve match filtering compared with raw brute-force matching.
- Quantitative evaluation under synthetic transforms makes it possible to compare precision under scale, rotation, and viewpoint changes in a controlled way.

## Installation

Use Python 3 and install the dependencies listed in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main dependencies:

- `opencv-python`
- `numpy`
- `matplotlib`

## How to run

Run the scripts from the repository root.

### Convolutions and gradient analysis

```bash
python3 Convolutions.py
```

### Harris corner detector

Single execution:

```bash
python3 Harris.py
```

Benchmark mode:

```bash
python3 Harris.py -stats 50
python3 Harris.py -stats 50 -plots
```

### ORB or KAZE keypoint detection

```bash
python3 Features_Detect.py orb
python3 Features_Detect.py kaze
python3 Features_Detect.py orb -stats
python3 Features_Detect.py kaze -stats 100
```

### Descriptor matching

Cross-check:

```bash
python3 Features_Match_CrossCheck.py orb
python3 Features_Match_CrossCheck.py kaze
```

Ratio test:

```bash
python3 Features_Match_RatioTest.py orb
python3 Features_Match_RatioTest.py kaze
```

FLANN:

```bash
python3 Features_Match_FLANN.py orb
python3 Features_Match_FLANN.py kaze
```

### Quantitative evaluation plots

```bash
python3 "graph generators/scale_precision_eval.py"
python3 "graph generators/rotation_precision_eval.py"
python3 "graph generators/viewpoint_precision_eval.py"
```

## Outputs

Generated figures are saved in places such as:

- `results/`
- `results/method_features/`
- `docs/rappport/imgs/descriptors/`
- `docs/rappport/imgs/convolutions/`

The final written report is available here:

- `docs/rappport/rap.tex`
- `docs/rappport/rap.pdf`

## Notes

- Most scripts assume the provided image pairs remain in the repository structure used here.
- Some plotting code uses a non-interactive Matplotlib backend so figures can be saved even in headless environments.
- The project is intended as an educational and experimental codebase rather than a packaged library.
