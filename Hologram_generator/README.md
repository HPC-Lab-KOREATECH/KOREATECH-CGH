# Hologram generation - pre-release (work in progress)

> **Status:** in progress, on the way to release. The scripts run end-to-end
> and reproduce the published methods, but interfaces, defaults, and outputs
> may still change before the final release. Treat this as a preview.

Self-contained reproduction of the four layer-based hologram generation
methods reported in our paper. Two scripts:

- **Research note:** the hologram generator is based on the
 research paper "A Large-Depth-Range Layer-Based Hologram
  Dataset Generation for Machine Learning-Based 3D Computer-Generated
  Holography".

- [generate_hologram.py](generate_hologram.py) — RGB+D → complex hologram (`.npy`)
- [dpm.py](dpm.py) — complex hologram (`.npy`) → phase-only image (`.png`) via the double-phase method (DPM)

| Flag                  | Method                                                          |
| --------------------- | --------------------------------------------------------------- |
| `edge-padded-AP-LBM`  | Proposed AP-LBM with edge padding                               |
| `SM-LBM`              | Standard masked layer-based method (baseline)                   |
| `ADV-LBM`             | Amplitude-compensated bidirectional masking                     |
| `AP-LBM`              | Proposed AP-LBM with constant (zero) padding                    |

## Requirements

- An NVIDIA GPU and a matching CUDA toolkit (11.x or 12.x)
- Python 3.10+

CuPy is **not** listed in [requirements.txt](requirements.txt) because the
correct wheel depends on your CUDA version. Install it manually:

```bash
# CUDA 12.x
pip install cupy-cuda12x

# CUDA 11.x
pip install cupy-cuda11x
```

Then install the rest:

```bash
pip install -r requirements.txt
```

Verify the GPU is visible to CuPy:

```bash
python -c "import cupy; cupy.show_config()"
```

Tested with `cupy-cuda12x==13.2.0` on CUDA 12.x.

## Inputs

- **RGB image**: `.png` / `.jpg` / `.webp` / `.exr`
- **Depth map**: same extensions; single-channel (or first channel used)
  - For `.png`/`.jpg`: pixel value `255` = closest, `0` = farthest
  - For `.exr`: float depth (smaller = closer); use `--auto_depth` to derive
    `min_z`/`max_z` from the EXR depth range

When a directory is given, an image and depth map are paired by **basename**
(e.g. `img/scene01.png` ↔ `depth/scene01.png`).

## Output

One `.npy` file per input, named `<input_stem>.npy`, containing a
`complex64` array of shape `(3, height, width)` ordered as `(R, G, B)`.

## Examples

Single image:

```bash
python generate_hologram.py \
    --input  img.png    --depth  depth.png \
    --output ./holograms \
    --gen_method AP-LBM --uniform_phase \
    --width 1024 --height 1024 --pp 3.6e-6 \
    --min_z 0.01 --max_z 0.05 --depth_level 256
```

Batch (directory of pairs):

```bash
python generate_hologram.py \
    --input  ./images   --depth  ./depths \
    --output ./holograms \
    --gen_method edge-padded-AP-LBM --uniform_phase \
    --width 1920 --height 1080 --pp 3.6e-6 \
    --min_z 0.0 --max_z 0.0226912 --depth_level 256
```

Run all four methods on the same scene for comparison:

```bash
for m in edge-padded-AP-LBM SM-LBM ADV-LBM AP-LBM; do
    python generate_hologram.py \
        --input img.png --depth depth.png \
        --output ./holograms_$m \
        --gen_method $m --uniform_phase \
        --width 1024 --height 1024 --pp 3.6e-6 \
        --min_z 0.01 --max_z 0.05 --depth_level 256
done
```

### Generation CLI options

```text
--input          input image file or directory
--depth          depth file or directory
--output         output directory for .npy holograms (default: ./holograms)
--gen_method    {edge-padded-AP-LBM, SM-LBM, ADV-LBM, AP-LBM}

--width          hologram width  in pixels  (default: 1024)
--height         hologram height in pixels  (default: 1024)
--pp             pixel pitch [m]            (default: 3.6e-6)
--wavelength R G B                          (default: 638e-9 532e-9 450e-9)

--min_z          nearest object distance [m]   (default: 0.01)
--max_z          farthest object distance [m]  (default: 0.05)
--post_z         extra propagation after generation [m] (default: 0)
--depth_level    number of depth layers        (default: 256)
--gamma          gamma applied to RGB input    (default: 1.0)
--auto_depth     (EXR only) derive min_z/max_z from depth values

--random_phase   use random initial phase
--no_phase_init  no initial phase (recommended starting point)
--uniform_phase  uniform phase per layer (recommended for our methods)

--gpu_id         CUDA device id (default: 0)
```

## Loading the output

```python
import numpy as np
holo = np.load('./holograms/scene01.npy')   # complex64, (3, H, W)
amp   = np.abs(holo)
phase = np.angle(holo)
```

## DPM encoding (phase-only output for SLM)

`dpm.py` converts a complex hologram into a phase-only PNG. It applies a
per-channel off-axis carrier (so R/G/B separate at the Fourier plane) and
encodes the residual amplitude as two interleaved phase samples.

Single hologram:

```bash
python dpm.py \
    --input  ./holograms/scene01.npy \
    --output ./dpm/scene01.png \
    --encoding Maimonev \
    --off_axis_x -1.1 --move_z -0.04
```

Whole directory:

```bash
python dpm.py \
    --input  ./holograms \
    --output ./dpm \
    --encoding Maimonev \
    --off_axis_x -1.1 --move_z -0.04
```

### DPM CLI options

```text
--input          input .npy hologram (file or directory)
--output         output PNG file or directory
--encoding      {Maimonev, Maimoneh, Anti-aliasing, phase}
                 Maimonev      : alternate clockwise/CCW phase row by row
                 Maimoneh      : alternate column by column
                 Anti-aliasing : 2x2 checkerboard interleave
                 phase         : drop amplitude (no DPM)

--off_axis_x     off-axis carrier angle (deg)  (default 0)
--off_axis_y     off-axis carrier angle (deg)  (default 0)
--move_z         ASM propagation [m] before encoding (default 0)

--pp             pixel pitch [m]               (default 3.6e-6)
--wavelength R G B                             (default 638e-9 532e-9 450e-9)
--max_pi         output phase range, units of pi (default 2 -> [0, 2pi])
--gaussian       sigma for phase smoothing      (default 1 = off)
--gamma          gamma before 8-bit quantize    (default 1.0)
--gpu_id         CUDA device id                 (default 0)
```

The output is an 8-bit PNG where pixel value `0..255` maps linearly to phase
`[0, max_pi * pi]`.
