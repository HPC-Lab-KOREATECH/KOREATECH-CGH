"""Hologram generation - release reproduction script.

A self-contained CLI for generating complex-amplitude holograms from RGB+D
inputs with four layer-based methods:

    edge-padded-AP-LBM  : Proposed AP-LBM with edge padding
    SM-LBM              : Standard masked layer-based method (baseline)
    ADV-LBM             : Amplitude-compensated bidirectional masking
    AP-LBM              : Proposed AP-LBM with constant (zero) padding

Output is one .npy file per input image, containing a complex64 array of
shape (3, H, W) ordered as (R, G, B) channels.

Example
-------
    python generate_hologram.py \
        --input  ./images  --depth ./depths \
        --output ./holograms \
        --gen_method AP-LBM --smooth_phase \
        --width 1024 --height 1024 --pp 3.6e-6 \
        --min_z 0.01 --max_z 0.05 --depth_level 256
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List

import cupy as cp
import cupyx.scipy.fft as cufft
import numpy as np
import scipy.fft
import skimage.transform
from tqdm import tqdm

scipy.fft.set_global_backend(cufft)


# ---------------------------------------------------------------------------
# Hologram parameter container
# ---------------------------------------------------------------------------
class HoloParam:
    def __init__(self, nx: int, ny: int, wavelength: List[float], dx: float, dy: float):
        self.nx_ = nx
        self.ny_ = ny
        self.wavelength_ = wavelength
        self.dx_ = dx
        self.dy_ = dy

    def print(self) -> None:
        print(f"  resolution : {self.nx_} x {self.ny_}")
        print(f"  pixel pitch: {self.dx_} m")
        for i, w in enumerate(self.wavelength_):
            print(f"  wavelength[{i}]: {w} m")


# ---------------------------------------------------------------------------
# Angular spectrum method (FP64 transfer function, FP32 storage)
# ---------------------------------------------------------------------------
_TRANSFER_KERNEL = cp.RawKernel(r'''
#include <cuComplex.h>
#define M_PI 3.14159265358979323846
__device__ __host__ __forceinline__ cuComplex      operator*(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
__device__ __host__ __forceinline__ cuDoubleComplex operator*(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }

__device__ __forceinline__ cuDoubleComplex my_cexp(double imag) {
    cuDoubleComplex res;
    sincos(imag, &res.y, &res.x);
    return res;
}

extern "C" __global__
void multiplyTransferFunctionD(cuComplex* src, double z,
                               int width, int height,
                               int half_width, int half_height,
                               double wavelength, double dfx, double dfy) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx >= width || idy >= height) return;

    double fx = ((double)idx - half_width) * dfx;
    double fy = ((double)idy - half_height) * dfy;

    double transfer = 1.0 - (wavelength*fx)*(wavelength*fx) - (wavelength*fy)*(wavelength*fy);
    if (transfer >= 0) {
        cuDoubleComplex v = make_cuDoubleComplex(src[idy*width+idx].x, src[idy*width+idx].y)
                            * my_cexp(2.0 * M_PI * z * sqrt(transfer) / wavelength);
        src[idy*width+idx] = make_cuComplex(v.x, v.y);
    } else {
        src[idy*width+idx] = make_cuComplex(0.0f, 0.0f);
    }
}
''', 'multiplyTransferFunctionD')


def angularSpectrumMethod(src, width, height, pixel_pitch, wavelength,
                          distance, padding_method='constant'):
    """ASM propagation. `src` shape (C, H, W) complex64. Returns same shape."""
    blocks = (16, 16)
    fft_width = width * 2
    fft_height = height * 2
    dst = cp.zeros(src.shape, dtype=cp.complex64)

    for index in range(src.shape[0]):
        if padding_method == 'constant':
            padded = cp.pad(src[index],
                            ((height // 2, height // 2), (width // 2, width // 2)),
                            'constant', constant_values=(0, 0))
        elif padding_method == 'none':
            padded = src[index]
            fft_width = width
            fft_height = height
        else:
            padded = cp.pad(src[index],
                            ((height // 2, height // 2), (width // 2, width // 2)),
                            padding_method)

        grid = (int(np.ceil(fft_width / 16)), int(np.ceil(fft_height / 16)))

        padded = cp.fft.fftshift(padded, (-1, -2))
        padded = scipy.fft.fft2(padded)
        padded = cp.fft.fftshift(padded, (-1, -2))

        _TRANSFER_KERNEL(grid, blocks, (
            padded,
            cp.float64(distance),
            cp.int32(padded.shape[1]), cp.int32(padded.shape[0]),
            cp.int32(padded.shape[1] // 2), cp.int32(padded.shape[0] // 2),
            cp.float64(wavelength[index]),
            cp.float64(1 / (fft_width * pixel_pitch)),
            cp.float64(1 / (fft_height * pixel_pitch)),
        ))

        padded = cp.fft.fftshift(padded, (-1, -2))
        padded = scipy.fft.ifft2(padded)
        padded = cp.fft.fftshift(padded, (-1, -2))

        if padding_method != 'none':
            dy = (padded.shape[-2] - dst.shape[-2]) // 2
            dx = (padded.shape[-1] - dst.shape[-1]) // 2
            dst[index] = padded[dy:padded.shape[-2] - dy, dx:padded.shape[-1] - dx]
        else:
            dst[index] = padded
    return dst


# ---------------------------------------------------------------------------
# Depth normalization (shared by all four methods)
# ---------------------------------------------------------------------------
def _prepare_arrays(image, depth, array_min, array_max):
    """Reorder image to (C,H,W); convert depth to inverse-normalized form."""
    if image.shape[-3] != 3:
        image = image.transpose((2, 0, 1))
    if image.dtype == cp.int8:
        image = image * (1 / 255)
    if image.dtype != cp.float32:
        image = image.astype(cp.float32)

    original_depth_level = 1.0
    if depth.dtype == np.uint16:
        original_depth_level = 65536
    elif depth.dtype == np.uint8:
        original_depth_level = 256
    else:
        if array_max is None:
            array_max = float(cp.max(depth))
        if array_min is None:
            array_min = float(cp.min(depth))
        depth = cp.where(depth == 0.0, array_max, depth)
        depth = original_depth_level - (depth - array_min) / (array_max - array_min)
    return image, depth, original_depth_level


def _initial_phase(image_plane, phase_method, image_shape, wavelength, depth_value):
    """Apply initial phase to a layer plane."""
    if phase_method == 'random phase':
        phase_map = cp.random.rand(*image_shape, dtype=cp.float32) * cp.pi * 2
        return image_plane * cp.exp(phase_map * 1j)
    if phase_method == 'smooth phase':
        wl = cp.asarray(wavelength)
        return image_plane * cp.expand_dims(
            cp.exp(-depth_value * cp.pi * 2 / wl * 1j), axis=(1, 2)
        ).astype(cp.complex64)
    return image_plane.astype(cp.complex64)  # 'no init'


# ---------------------------------------------------------------------------
# SM-LBM: standard masked layer-based method (baseline)
# ---------------------------------------------------------------------------
def hologram_SM_LBM(image, depth, param, phase_method,
                    min_depth, max_depth, post_z, depth_level,
                    array_min=0.0, array_max=1.0):
    image, depth, orig_lvl = _prepare_arrays(image, depth, array_min, array_max)
    width, height = image.shape[-1], image.shape[-2]
    depth_range = max_depth - min_depth
    depth_step = orig_lvl / depth_level
    depth = cp.expand_dims(depth, 0)

    image_plane = image * cp.logical_and(depth >= 0, depth < depth_step)
    next_plane = _initial_phase(image_plane, phase_method, image.shape,
                                param.wavelength_, depth_range)

    if depth_level > 1:
        prop_distance = depth_range / (depth_level - 1)
        for cur in tqdm(range(1, depth_level), total=depth_level - 1):
            cur_phys = ((depth_level - 1) - cur) / (depth_level - 1) * depth_range + min_depth
            mask = cp.logical_and(depth > cur * depth_step, depth <= (cur + 1) * depth_step)
            plane = image * mask
            plane = _initial_phase(plane, phase_method, image.shape,
                                   param.wavelength_, cur_phys)
            propagated = angularSpectrumMethod(
                next_plane, param.nx_, param.ny_, param.dx_, param.wavelength_,
                +prop_distance, padding_method='constant')
            next_plane = plane + propagated * (depth <= cur * depth_step)

    if min_depth != 0:
        next_plane = angularSpectrumMethod(next_plane, width, height,
                                           param.dx_, param.wavelength_, +min_depth)
    if post_z != 0:
        next_plane = angularSpectrumMethod(next_plane,
                                           next_plane.shape[-1], next_plane.shape[-2],
                                           param.dx_, param.wavelength_, post_z)
    return next_plane


# ---------------------------------------------------------------------------
# ADV-LBM: amplitude compensation + bidirectional masking
# ---------------------------------------------------------------------------
def hologram_ADV_LBM(image, depth, param, phase_method,
                    min_depth, max_depth, post_z, depth_level,
                    array_min=0.0, array_max=1.0):
    image, depth, orig_lvl = _prepare_arrays(image, depth, array_min, array_max)
    width, height = image.shape[-1], image.shape[-2]
    depth_range = max_depth - min_depth
    depth_step = orig_lvl / depth_level
    depth = cp.expand_dims(depth, 0)

    image_plane = image * cp.logical_and(depth >= 0, depth < depth_step * 2)
    next_plane = _initial_phase(image_plane, phase_method, image.shape,
                                param.wavelength_, max_depth)

    if depth_level > 1:
        prop_distance = depth_range / (depth_level - 1)
        for cur in tqdm(range(1, depth_level), total=depth_level - 1):
            cur_phys = ((depth_level - 1) - cur) / (depth_level - 1) * depth_range + min_depth
            mask_one = cp.logical_and(depth > cur * depth_step, depth <= (cur + 1) * depth_step)
            plane = image * mask_one
            plane = _initial_phase(plane, phase_method, image.shape,
                                   param.wavelength_, cur_phys)
            mask_two = cp.logical_and(depth > cur * depth_step, depth <= (cur + 2) * depth_step)
            propagated = angularSpectrumMethod(
                next_plane, width, height, param.dx_, param.wavelength_,
                +prop_distance, padding_method='constant')
            next_plane = plane + propagated * cp.logical_not(mask_two)

    if min_depth != 0:
        next_plane = angularSpectrumMethod(next_plane, width, height,
                                           param.dx_, param.wavelength_, +min_depth)
    if post_z != 0:
        next_plane = angularSpectrumMethod(next_plane,
                                           next_plane.shape[-1], next_plane.shape[-2],
                                           param.dx_, param.wavelength_, post_z)
    return next_plane


# ---------------------------------------------------------------------------
# AP-LBM: proposed amplitude-compensated amplitude projection LBM
# (padding='edge'    -> edge-padded-AP-LBM
#  padding='constant'-> AP-LBM)
# ---------------------------------------------------------------------------
def hologram_AP_LBM(image, depth, param, phase_method,
                    min_depth, max_depth, post_z, depth_level,
                    padding='constant', array_min=0.0, array_max=1.0):
    image, depth, orig_lvl = _prepare_arrays(image, depth, array_min, array_max)
    width, height = image.shape[-1], image.shape[-2]
    depth_range = max_depth - min_depth
    depth_step = orig_lvl / depth_level
    depth = cp.expand_dims(depth, 0)

    # ---- forward pass: accumulate planes from far to near ----
    image_plane = image * cp.logical_and(depth >= 0, depth < depth_step * 2)
    next_plane = _initial_phase(image_plane, phase_method, image.shape,
                                param.wavelength_, max_depth)

    if depth_level > 1:
        prop_distance = depth_range / (depth_level - 1)
        for cur in tqdm(range(1, depth_level), total=depth_level - 1):
            cur_phys = ((depth_level - 1) - cur) / (depth_level - 1) * depth_range + min_depth
            mask_two = cp.logical_and(depth > cur * depth_step, depth <= (cur + 2) * depth_step)
            plane = image * mask_two
            mask_one = cp.logical_and(depth > cur * depth_step, depth <= (cur + 1) * depth_step)
            if phase_method == 'random phase':
                pmap = mask_one.astype(cp.float32)
                pmap = (cp.random.rand(*pmap.shape, dtype=cp.float32) * cp.pi * 2) * pmap
                plane = plane * cp.exp(pmap * 1j)
            elif phase_method == 'smooth phase':
                wl = cp.asarray(param.wavelength_)
                plane = plane * cp.expand_dims(
                    cp.exp(-cur_phys * cp.pi * 2 / wl * 1j), axis=(1, 2)
                ).astype(cp.complex64)
            else:
                plane = plane.astype(cp.complex64)
            propagated = angularSpectrumMethod(
                next_plane, width, height, param.dx_, param.wavelength_,
                +prop_distance, padding_method=padding)
            next_plane = plane + propagated * cp.logical_not(mask_two)

    # ---- back-propagate to hologram plane ----
    next_plane = angularSpectrumMethod(next_plane, width, height,
                                       param.dx_, param.wavelength_,
                                       -max_depth, padding_method=padding)

    # ---- backward pass: amplitude replacement, near to far in reconstruction ----
    if depth_level > 1:
        cur_mask = cp.logical_and(depth >= 0, depth < depth_step * 1)
        amp = cp.abs(next_plane)
        cmask = cp.broadcast_to(cur_mask, amp.shape)
        amp[cmask] = image[cmask]
        ang = cp.angle(next_plane)
        next_plane = amp * cp.exp(1j * ang)

        prop_distance = depth_range / (depth_level - 1)
        for cur in tqdm(range(1, depth_level), total=depth_level - 1):
            cur_mask = cp.logical_and(depth > cur * depth_step, depth <= (cur + 1) * depth_step)
            next_plane = angularSpectrumMethod(next_plane, width, height,
                                               param.dx_, param.wavelength_,
                                               +prop_distance, padding_method=padding)
            amp = cp.abs(next_plane)
            cmask = cp.broadcast_to(cur_mask, amp.shape)
            amp[cmask] = image[cmask]
            ang = cp.angle(next_plane)
            next_plane = amp * cp.exp(1j * ang)

    if min_depth != 0:
        next_plane = angularSpectrumMethod(next_plane, width, height,
                                           param.dx_, param.wavelength_, +min_depth)
    if post_z != 0:
        next_plane = angularSpectrumMethod(next_plane,
                                           next_plane.shape[-1], next_plane.shape[-2],
                                           param.dx_, param.wavelength_, post_z)
    return next_plane


METHODS = {
    'edge-padded-AP-LBM': lambda *a, **kw: hologram_AP_LBM(*a, padding='edge', **kw),
    'SM-LBM':             hologram_SM_LBM,
    'ADV-LBM':            hologram_ADV_LBM,
    'AP-LBM':             lambda *a, **kw: hologram_AP_LBM(*a, padding='constant', **kw),
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
SUPPORTED_EXT = ('.png', '.jpg', '.jpeg', '.webp', '.exr')


def load_rgb(path, target_hw):
    """Load an RGB image, resize to target (h,w), return float32 [0,1] HxWx3."""
    ext = Path(path).suffix.lower()
    if ext == '.exr':
        import openexr_numpy as exrio
        img = exrio.imread(str(path))
    else:
        import imageio.v3 as iio
        img = iio.imread(str(path))
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if (img.shape[0], img.shape[1]) != target_hw:
        img = skimage.transform.resize(img, (*target_hw, 3), order=3,
                                       preserve_range=True, anti_aliasing=True)
    return img.astype(np.float32)


def load_depth(path, target_hw):
    """Load a single-channel depth map, resize to target (h,w)."""
    ext = Path(path).suffix.lower()
    if ext == '.exr':
        import openexr_numpy as exrio
        d = exrio.imread(str(path))
    else:
        import imageio.v3 as iio
        d = iio.imread(str(path))
    if d.ndim == 3:
        d = d[..., 0]
    if d.shape != target_hw:
        d = skimage.transform.resize(d, target_hw, order=0,
                                     anti_aliasing=False, preserve_range=True, clip=False)
    return d


# ---------------------------------------------------------------------------
# Per-image driver
# ---------------------------------------------------------------------------
def process_one(image_path, depth_path, output_dir, param, args, phase_init):
    target_hw = (param.ny_, param.nx_)
    img = load_rgb(image_path, target_hw)
    img = img ** args.gamma
    depth = load_depth(depth_path, target_hw)

    min_z, max_z = args.min_z, args.max_z
    if args.auto_depth and Path(image_path).suffix.lower() == '.exr':
        # use EXR depth range, scaled to physical extent of the hologram
        original_size = depth.shape[0]
        min_z = float(np.min(depth)) / original_size * param.nx_ * param.dx_
        max_z = float(np.max(depth)) / original_size * param.nx_ * param.dx_

    img_cp = cp.asarray(img)
    depth_cp = cp.asarray(depth)

    fn = METHODS[args.gen_method]
    holo = fn(img_cp, depth_cp, param, phase_init,
              min_z, max_z, args.post_z, args.depth_level)
    out_name = Path(image_path).stem + '.npy'
    np.save(os.path.join(output_dir, out_name),
            cp.asnumpy(holo).astype(np.complex64))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Layer-based hologram generation (release)")
    parser.add_argument('--input', required=True,
                        help="Input image file or directory")
    parser.add_argument('--depth', required=True,
                        help="Depth file or directory (filenames must match --input)")
    parser.add_argument('--output', default='./holograms',
                        help="Output directory for .npy holograms")
    parser.add_argument('--gen_method', required=True,
                        choices=list(METHODS.keys()),
                        help="AP-LBM | SM-LBM | ADV-LBM | edge-padded-AP-LBM")

    parser.add_argument('--width',  type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--pp',     type=float, default=3.6e-6,
                        help="Hologram pixel pitch [m]")
    parser.add_argument('--wavelength', type=float, nargs=3,
                        default=[638e-9, 532e-9, 450e-9],
                        help="Wavelengths of (R, G, B) channels [m]")

    parser.add_argument('--min_z', type=float, default=0.01,
                        help="Nearest object distance from hologram plane [m]")
    parser.add_argument('--max_z', type=float, default=0.05,
                        help="Farthest object distance from hologram plane [m]")
    parser.add_argument('--post_z', type=float, default=0.0,
                        help="Extra propagation after hologram generation [m]")
    parser.add_argument('--depth_level', type=int, default=256,
                        help="Number of depth layers")
    parser.add_argument('--gamma', type=float, default=1.0,
                        help="Gamma correction applied to RGB input")
    parser.add_argument('--auto_depth', action='store_true',
                        help="(EXR only) derive min_z/max_z from depth values")
    parser.add_argument('--gpu_id', type=int, default=0)

    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument('--random_phase',  action='store_true')
    phase.add_argument('--no_phase_init', action='store_true')
    phase.add_argument('--smooth_phase', action='store_true')

    args = parser.parse_args()

    cp.cuda.runtime.setDevice(args.gpu_id)
    param = HoloParam(args.width, args.height, args.wavelength, args.pp, args.pp)
    print("Hologram parameters:")
    param.print()

    phase_init = ('random phase' if args.random_phase
                  else 'smooth phase' if args.smooth_phase
                  else 'no init')
    print(f"Method: {args.gen_method}  |  Phase init: {phase_init}")

    os.makedirs(args.output, exist_ok=True)

    # Resolve input/depth into matching pairs
    in_path, dp_path = Path(args.input), Path(args.depth)
    if in_path.is_file():
        if not dp_path.is_file():
            sys.exit("ERROR: --input is a file but --depth is not")
        pairs = [(in_path, dp_path)]
    else:
        imgs = sorted([p for p in in_path.iterdir() if p.suffix.lower() in SUPPORTED_EXT])
        depths = {p.stem: p for p in dp_path.iterdir() if p.suffix.lower() in SUPPORTED_EXT}
        pairs = []
        for p in imgs:
            if p.stem not in depths:
                print(f"  [skip] no depth match for {p.name}")
                continue
            pairs.append((p, depths[p.stem]))
        print(f"{len(pairs)} (image, depth) pairs found")

    for img_path, dp_path in tqdm(pairs, total=len(pairs)):
        process_one(str(img_path), str(dp_path), args.output, param, args, phase_init)

    print(f"Done. Holograms written to: {args.output}")


if __name__ == '__main__':
    main()
