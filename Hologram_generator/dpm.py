"""Double-phase method (DPM) encoder.

Converts a complex-amplitude hologram (from `generate_hologram.py`) into a
phase-only hologram suitable for display on a phase-only SLM. Applies a
per-channel off-axis carrier so the three colour channels separate at the
Fourier plane, then encodes the residual amplitude as two interleaved phase
samples.

Output is an 8-bit PNG where pixel value 0..255 maps linearly to the
phase range [0, max_pi * pi].

Example
-------
    python dpm.py \
        --input  ./holograms/scene01.npy \
        --output ./dpm/scene01.png \
        --encoding Maimonev --off_axis_x -1.1 --move_z -0.04
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import cupy as cp
import cupyx.scipy.fft as cufft
import cupyx.scipy.ndimage
import imageio.v3 as iio
import numpy as np
import scipy.fft

from generate_hologram import angularSpectrumMethod

scipy.fft.set_global_backend(cufft)


def _shifted_fft(x):
    x = cp.fft.fftshift(x, (-1, -2))
    x = scipy.fft.fft2(x)
    return cp.fft.fftshift(x, (-1, -2))


def _shifted_ifft(x):
    x = cp.fft.fftshift(x, (-1, -2))
    x = scipy.fft.ifft2(x)
    return cp.fft.fftshift(x, (-1, -2))


def off_axis(hologram, wavelength, pp, off_axis_angle):
    """Apply per-channel off-axis carrier and angular-band filter."""
    off_x, off_y = off_axis_angle
    width = hologram.shape[-1]
    height = hologram.shape[-2]
    wavenumber = 2 * cp.pi / cp.array(wavelength)[:, None, None]

    x = cp.linspace(-width  * pp / 2, width  * pp / 2, width )[None, :]
    y = cp.linspace(-height * pp / 2, height * pp / 2, height)[:, None]

    if isinstance(off_x, Sequence) or isinstance(off_y, Sequence):
        ref_x = cp.deg2rad(cp.array(off_x))
        ref_y = cp.deg2rad(cp.array(off_y))
        ref = cp.exp(
            1j * wavenumber *
            cp.expand_dims(cp.sin(ref_x), (1, 2)) * x +
            cp.expand_dims(cp.sin(ref_y), (1, 2)) * cp.expand_dims(y, 0)
        )
    else:
        ref = cp.exp(
            1j * wavenumber *
            cp.expand_dims(cp.sin(np.deg2rad(off_x)) * x +
                           cp.sin(np.deg2rad(off_y)) * y, 0)
        )

    as_ref = _shifted_fft(hologram * cp.conjugate(ref))

    dfx, dfy = 1 / pp / width, 1 / pp / height
    fx = cp.linspace(-width  * dfx / 2, width  * dfx / 2, width )
    fy = cp.linspace(-height * dfy / 2, height * dfy / 2, height)
    Fx, Fy = cp.meshgrid(fx, fy)
    Fx, Fy = cp.expand_dims(Fx, 0), cp.expand_dims(Fy, 0)
    sin_x = cp.array(np.sin(np.deg2rad(off_x)) / np.array(wavelength))[:, None, None]
    sin_y = cp.array(np.sin(np.deg2rad(off_y)) / np.array(wavelength))[:, None, None]

    band = ((Fx >= (fx[0]  - sin_x)) & (Fy >= (fy[0]  - sin_y)) &
            (Fx <= (fx[-1] - sin_x)) & (Fy <= (fy[-1] - sin_y)))
    return _shifted_ifft(as_ref * band)


def double_phase_method(hologram, encoding='Maimonev', max_pi_multiplier=2.0,
                        gaussian_size=1, move_z=0.0, wavelength=None,
                        pp=3.6e-6, off_axis_angle=(0.0, 0.0), gamma=1.0):
    """Encode a complex hologram as a phase-only image.

    Parameters
    ----------
    hologram : ndarray, complex, shape (3, H, W) or (H, W, 3)
    encoding : 'Maimonev' | 'Maimoneh' | 'Anti-aliasing' | 'phase'
    max_pi_multiplier : float    output phase range [0, max_pi * pi]
    gaussian_size : int          Gaussian sigma applied to clockwise/ccw
                                 phases before interleaving (1 = off)
    move_z : float               extra ASM propagation [m] before encoding
    wavelength : list of 3 float wavelengths [m]
    pp : float                   pixel pitch [m]
    off_axis_angle : (x, y)      off-axis carrier angles in degrees
    gamma : float                gamma before 8-bit quantization

    Returns
    -------
    uint8 ndarray of shape (H, W, 3)
    """
    holo = cp.asarray(hologram, dtype=cp.complex64)
    if holo.shape[-1] == 3 and holo.ndim == 3:
        holo = cp.transpose(holo, (2, 0, 1))
    holo = cp.conj(holo)

    if move_z != 0:
        holo = angularSpectrumMethod(holo, holo.shape[-1], holo.shape[-2],
                                     pp, list(wavelength), move_z)

    holo = off_axis(holo, list(wavelength), pp, off_axis_angle)

    max_pi = max_pi_multiplier * cp.pi
    phase = cp.angle(holo)
    phase = phase - cp.average(phase)
    amp = cp.abs(holo)
    amp = amp / cp.max(amp)

    p_cw  = phase - cp.arccos(amp) + max_pi / 2
    p_ccw = phase + cp.arccos(amp) + max_pi / 2
    p_cw[p_cw  > max_pi] -= 2.0 * cp.pi
    p_ccw[p_ccw > max_pi] -= 2.0 * cp.pi
    p_cw[p_cw  < 0]      += 2.0 * cp.pi
    p_ccw[p_ccw < 0]     += 2.0 * cp.pi

    if gaussian_size > 1:
        sig = (1, gaussian_size, gaussian_size)
        p_cw  = cupyx.scipy.ndimage.gaussian_filter(p_cw,  sigma=sig)
        p_ccw = cupyx.scipy.ndimage.gaussian_filter(p_ccw, sigma=sig)

    result = cp.zeros(holo.shape, dtype=cp.float32)
    if encoding == 'Maimonev':                 # vertical alternation
        result[:, 0::2, :]  = p_cw [:, 0::2, :]
        result[:, 1::2, :]  = p_ccw[:, 1::2, :]
    elif encoding == 'Maimoneh':               # horizontal alternation
        result[:, :, 0::2]  = p_cw [:, :, 0::2]
        result[:, :, 1::2]  = p_ccw[:, :, 1::2]
    elif encoding == 'Anti-aliasing':          # checkerboard
        result[:, 0::2, 0::2] = p_cw [:, 0::2, 0::2]
        result[:, 1::2, 1::2] = p_cw [:, 1::2, 1::2]
        result[:, 1::2, 0::2] = p_ccw[:, 1::2, 0::2]
        result[:, 0::2, 1::2] = p_ccw[:, 0::2, 1::2]
    elif encoding == 'phase':
        result = phase
    else:
        raise ValueError(f"unknown encoding: {encoding}")

    img = ((result / max_pi) ** gamma * 255).astype(cp.uint8)
    img = cp.transpose(img, (1, 2, 0))
    return cp.asnumpy(img)


def main():
    parser = argparse.ArgumentParser(
        description="Double-phase encoding for complex holograms (release)")
    parser.add_argument('--input',  required=True,
                        help="Input .npy hologram or directory of .npy files")
    parser.add_argument('--output', default=None,
                        help="Output PNG file or directory "
                             "(default: alongside input as <stem>_dpm.png)")
    parser.add_argument('--encoding', default='Maimonev',
                        choices=['Maimonev', 'Maimoneh', 'Anti-aliasing', 'phase'],
                        help="Spatial multiplexing pattern of the two phase samples")

    parser.add_argument('--off_axis_x', type=float, default=0.0,
                        help="Off-axis carrier angle in degrees (x)")
    parser.add_argument('--off_axis_y', type=float, default=0.0,
                        help="Off-axis carrier angle in degrees (y)")
    parser.add_argument('--move_z',     type=float, default=0.0,
                        help="ASM propagation [m] applied before encoding")

    parser.add_argument('--pp',         type=float, default=3.6e-6,
                        help="Pixel pitch [m]")
    parser.add_argument('--wavelength', type=float, nargs=3,
                        default=[638e-9, 532e-9, 450e-9],
                        help="Wavelengths of (R, G, B) channels [m]")
    parser.add_argument('--max_pi',   type=float, default=2.0,
                        help="Output phase range, in units of pi (default 2 -> [0, 2pi])")
    parser.add_argument('--gaussian', type=int,   default=1,
                        help="Gaussian sigma applied to phases (1 = off)")
    parser.add_argument('--gamma',    type=float, default=1.0,
                        help="Gamma before 8-bit quantization")
    parser.add_argument('--gpu_id',   type=int,   default=0)
    args = parser.parse_args()

    cp.cuda.runtime.setDevice(args.gpu_id)

    in_path = Path(args.input)
    if in_path.is_file():
        items = [in_path]
        out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_dpm.png")
        out_is_dir = False
    else:
        items = sorted([p for p in in_path.iterdir() if p.suffix.lower() == '.npy'])
        out_path = Path(args.output) if args.output else in_path
        out_path.mkdir(parents=True, exist_ok=True)
        out_is_dir = True

    print(f"Encoding {len(items)} hologram(s) | encoding={args.encoding} | "
          f"off-axis=({args.off_axis_x},{args.off_axis_y}) deg | move_z={args.move_z} m")

    for src in items:
        holo = np.load(src)
        img = double_phase_method(
            holo,
            encoding=args.encoding,
            max_pi_multiplier=args.max_pi,
            gaussian_size=args.gaussian,
            move_z=args.move_z,
            wavelength=args.wavelength,
            pp=args.pp,
            off_axis_angle=(args.off_axis_x, args.off_axis_y),
            gamma=args.gamma,
        )
        dst = out_path / (src.stem + "_dpm.png") if out_is_dir else out_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(str(dst), img)
        print(f"  {src.name}  ->  {dst}")


if __name__ == '__main__':
    main()
