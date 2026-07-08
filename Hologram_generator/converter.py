#!/usr/bin/env python3
"""
npy_convert.py — Dataset conversion utilities for the hologram generation pipeline.

Commands
--------
rgb      : Resize float32 RGB EXR files to a target resolution
depth    : Normalize and/or resize float32 depth EXR files
complex  : Convert complex hologram .npy files to amplitude and phase EXR images
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
import cupy as cp
from openexr_numpy import imread, imwrite
from skimage.transform import resize
from PIL import Image


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _list_files(directory: Path, ext: str) -> list[Path]:
    return [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() == ext]


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)


def _load_depth_exr(path: Path) -> np.ndarray:
    """Load a single-channel depth EXR, trying common channel names."""
    import OpenEXR
    exr = OpenEXR.InputFile(str(path))
    available = list(exr.header()["channels"].keys())
    exr.close()

    preferred = ["R", "Y", "Value", "Value.V"]
    candidates = [ch for ch in preferred if ch in available]
    candidates += [ch for ch in available if ch not in preferred]
    candidates.append(None)

    last_err = None
    for ch in candidates:
        try:
            arr = imread(str(path)) if ch is None else imread(str(path), [ch])
            arr = arr[..., 0] if arr.ndim == 3 else arr
            return arr.astype(np.float32)
        except (ValueError, KeyError) as e:
            last_err = e
    raise RuntimeError(f"Cannot read depth channel from {path}") from last_err


def _load_rgb_exr(path: Path) -> np.ndarray:
    """Load an RGB EXR as a float32 H×W×3 array."""
    last_err = None
    for names in [None, ["R", "G", "B"], ["R", "G", "B", "A"]]:
        try:
            arr = imread(str(path)) if names is None else imread(str(path), names)
            if arr.ndim == 2:
                return np.stack([arr, arr, arr], axis=-1).astype(np.float32)
            if arr.ndim == 3 and arr.shape[2] == 1:
                return np.repeat(arr, 3, axis=2).astype(np.float32)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                return arr[..., :3].astype(np.float32)
        except (ValueError, KeyError) as e:
            last_err = e
    raise RuntimeError(f"Cannot read RGB channels from {path}") from last_err


# ---------------------------------------------------------------------------
# RGB — resize
# ---------------------------------------------------------------------------

def _rgb_one(src: Path, dst: Path, size: tuple[int, int] | None, preview_dir: Path | None) -> None:
    rgb = _load_rgb_exr(src)
    if size is not None:
        w, h = size
        rgb = resize(rgb, (h, w, 3), order=3, preserve_range=True, anti_aliasing=True).astype(np.float32)
    imwrite(str(dst), rgb, channel_names="RGB")
    if preview_dir is not None:
        Image.fromarray(_to_uint8(rgb)).save(preview_dir / dst.with_suffix(".png").name)
    print(f"rgb   {src.name} → {dst.name}")


def rgb_entry(args: argparse.Namespace) -> None:
    size = tuple(args.size) if args.size else None
    preview_dir = args.preview
    if preview_dir is not None:
        _ensure_dir(preview_dir)

    if args.input.is_file():
        out = args.output if args.output is not None else args.input
        _ensure_dir(out.parent)
        _rgb_one(args.input, out, size, preview_dir)
        return

    out_dir = args.output if args.output is not None else args.input
    if not out_dir.is_dir():
        raise ValueError("--output must be a directory when --input is a directory")
    _ensure_dir(out_dir)

    files = _list_files(args.input, ".exr")
    work = partial(_rgb_one, size=size, preview_dir=preview_dir)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in as_completed([pool.submit(work, f, out_dir / f.name) for f in files]):
            pass


# ---------------------------------------------------------------------------
# Depth — normalize and/or resize
# ---------------------------------------------------------------------------

def _depth_one(
    src: Path,
    dst: Path,
    dmin: float,
    dmax: float,
    size: tuple[int, int] | None,
    preview_dir: Path | None,
    no_normalize: bool = False,
) -> None:
    arr = _load_depth_exr(src)

    if size is not None:
        w, h = size
        arr = resize(arr, (h, w), order=0, preserve_range=True, anti_aliasing=True).astype(np.float32)

    if no_normalize:
        out = arr
    else:
        effective_dmax = float(np.max(arr)) if dmax == 0 else dmax
        out = cp.asnumpy(
            cp.clip(
                (cp.asarray(arr, dtype=cp.float32) - dmin) / (effective_dmax - dmin),
                0, 1,
            )
        )

    imwrite(str(dst), out.astype(np.float32), channel_names="Y")
    if preview_dir is not None:
        Image.fromarray(_to_uint8(out), "L").save(preview_dir / dst.with_suffix(".png").name)
    print(f"depth {src.name} → {dst.name}")


def depth_entry(args: argparse.Namespace) -> None:
    size = tuple(args.size) if args.size else None
    preview_dir = args.preview
    if preview_dir is not None:
        _ensure_dir(preview_dir)

    if args.input.is_file():
        _ensure_dir(args.output.parent)
        _depth_one(args.input, args.output, args.dmin, args.dmax, size, preview_dir, args.no_normalize)
        return

    if not args.output.is_dir():
        raise ValueError("--output must be a directory when --input is a directory")
    _ensure_dir(args.output)

    files = _list_files(args.input, ".exr")
    work = partial(_depth_one, dmin=args.dmin, dmax=args.dmax, size=size,
                   preview_dir=preview_dir, no_normalize=args.no_normalize)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in as_completed([pool.submit(work, f, args.output / f.name) for f in files]):
            pass


# ---------------------------------------------------------------------------
# Complex .npy → amplitude + phase EXR
# ---------------------------------------------------------------------------

def _complex_one(src: Path, amp_dst: Path, phase_dst: Path) -> None:
    c = np.load(src)  # expected: complex64, shape (3, H, W)
    if c.ndim == 3 and c.shape[2] == 3 and np.iscomplexobj(c):
        c = np.transpose(c, (2, 0, 1))  # (H, W, 3) → (3, H, W)
    c_hwc = np.transpose(c, (1, 2, 0))  # (3, H, W) → (H, W, 3) for EXR

    g = cp.asarray(c_hwc)
    amp = cp.abs(g)
    phase = (cp.angle(g) + cp.pi) / (2 * cp.pi)  # [-π, π] → [0, 1]

    imwrite(str(amp_dst),   cp.asnumpy(amp).astype(np.float32),   channel_names="RGB")
    imwrite(str(phase_dst), cp.asnumpy(phase).astype(np.float32), channel_names="RGB")
    print(f"npy   {src.name} → amp: {amp_dst.name}, phase: {phase_dst.name}")


def complex_entry(args: argparse.Namespace) -> None:
    if args.input.is_file():
        _ensure_dir(args.amp.parent)
        _ensure_dir(args.phase.parent)
        _complex_one(args.input, args.amp, args.phase)
        return

    _ensure_dir(args.amp)
    _ensure_dir(args.phase)

    files = _list_files(args.input, ".npy")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _complex_one,
                f,
                args.amp   / f.with_suffix(".exr").name,
                args.phase / f.with_suffix(".exr").name,
            )
            for f in files
        ]
        for _ in as_completed(futures):
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--workers", type=int, default=os.cpu_count(),
        help="threads for batch mode (default: CPU core count)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # rgb
    r = sub.add_parser("rgb", help="Resize float32 RGB EXR files")
    r.add_argument("--input",   required=True, type=Path, help="input EXR file or directory")
    r.add_argument("--output",  type=Path,                help="output EXR file or directory (default: overwrite input)")
    r.add_argument("--size",    nargs=2, metavar=("W", "H"), type=int, help="target resolution, e.g. --size 1024 1024")
    r.add_argument("--preview", type=Path,                help="directory for 8-bit PNG previews")

    # depth
    d = sub.add_parser("depth", help="Normalize and/or resize float32 depth EXR files")
    d.add_argument("--input",        required=True, type=Path,  help="input EXR file or directory")
    d.add_argument("--output",       required=True, type=Path,  help="output EXR file or directory")
    d.add_argument("--dmin",         type=float, default=0.0,   help="minimum depth for normalization (default: 0.0)")
    d.add_argument("--dmax",         type=float, default=0.0,   help="maximum depth for normalization (default: auto from file)")
    d.add_argument("--no_normalize", action="store_true",       help="write raw depth values without normalization")
    d.add_argument("--size",         nargs=2, metavar=("W", "H"), type=int, help="target resolution, e.g. --size 1024 1024")
    d.add_argument("--preview",      type=Path,                 help="directory for 8-bit PNG previews")

    # complex
    c = sub.add_parser("complex", help="Convert complex .npy holograms to amplitude and phase EXR")
    c.add_argument("--input", required=True, type=Path, help="input .npy file or directory")
    c.add_argument("--amp",   required=True, type=Path, help="amplitude output EXR file or directory")
    c.add_argument("--phase", required=True, type=Path, help="phase output EXR file or directory (mapped to [0, 1])")

    return p


def main(argv=None) -> None:
    args = make_parser().parse_args(argv)
    args.workers = max(1, args.workers)
    {"rgb": rgb_entry, "depth": depth_entry, "complex": complex_entry}[args.cmd](args)


if __name__ == "__main__":
    main()
