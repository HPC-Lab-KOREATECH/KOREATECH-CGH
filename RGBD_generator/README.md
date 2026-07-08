# RUCGH RGB+D Generator

> **Status:** in progress, on the way to release. The scripts run end-to-end
> and reproduce the published methods, but interfaces, defaults, and outputs
> may still change before the final release. Treat this as a preview.

RUCGH samples textured 3D object scenes and exports orthographic RGB images
with matching per-pixel depth maps. This release supports only the RGB+D
export path and requires `gen --rgbd_only`.

This RGB+D generator was developed as part of the research paper
"A Large-Depth-Range Layer-Based Hologram Dataset Generation for Machine
Learning-Based 3D Computer-Generated Holography".

## Notice

This project includes code modified from Ingo Wald's OptiX 7 course:
<https://github.com/ingowald/optix7course>. The upstream OptiX course code and
vendored `gdt` helper library are copyright Ingo Wald and licensed under the
Apache License, Version 2.0, as noted in the original source headers.

## Requirements

- NVIDIA GPU with an OptiX-capable driver
- CUDA Toolkit 12.x
- OptiX SDK 8.0.0
- OpenCV
- CMake 3.25.2 or newer
- C++20-capable compiler

OptiX must be downloaded manually from
<https://developer.nvidia.com/designworks/optix/downloads>. Set
`OPTIX_INSTALL_DIR` to the directory containing
`NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64`.

The `gdt` math headers from Ingo Wald's OptiX-7 course are vendored at
[RUCGH/common/gdt](RUCGH/common/gdt).

## Build

Linux:

```bash
export OPTIX_INSTALL_DIR=/opt/optix
cmake -S . -B build
cmake --build build -j
```

Windows, from an x64 Native Tools Command Prompt or Developer PowerShell for
Visual Studio:

```powershell
cmake -S . -B build -DOPTIX_INSTALL_DIR="C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0"
cmake --build build --config Release
```

`OPTIX_INSTALL_DIR` may point either to the extracted SDK root containing
`include/optix.h`, or to a parent directory containing
`NVIDIA-OptiX-SDK-8.0.0-win64` / `NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64`.

If Visual Studio reports `Cannot open include file: 'gdt/math/vec.h'`, delete
the old generated build folder and re-run CMake so the target include paths are
regenerated. The vendored header root is `RUCGH/common/gdt`.

The executable is `build/RUCGH/RUCGH_exe`. The orthographic OptiX PTX file is
emitted under `build/RUCGH/CMakeFiles/myptx.dir/`; pass that directory as
`--ptx_path` at runtime.

### Docker

Place `NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh` next to `dockerfile`, then:

```bash
docker build -t rucgh-release .
```

## Inputs

`--obj_path` should contain one subdirectory per object. Each object directory
must include a `meshes/` folder with an `.obj`. Textures are loaded first from
`materials/textures/texture.png`; if that file is not present, RUCGH falls back
to the first `.png`, `.jpg`, or `.jpeg` in `materials/textures/`, then to an
image beside the `.obj`.

```text
obj_path/
  chair_01/
    meshes/
      model.obj
      model.mtl
    materials/
      textures/
        texture.png
  lamp_03/
    meshes/
      model.obj
    materials/
      textures/
        texture.jpg
```

At each iteration RUCGH picks `--num_objects` object directories at random,
applies random scale, rotation, and translation, then writes the sampled scene
to `<csv_path>/i.csv`. Use `--load_csv` to replay existing CSV scenes.

`--floor_path` is optional. When provided, one texture from the directory is
placed on a flat floor plane at `--object_maxdepth`.

## Outputs

For each index `i` in `[begin_index, end_index)`, `gen --rgbd_only` writes:

- `<rgb_output_path>/i.exr` and `<rgb_output_path>/i.png`
- `<depth_output_path>/i.exr` and `<depth_output_path>/i.png`
- `<csv_path>/i.csv`

The EXR files are the canonical float outputs. PNG files are previews; depth
PNGs are normalized to `[0, 255]`.

## Examples

Windows 512x512 single-GPU render:

```powershell
.\build\RUCGH\Release\RUCGH_exe.exe gen `
  --precision fp32 `
  --width 512 --height 512 `
  --wavelength 638e-9 532e-9 450e-9 `
  --pixel_pitch 3.6e-6 `
  --object_mindepth 0.000 `
  --object_maxdepth 0.0203361 `
  --rgb_output_path "rgb" `
  --depth_output_path "depth" `
  --obj_path "D:\scanned_objects" `
  --csv_path "csv" `
  --ptx_path ".\build\RUCGH\CMakeFiles\myptx.dir\Release\deviceProgramOrthogonal.ptx" `
  --begin_index 0 --end_index 6000 `
  --num_objects 1 `
  --device 0 --num_devices 1 `
  --rgbd_only
```

Single-scene RGB+D render:

```bash
./build/RUCGH/RUCGH_exe gen \
    --width 1024 --height 1024 \
    --wavelength 638e-9 532e-9 450e-9 \
    --pixel_pitch 3.6e-6 \
    --object_mindepth 0.01 --object_maxdepth 0.05 \
    --obj_path ./data/objects \
    --rgb_output_path ./out/rgb \
    --depth_output_path ./out/depth \
    --csv_path ./out/csv \
    --ptx_path ./build/RUCGH/CMakeFiles/myptx.dir \
    --begin_index 0 --end_index 1 \
    --num_objects 4 \
    --device 0 --num_devices 1 \
    --rgbd_only
```

Batch render on two GPUs:

```bash
./build/RUCGH/RUCGH_exe gen \
    --width 1920 --height 1080 \
    --wavelength 638e-9 532e-9 450e-9 \
    --pixel_pitch 3.6e-6 \
    --object_mindepth 0.0 --object_maxdepth 0.0226912 \
    --obj_path ./data/objects \
    --floor_path ./data/floor_textures \
    --rgb_output_path ./out/rgb \
    --depth_output_path ./out/depth \
    --csv_path ./out/csv \
    --ptx_path ./build/RUCGH/CMakeFiles/myptx.dir \
    --begin_index 0 --end_index 100 \
    --num_objects 8 \
    --num_devices 2 \
    --rgbd_only
```

Replay existing scene CSVs:

```bash
./build/RUCGH/RUCGH_exe gen \
    ... same flags as above ... \
    --load_csv \
    --rgbd_only
```

## CLI Options

```text
--width                output width in pixels
--height               output height in pixels
--wavelength R G B     wavelengths [m], 3 values
--pixel_pitch          pixel pitch [m]

--object_mindepth      nearest object distance [m]
--object_maxdepth      farthest object distance [m]
--num_objects          objects sampled per scene

--obj_path             directory of object subfolders
--floor_path           directory of floor textures
--rgb_output_path      RGB render directory
--depth_output_path    depth render directory
--csv_path             scene-description directory
--ptx_path             directory containing the orthographic PTX

--begin_index          first scene index, inclusive
--end_index            last scene index, exclusive
--load_csv             replay scenes from <csv_path>/i.csv
--rgbd_only            required in this release

--device               CUDA device id
--num_devices          number of GPUs to dispatch over
--precision            accepted for compatibility; fp32 is used
```

## Loading Output

```python
import cv2

rgb = cv2.imread("out/rgb/0.exr", cv2.IMREAD_UNCHANGED)
depth = cv2.imread("out/depth/0.exr", cv2.IMREAD_UNCHANGED)
```
