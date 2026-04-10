# Cellpose Inference Tool (v0.1.1-dev0)

A plugin that segments cells and nuclei in fluorescence microscopy images
using the [Cellpose](https://github.com/MouseLand/cellpose) deep-learning
model. Images are read and written through
[bfio](https://github.com/PolusAI/bfio), enabling support for the full range
of BioFormats-compatible file types, and output label masks are stored in
OME-TIFF or OME-Zarr format.

## Overview

Cellpose is a generalist algorithm for cell segmentation.  This tool wraps the
`cellpose.models.CellposeModel` inference API and exposes all key parameters as CLI
flags so it can run inside a WIPP workflow or standalone via Docker.

Features:
- **Batch processing** — all images matching a filepattern are processed in one
  run.
- **2-D and 3-D segmentation** — use `--do3D` to enable volumetric
  segmentation; otherwise each Z-plane is processed independently.
- **Multi-channel support** — specify separate cytoplasm and nucleus channels.
- **GPU-ready** — pass `--useGpu` when running on a CUDA-enabled host (requires
  the GPU PyTorch wheel; see [GPU note](#gpu-support) below).
- **OME output** — label masks are written as `uint32` OME-TIFF / OME-Zarr,
  compatible with downstream WIPP plugins.

## Local Development Install

These steps install the tool in an isolated environment for local development
or testing (without Docker).

**Prerequisites:** Python ≥ 3.10, [uv](https://github.com/astral-sh/uv)

```bash
# 1. Clone the monorepo
git clone https://github.com/PolusAI/image-tools.git
cd image-tools/segmentation/cellpose-inference-tool

# 2. Install uv (if not already present)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create a virtual environment and install dependencies
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[dev]"
# 4. Run the CLI
python -m polus.images.segmentation.cellpose_inference \
  --inpDir /path/to/images \
  --filePattern ".+" \
  --outDir /path/to/masks
```

## Building

```bash
./build-docker.sh
```

## Install WIPP Plugin

Navigate to the WIPP plugins page, click *Add plugin*, and paste the contents
of `plugin.json`.

## Options

### I/O

| Name            | Description                                    | I/O    | Type   | Default  |
|-----------------|------------------------------------------------|--------|--------|----------|
| `--inpDir`      | Input image collection                         | Input  | path   | required |
| `--filePattern` | Filepattern to select images inside inpDir     | Input  | string | `.+`     |
| `--outDir`      | Output directory for uint32 label masks        | Output | path   | required |
| `--preview`     | Write `preview.json` without running inference | Input  | bool   | `False`  |

### Model

| Name          | Description                                                                         | Type   | Default  |
|---------------|-------------------------------------------------------------------------------------|--------|----------|
| `--modelType` | Pretrained model. Default `cpsam` (SAM-based, v4+). Legacy: `cyto3`, `cyto2`, `cyto`, `nuclei`, `bact_omni`, `cyto2_omni` | string | `cpsam` |

### Channels

| Name            | Description                                         | Type | Default |
|-----------------|-----------------------------------------------------|------|---------|
| `--channelCyto` | 0-indexed channel for cytoplasm signal              | int  | `0`     |
| `--channelNuc`  | 0-indexed channel for nucleus signal (`-1` = none)  | int  | `-1`    |

### Diameter & size

| Name        | Description                                                   | Type  | Default |
|-------------|---------------------------------------------------------------|-------|---------|
| `--diameter` | Expected cell diameter in pixels; `0` = automatic estimation | float | `0.0`   |
| `--minSize`  | Minimum pixels per mask; `-1` disables the filter            | int   | `15`    |

### Segmentation thresholds

| Name                  | Description                                                    | Type  | Default |
|-----------------------|----------------------------------------------------------------|-------|---------|
| `--flowThreshold`     | Maximum allowed flow error; `0` disables this QC step          | float | `0.4`   |
| `--cellprobThreshold` | Cell-probability threshold; lower keeps more detections        | float | `0.0`   |

### Dynamics

| Name      | Description                                                                          | Type | Default |
|-----------|--------------------------------------------------------------------------------------|------|---------|
| `--niter` | Dynamics iterations; `0` = auto (proportional to diameter); increase for long ROIs  | int  | `0`     |

### 3-D options

| Name                | Description                                                                    | Type   | Default |
|---------------------|--------------------------------------------------------------------------------|--------|---------|
| `--do3D`            | Run full 3-D segmentation across the Z-stack                                   | bool   | `False` |
| `--stitchThreshold` | IoU threshold for stitching 2-D masks across Z; `0` = off (ignored if `--do3D`) | float | `0.0`  |
| `--anisotropy`      | Z/XY voxel size ratio; `1.0` = isotropic; critical for non-isotropic volumes   | float  | `1.0`   |
| `--flow3dSmooth`    | Gaussian σ for 3-D flow smoothing; single value or ZYX triple `"z,y,x"`; `0` = off | string | `None` |

### Normalisation

| Name               | Description                                                            | Type   | Default |
|--------------------|------------------------------------------------------------------------|--------|---------|
| `--noNorm`         | Disable image normalisation                                            | bool   | `False` |
| `--normPercentile` | Normalisation range as `"low,high"` (e.g. `"1.0,99.0"`); ignored if `--noNorm` | string | `None` |

### Inference performance

| Name          | Description                                                         | Type | Default |
|---------------|---------------------------------------------------------------------|------|---------|
| `--batchSize` | Tiles per network forward pass; increase on high-VRAM GPUs          | int  | `8`     |
| `--augment`   | Overlapping-tile augmentation for slightly improved accuracy         | bool | `False` |

### Hardware

| Name          | Description                                                          | Type   | Default |
|---------------|----------------------------------------------------------------------|--------|---------|
| `--useGpu`    | Use GPU acceleration. Cellpose auto-selects CUDA → MPS → CPU         | bool   | `False` |

### Miscellaneous

| Name                | Description                                 | Type | Default |
|---------------------|---------------------------------------------|------|---------|
| `--excludeOnEdges`  | Discard masks that touch the image border   | bool | `False` |

## Docker Command

```bash
docker run \
  -e POLUS_IMG_EXT=".ome.tif" \
  -e NUM_WORKERS=1 \
  -v /path/to/data:/data \
  polusai/cellpose-inference-tool:0.1.1-dev0 \
  --inpDir=/data/images \
  --filePattern=".*\.tif" \
  --modelType=cyto3 \
  --diameter=0 \
  --minSize=15 \
  --channelCyto=0 \
  --channelNuc=-1 \
  --flowThreshold=0.4 \
  --cellprobThreshold=0.0 \
  --batchSize=8 \
  --outDir=/data/masks
```

### Multi-channel example (cytoplasm in ch0, nucleus in ch1)

```bash
docker run \
  -v /path/to/data:/data \
  polusai/cellpose-inference-tool:0.1.1-dev0 \
  --inpDir=/data/images \
  --filePattern=".+" \
  --modelType=cyto3 \
  --channelCyto=0 \
  --channelNuc=1 \
  --outDir=/data/masks
```

### 3-D segmentation

```bash
docker run \
  -v /path/to/data:/data \
  polusai/cellpose-inference-tool:0.1.1-dev0 \
  --inpDir=/data/z_stacks \
  --filePattern=".+" \
  --modelType=cyto3 \
  --do3D \
  --outDir=/data/masks_3d
```

## GPU Support

Cellpose supports three compute backends — CUDA (NVIDIA), MPS (Apple Silicon),
and CPU.

### NVIDIA GPU (Linux x86\_64 — Docker)

The Docker image auto-detects the architecture at build time:
- **x86\_64** → PyTorch CUDA 12.4 wheel is installed.
- **ARM64** → standard PyTorch wheel is installed (CPU only inside Docker; see
  MPS section below for Apple Silicon GPU acceleration).

No CUDA installation inside the container is required. The NVIDIA Container
Toolkit mounts the host CUDA libraries at runtime.

**Host requirements**

| Requirement | Minimum version |
|---|---|
| NVIDIA driver | 525.60.13 (CUDA 12.0+) |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | latest |
| Docker | 20.10+ |

**Run command**

```bash
docker run --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v /path/to/data:/data \
  polusai/cellpose-inference-tool:0.1.1-dev0 \
  --inpDir=/data/images \
  --filePattern=".+" \
  --modelType=cyto3 \
  --batchSize=16 \
  --useGpu \
  --outDir=/data/masks
```

Use `-e CUDA_VISIBLE_DEVICES=1` to target a specific GPU on a multi-GPU host.

---

### Multi-GPU (Linux — Docker)

The tool automatically distributes images across all available GPUs using
round-robin assignment. No extra flags are needed — passing `--gpus all`
exposes all GPUs to the container and the tool detects them via
`torch.cuda.device_count()` at runtime.

Set `NUM_WORKERS` to match the number of GPUs so each GPU gets its own
dedicated worker process:

```bash
docker run --gpus all \
  -e NUM_WORKERS=4 \
  -v /path/to/data:/data \
  polusai/cellpose-inference-tool:0.1.1-dev0 \
  --inpDir=/data/images \
  --filePattern=".+" \
  --modelType=cyto3 \
  --batchSize=16 \
  --useGpu \
  --outDir=/data/masks
```

**How it works:**

| Image index | Assigned GPU (`idx % num_gpus`) |
|---|---|
| 0, 4, 8, … | GPU 0 |
| 1, 5, 9, … | GPU 1 |
| 2, 6, 10, … | GPU 2 |
| 3, 7, 11, … | GPU 3 |

The tool logs the GPU assignment at startup:
```
Multi-GPU mode: distributing across 4 GPU(s).
```
And prints a summary when done:
```
Batch complete: 100/100 succeeded, 0 failed | total time: 240.3s (2.4s/image avg)
```

**Tip:** set `NUM_WORKERS` equal to the number of GPUs. Using more workers
than GPUs causes multiple workers to share a GPU, which may reduce throughput.

---

### Apple Silicon GPU — MPS (M1/M2/M3/M4, native only)

Docker on macOS runs inside a Linux VM and **cannot access the Metal GPU**.
To use the M-series GPU, follow the [Local Development Install](#local-development-install)
steps and pass `--useGpu` when running the CLI.

PyTorch's MPS backend is supported on macOS 12.3+ with any Apple Silicon chip.

---

### CPU-only

Omit `--useGpu` (or pass `--useGpu false`). Inside Docker on ARM64 (M-series
Mac) the image automatically uses the CPU-only PyTorch wheel.

## Output

Each input image produces one label mask with the same stem name and the
extension controlled by `POLUS_IMG_EXT` (default `.ome.tif`). The mask is a
`uint32` image where each unique non-zero integer identifies one segmented cell.
Background pixels are `0`.

## Authors

- Hamdah Shafqat Abbasi <hamdahshafqat.abbasi@nih.gov>
- Nick Schaub <nick.schaub@nih.gov>

## References

- Stringer, C., Wang, T., Michaelos, M. & Pachitariu, M. **Cellpose: a
  generalist algorithm for cellular segmentation.** *Nature Methods* 18,
  100–106 (2021). https://doi.org/10.1038/s41592-020-01018-x
- Pachitariu, M. & Stringer, C. **Cellpose 2.0: how to train your own model.**
  *Nature Methods* 19, 1634–1641 (2022).
  https://doi.org/10.1038/s41592-022-01663-4
