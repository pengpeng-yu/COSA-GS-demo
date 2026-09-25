3D Gaussian Splatting (3DGS) enables high-quality novel-view synthesis but requires substantial storage.
Existing compression methods often rely on spatial context modeling over irregular 3D representations, increasing the complexity of training and coding.
Meanwhile, floating-point context inference can introduce numerical inconsistencies across platforms, causing entropy-decoding failures.
To address these practical challenges, we propose **COSA-GS**, which constructs **C**ontext with**O**ut **S**patial **A**ggregation through anchor-wise causal factorization.
Specifically, we use geometry context derived from each anchor's coordinates to model a compact learnable anchor latent.
The anchor latent is then fused with the geometry context to form an anchor context for attribute coding.
The resulting context model features a simple architecture composed solely of linear transformations and activations.
We train COSA-GS using rate-distortion optimization with adaptive Gaussian pruning.
Further, we develop quantization-aware training and integer inference for the context model to achieve bit-exact consistency of entropy-decoded symbols across platforms.
Experiments demonstrate that COSA-GS achieves state-of-the-art compression performance while retaining fast and consistent cross-platform decoding, providing a simple yet effective framework for practical 3DGS compression.

## Installation

The code is developed with Python 3.10 and PyTorch 2.9.1. Please refer to the official
[PyTorch installation guide](https://pytorch.org/get-started/locally/) if your
CUDA version or platform differs from the example below.

```bash
conda create -y -n py310torch291 python=3.10
conda activate py310torch291

pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu130
```

Run the following commands from the project root:

```bash
pip install -r requirements.txt
pip install --no-build-isolation \
  ./submodules/diff-gaussian-rasterization \
  ./submodules/simple-knn

git clone https://github.com/NVIDIA/cutlass.git /your/path/to/cutlass
export CUTLASS_HOME=/your/path/to/cutlass
python scaffold_codec_int/int_nn_ops/setup.py build

# Optional: precompile extensions; otherwise they compile on first import.
python -c "from scaffold_codec import rans_coder"
python -c "from scaffold_codec import space_filling_curves"
```

## Demo

The demo includes a complete compressed Flowers scene and one test image,
`_DSC9144.JPG`, under `datasets/mipnerf360_flowers_DSC9144/`. After installation, run from the project root:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scaffold_codec_int.decompress_and_evaluate \
  --config config/codec30k_mipnerf360.yaml \
  --source-path datasets/mipnerf360_flowers_DSC9144 \
  --bitstream outputs/mipnerf360_30k_int_flowers_lmb0.0006/scene.bin \
  --output outputs/flowers_DSC9144_demo \
  --save-images
```

This decodes the scene and evaluates only `_DSC9144.JPG`. The output directory
contains `results.json`, `per_view.json`, `render/00000__DSC9144.png`, and
`target/00000__DSC9144.png`.

Reference metrics for this view: **PSNR 21.3739 dB**, **SSIM 0.6027**,
**LPIPS 0.3590**.

## Data

Place COLMAP-format scenes under `datasets/`, or link that directory to your
dataset path. For example:

```text
datasets/
  mipnerf360/
    bicycle/
      images/
      sparse/0/
  deep_blending/
    drjohnson/
      images/
      sparse/0/
  tandt/
    train/
      images/
      sparse/0/
```

Dataset-specific training configurations are provided in `config/`.

## Training

First train a Scaffold-GS representation (15K iterations):

```bash
CUDA_VISIBLE_DEVICES=0 python -m scaffold.train \
  config/scaffold15k_mipnerf360.yaml \
  data.scene=bicycle \
  data.model_path=outputs/scaffold15k_mipnerf360/bicycle
```

Then perform rate-distortion optimization for compression:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scaffold_codec_int.train \
  config/codec30k_mipnerf360.yaml \
  data.source_path=datasets/mipnerf360/bicycle \
  data.scaffold_model=outputs/scaffold15k_mipnerf360/bicycle/point_cloud/iteration_15000 \
  data.model_path=outputs/mipnerf360_30k_int/bicycle_lmb0.0006 \
  train.lambda_rate=0.0006
```

Change `train.lambda_rate` to select a rate-distortion trade-off. Training
automatically performs final compression, decompression, and evaluation,
saving `checkpoints/final.pt`, `scene.bin`, `results.json`, and `per_view.json`.

## Evaluation

Compress and evaluate a trained checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scaffold_codec_int.compress_and_evaluate \
  --config config/codec30k_mipnerf360.yaml \
  --source-path datasets/mipnerf360/bicycle \
  --checkpoint outputs/mipnerf360_30k_int/bicycle_lmb0.0006/checkpoints/final.pt \
  --output outputs/mipnerf360_30k_int/bicycle_lmb0.0006/eval \
  --save-images
```

To decode and evaluate an integer bitstream:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scaffold_codec_int.decompress_and_evaluate \
  --config config/codec30k_mipnerf360.yaml \
  --source-path datasets/mipnerf360/bicycle \
  --bitstream outputs/mipnerf360_30k_int/bicycle_lmb0.0006/scene.bin \
  --output outputs/mipnerf360_30k_int/bicycle_lmb0.0006/decoded_eval \
  --save-images
```

Both evaluation scripts run one warmup by default (`--warmup 0` disables it).
The checkpoint evaluation repeats encoding, decoding, and evaluation. Warmup results
are saved under `warmup_1/`, etc.; the final results are saved directly to `--output`.
For bitstream-only evaluation, decoded symbols, integer anchor coordinates, and reconstructed attributes are
saved as NumPy arrays in each result directory's `cross_platform_check/`.

## Acknowledgements

Our implementation builds on [Scaffold-GS](https://github.com/city-super/Scaffold-GS),
[HAC++](https://github.com/YihangChen-ee/HAC-plus),
and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
We thank their authors for making the code available.
