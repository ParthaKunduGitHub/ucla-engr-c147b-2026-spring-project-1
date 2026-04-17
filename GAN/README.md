# GAN Project

Generative Adversarial Networks (GANs) for MNIST digit generation. This project implements three GAN variants — a vanilla fully-connected GAN, a Least-Squares GAN (LSGAN), and a Deep Convolutional GAN (DCGAN) — trained on the MNIST handwritten digit dataset.

Based on Stanford University's CS231n course materials, adapted for UCLA ENGR C147B Spring 2026.

## Project Structure

| File | Description |
|---|---|
| `gan.py` | Core implementations: Discriminator, Generator, loss functions, DC variants, and training loop |
| `gan.ipynb` | Jupyter notebook with instructions, sanity checks, training cells, and inline questions |
| `assets/` | Reference images and precomputed check values (`gan-checks.npz`) |
| `mnist_data/` | MNIST dataset (auto-downloaded on first run) |
| `pyproject.toml` | Project metadata and dependencies |

## Implemented Components

- **Discriminator** — 3-layer MLP (784 → 256 → 256 → 1) with LeakyReLU
- **Generator** — 3-layer MLP (noise_dim → 1024 → 1024 → 784) with ReLU and Tanh
- **BCE GAN Loss** — Standard binary cross-entropy discriminator and generator losses
- **LSGAN Loss** — Least-squares discriminator and generator losses for more stable training
- **DCDiscriminator** — Convolutional discriminator with Conv2d, MaxPool, and LeakyReLU
- **DCGenerator** — Transposed-convolution generator with BatchNorm, following the InfoGAN architecture

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision 0.15+
- numpy
- matplotlib
- jupyterlab

## Setup

Pick any method you prefer.

### Using requirements.txt (from project root)
```bash
pip install -r ../requirements.txt
```

### uv
```bash
uv sync
```

### conda
```bash
conda create -n gan python=3.11
conda activate gan
conda install pytorch torchvision -c pytorch
pip install matplotlib jupyterlab
```

### venv
```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install .
```

## Running the Notebook

1. Start Jupyter from the `GAN/` directory:
   ```bash
   cd GAN
   jupyter lab
   ```
2. Open `gan.ipynb` and run cells sequentially.
3. The notebook will:
   - Load the MNIST dataset (downloads automatically on first run)
   - Verify model architectures via parameter count checks
   - Validate loss functions against precomputed reference values
   - Train three GAN variants (Vanilla, LSGAN, DCGAN) and display generated images

## Training Times

| Model | CPU | GPU |
|---|---|---|
| Vanilla GAN (10 epochs) | ~7 min | ~1-2 min |
| LSGAN (10 epochs) | ~7 min | ~1-2 min |
| DCGAN (5 epochs) | ~35 min | ~1 min |

## GPU Support

To use GPU acceleration, ensure CUDA is available. The code automatically detects and uses CUDA, MPS (Apple Silicon), or falls back to CPU.
