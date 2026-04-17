# VAE Project

Variational Autoencoder (VAE) and Conditional VAE (CVAE) for MNIST digit generation. This project implements a fully-connected VAE that learns a compact latent representation of handwritten digits and can generate new images by sampling from the learned latent space. The CVAE extends this by conditioning on digit class labels, enabling targeted digit generation.

Based on University of Michigan's EECS 498-007/598-005 course materials, adapted for UCLA ENGR C147B Spring 2026.

## Project Structure

| File | Description |
|---|---|
| `vae.py` | Core implementations: VAE, CVAE, reparametrize, and loss function |
| `vae_utils.py` | Provided utilities: training loop, visualization, helper functions (do not modify) |
| `vae.ipynb` | Jupyter notebook with instructions, sanity checks, training cells, and visualizations |

## Implemented Components

- **VAE Encoder** — 3-layer MLP (input_size → 400 → 400 → 400) with ReLU, plus separate mu and logvar linear heads
- **VAE Decoder** — 4-layer MLP (latent_size → 400 → 400 → 400 → input_size) with ReLU and Sigmoid output
- **VAE Forward** — Encode → reparametrize → decode pipeline
- **Reparametrization Trick** — Differentiable sampling: z = mu + sigma * epsilon
- **Loss Function** — Binary cross-entropy reconstruction loss + KL divergence, averaged over batch
- **CVAE** — Conditional VAE that concatenates one-hot class labels with encoder input and decoder input

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
conda create -n vae python=3.11
conda activate vae
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

1. Start Jupyter from the `VAE/` directory:
   ```bash
   cd VAE
   jupyter lab
   ```
2. Open `vae.ipynb` and run cells sequentially.
3. The notebook will:
   - Load the MNIST dataset (downloads automatically on first run)
   - Verify encoder, decoder, mu_layer, and logvar_layer parameter counts
   - Validate reparametrization and loss function against reference values
   - Train the VAE and CVAE (10 epochs each, ~2 min each)
   - Visualize generated digits and latent space interpolations
   - Generate specific digits using the CVAE by conditioning on class labels

## GPU Support

To use GPU acceleration, ensure CUDA is available. The code automatically detects and uses CUDA, MPS (Apple Silicon), or falls back to CPU.
