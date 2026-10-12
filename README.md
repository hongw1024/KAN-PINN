# KAN-PINN: Kolmogorov-Arnold Network Enhanced Physics-Informed Neural Network

A physics-informed neural network (PINN) framework that integrates the Kolmogorov-Arnold Network (KAN) for solving coupled hydrodynamic and solute transport equations, specifically the kinematic wave equation (KWE) and advection-dispersion equation (ADE).

## Repository Structure

```
KAN-PINN/
├── benchmarks/                    # Numerical benchmark code
│   ├── 01_Benchmark_ADE_InstantaneousRelease.py    # Case 1: ADE instantaneous release
│   ├── 02_Benchmark_ADE_ContinuousRelease.py       # Case 2: ADE continuous release
│   ├── 03_Benchmark_ADE_FiniteTimeRelease.py        # Case 3: ADE finite-time release
│   ├── 04_Benchmark_ADE_NumericalReference.py       # Case 4: ADE numerical reference
│   ├── 05_Benchmark_Coupled_KWE_ADE.ipynb           # Coupled KWE-ADE (with results)
│   └── 05_Benchmark_Coupled_KWE_ADE.py              # Same as above (.py format)
├── kan/                           # KAN neural network implementation
│   ├── __init__.py
│   ├── KANLayer.py                # KAN layer with B-spline activations
│   ├── MultKAN.py                 # Multi-layer KAN model
│   ├── spline.py                  # B-spline basis functions
│   ├── utils.py                   # KAN utilities
│   ├── compiler.py                # Symbolic regression compiler
│   ├── LBFGS.py                   # L-BFGS optimizer wrapper
│   ├── MLP.py                     # MLP baseline model
│   ├── Symbolic_KANLayer.py       # Symbolic KAN layer
│   └── experiment.py              # Experiment utilities
├── utils/                         # Shared utilities
│   ├── plotting.py                # Colormap and plotting functions
│   ├── logger.py                  # Logging utilities
│   └── random_sampling.py         # Random sampling methods
├── fast_bspline_kan.py            # Optimized B-spline KAN for PINN
├── requirements.txt               # Python dependencies
└── README.md
```

> **Note:** Additional code for the SVE kinematic wave benchmark and real-world case studies will be released upon publication of the accompanying manuscript.

## Key Features

- **KAN-PINN architecture**: Replaces fixed node activations (MLP) with learnable B-spline functions on network edges, overcoming spectral bias
- **Parameter efficiency**: Achieves comparable or better accuracy with ~36% fewer parameters than MLP-PINN
- **Physics-informed**: Embeds governing PDEs directly into the loss function via automatic differentiation
- **Forward and inverse modeling**: Supports both forward simulation and inverse parameter estimation

## Governing Equations

### Advection-Dispersion Equation (ADE)

```
dc/dt + v * dc/dx - D * d2c/dx2 + k * c = 0
```

where:
- `c` is the solute concentration (M L^-3)
- `v` is the cross-sectional average flow velocity (L T^-1)
- `D` is the dispersion coefficient (L^2 T^-1)
- `k` is the first-order decay rate (T^-1)

### Kinematic Wave Equation (KWE)

```
dA/dt + dQ/dx = ql  (continuity)
Q = (1/n) * A^(5/3) * S0^(1/2) / P^(2/3)  (Manning's equation)
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.4.1+
- CUDA 11.8 (for GPU acceleration)

### Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

Each benchmark script can be run independently:

```bash
# Case 1: Instantaneous release (Gaussian pulse)
python benchmarks/01_Benchmark_ADE_InstantaneousRelease.py

# Case 2: Continuous release at constant rate
python benchmarks/02_Benchmark_ADE_ContinuousRelease.py

# Case 3: Finite-time release
python benchmarks/03_Benchmark_ADE_FiniteTimeRelease.py

# Case 4: Numerical reference solution
python benchmarks/04_Benchmark_ADE_NumericalReference.py

# Case 5: Coupled KWE-ADE
python benchmarks/05_Benchmark_Coupled_KWE_ADE.py
```

### Benchmark Descriptions

| Case | Description | Equation | Analytical Solution |
|------|-------------|----------|---------------------|
| 1 | ADE instantaneous release | ADE (Gaussian pulse) | mp / (2*sqrt(pi*D*t)) * exp(...) |
| 2 | ADE continuous release | ADE (erfc-based) | c0/2 * [exp(...) * erfc(...)] |
| 3 | ADE finite-time release | ADE (erfc superposition) | Piecewise erfc solution |
| 4 | ADE numerical reference | ADE (.mat data) | Numerical reference |
| 5 | Coupled KWE-ADE | KWE + ADE | Finite difference reference |

### Network Configurations

- **MLP-PINN**: [2, 20, 20, 20, 20, 20, 20, 20, 20, 1] with Tanh activation
- **KAN-PINN**: [2, 8, 8, 1] with grid=20, k=5 (5th-order B-spline)

### Training

- Optimizer: Adam (lr=1e-3) for 10,001 iterations
- Optional: L-BFGS refinement
- Loss function: Data loss (MSE) + Physics residual loss (PDE)

## Results

The KAN-PINN consistently outperforms MLP-PINN across all benchmarks:
- Reduced L2 errors by up to 82.76%
- Used ~36% fewer trainable parameters
- Faster convergence with smoother loss landscapes

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lei2026kanpinn,
  title={From Numerical Benchmarks to Real-World Floods Routing: A KAN-Enhanced Physics-Informed Neural Network for solving Coupled 1D Kinematic Wave and Advection-Dispersion Equations},
  author={Lei, Hongwei and Li, Gaoyuan and Niu, Jie and He, Mingxia and He, Bin and Zhu, Qingmin and Wu, Chuanhao and Hu, Bill X. and Wu, Pan},
  journal={Environmental Modelling and Software},
  year={2026},
  publisher={Elsevier}
}
```

## License

MIT License

## Contact

- Jie Niu: jay_niu@hotmail.com
- Pan Wu: pwu@gzu.edu.cn

## Acknowledgments

This work was supported by the National Natural Science Foundation of China [41972244].
