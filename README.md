# Probabilistic Energy Load Forecasting via Bayesian LSTM

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)
![Pyro](https://img.shields.io/badge/Pyro-PPL-orange.svg)
![Status](https://img.shields.io/badge/Status-Prototype-green.svg)

## Executive Summary & Business Value

In high-stakes sectors like energy trading and grid management, **point estimates are insufficient**. Knowing *that* the load will be 15,000 MW is less valuable than knowing it will be between 14,800 and 15,200 MW with 95% confidence.

This project implements a **Bayesian Long Short-Term Memory (LSTM)** network to predict energy consumption. Unlike standard deep learning models, this solution leverages **Variational Inference** to decompose uncertainty into two actionable components:
1.  **Epistemic Uncertainty:** Uncertainty derived from the model itself (lack of training data).
2.  **Aleatoric Uncertainty:** Noise inherent in the data generation process.

**Use Case:** This approach allows decision-makers to distinguish between "we need more data" (epistemic) and "the system is naturally volatile" (aleatoric), enabling better risk management and resource allocation.

---

## Results & Visualization

The model successfully captures the periodic nature of energy consumption while providing dynamic confidence intervals.

![Uncertainty Decomposition](images/forecast_plot.jpg)
*Figure 1: Probabilistic forecast showing the ground truth (green) vs. predicted mean (navy). The teal area represents model uncertainty (reducible with data), while light blue represents data noise (irreducible).*

---

## Technical Methodology

The core architecture combines the sequence modeling capabilities of LSTMs with the probabilistic framework of **Pyro (built on PyTorch)**.

### Key Components:
* **Bayesian Layers:** Weights and biases are not fixed values but probability distributions (Gaussian priors).
* **Variational Inference (SVI):** We approximate the complex posterior distribution using a diagonal normal guide and optimize the Evidence Lower Bound (ELBO).
* **Modular Design:** The project utilizes distinct modules for data loading, model definition, and configuration management using Python `dataclasses`.

### Tech Stack:
* **Deep Learning:** PyTorch, Pyro (Probabilistic Programming)
* **Data Handling:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib
* **Environment:** Managed via `uv` (modern Python package manager)

---

## How to Run

This project uses `uv` for fast and reliable dependency management.

### 1. Clone & Setup
```bash
git clone https://github.com/m0r1tzwa/probabilistic-forecaster.git
cd probabilistic-forecaster

# Install dependencies from lock file
uv sync
```

### 2. Train the Model
To train the Bayesian LSTM with Variational Inference:
```bash
uv run main.py --mode train --look_back 48 --horizon 24
```

### 3. Evaluate & Visualize
To generate the uncertainty decomposition plots:
```bash
uv run main.py --mode eval
```

---

## Project Structure

```text
probabilistic-forecaster/
├── data/
│   └── AEP_hourly.csv      # Time-series data
├── src/
│   ├── config.py           # Configuration via Dataclasses
│   ├── data_loader.py      # Preprocessing & Windowing logic
│   ├── model.py            # Bayesian LSTM (Pyro Module)
│   └── visualize.py        # Plotting logic for uncertainty
├── images/                 # Generated plots
├── main.py                 # Entry point for training/eval
├── pyproject.toml          # Project dependencies
└── uv.lock                 # Exact dependency tree
```

## Contact

**Moritz Wagenknecht** [\[LinkedIn\]](https://www.linkedin.com/in/moritz-wagenknecht-5174812ab/)