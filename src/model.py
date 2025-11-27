import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


class BayesianLSTM(PyroModule):
    """
    A Bayesian LSTM model using Pyro for probabilistic forecasting.

    Attributes:
        lstm (nn.LSTM): Standard PyTorch LSTM layer.
        linear_mu (PyroModule): Probabilistic linear layer for mean prediction.
        linear_sigma (PyroModule): Probabilistic linear layer for uncertainty prediction.
    """

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1, dropout: float = 0.0
    ):
        super().__init__()
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Probabilistic Output Heads (Bayesian Regression)
        # Using Normal priors for weights and biases
        self.linear_mu = PyroModule[nn.Linear](hidden_size, output_size)
        self.linear_mu.weight = PyroSample(
            dist.Normal(0.0, 1.0).expand([output_size, hidden_size]).to_event(2)
        )
        self.linear_mu.bias = PyroSample(
            dist.Normal(0.0, 1.0).expand([output_size]).to_event(1)
        )

        self.linear_sigma = PyroModule[nn.Linear](hidden_size, output_size)
        self.linear_sigma.weight = PyroSample(
            dist.Normal(0.0, 1.0).expand([output_size, hidden_size]).to_event(2)
        )
        self.linear_sigma.bias = PyroSample(
            dist.Normal(0.0, 1.0).expand([output_size]).to_event(1)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass defining the generative process.

        Args:
            x (torch.Tensor): Input sequence (Batch, Lookback, Features).
            y (torch.Tensor, optional): Observed values for training (Batch, Horizon).

        Returns:
            torch.Tensor: Predicted mean values.
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]

        mu = self.linear_mu(last_hidden)
        pyro.deterministic("linear_mu", mu)

        sigma_pre = self.linear_sigma(last_hidden)
        pyro.deterministic("linear_sigma", sigma_pre)

        # Softplus to ensure positive standard deviation
        sigma = F.softplus(sigma_pre) + 1e-5

        # Likelihood definition
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), obs=y)

        return mu
