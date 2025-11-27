import argparse
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os
import pathlib as Path

from src.config import ProjectConfig
from src.data_loader import get_data_loaders
from src.model import BayesianLSTM
from src.visualize import plot_uncertainty_decomposition


def train(cfg: ProjectConfig):
    pyro.set_rng_seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    pyro.clear_param_store()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, _ = get_data_loaders(
        filepath=cfg.data_path,
        location_col=cfg.column_name,
        look_back=cfg.train.look_back,
        horizon=cfg.train.horizon,
        batch_size=cfg.train.batch_size,
        train_split=cfg.train.train_split,
        val_split=cfg.train.val_split,
    )

    model = BayesianLSTM(
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_size,
        output_size=cfg.train.horizon,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    ).to(device)
    guide = AutoDiagonalNormal(model)

    optimizer = pyro.optim.ClippedAdam(
        {"lr": cfg.train.learning_rate, "clip_norm": cfg.train.clip_norm}
    )
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    print(f"Starting training for {cfg.train.num_epochs} epochs...")
    best_val_loss = float("inf")

    os.makedirs(os.path.dirname(cfg.param_save_path), exist_ok=True)

    for epoch in range(cfg.train.num_epochs):
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            train_loss += svi.step(x, y)
        train_loss = train_loss / len(train_loader.dataset)

        val_loss = 0.0
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            val_loss += svi.evaluate_loss(x, y)
        val_loss = val_loss / len(val_loader.dataset)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            pyro.get_param_store().save(cfg.param_save_path)
            torch.save(model.state_dict(), cfg.model_save_path)


def evaluate(cfg: ProjectConfig):
    pyro.clear_param_store()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, scaler = get_data_loaders(
        cfg.data_path,
        cfg.column_name,
        cfg.train.look_back,
        cfg.train.horizon,
        batch_size=1,  # Batch size 1 for viz
        train_split=cfg.train.train_split,
        val_split=cfg.train.val_split,
    )

    model = BayesianLSTM(
        cfg.model.input_size, cfg.model.hidden_size, cfg.train.horizon
    ).to(device)

    model.load_state_dict(torch.load(cfg.model_save_path, map_location=device))

    guide = AutoDiagonalNormal(pyro.poutine.block(model, hide=["obs"]))

    saved_params = torch.load(cfg.param_save_path, weights_only=False)
    pyro.get_param_store().set_state(saved_params)

    x_test, y_test = next(iter(test_loader))
    x_test = x_test.to(device)

    predictive = Predictive(
        model,
        guide=guide,
        num_samples=100,
        return_sites=("linear_mu", "linear_sigma", "obs"),
    )
    preds = predictive(x_test)

    mus = preds["linear_mu"].detach().cpu().numpy().squeeze(1)
    sigmas_pre = preds["linear_sigma"].detach().cpu().numpy().squeeze(1)
    sigmas = np.log(1 + np.exp(sigmas_pre))

    epistemic_var = np.var(mus, axis=0)
    aleatoric_var = np.mean(sigmas**2, axis=0)
    total_var = epistemic_var + aleatoric_var

    scale = scaler.scale_[0]
    mean_mw = scaler.inverse_transform(np.mean(mus, axis=0).reshape(-1, 1)).flatten()
    epi_std_mw = np.sqrt(epistemic_var).flatten() * scale
    total_std_mw = np.sqrt(total_var).flatten() * scale

    hist_mw = scaler.inverse_transform(x_test[0].reshape(-1, 1)).flatten()
    true_mw = scaler.inverse_transform(y_test[0].reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(true_mw, mean_mw))
    mae = mean_absolute_error(true_mw, mean_mw)

    lower_bound = mean_mw - 2 * total_std_mw
    upper_bound = mean_mw + 2 * total_std_mw

    inside_interval = ((true_mw >= lower_bound) & (true_mw <= upper_bound)).mean()

    print("-" * 30)
    print(f"Model Evaluation Results:")
    print(f"1. Accuracy:")
    print(f"   RMSE: {rmse:.2f} MW (deviation from target)")
    print(f"   MAE:  {mae:.2f} MW")
    print(f"2. Uncertainty Calibration:")
    print(f"   95% Coverage: {inside_interval * 100:.2f}% (Target: ~95%)")
    print("-" * 30)

    x_test = x_test.cpu()
    plot_uncertainty_decomposition(hist_mw, true_mw, mean_mw, epi_std_mw, total_std_mw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian LSTM Forecasting")

    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    parser.add_argument("--look_back", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)

    args = parser.parse_args()

    config = ProjectConfig()

    if args.look_back is not None:
        config.train.look_back = args.look_back

    if args.horizon is not None:
        config.train.horizon = args.horizon

    lb = config.train.look_back
    hz = config.train.horizon

    base_path = Path.Path("checkpoints")
    config.model_save_path = str(base_path / f"best_model_lb{lb}_h{hz}.pt")
    config.param_save_path = str(base_path / f"best_params_lb{lb}_h{hz}.pyro")

    if args.mode == "train":
        train(config)
    elif args.mode == "eval":
        evaluate(config)
