import matplotlib.pyplot as plt
import numpy as np


def plot_uncertainty_decomposition(
    history: np.ndarray,
    true_future: np.ndarray,
    pred_mean: np.ndarray,
    epistemic_std: np.ndarray,
    total_std: np.ndarray,
    save_path: str = "forecast_plot.jpg",
):
    """
    Plots the forecast with separated epistemic and aleatoric uncertainty intervals.
    """
    plt.figure(figsize=(12, 6))

    look_back = len(history)
    horizon = len(true_future)

    # Time axes
    t_hist = range(0, look_back)
    t_future = range(look_back, look_back + horizon)

    # Connecting lines for visual continuity
    t_future_conn = [t_hist[-1]] + list(t_future)
    pred_mean_conn = [history[-1]] + list(pred_mean)
    true_future_conn = [history[-1]] + list(true_future)

    # Plotting
    plt.plot(t_hist, history, "k:", label="History")
    plt.plot(t_future_conn, true_future_conn, "g--", linewidth=2, label="Ground Truth")
    plt.plot(
        t_future_conn, pred_mean_conn, "navy", linewidth=2, label="Prediction (Mean)"
    )

    # Epistemic Uncertainty (Model Knowledge)
    upper_epi = np.concatenate(([pred_mean_conn[0]], pred_mean + 2 * epistemic_std))
    lower_epi = np.concatenate(([pred_mean_conn[0]], pred_mean - 2 * epistemic_std))

    plt.fill_between(
        t_future_conn,
        lower_epi,
        upper_epi,
        color="teal",
        alpha=0.6,
        label="Epistemic Uncertainty (Model)",
    )

    # Total Uncertainty (Aleatoric + Epistemic)
    upper_total = np.concatenate(([pred_mean_conn[0]], pred_mean + 2 * total_std))
    lower_total = np.concatenate(([pred_mean_conn[0]], pred_mean - 2 * total_std))

    plt.fill_between(
        t_future_conn,
        lower_epi,
        lower_total,
        color="lightblue",
        alpha=0.4,
        label="Aleatoric Uncertainty (Data Noise)",
    )
    plt.fill_between(
        t_future_conn, upper_epi, upper_total, color="lightblue", alpha=0.4
    )

    plt.title("Probabilistic Forecast: Uncertainty Decomposition")
    plt.xlabel("Time Steps")
    plt.ylabel("Load (MW)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.savefig(f"images/{save_path}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to images/{save_path}")