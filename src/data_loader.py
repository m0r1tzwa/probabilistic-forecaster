import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def load_preprocess_series(filepath: str, location_col: str) -> pd.Series:
    df = pd.read_csv(filepath)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")
    df = df.sort_index()

    if location_col not in df.columns:
        raise ValueError(f"Column '{location_col}' not found in data.")

    series = df[location_col].rename("MW")
    series = series.interpolate(method="linear")
    series = series.bfill().ffill()

    return series


def create_supervised_dataset(
    data: np.ndarray, look_back: int, prediction_horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    stop_index = len(data) - look_back - prediction_horizon + 1

    for i in range(stop_index):
        X.append(data[i : (i + look_back)])
        y.append(data[(i + look_back) : (i + look_back + prediction_horizon)])

    X = np.array(X).reshape(-1, look_back, 1)
    y = np.array(y).reshape(-1, prediction_horizon)

    return X, y


def get_data_loaders(
    filepath: str,
    location_col: str,
    look_back: int,
    horizon: int,
    batch_size: int,
    train_split: float = 0.7,
    val_split: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:

    # 1. Data loading
    series = load_preprocess_series(filepath, location_col)

    # 2. Data splitting
    n = len(series)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    train_data = series.iloc[:train_end].values.reshape(-1, 1)
    val_data = series.iloc[train_end:val_end].values.reshape(-1, 1)
    test_data = series.iloc[val_end:].values.reshape(-1, 1)

    # 3. Normalization
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # 4. Windowed datasets
    X_train, y_train = create_supervised_dataset(train_scaled, look_back, horizon)
    X_val, y_val = create_supervised_dataset(val_scaled, look_back, horizon)
    X_test, y_test = create_supervised_dataset(test_scaled, look_back, horizon)

    # 5. Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train).float(), torch.tensor(y_train).float()
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val).float(), torch.tensor(y_val).float()
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test).float(), torch.tensor(y_test).float()
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Data loaded: {len(X_train)} training samples.")
    print(f"Input-Form (X): (Batch, {look_back}, 1)")
    print(f"Output-Form (y): (Batch, {horizon})")

    return train_loader, val_loader, test_loader, scaler
