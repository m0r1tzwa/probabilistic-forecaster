from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    input_size: int = 1
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    batch_size: int = 1024
    learning_rate: float = 0.001
    num_epochs: int = 200
    look_back: int = 48
    horizon: int = 24
    train_split: float = 0.7
    val_split: float = 0.15
    seed: int = 42
    clip_norm: float = 10.0


@dataclass
class ProjectConfig:
    data_path: str = "data/AEP_hourly.csv"
    column_name: str = "AEP_MW"
    model_save_path: str = "checkpoints/best_model.pt"
    param_save_path: str = "checkpoints/best_params.pyro"

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
