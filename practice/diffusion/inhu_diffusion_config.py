from typing import TypedDict
from pathlib import Path
import torch

class DiffusionModelConfig(TypedDict):
    TOTAL_TIME: int
    TIME_DIM: int
    CHANNEL: list[int]
    MODEL_NAME: str


TRAINED_MODEL_ROOT = Path("trained_model")


def get_model_save_dir(config: DiffusionModelConfig) -> Path:
    """Return the checkpoint directory assigned to a diffusion config."""
    return TRAINED_MODEL_ROOT / config["MODEL_NAME"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIFFUSION_MODEL128_CONFIG: DiffusionModelConfig = {
    "TOTAL_TIME": 4000,
    "TIME_DIM": 256,
    "CHANNEL": [64, 128, 256, 384, 512, 1024],
    "MODEL_NAME": "model_128_1024",
}

DIFFUSION_MODEL128_CONFIG: DiffusionModelConfig = {
    "TOTAL_TIME": 4000,
    "TIME_DIM": 256,
    "CHANNEL": [64, 128, 256, 384, 512, 1024],
    "MODEL_NAME": "model_128_1024",
}

DIFFUSION_MODEL_CONFIG: DiffusionModelConfig = {
    "TOTAL_TIME": 4000,
    "TIME_DIM": 256,
    "CHANNEL": [32, 64, 128, 256, 512, 512],
    "MODEL_NAME": "model_small",
}
