# after saved model
from pathlib import Path
import torch

from practice.diffusion.inhu_diffusion_config import (
    DEVICE,
    DIFFUSION_MODEL128_CONFIG,
    get_model_save_dir,
)
from practice.diffusion.inhu_diffusijon_model import InhuDiffusionCheckpoint, InhuDiffusionModel
from practice.diffusion.inhu_res_unet import InhuResUnet

device = DEVICE
save_dir = get_model_save_dir(DIFFUSION_MODEL128_CONFIG)

def build_model(
    time_dim: int = DIFFUSION_MODEL128_CONFIG["TIME_DIM"],
    total_timestep: int = DIFFUSION_MODEL128_CONFIG["TOTAL_TIME"],
):
    unet = InhuResUnet(
        time_dim=time_dim,
        channels=DIFFUSION_MODEL128_CONFIG["CHANNEL"],
        attention_start_idx=3,
    ).to(device)

    model = InhuDiffusionModel(
        unet=unet,
        time_dim=time_dim,
        total_timestep=total_timestep,
    ).to(device)

    return model


def create_dummy_checkpoint(
    path,
    time_dim: int = DIFFUSION_MODEL128_CONFIG["TIME_DIM"],
    total_timestep: int = DIFFUSION_MODEL128_CONFIG["TOTAL_TIME"],
):
    model = build_model(time_dim, total_timestep)

    checkpoint: InhuDiffusionCheckpoint = {
        "epoch": -1,
        "best_epoch": -1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "best_val_loss": float("inf"),
        "history": {
            "train_loss": [],
            "val_loss": [],
        },
        "total_timestep": total_timestep,
        "time_dim": time_dim,
    }

    torch.save(checkpoint, path)
    print("Created dummy checkpoint:", path)

    return checkpoint, model


def load_best_model(checkpoint_dir: str | Path = save_dir):
    path = Path(checkpoint_dir) / "best_model.pth"
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        return create_dummy_checkpoint(path)

    checkpoint: InhuDiffusionCheckpoint = torch.load(
        path,
        map_location=device
    )

    model = build_model(
        time_dim=checkpoint["time_dim"],
        total_timestep=checkpoint["total_timestep"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded best model")
    print("current epoch idx:", checkpoint["epoch"])
    print("best epoch idx:", checkpoint["best_epoch"])
    print("Best val loss:", checkpoint["best_val_loss"])

    return checkpoint, model

def load_latest_model(checkpoint_dir: str | Path = save_dir):
    path = Path(checkpoint_dir) / "last_model.pth"
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        return create_dummy_checkpoint(path)

    checkpoint: InhuDiffusionCheckpoint = torch.load(
        path,
        map_location=device
    )

    model = build_model(
        time_dim=checkpoint["time_dim"],
        total_timestep=checkpoint["total_timestep"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded last model")
    print("current epoch idx:", checkpoint["epoch"])
    print("best epoch idx:", checkpoint["best_epoch"])
    print("Best val loss:", checkpoint["best_val_loss"])

    return checkpoint, model

def load_target_epoch_model(checkpoint_dir: str | Path = save_dir, target_epoch:int = 0):
    path = Path(checkpoint_dir) / f"model_epoch_{target_epoch}.pth"
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        return create_dummy_checkpoint(path)

    checkpoint: InhuDiffusionCheckpoint = torch.load(
        path,
        map_location=device
    )

    model = build_model(
        time_dim=checkpoint["time_dim"],
        total_timestep=checkpoint["total_timestep"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model_epoch_{target_epoch}.pth")
    print("current epoch idx:", checkpoint["epoch"])
    print("best epoch idx:", checkpoint["best_epoch"])
    print("Best val loss:", checkpoint["best_val_loss"])

    return checkpoint, model

