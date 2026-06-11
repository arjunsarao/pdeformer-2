r"""This module provides the PDEformer inference model factory."""

from omegaconf import DictConfig
import torch
from torch import nn

from .pdeformer import PDEformer
from ..utils.tools import calculate_num_params


def get_model(config: DictConfig,
              compute_dtype=None) -> nn.Module:
    r"""Create a PDEformer model and load its inference checkpoint."""
    if compute_dtype is None:  # set automatically
        compute_dtype = torch.float32

    if config.model_type != "pdeformer":
        raise ValueError(
            "This inference-only build only supports model_type 'pdeformer'.")

    model = PDEformer(config.model, compute_dtype=compute_dtype)

    # load pre-trained model weights
    load_ckpt = str(config.model.get("load_ckpt", "none"))
    if load_ckpt.lower() != "none":
        try:
            checkpoint = torch.load(load_ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(load_ckpt, map_location="cpu")
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the checkpoint with PyTorch. This build expects "
                "a torch checkpoint/state_dict; native MindSpore .ckpt files "
                "must be converted before loading."
            ) from exc

        if isinstance(checkpoint, dict):
            state_dict = (checkpoint.get("state_dict")
                          or checkpoint.get("model_state_dict")
                          or checkpoint.get("model")
                          or checkpoint)
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, dict):
            raise TypeError(
                "Checkpoint must be a torch state_dict or a dict containing one.")

        state_dict = {
            key.removeprefix("module.").removeprefix("model."): value
            for key, value in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            warning_str = ("WARNING: These model parameters are not loaded: "
                           + str(missing)
                           + "\nWARNING: These checkpoint parameters are not loaded: "
                           + str(unexpected))
            print(warning_str)

    model.to(dtype=compute_dtype)
    model.eval()
    print(f"model_type: {config.model_type}, num_parameters: "
          + calculate_num_params(model))

    return model
