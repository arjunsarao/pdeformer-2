r"""This module provides the PDEformer inference model factory."""

from pathlib import Path

from omegaconf import DictConfig
import torch
from torch import nn

from .pdeformer import PDEformer
from ..utils.tools import calculate_num_params


def _normalize_checkpoint_key(key: str) -> str:
    r"""Map common MindSpore checkpoint parameter names to PyTorch names."""
    key = key.removeprefix("module.").removeprefix("model.")
    if key.endswith(".embedding_table"):
        return key.removesuffix(".embedding_table") + ".weight"

    layer_norm_names = ("attn_layer_norm", "ffn_layer_norm", "emb_layer_norm")
    if any(f".{name}." in key for name in layer_norm_names):
        if key.endswith(".gamma"):
            return key.removesuffix(".gamma") + ".weight"
        if key.endswith(".beta"):
            return key.removesuffix(".beta") + ".bias"

    return key


def _to_torch_tensor(value):
    r"""Convert supported checkpoint array values to torch tensors."""
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


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
        ckpt_path = Path(load_ckpt)
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Configured checkpoint does not exist: {load_ckpt}. "
                "The PyTorch notebook needs a converted .pt state_dict. "
                "If you only have a native MindSpore .ckpt file, convert it "
                "to a PyTorch state_dict first, then update model.load_ckpt."
            )
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
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
            _normalize_checkpoint_key(key): _to_torch_tensor(value)
            for key, value in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            warning_str = ("WARNING: These model parameters are not loaded: "
                           + str(missing)
                           + "\nWARNING: These checkpoint parameters are not loaded: "
                           + str(unexpected))
            print(warning_str)
    elif not config.model.get("allow_untrained", False):
        raise ValueError(
            "No checkpoint was configured. Set model.load_ckpt to a converted "
            "PyTorch state_dict path before running inference, or set "
            "model.allow_untrained: true only for architecture/debug checks. "
            "Running an untrained model produces misleading repeated-looking plots."
        )

    model.to(dtype=compute_dtype)
    model.eval()
    print(f"model_type: {config.model_type}, num_parameters: "
          + calculate_num_params(model))

    return model
