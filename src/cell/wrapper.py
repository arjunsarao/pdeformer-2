r"""This module provides the PDEformer inference model factory."""

from omegaconf import DictConfig
import mindspore as ms
from mindspore import nn, context
from mindspore import dtype as mstype

from .pdeformer import PDEformer
from ..utils.tools import calculate_num_params


def get_model(config: DictConfig,
              compute_dtype=None) -> nn.Cell:
    r"""Create a PDEformer model and load its inference checkpoint."""
    if compute_dtype is None:  # set automatically
        if context.get_context(attr_key='device_target') == "Ascend":
            compute_dtype = mstype.float16
        else:
            compute_dtype = mstype.float32

    if config.model_type != "pdeformer":
        raise ValueError(
            "This inference-only build only supports model_type 'pdeformer'.")

    model = PDEformer(config.model, compute_dtype=compute_dtype)

    # load pre-trained model weights
    load_ckpt = config.model.get("load_ckpt", "none")
    if load_ckpt.lower() != "none":
        param_dict = ms.load_checkpoint(load_ckpt)
        param_not_load, checkpoint_not_load = ms.load_param_into_net(model, param_dict)
        if param_not_load or checkpoint_not_load:  # either list is non-empty
            warning_str = ("WARNING: These model parameters are not loaded: "
                           + str(param_not_load)
                           + "\nWARNING: These checkpoint parameters are not loaded: "
                           + str(checkpoint_not_load))
            print(warning_str)

    print(f"model_type: {config.model_type}, num_parameters: "
          + calculate_num_params(model))

    return model
