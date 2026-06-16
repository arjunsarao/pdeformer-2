r"""PyTorch-backed compatibility helpers for the original MindSpore code.

The project was initially written against MindSpore's ``nn.Cell``/``ops`` API.
This module keeps that small dialect available while the implementation runs on
PyTorch tensors, modules, optimizers, and dataloaders.
"""
from __future__ import annotations

import math
import random
import types
from typing import Any, Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as torch_f
from torch import nn as torch_nn
from torch.utils.data import DataLoader, Dataset as TorchDataset


__version__ = torch.__version__

float16 = torch.float16
float32 = torch.float32
float64 = torch.float64
int32 = torch.int32
int64 = torch.int64
bool_ = torch.bool


class _DTypeNamespace:
    float16 = torch.float16
    float32 = torch.float32
    float64 = torch.float64
    int32 = torch.int32
    int64 = torch.int64
    bool_ = torch.bool


dtype = _DTypeNamespace()


def _canonical_dtype(dt):
    return dt


def _as_tensor(value, dtype=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor
    return torch.as_tensor(value, dtype=_canonical_dtype(dtype))


class Zero:
    pass


class One:
    pass


class Uniform:
    def __init__(self, scale: float = 0.07):
        self.scale = scale


class HeUniform:
    def __init__(self, negative_slope: float = 0.0):
        self.negative_slope = negative_slope


class XavierUniform:
    def __init__(self, gain: float = 1.0):
        self.gain = gain


class Normal:
    def __init__(self, sigma: float = 0.01, mean: float = 0.0):
        self.sigma = sigma
        self.mean = mean


def initializer(init, shape, dtype=torch.float32):
    r"""Create an initialized torch tensor."""
    if isinstance(shape, torch.Size):
        shape = tuple(shape)
    if isinstance(shape, int):
        shape = (shape,)

    if init is None:
        out = torch.empty(shape, dtype=dtype)
        torch_nn.init.kaiming_uniform_(out, a=math.sqrt(5)) if out.ndim >= 2 else torch_nn.init.zeros_(out)
        return out
    if isinstance(init, torch.Tensor):
        return init.detach().clone().to(dtype=dtype)
    if isinstance(init, np.ndarray):
        return torch.as_tensor(init, dtype=dtype)
    if isinstance(init, (int, float, np.number)):
        return torch.full(shape, float(init), dtype=dtype)
    if isinstance(init, str):
        name = init.lower()
        if name in {"zero", "zeros"}:
            return torch.zeros(shape, dtype=dtype)
        if name in {"one", "ones"}:
            return torch.ones(shape, dtype=dtype)
        if name in {"heuniform", "he_uniform"}:
            out = torch.empty(shape, dtype=dtype)
            torch_nn.init.kaiming_uniform_(out, a=math.sqrt(5))
            return out
        raise ValueError(f"Unsupported initializer string: {init}")
    if isinstance(init, Zero):
        return torch.zeros(shape, dtype=dtype)
    if isinstance(init, One):
        return torch.ones(shape, dtype=dtype)
    if isinstance(init, Uniform):
        return torch.empty(shape, dtype=dtype).uniform_(-init.scale, init.scale)
    if isinstance(init, HeUniform):
        out = torch.empty(shape, dtype=dtype)
        torch_nn.init.kaiming_uniform_(out, a=init.negative_slope)
        return out
    if isinstance(init, XavierUniform):
        out = torch.empty(shape, dtype=dtype)
        torch_nn.init.xavier_uniform_(out, gain=init.gain)
        return out
    if isinstance(init, Normal):
        return torch.empty(shape, dtype=dtype).normal_(mean=init.mean, std=init.sigma)
    raise TypeError(f"Unsupported initializer type: {type(init)!r}")


def Tensor(input_data=None, dtype=None, shape=None, init=None):
    r"""MindSpore-style tensor constructor."""
    if shape is not None:
        return initializer(init if init is not None else Zero(), shape, dtype or torch.float32)
    if input_data is None:
        return torch.tensor([], dtype=dtype or torch.float32)
    if isinstance(input_data, torch.nn.Parameter):
        return input_data.detach().clone().to(dtype=dtype or input_data.dtype)
    if isinstance(input_data, torch.Tensor):
        return input_data.detach().clone().to(dtype=dtype or input_data.dtype)
    return torch.as_tensor(input_data, dtype=dtype)


def Parameter(default_input, name: Optional[str] = None, requires_grad: bool = True):
    param = torch_nn.Parameter(Tensor(default_input).detach().clone(), requires_grad=requires_grad)
    if name is not None:
        _set_param_name(param, name)
    return param


def _set_param_name(param: torch.Tensor, name: str) -> None:
    try:
        setattr(param, "_compat_name", name)
    except Exception:
        pass


def param_name(param: torch.Tensor) -> str:
    return getattr(param, "_compat_name", "")


def assign_parameter_names(module: torch_nn.Module) -> None:
    for name, param in module.named_parameters():
        _set_param_name(param, name)


def _tensor_asnumpy(self):
    return self.detach().cpu().numpy()


def _tensor_astype(self, dtype):
    return self.to(dtype=dtype)


def _tensor_expand_dims(self, axis):
    return self.unsqueeze(axis)


def _tensor_copy(self):
    return self.detach().clone()


def _tensor_set_data(self, data):
    data_tensor = _as_tensor(data, dtype=self.dtype).to(device=self.device)
    with torch.no_grad():
        self.data.copy_(data_tensor.reshape_as(self.data))


_ORIG_REPEAT = torch.Tensor.repeat
_ORIG_TRANSPOSE = torch.Tensor.transpose
_ORIG_VIEW = torch.Tensor.view


def _tensor_repeat(self, *sizes, axis=None):
    if axis is not None:
        return torch.repeat_interleave(self, repeats=int(sizes[0]), dim=axis)
    if len(sizes) == 2 and all(isinstance(item, int) for item in sizes):
        repeats, dim = sizes
        if -self.ndim <= dim < self.ndim:
            return torch.repeat_interleave(self, repeats=int(repeats), dim=dim)
    return _ORIG_REPEAT(self, *sizes)


def _tensor_transpose(self, *dims):
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        return self.permute(*dims[0])
    if len(dims) > 2:
        return self.permute(*dims)
    return _ORIG_TRANSPOSE(self, *dims)


def _tensor_view(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
        shape = tuple(shape[0])
    return self.reshape(*shape)


for _name, _fn in {
        "asnumpy": _tensor_asnumpy,
        "astype": _tensor_astype,
        "expand_dims": _tensor_expand_dims,
        "copy": _tensor_copy,
        "set_data": _tensor_set_data,
        "repeat": _tensor_repeat,
        "transpose": _tensor_transpose,
        "view": _tensor_view,
}.items():
    setattr(torch.Tensor, _name, _fn)


def _module_to_float(self, dtype=torch.float32):
    return self.to(dtype=dtype)


def _module_set_train(self, mode: bool = True):
    return self.train(mode)


def _module_trainable_params(self):
    assign_parameter_names(self)
    return [param for param in self.parameters() if param.requires_grad]


def _module_get_parameters(self):
    assign_parameter_names(self)
    return list(self.parameters())


def _module_parameters_dict(self):
    assign_parameter_names(self)
    return dict(self.named_parameters())


def _module_update_parameters_name(self):
    assign_parameter_names(self)


torch_nn.Module.to_float = _module_to_float
torch_nn.Module.set_train = _module_set_train
torch_nn.Module.trainable_params = _module_trainable_params
torch_nn.Module.get_parameters = _module_get_parameters
torch_nn.Module.parameters_dict = _module_parameters_dict
torch_nn.Module.update_parameters_name = _module_update_parameters_name
torch_nn.Module._cells = property(lambda self: self._modules)


class Cell(torch_nn.Module):
    def construct(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.construct is not implemented")

    def forward(self, *args, **kwargs):
        return self.construct(*args, **kwargs)


class SequentialCell(Cell):
    def __init__(self, *cells):
        super().__init__()
        if len(cells) == 1 and isinstance(cells[0], (list, tuple)):
            cells = tuple(cells[0])
        for cell in cells:
            self.append(cell)

    def append(self, cell):
        self.add_module(str(len(self._modules)), cell)

    def __iter__(self) -> Iterator[torch_nn.Module]:
        return iter(self._modules.values())

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, idx):
        return list(self._modules.values())[idx]

    def construct(self, x):
        for cell in self._modules.values():
            x = cell(x)
        return x


CellList = torch_nn.ModuleList


class Dense(Cell):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            has_bias: bool = True,
            weight_init=None,
            bias_init=None,
            activation=None,
            **_: Any) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_bias = has_bias
        self.weight = Parameter(initializer(weight_init or HeUniform(math.sqrt(5)),
                                           (out_channels, in_channels),
                                           dtype=torch.float32))
        if has_bias:
            if bias_init is None:
                bound = 1 / math.sqrt(in_channels) if in_channels > 0 else 0
                bias_value = initializer(Uniform(bound), (out_channels,), dtype=torch.float32)
            else:
                bias_value = initializer(bias_init, (out_channels,), dtype=torch.float32)
            self.bias = Parameter(bias_value)
        else:
            self.register_parameter("bias", None)
        self.activation = _activation_from_any(activation)
        self.activation_flag = self.activation is not None
        self.shape_op = lambda x: x.shape
        self.reshape = lambda x, shape: x.reshape(shape)
        self.matmul = MatMul(transpose_b=True)
        self.bias_add = lambda x, b: x + b

    def construct(self, x):
        out = torch_f.linear(x, self.weight, self.bias)
        if self.activation_flag:
            out = self.activation(out)
        return out


def _same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        return tuple(k // 2 for k in kernel_size)
    return kernel_size // 2


def _default_padding(kernel_size, stride, pad_mode):
    if pad_mode == "valid":
        return 0
    if stride == kernel_size:
        return 0
    if isinstance(kernel_size, tuple) and isinstance(stride, tuple) and stride == kernel_size:
        return tuple(0 for _ in kernel_size)
    return _same_padding(kernel_size)


def _init_weight_bias(module, weight_init=None, bias_init=None):
    if weight_init is not None:
        module.weight.set_data(initializer(weight_init, module.weight.shape, module.weight.dtype))
    if module.bias is not None and bias_init is not None:
        module.bias.set_data(initializer(bias_init, module.bias.shape, module.bias.dtype))


class Conv1d(torch_nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode="same",
                 padding=None, has_bias=True, weight_init=None, bias_init=None, **kwargs):
        if padding is None:
            padding = _default_padding(kernel_size, stride, pad_mode)
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, bias=has_bias, **kwargs)
        _init_weight_bias(self, weight_init, bias_init)


class Conv2d(torch_nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode="same",
                 padding=None, has_bias=True, weight_init=None, bias_init=None, **kwargs):
        if padding is None:
            padding = _default_padding(kernel_size, stride, pad_mode)
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, bias=has_bias, **kwargs)
        _init_weight_bias(self, weight_init, bias_init)


class Conv3d(torch_nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode="same",
                 padding=None, has_bias=True, weight_init=None, bias_init=None, **kwargs):
        if padding is None:
            padding = _default_padding(kernel_size, stride, pad_mode)
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, bias=has_bias, **kwargs)
        _init_weight_bias(self, weight_init, bias_init)


class Embedding(torch_nn.Embedding):
    @property
    def embedding_table(self):
        return self.weight


class LayerNorm(torch_nn.LayerNorm):
    def __init__(self, normalized_shape, epsilon=1e-5, **kwargs):
        super().__init__(normalized_shape, eps=epsilon, **kwargs)


class FastGelu(torch_nn.GELU):
    pass


class HSwish(torch_nn.Hardswish):
    pass


class HSigmoid(torch_nn.Hardsigmoid):
    pass


class ReLU6(torch_nn.ReLU6):
    pass


class LogSoftmax(torch_nn.LogSoftmax):
    def __init__(self, axis=-1):
        super().__init__(dim=axis)


class Softmax(torch_nn.Softmax):
    def __init__(self, axis=-1):
        super().__init__(dim=axis)


def _activation_from_any(activation_obj):
    if activation_obj is None:
        return None
    if isinstance(activation_obj, str):
        return get_activation(activation_obj)
    if isinstance(activation_obj, type):
        return activation_obj()
    return activation_obj


def get_activation(name):
    if name is None:
        return None
    table = {
        "softmax": Softmax,
        "logsoftmax": LogSoftmax,
        "relu": torch_nn.ReLU,
        "relu6": ReLU6,
        "tanh": torch_nn.Tanh,
        "gelu": torch_nn.GELU,
        "fast_gelu": FastGelu,
        "elu": torch_nn.ELU,
        "sigmoid": torch_nn.Sigmoid,
        "prelu": torch_nn.PReLU,
        "leakyrelu": torch_nn.LeakyReLU,
        "hswish": HSwish,
        "hsigmoid": HSigmoid,
        "logsigmoid": torch_nn.LogSigmoid,
        "identity": torch_nn.Identity,
    }
    return table[name.lower()]()


activation = types.SimpleNamespace(
    Softmax=Softmax,
    LogSoftmax=LogSoftmax,
    ReLU=torch_nn.ReLU,
    ReLU6=ReLU6,
    Tanh=torch_nn.Tanh,
    GELU=torch_nn.GELU,
    FastGelu=FastGelu,
    ELU=torch_nn.ELU,
    Sigmoid=torch_nn.Sigmoid,
    PReLU=torch_nn.PReLU,
    LeakyReLU=torch_nn.LeakyReLU,
    HSwish=HSwish,
    HSigmoid=HSigmoid,
    LogSigmoid=torch_nn.LogSigmoid,
    get_activation=get_activation,
)


class Cast:
    def __call__(self, x, dst_type):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dst_type)
        return torch.as_tensor(x, dtype=dst_type)


class Mul:
    def __call__(self, a, b):
        return a * b


class Add:
    def __call__(self, a, b):
        return a + b


class Sub:
    def __call__(self, a, b):
        return a - b


class RealDiv:
    def __call__(self, a, b):
        return a / b


class Square:
    def __call__(self, x):
        return torch.square(x)


class Sqrt:
    def __call__(self, x):
        return torch.sqrt(_as_tensor(x))


class Abs:
    def __call__(self, x):
        return torch.abs(x)


class Identity(Cell):
    def construct(self, x):
        return x


class ReduceSum:
    def __init__(self, keep_dims=False):
        self.keep_dims = keep_dims

    def __call__(self, x, axis=None):
        return torch.sum(x, dim=axis, keepdim=self.keep_dims)


class ReduceMean:
    def __init__(self, keep_dims=False):
        self.keep_dims = keep_dims

    def __call__(self, x, axis=None):
        return torch.mean(x, dim=axis, keepdim=self.keep_dims)


class Concat:
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, tensors):
        return torch.cat(tuple(tensors), dim=self.axis)


class MatMul:
    def __init__(self, transpose_b=False):
        self.transpose_b = transpose_b

    def __call__(self, a, b):
        if self.transpose_b:
            b = b.transpose(-1, -2)
        return torch.matmul(a, b)


class Sin(Cell):
    def construct(self, x):
        return torch.sin(x)


class AllGather(Cell):
    def construct(self, x):
        return x


def _maybe_tensor(value, like: Optional[torch.Tensor] = None):
    if isinstance(value, torch.Tensor):
        return value
    dtype_ = like.dtype if like is not None and like.is_floating_point() else None
    device = like.device if like is not None else None
    return torch.as_tensor(value, dtype=dtype_, device=device)


def _sum(x, axis=None, dim=None, keep_dims=False, keepdim=False):
    return torch.sum(x, dim=dim if dim is not None else axis, keepdim=keep_dims or keepdim)


def _mean(x, axis=None, dim=None, keep_dims=False, keepdim=False):
    return torch.mean(x, dim=dim if dim is not None else axis, keepdim=keep_dims or keepdim)


def _nansum(x, axis=None, dim=None, keep_dims=False, keepdim=False):
    return torch.nansum(x, dim=dim if dim is not None else axis, keepdim=keep_dims or keepdim)


def _nanmean(x, axis=None, dim=None, keep_dims=False, keepdim=False):
    return torch.nanmean(x, dim=dim if dim is not None else axis, keepdim=keep_dims or keepdim)


def _maximum(a, b):
    return torch.maximum(_maybe_tensor(a, b if isinstance(b, torch.Tensor) else None),
                         _maybe_tensor(b, a if isinstance(a, torch.Tensor) else None))


def _where(cond, x, y):
    like = y if isinstance(y, torch.Tensor) else x if isinstance(x, torch.Tensor) else None
    return torch.where(cond, _maybe_tensor(x, like), _maybe_tensor(y, like))


def _transpose(x, input_perm=None, axes=None):
    perm = input_perm if input_perm is not None else axes
    return x.permute(*perm)


def _interpolate(x, size=None, scale_factor=None, mode="nearest"):
    kwargs = {}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    return torch_f.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, **kwargs)


def _conv2d(x, weight, bias=None, pad_mode="valid", stride=1, padding=0, dilation=1, groups=1):
    if pad_mode == "same":
        padding = _same_padding(weight.shape[-2:])
    return torch_f.conv2d(x, weight, bias=bias, stride=stride, padding=padding,
                          dilation=dilation, groups=groups)


ops = types.SimpleNamespace(
    Cast=Cast,
    Mul=Mul,
    Add=Add,
    Sub=Sub,
    RealDiv=RealDiv,
    Square=Square,
    Sqrt=Sqrt,
    Abs=Abs,
    Identity=Identity,
    ReduceSum=ReduceSum,
    ReduceMean=ReduceMean,
    Concat=Concat,
    MatMul=MatMul,
    Sin=Sin,
    AllGather=AllGather,
    cast=Cast(),
    sum=_sum,
    mean=_mean,
    nansum=_nansum,
    nanmean=_nanmean,
    matmul=torch.matmul,
    bmm=torch.bmm,
    zeros_like=torch.zeros_like,
    ones=lambda shape, dtype=torch.float32: torch.ones(shape, dtype=dtype),
    sin=torch.sin,
    cos=torch.cos,
    exp=torch.exp,
    sqrt=lambda x: torch.sqrt(_as_tensor(x)),
    abs=torch.abs,
    square=torch.square,
    softmax=lambda x, axis=-1, dim=None: torch.softmax(x, dim=axis if dim is None else dim),
    concat=lambda tensors, axis=0, dim=None: torch.cat(tuple(tensors), dim=axis if dim is None else dim),
    cat=lambda tensors, axis=0, dim=None: torch.cat(tuple(tensors), dim=axis if dim is None else dim),
    stack=lambda tensors, axis=0, dim=None: torch.stack(tuple(tensors), dim=axis if dim is None else dim),
    transpose=_transpose,
    diff=lambda x, axis=-1, n=1: torch.diff(x, n=n, dim=axis),
    select=lambda cond, x, y: torch.where(cond, x, y),
    maximum=_maximum,
    where=_where,
    equal=torch.eq,
    clamp=torch.clamp,
    flip=lambda x, axis: torch.flip(x, dims=(axis,)),
    unsqueeze=lambda x, dim: x.unsqueeze(dim),
    interpolate=_interpolate,
    conv2d=_conv2d,
)

numpy = types.SimpleNamespace(
    flip=lambda x, axis: torch.flip(x, dims=(axis,)),
)

operations = ops
functional = types.SimpleNamespace(
    dtype=lambda x: x.dtype,
    square=torch.square,
    sqrt=lambda x: torch.sqrt(_as_tensor(x)),
    abs=torch.abs,
)


nn = types.SimpleNamespace(
    Cell=Cell,
    SequentialCell=SequentialCell,
    CellList=CellList,
    Dense=Dense,
    Conv1d=Conv1d,
    Conv2d=Conv2d,
    Conv3d=Conv3d,
    BatchNorm2d=torch_nn.BatchNorm2d,
    ReLU=torch_nn.ReLU,
    ReLU6=ReLU6,
    GELU=torch_nn.GELU,
    Dropout=torch_nn.Dropout,
    LayerNorm=LayerNorm,
    Embedding=Embedding,
    LeakyReLU=torch_nn.LeakyReLU,
    Identity=torch_nn.Identity,
    ConstantPad1d=torch_nn.ConstantPad1d,
    MaxPool2d=torch_nn.MaxPool2d,
    AdaptiveAvgPool2d=torch_nn.AdaptiveAvgPool2d,
    Flatten=torch_nn.Flatten,
    Adam=torch.optim.Adam,
)


class _Context:
    GRAPH_MODE = 0
    PYNATIVE_MODE = 1

    def __init__(self):
        self._values = {"device_target": "CPU", "mode": self.PYNATIVE_MODE}

    def set_context(self, **kwargs):
        self._values.update(kwargs)

    def get_context(self, attr_key=None):
        if attr_key is None:
            return dict(self._values)
        return self._values.get(attr_key)

    def set_auto_parallel_context(self, **kwargs):
        self._values.update({f"parallel_{key}": value for key, value in kwargs.items()})


context = _Context()


class ParallelMode:
    DATA_PARALLEL = "data_parallel"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jit(fn=None, **_):
    if fn is None:
        return lambda inner: inner
    return fn


class DynamicLossScaler:
    def __init__(self, *_, **__):
        pass

    def scale(self, value):
        return value

    def unscale(self, value):
        return value


def auto_mixed_precision(model, *_args, **_kwargs):
    return model


class GeneratorDataset:
    def __init__(self, source, shuffle=False, column_names=None, num_parallel_workers=0,
                 python_multiprocessing=False, num_shards=None, shard_id=None):
        del column_names, python_multiprocessing
        if num_shards is not None and shard_id is not None and num_shards > 1:
            source = _ShardDataset(source, num_shards, shard_id)
        self.source = source
        self.shuffle = shuffle
        self.num_workers = num_parallel_workers or 0

    def batch(self, batch_size):
        return BatchDataset(self.source, batch_size, self.shuffle, self.num_workers)


class _ShardDataset(TorchDataset):
    def __init__(self, source, num_shards: int, shard_id: int):
        self.source = source
        self.indices = list(range(shard_id, len(source), num_shards))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.source[self.indices[idx]]


class BatchDataset:
    def __init__(self, source, batch_size: int, shuffle: bool = False, num_workers: int = 0):
        self.source = source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.loader = DataLoader(source, batch_size=batch_size, shuffle=shuffle,
                                 num_workers=num_workers, drop_last=False)

    def create_tuple_iterator(self):
        return iter(self)

    def get_dataset_size(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            if isinstance(batch, list):
                batch = tuple(batch)
            yield batch

    def __len__(self):
        return len(self.loader)


TupleIterator = Iterator


def init():
    return None


def get_rank():
    return 0


def get_group_size():
    return 1


def save_checkpoint(model: torch_nn.Module, path: str):
    assign_parameter_names(model)
    torch.save(model.state_dict(), path)


def load_checkpoint(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    return unwrap_checkpoint_state_dict(checkpoint)


def unwrap_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                return state_dict
    return checkpoint


def load_param_into_net(model: torch_nn.Module, param_dict: dict):
    param_dict = unwrap_checkpoint_state_dict(param_dict)
    state = model.state_dict()
    loadable = {}
    unexpected = []
    for name, value in param_dict.items():
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if name in state and state[name].shape == tensor.shape:
            loadable[name] = tensor.to(dtype=state[name].dtype)
        else:
            unexpected.append(name)
    missing = [name for name in state if name not in loadable]
    model.load_state_dict(loadable, strict=False)
    return missing, unexpected


class SummaryRecord:
    def __init__(self, *_args, **_kwargs):
        pass

    def add_value(self, *_args, **_kwargs):
        pass

    def record(self, *_args, **_kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass
