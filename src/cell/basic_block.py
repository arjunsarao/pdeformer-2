r"""Some basic network blocks."""

import math

import torch
from torch import Tensor, nn


class UniformInitDense(nn.Linear):
    r"""Linear layer (nn.Linear) with Uniform initialization.

    Args:
        dim_in (int): Dimension of the input features.
        dim_out (int): Dimension of the output features.
        has_bias (bool): Whether bias is involved. Default: ``True``.
        scale (float): Scale of initialized weights and biases.
            Default: ``None``, use Kaiming uniform initialization.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, dim\_in)`.


    Outputs:
        Tensor of shape :math:`(*, dim\_out)`.

    Supported Platforms:
        ``CPU`` ``CUDA``

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from src.cell.basic_block import UniformInitDense
        >>> dense = UniformInitDense(10, 5, has_bias=True, scale=0.1)
        >>> x = torch.tensor(np.random.rand(16, 10), dtype=torch.float32)
        >>> y = dense(x)
        >>> print(y.shape)
        (16, 5)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 has_bias: bool = True,
                 scale: float = None,
                 bias_scale: float = None,
                 modify_he_init: bool = False,
                 neg_slope: float = 0.0) -> None:
        super().__init__(dim_in, dim_out, bias=has_bias)

        # initialize parameters
        if dim_in <= 0:
            raise ValueError(
                f"'dim_in' should be greater than 0, but got {dim_in}.")
        if modify_he_init:
            if scale is None:
                # Modified He-Kaiming uniform initialization
                scale = math.sqrt(6.0 / ((1.0 + neg_slope ** 2) * dim_in))
            nn.init.uniform_(self.weight, -scale, scale)
            if has_bias:
                if bias_scale is None:
                    nn.init.zeros_(self.bias)
                else:
                    nn.init.uniform_(self.bias, -bias_scale, bias_scale)
        else:
            if scale is None:
                scale = math.sqrt(1 / dim_in)  # Kaiming uniform initialization
            nn.init.uniform_(self.weight, -scale, scale)
            if has_bias:
                nn.init.uniform_(self.bias, -scale, scale)


class MLP(nn.Module):
    r"""Multi-layer perceptron (MLP).

    Args:
        dim_in (int): Dimension of the input features.
        dim_out (int): Dimension of the output features.
        dim_hidden (int): Dimension of hidden layer features.
        num_layers (int): Number of Layers. Default: ``3``.
        compute_dtype (torch.dtype): The floating point precision of the
            layer. Default: ``torch.float16``.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, dim\_in)`.

    Outputs:
        Tensor of shape :math:`(*, dim\_out)`.

    Supported Platforms:
        ``CPU`` ``CUDA``

    Examples:
        >>> import numpy as np
        >>> import torch
        >>>
        >>> from src.cell.basic_block import MLP
        >>> mlp = MLP(dim_in=10, dim_out=5, dim_hidden=128, num_layers=3)
        >>> x = torch.tensor(np.random.rand(16, 10), dtype=torch.float32)
        >>> y = mlp(x)
        >>> print(y.shape)
        (16, 5)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_hidden: int,
                 num_layers: int = 3,
                 *,  # keyword-only arguments afterwards
                 mode: str = None,
                 modify_he_init: bool = False,
                 compute_dtype=torch.float16) -> None:
        super().__init__()

        if num_layers > 1:
            layers = []
            layers.append(UniformInitDense(
                dim_in, dim_hidden, has_bias=True,
                modify_he_init=modify_he_init).to(dtype=compute_dtype))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 2):
                layers.append(UniformInitDense(
                    dim_hidden, dim_hidden, has_bias=True,
                    modify_he_init=modify_he_init).to(dtype=compute_dtype))
                layers.append(nn.ReLU())
            if mode in ['shift', 'scale', 'affine'] and modify_he_init:
                layers.append(UniformInitDense(
                    dim_hidden, dim_out, has_bias=True, scale=0.0,
                    modify_he_init=modify_he_init).to(dtype=compute_dtype))
            else:
                layers.append(UniformInitDense(
                    dim_hidden, dim_out, has_bias=True,
                    modify_he_init=modify_he_init).to(dtype=compute_dtype))
            self.net = nn.Sequential(*layers)
        elif num_layers == 1:
            if mode in ['shift', 'scale', 'affine'] and modify_he_init:
                self.net = UniformInitDense(
                    dim_in, dim_out, has_bias=True, scale=0.0,
                    modify_he_init=modify_he_init).to(dtype=compute_dtype)
            else:
                self.net = UniformInitDense(
                    dim_in, dim_out, has_bias=True,
                    modify_he_init=modify_he_init).to(dtype=compute_dtype)
        elif num_layers == 0 and dim_in == dim_out:
            self.net = nn.Identity()
        else:
            raise ValueError(
                f"'num_layers' should be greater than 0, but got {num_layers}.")

    def forward(self, x: Tensor) -> Tensor:
        r"""forward"""
        return self.net(x)


class CoordPositionalEncoding(nn.Module):
    r"""Coordinate positional encoding used in implicit neural representations
    (INRs): x -> [x, sin(x), cos(x), .., sin(2**k * x), cos(2**k * x)] for example.

    Args:
        num_pos_enc (int): Number of frequencies involved in the position
                        encoding. Default: ``0``, no position encoding.
        period (float): Period of the fourier features in the positional
                        encoding. Default: ``2``, for coordinates normalized
                        to [-1, 1].

    Input:
        - **x** (Tensor) - Tensor of shape :math:`(*, dim\_in)`.

    Output:
        Output features of shape (.., dim_out), where
        dim_out = (1 + 2 * num_pos_enc) * dim_in

    Supported Platforms:
        ``CPU`` ``CUDA``

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from src.cell.basic_block import CoordPositionalEncoding
        >>> pos_enc = CoordPositionalEncoding(num_pos_enc=2, period=2.0)
        >>> x = torch.tensor(np.random.rand(16, 10), dtype=torch.float32)
        >>> y = pos_enc(x)
        >>> print(y.shape)
        (16, 50)
    """

    def __init__(self, num_pos_enc: int = 0, period: float = 2.0) -> None:
        super().__init__()
        if period == 0:
            raise ValueError("'period' should be non-zero.")
        omega_0 = 2 * math.pi / period
        self.omegas = [2**k * omega_0 for k in range(num_pos_enc)]

    def forward(self, x: Tensor) -> Tensor:
        r"""forward"""
        pos_enc_list = [x]
        for omega in self.omegas:
            pos_enc_list.extend([torch.sin(omega * x), torch.cos(omega * x)])
        x = torch.cat(pos_enc_list, dim=-1)
        return x


class Sine(nn.Module):
    r"""Sine activation with scaling factor.

    Args:
        w0 (float): scaling factor. Default: ``1.0``.

    Input:
        - **x** (Tensor) - Tensor of shape :math:`(*, dim\_in)`.

    Output:
        Output features of shape :math:`(*, dim\_in)`.

    Supported Platforms:
        ``CPU`` ``CUDA`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from src.cell.basic_block import Sine
        >>> sine = Sine(w0=1.0)
        >>> x = torch.tensor(np.random.rand(16, 10), dtype=torch.float32)
        >>> y = sine(x)
        >>> print(y.shape)
        (16, 10)
    """

    def __init__(self, w0: float = 1.0) -> None:
        super().__init__()
        self.omega_0 = w0

    def forward(self, x: Tensor) -> Tensor:
        r"""forward"""
        return torch.sin(self.omega_0 * x)


class Scale(nn.Module):
    r"""Scale the input Tensor.

    Args:
        a (float): scaling factor. Default: ``1.0``.

    Input:
        - **x** (Tensor) - Tensor of shape :math:`(*, dim\_in)`.

    Output:
        Output features of shape :math:`(*, dim\_in)`.

    Supported Platforms:
        ``CPU`` ``CUDA`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from src.cell.basic_block import Sine
        >>> scale = Scale(a=1.0)
        >>> x = torch.tensor(np.random.rand(16, 10), dtype=torch.float32)
        >>> y = sine(x)
        >>> print(y.shape)
        (16, 10)
    """

    def __init__(self, a: float = 1.0) -> None:
        super().__init__()
        self.a = a

    def forward(self, x: Tensor) -> Tensor:
        r"""forward"""
        return self.a * x
