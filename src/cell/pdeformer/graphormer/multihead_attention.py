r"""Multi-headed attention."""
from typing import Optional
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ...env import ENABLE_DROPOUT


class MultiheadAttention(nn.Module):
    r"""
    Multi-headed attention. See "Attention Is All You Need" paper for more details.

    Args:
        embed_dim (int): The dimension of embedding.
        num_heads (int): The number of heads.
        dropout (float): The discard rate of dropout layer. Default: ``0.0``.
        bias (bool): Determine whether bias is included in the nn.Linear layer. Default: ``True``.
        compute_dtype (torch.dtype): The computation type. Default: torch.float16.

    Inputs:
        - **x** (Tensor) - Input Tensor, shape is : math:`(n\_node, n\_graph, embed\_dim)`.
        - **attn_bias** (Tensor, optional) - Graphormer's self-attention bias for encoding graph
          structure information, shape is : math:`(n\_graph * num\_heads,, n\_node, n\_node)`.
        - **key_padding_mask** (ByteTensor, optional) - Mask to exclude keys that are pads where
          padding elements are indicated by 1s, shape is : math:`(n\_graph, n\_node)`.
        - **attn_mask** (ByteTensor, optional) - Used to implement causal attention, where the mask
          prevents the attention from looking forward in time, shape is : math:`(n\_node, n\_node)`.

    Outputs:
        Tensor of shape :math:`(n\_node, n\_graph, embed\_dim)`.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from src.cell.pdeformer.graphormer.multihead_attention import MultiheadAttention
        >>> x = torch.tensor(np.random.randn(16, 8, 128), dtype=torch.float32)
        >>> mha = MultiheadAttention(embed_dim=128, num_heads=8)
        >>> output = mha(x)
        >>> print(output.shape)
        (16, 8, 128)
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            compute_dtype=torch.float16) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        if self.embed_dim <= 0:
            raise ValueError("'embed_dim' must be a positive integer.")
        self.compute_dtype = compute_dtype
        self.num_heads = num_heads

        self.dropout_module = nn.Dropout(p=dropout)

        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("'embed_dim' must be divisible by 'num_heads'")
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias).to(dtype=compute_dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias).to(dtype=compute_dtype)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias).to(dtype=compute_dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias).to(dtype=compute_dtype)

        self.init_params()

    def init_params(self) -> None:
        """
        Set initializer to parameters, Empirically observed the convergence to be much
        better with the scaled initialization.
        """

        scale = math.sqrt(1 / self.embed_dim)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        if self.k_proj.bias is not None:
            nn.init.uniform_(self.k_proj.bias, -scale, scale)

        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        if self.v_proj.bias is not None:
            nn.init.uniform_(self.v_proj.bias, -scale, scale)

        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        if self.q_proj.bias is not None:
            nn.init.uniform_(self.q_proj.bias, -scale, scale)

        nn.init.xavier_uniform_(self.out_proj.weight, gain=1)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
            self,
            x: Tensor,
            attn_bias: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None) -> Tensor:
        r"""forward"""
        n_node, n_graph, embed_dim = x.shape

        # [n_node, n_graph, embed_dim] * [embed_dim, embed_dim] -> [n_node, n_graph, embed_dim]
        query = self.q_proj(x)

        # [n_node, n_graph, embed_dim] * [embed_dim, embed_dim] -> [n_node, n_graph, embed_dim]
        key = self.k_proj(x)

        # [n_node, n_graph, embed_dim] * [embed_dim, embed_dim] -> [n_node, n_graph, embed_dim]
        value = self.v_proj(x)

        query *= self.scaling

        # [n_node, n_graph, embed_dim] -> [n_graph*num_heads, n_node, head_dim]
        query = query.reshape(n_node, n_graph * self.num_heads,
                              self.head_dim).permute(1, 0, 2)

        # [n_node, n_graph, embed_dim] -> [n_graph*num_heads, n_node, head_dim]
        key = key.reshape(n_node, n_graph * self.num_heads,
                          self.head_dim).permute(1, 0, 2)

        # [n_node, n_graph, embed_dim] -> [n_graph*num_heads, n_node, head_dim]
        value = value.reshape(n_node, n_graph * self.num_heads,
                              self.head_dim).permute(1, 0, 2)

        # [n_graph*num_heads, n_node, head_dim] x [n_graph*num_heads, head_dim, n_node]
        # -> [n_graph*num_heads, n_node, n_node]
        attn_weights = torch.bmm(query, key.transpose(1, 2))

        # Core code of Graphormer
        if attn_bias is not None:
            # Shape is [n_graph*num_heads, n_node, n_node].
            attn_weights += attn_bias.reshape(n_graph * self.num_heads, n_node, n_node)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(dim=0)  # [n_node, n_node] -> [1, n_node, n_node]
            attn_weights += attn_mask  # [n_graph*num_heads, n_node, n_node]

        if key_padding_mask is not None and key_padding_mask.ndim == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            if key_padding_mask.shape[0] != n_graph or key_padding_mask.shape[1] != n_node:
                raise ValueError(
                    f"'key_padding_mask' shape error: Expected ({n_graph}, {n_node}), "
                    f"but got {key_padding_mask.shape}.")

            # don't attend to padding symbols
            # [n_graph*num_heads, n_node, n_node] -> [n_graph, num_heads, n_node, n_node]
            attn_weights = attn_weights.reshape(n_graph, self.num_heads, n_node, n_node)

            # [n_graph, n_node] -> [n_graph, 1, 1, n_node]
            key_padding_mask = key_padding_mask.unsqueeze(dim=1).unsqueeze(dim=2).to(torch.bool)
            attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))

            # [n_graph, num_heads, n_node, n_node] -> [n_graph*num_heads, n_node, n_node]
            attn_weights = attn_weights.reshape(n_graph * self.num_heads, n_node, n_node)

        attn_weights = attn_weights.to(torch.float32)
        attn_probs = F.softmax(attn_weights, dim=-1)  # [n_graph*num_heads, n_node, n_node]
        attn_probs = attn_probs.to(self.compute_dtype)
        if ENABLE_DROPOUT:
            attn_probs = self.dropout_module(attn_probs)  # [n_graph*num_heads, n_node, n_node]

        # [n_graph*num_heads, n_node, n_node] x [n_graph*num_heads, n_node, head_dim]
        # -> [n_graph*num_heads, n_node, head_dim]
        attn = torch.bmm(attn_probs, value)

        # [n_graph*num_heads, n_node, head_dim] -> [n_node, n_graph, embed_dim]
        attn = attn.permute(1, 0, 2).reshape(n_node, n_graph, embed_dim)

        # [n_node, n_graph, embed_dim] * [embed_dim, embed_dim] -> [n_node, n_graph, embed_dim]
        attn = self.out_proj(attn)

        return attn  # [n_node, n_graph, embed_dim]
