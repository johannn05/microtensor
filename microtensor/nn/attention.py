import math
from typing import *
from ..core import Tensor
from ._module import Module
from .linear import Linear
from .dropout import Dropout
from ._activations import softmax
import numpy as np  


class CausalSelfAttention(Module):
    """
    Causal Self-Attention module for transformer models.
    Applies scaled dot-product attention with a causal mask to prevent information
    flow from future tokens.
    """
    def __init__(
        self,
        context_size: int,
        d_embed: int,
        n_heads: int,
        attn_pdrop: float = 0.7,
        resd_pdrop: float = 0.6,
        device: str = "cpu"
    ):
        super().__init__(device)
        self.context_size = context_size
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.attn_pdrop = attn_pdrop
        self.resd_pdrop = resd_pdrop

        self.w_proj = Linear(d_embed, 3 * d_embed, device=self.device)
        self.o_proj = Linear(d_embed, d_embed, device=self.device)
        self.attn_drop = Dropout(p=attn_pdrop, device=self.device)
        self.resd_drop = Dropout(p=resd_pdrop, device=self.device)

        # Causal mask for attention
        self.mask = Tensor(
            (1 - np.tril(np.ones((context_size, context_size)))),
            dtype=np.float32,
            requires_grad=False
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, embed_dim = x.shape

        # Project input tensor into query, key, and value tensors
        q, k, v = self.w_proj(x).split(sections=3, dim=-1)
        q = q.reshape((batch_size, seq_len, self.n_heads, embed_dim // self.n_heads)).T(axes=[0, 2, 1, 3])
        k = k.reshape((batch_size, seq_len, self.n_heads, embed_dim // self.n_heads)).T(axes=[0, 2, 1, 3])
        v = v.reshape((batch_size, seq_len, self.n_heads, embed_dim // self.n_heads)).T(axes=[0, 2, 1, 3])

        # Compute scaled dot-product attention
        attn = (q @ k.T(axes=[0, 1, 3, 2])) * (1.0 / np.sqrt(embed_dim // self.n_heads))

        # Create the causal mask with the correct shape
        causal_mask_data = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
        causal_mask = Tensor(
            np.expand_dims(np.expand_dims(causal_mask_data, axis=0), axis=0),
            dtype=attn.dtype,
            requires_grad=False
        )  # Shape: (1, 1, seq_len, seq_len)

        # Expand the causal mask to match the attention tensor's shape
        causal_mask = causal_mask.broadcast_to(attn.shape)

        # Apply the causal mask
        attn = attn.masked_fill(causal_mask.data == 0, float('-inf'))

        # Compute softmax and apply dropout
        attn = softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values
        y = attn @ v
        y = y.T(axes=[0, 2, 1, 3]).reshape((batch_size, seq_len, embed_dim))

        # Output projection
        y = self.o_proj(y)
        y = self.resd_drop(y)
        return y


