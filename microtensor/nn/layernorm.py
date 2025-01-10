from typing import Union, List
from ..core import Tensor
from ._module import Module
import numpy as np


class LayerNorm(Module):
    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-5, device: str = "cpu"):
        super().__init__(device)
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, (list, tuple)) else [normalized_shape]
        self.eps = eps

        # Initialize gamma and beta parameters
        self.weight = Tensor(
            np.ones(self.normalized_shape, dtype=np.float32), requires_grad=True
        )
        self.bias = Tensor(
            np.zeros(self.normalized_shape, dtype=np.float32), requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        # Calculate the dimensions to normalize over
        norm_dims = tuple(range(len(x.shape) - len(self.normalized_shape), len(x.shape)))

        # Calculate mean and variance along the normalized dimensions
        mean = x.data.mean(axis=norm_dims, keepdims=True)
        var = ((x.data - mean) ** 2).mean(axis=norm_dims, keepdims=True)

        # Normalize the input
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)

        # Apply the learnable parameters
        out = Tensor(
            self.weight.data * x_norm + self.bias.data,
            dtype=x.dtype,
            _children=(x, self.weight, self.bias),
            _op="layernorm"
        )

        if x.requires_grad and Tensor.grad_is_enabled:
            def _layernorm_backward():
                N = np.prod([x.shape[dim] for dim in norm_dims])
                grad_input = (
                    (out.grad - out.grad.mean(axis=norm_dims, keepdims=True)) -
                    (x_norm * (out.grad * x_norm).mean(axis=norm_dims, keepdims=True))
                ) / np.sqrt(var + self.eps)
                x.grad += grad_input

            out.grad_fn = _layernorm_backward
            out.set_requires_grad(True)

        return out
