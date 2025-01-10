from typing import Optional
from ..core import Tensor
from ._module import Module
import numpy as np

class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: str = "cpu"
    ):
        super().__init__(device)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize the embedding weight matrix
        self.weight = Tensor(
            np.random.normal(size=(num_embeddings, embedding_dim)).astype(np.float32),
            requires_grad=True
        )

    def forward(self, input: Tensor) -> Tensor:
        # Ensure input is of integer type
        if input.dtype not in [np.int32, np.int64]:
            input.data = input.data.astype(np.int32)
            input.dtype = np.int32

        # Perform the embedding lookup
        out = Tensor(
            self.weight.data[input.data],
            dtype=self.weight.dtype,
            _children=(self.weight,),
            _op="embedding"
        )

        if input.requires_grad and Tensor.grad_is_enabled:
            def _embedding_backward():
                grad = np.zeros_like(self.weight.data)
                np.add.at(grad, input.data, out.grad)
                self.weight.grad += grad

            out.grad_fn = _embedding_backward
            out.set_requires_grad(True)

        return out
