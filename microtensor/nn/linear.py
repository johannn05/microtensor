from ..core import Tensor
from ._module import Module

class Linear(Module):
    def __init__(
        self, in_features: int, out_features: int, 
        use_bias: bool = True, device: str = "gpu", dtype=None
    ):
        """
        fully connected linear layer.

        Args:
            in_features (int): number of input features.
            out_features (int): number of output features.
            bias (bool, optional): whether to include bias terms. default is true.
            device (str, optional): device on which the layer's tensors should reside. default is "gpu".
            dtype (str, optional): data type of the tensors. default is "float32".
        """

        # call the parent class's __init__ method
        super().__init__(device=device)

        self.use_bias = use_bias
        self.use_np = self.device == "cpu"

        self.weight = Tensor(
            self._d.random.uniform(-1, 1, (out_features, in_features)),
            dtype=dtype, requires_grad=True
        )
        if self.use_bias:
            self.bias = Tensor(
                self._d.random.uniform(-1, 1, (out_features, )),
                dtype=dtype, requires_grad=True
            )


    def forward(self, X: Tensor) -> Tensor:
        """
        fully connected layer's output is:
        
        `o = w @ x + b`
        """
        out = X @ self.weight.T()
        if self.use_bias:
            out += self.bias
        return out
