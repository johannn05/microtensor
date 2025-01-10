from ..core import Tensor, DEFAULT_MIN
from ._module import Module


def _check_tensor_types(a: Tensor, b: Tensor):
    """
    Check if both tensors are compatible for operations.

    Args:
        a (Tensor): First tensor.
        b (Tensor): Second tensor.

    Raises:
        RuntimeError: If the tensors are incompatible.
    """
    if type(a.data) != type(b.data):
        raise RuntimeError("Expected both Tensors to be of the same type.")



class MSELoss(Module):
    """
    creates a criterion that measures the mean squared error (squared l2 norm) 
    between each element in pred and target of size n based on reduction.

    if reduction is "sum", it doesn't divide by n to get mean.
    if reduction is "mean", it divides by n to get mean.
    """
    def __init__(self, reduction: str = "sum") -> None:
        super().__init__()  # ensure proper superclass initialization
        self.reduction = reduction

    def forward(self, pred: Tensor, actual: Tensor) -> Tensor:
        """
        computes the mean squared error loss between predictions and targets.
        """
        _check_tensor_types(pred, actual)

        l2sum = ((pred - actual) ** 2).sum()

        if self.reduction == "sum":
            return l2sum
        elif self.reduction == "mean":
            return l2sum / actual.shape[0]
        else:
            raise ValueError(f"invalid reduction type '{self.reduction}' found.")


class BCELoss(Module):
    """
    calculates binary cross-entropy loss between predictions and targets.
    """
    def __init__(self, eps: float = DEFAULT_MIN) -> None:
        super().__init__()  # ensure proper superclass initialization
        self.eps = eps

    def forward(self, pred: Tensor, actual: Tensor) -> Tensor:
        """
        computes binary cross-entropy loss with numerical stability.
        """
        _check_tensor_types(pred, actual)

        # clip the predictions and targets to avoid numerical instability
        a: Tensor = pred * actual.clip(self.eps, 1 - self.eps).log()
        b: Tensor = (1 - pred) * (1 - actual).clip(self.eps, 1 - self.eps).log()

        return -(a + b).sum() / pred.shape[0]
