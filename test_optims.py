import numpy as np
from microtensor.optims import Adam, RMSProp, SGD
from microtensor.nn import Linear
from microtensor.core import Tensor
from microtensor.nn._losses import MSELoss

def test_optimizer(optimizer_cls, optimizer_name):
    print(f"\nTesting {optimizer_name} Optimizer...")

    # Create a simple model (Linear Layer)
    linear = Linear(2, 1, device="cpu")

    # dummy dataset
    x = Tensor(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]), requires_grad=True)  # Requires gradients
    y_true = Tensor(np.array([[5.0], [8.0], [11.0]]))  # Ground truth


    # Loss function (Mean Squared Error)
    def mse_loss(y_pred, y_true):
        return ((y_pred - y_true) ** 2).mean()

    # Optimizer
    optimizer = optimizer_cls(parameters=linear.parameters(), lr=0.01)

    # Training loop
    for epoch in range(5):
        # Forward pass
        y_pred = linear(x)

        # Compute loss
        loss = mse_loss(y_pred, y_true)
        
        # Zero gradients
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.data:.4f}")

if __name__ == "__main__":
    test_optimizer(lambda parameters, lr: Adam(parameters=parameters, lr=lr), "Adam")
    test_optimizer(RMSProp, "RMSProp")
    test_optimizer(SGD, "SGD")
