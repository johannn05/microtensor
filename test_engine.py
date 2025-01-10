from microtensor.core.engine import Tensor, no_grad
import numpy as np
import torch


def test_operations():
    print("Testing Tensor Operations...\n")

    t1 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    t2 = Tensor([4.0, 5.0, 6.0], requires_grad=True)

    # PyTorch equivalents
    t1_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    t2_torch = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

    # basic operations
    for fn, name in [(lambda x, y: x + y, "t1 + t2"),
                     (lambda x, y: x - y, "t1 - t2"),
                     (lambda x, y: x * y, "t1 * t2"),
                     (lambda x, y: x / y, "t2 / t1"),
                     (lambda x, _: x ** 2, "t1 ** 2"),
                     (lambda x, _: -x, "-t1")]:
        result = fn(t1, t2 if name != "t1 ** 2" and name != "-t1" else None).data
        result_torch = fn(t1_torch, t2_torch if name != "t1 ** 2" and name != "-t1" else None)
        assert np.allclose(result, result_torch.detach().numpy()), f"{name} failed"
        print(f"{name}: {result}")

    print("\nAll operation tests passed!\n")


def test_gradients():
    print("Testing Gradients...\n")

    t1 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    t2 = Tensor([4.0, 5.0, 6.0], requires_grad=True)

    t1_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    t2_torch = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

    t3 = t1 * t2  # Elementwise multiplication
    t3_torch = t1_torch * t2_torch

    print(f"t3 (result): {t3.data}")

    # Backpropagation
    t3.backward()
    t3_torch.sum().backward()

    # check gradients
    assert np.allclose(t1.grad, t1_torch.grad.numpy()), "Gradient check failed for t1"
    assert np.allclose(t2.grad, t2_torch.grad.numpy()), "Gradient check failed for t2"

    print(f"t1.grad: {t1.grad}")
    print(f"t2.grad: {t2.grad}\n")

    print("All gradient tests passed!\n")


def test_no_grad():
    print("Testing no_grad Context...\n")

    t1 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    t2 = Tensor([4.0, 5.0, 6.0], requires_grad=True)

    with no_grad():
        t3 = t1 * t2
        print(f"t3 (result in no_grad): {t3.data}")

        try:
            t3.backward()
        except ValueError as e:
            print(f"Expected error: {e}")

    print()


def test_broadcasting():
    print("Testing Broadcasting...\n")

    t1 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    t2 = Tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True)

    t1_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    t2_torch = torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True)

    # Perform operation
    t3 = t1 + t2  # Broadcasting addition
    t3_torch = t1_torch + t2_torch

    print(f"t3 (broadcasted result): {t3.data}")

    # Backpropagation
    t3.backward()
    t3_torch.sum().backward()

    # Verify gradients
    assert np.allclose(t1.grad, t1_torch.grad.numpy()), "Gradient check failed for broadcasted t1"
    assert np.allclose(t2.grad, t2_torch.grad.numpy()), "Gradient check failed for broadcasted t2"

    print(f"t1.grad: {t1.grad}")  # Gradient aggregated along the broadcasted dimension
    print(f"t2.grad: {t2.grad}\n")

    print("All broadcasting tests passed!\n")


def run_tests():
    test_operations()
    test_gradients()
    test_no_grad()
    test_broadcasting()


if __name__ == "__main__":
    run_tests()
