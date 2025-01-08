from microtensor.core.engine import Tensor, no_grad

def test_operations():
    print("Testing Tensor Operations...\n")

    t1 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    t2 = Tensor([4.0, 5.0, 6.0], requires_grad=True)

    # Basic operations
    print(f"t1 + t2: {(t1 + t2).data}")
    print(f"t1 - t2: {(t1 - t2).data}")
    print(f"t1 * t2: {(t1 * t2).data}")
    print(f"t2 / t1: {(t2 / t1).data}")
    print(f"t1 ** 2: {(t1 ** 2).data}")
    print(f"-t1: {(-t1).data}\n")


def test_gradients():
    print("Testing Gradients...\n")

    # Initialize tensors
    t1 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    t2 = Tensor([4.0, 5.0, 6.0], requires_grad=True)

    # Perform operation
    t3 = t1 * t2  # Elementwise multiplication
    print(f"t3 (result): {t3.data}")

    # Backpropagation
    t3.backward()

    # Check gradients
    print(f"t1.grad: {t1.grad}")  # Should be t2.data
    print(f"t2.grad: {t2.grad}")  # Should be t1.data\n")


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

    # Perform operation
    t3 = t1 + t2  # Broadcasting addition
    print(f"t3 (broadcasted result): {t3.data}")

    # Backpropagation
    t3.backward()

    print(f"t1.grad: {t1.grad}")  # Gradient aggregated along the broadcasted dimension
    print(f"t2.grad: {t2.grad}\n")


def run_tests():
    test_operations()
    test_gradients()
    test_no_grad()
    test_broadcasting()


if __name__ == "__main__":
    run_tests()
