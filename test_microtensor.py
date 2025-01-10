import numpy as np
from microtensor.nn import Linear, ReLU, Sigmoid, Softmax, Dropout, LayerNorm, Sequential, Embedding
from microtensor.nn._losses import MSELoss, BCELoss
from microtensor.nn.attention import CausalSelfAttention
from microtensor.core import Tensor
from microtensor.core.engine import _get_d


def test_linear_layer():
    print("\nTesting Linear Layer...")
    linear = Linear(5, 3, device="cpu")
    x = Tensor(np.random.randn(4, 5), requires_grad=True)
    output = linear(x)
    print("Output Shape:", output.shape)

def test_relu():
    print("\nTesting ReLU Activation...")
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), requires_grad=True)
    output = ReLU()(x)
    print("Input:", x.data)
    print("Output:", output.data)

def test_sigmoid():
    print("\nTesting Sigmoid Activation...")
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), requires_grad=True)
    output = Sigmoid()(x)
    print("Input:", x.data)
    print("Output:", output.data)

def test_softmax():
    print("\nTesting Softmax Activation...")
    x = Tensor(np.array([[1.0, 2.0, 3.0], [1.0, 3.0, 5.0]]), requires_grad=True)
    output = Softmax(dim=-1)(x)
    print("Input:", x.data)
    print("Output:", output.data)

def test_dropout():
    print("\nTesting Dropout...")
    x = Tensor(np.ones((4, 4)), requires_grad=True)
    dropout = Dropout(p=0.5, device="cpu")
    dropout.train()
    output = dropout(x)
    print("Input:", x.data)
    print("Output:", output.data)

def test_layernorm():
    print("\nTesting LayerNorm...")
    x = Tensor(np.random.randn(3, 4, 5), requires_grad=True)
    layernorm = LayerNorm([4, 5], device="cpu")
    output = layernorm(x)
    print("Input Shape:", x.shape)
    print("Output Shape:", output.shape)

def test_embedding():
    print("\nTesting Embedding Layer...")
    embedding = Embedding(10, 3, device="cpu")
    indices = Tensor(np.array([1, 2, 3, 4]), dtype=np.int32)
    output = embedding(indices)
    print("Embedding Weights Shape:", embedding.weight.shape)
    print("Output Shape:", output.shape)

def test_sequential():
    print("\nTesting Sequential Module...")
    model = Sequential(
        Linear(5, 10, device="cpu"),
        ReLU(),
        Linear(10, 1, device="cpu"),
        Sigmoid()
    )
    x = Tensor(np.random.randn(4, 5), requires_grad=True)
    output = model(x)
    print("Input Shape:", x.shape)
    print("Output Shape:", output.shape)

def test_mseloss():
    print("\nTesting MSELoss...")
    loss_fn = MSELoss(reduction="mean")
    pred = Tensor(np.array([2.0, 3.0, 4.0]), requires_grad=True)
    actual = Tensor(np.array([1.0, 2.0, 3.0]))
    loss = loss_fn(pred, actual)
    print("MSE Loss:", loss.data)

def test_bceloss():
    print("\nTesting BCELoss...")
    loss_fn = BCELoss()
    pred = Tensor(np.array([0.9, 0.1, 0.8]), requires_grad=True)
    actual = Tensor(np.array([1.0, 0.0, 1.0]))
    loss = loss_fn(pred, actual)
    print("BCE Loss:", loss.data)

def test_attention():
    print("\nTesting Causal Self-Attention...")
    attention = CausalSelfAttention(context_size=4, d_embed=8, n_heads=2, device="cpu")
    x = Tensor(np.random.randn(2, 4, 8), requires_grad=True)
    output = attention(x)
    print("Input Shape:", x.shape)
    print("Output Shape:", output.shape)

if __name__ == "__main__":
    test_linear_layer()
    test_relu()
    test_sigmoid()
    test_softmax()
    test_dropout()
    test_layernorm()
    test_embedding()
    test_sequential()
    test_mseloss()
    test_bceloss()
    test_attention()
