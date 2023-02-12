# TorchRNG

TorchRNG provides deterministic generation of PyTorch's pseudo random operations with a few extra lines.
This feature is inspired from JAX library.

## Installation

TorchRNG only depends on PyTorch. You can install TorchRNG via pip as follows:
```bash
$ pip install git+https://github.com/h-terao/torchrng
```

## Usage

Take the following steps to use TorchRNG.

1. Create PRNG key using `torchrng.PRNGKey`.
2. Transform stochastic functions or methods of `nn.Module` using `torchrng.deterministic`. The transformed function has a new argument `key` as the first positional argument. (See the below example.) If you specify the same PRNGKey as `key`, the transformed function will return same results.
3. If you need multiple PRNGKey, split PRNGKey using `torchrng.split`.

This is the short demo code of TorchRNG.

```python
import torch
import torchrng

def random_fun(x: torch.Tensor) -> torch.Tensor:
    # Example of the stochastic function.
    return x * torch.rand_like(x)

tensor = torch.ones(size=(10, 10))

# This is a stochastic function, and results become different.
x = random_fun(tensor)
y = random_fun(tensor)
print("Estimated: False, Actual:", (x == y).all())

# Example of Step 1. If you use `torch.manual_seed` in advance, `seed=None` also works well.
key = torchrng.PRNGKey(seed=100)

# Example of Step 2. Because new_fun is deterministic and same key is passed, results are same.
new_fun = torchrng.deterministic(random_fun)
x = new_fun(key, tensor)
y = new_fun(key, tensor)
print("Estimated: True, Actual:", (x == y).all())

# Example of Step 3. If different PRNGKey is passed, results are different.
new_key, key = torchrng.split(key)
y = new_fun(new_key, tensor)
print("Estimated: False, Actual:", (x == y).all())
```

## Limitation

- If your machine has a lot of devices, TorchRNG becomes very slowly because TorchRNG uses `torch.random.folk_rng` many times. See details in [PyTorch documentation](https://pytorch.org/docs/stable/random.html).
- Currently, the implementation of PRNGKey split is very simple and may not enough to keep randomness as much as PyTorch.