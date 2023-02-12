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
