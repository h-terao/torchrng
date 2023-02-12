from __future__ import annotations
import typing as tp
import torch


def PRNGKey(seed: int | None = None) -> torch.Tensor:
    """Create a PRNG key from the given seed.

    Args:
        seed: Seed value. If None, use default rng state of PyTorch.

    Returns:
        A PRNG key.
    """

    with torch.random.fork_rng():
        if seed is not None:
            torch.random.manual_seed(seed)
        key = torch.random.get_rng_state()
    return key


def halve(key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Halve PRNG key.

    Args:
        key: A PRNGKey to halve.

    Returns:
        A tuple of two new keys.

    NOTE:
        This implementation is not enough to keep good randomness.
        Necessary to use better algorithm.
    """
    with torch.random.fork_rng():
        torch.random.set_rng_state(key)
        seed1, seed2 = map(int, torch.randint(0, 2**63 - 1, size=(2,)))
        size1, size2 = map(int, torch.randint(0, 2**20, size=(2,)))

        torch.manual_seed(seed1)
        torch.rand(size1)  # Step.
        key1 = torch.random.get_rng_state()

        torch.manual_seed(seed2)
        torch.rand(size2)  # Step.
        key2 = torch.random.get_rng_state()

    return key1, key2


def split(key: torch.Tensor, num: int = 2) -> tp.Sequence[torch.Tensor]:
    """Split PRNG key into `num` new keys.

    Args:
        key: A PRNGKey to split.
        num: Number of desired keys to split.

    Returns:
        A list of `num` new keys.

    NOTE:
        This implementation is not enough to keep good randomness.
        Necessary to use better algorithm.
    """
    new_keys = []
    for _ in range(num - 1):
        new_key, key = halve(key)
        new_keys.append(new_key)
    new_keys.append(key)
    return new_keys


def deterministic(fun: tp.Callable) -> tp.Callable:
    """Create a deterministic function.
        The stochastic operations in `fun` are controlloed by the new `key` argument.

    Args:
        fun: A callable to be deterministic.

    Returns:
        A wrapped version of fun.

    TODO:
        Automatic docstring generation of `wrapped`.
    """

    def wrapped(key: torch.Tensor, *args, **kwargs):
        with torch.random.fork_rng():
            torch.random.set_rng_state(key)
            return fun(*args, **kwargs)

    return wrapped


if __name__ == "__main__":

    def random_fun(x: torch.Tensor) -> torch.Tensor:
        # decay x with random factor.
        return x * torch.rand_like(x)

    tensor = torch.ones(size=(10, 10))

    x = random_fun(tensor)
    y = random_fun(tensor)
    print("Estimated: False, Actual:", (x == y).all())

    rng = PRNGKey(seed=100)
    new_fun = deterministic(random_fun)
    x = new_fun(rng, tensor)
    y = new_fun(rng, tensor)
    print("Estimated: True, Actual:", (x == y).all())

    new_rng, rng = split(rng)
    y = new_fun(new_rng, tensor)
    print("Estimated: False, Actual:", (x == y).all())
