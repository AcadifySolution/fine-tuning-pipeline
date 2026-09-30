import torch

from src.utils import count_trainable_parameters, set_seed


def test_seed_is_reproducible():
    set_seed(123)
    first = torch.rand(3)
    set_seed(123)
    second = torch.rand(3)
    assert torch.equal(first, second)


def test_trainable_parameter_count():
    model = torch.nn.Linear(4, 2)
    assert count_trainable_parameters(model) == sum(p.numel() for p in model.parameters())
