import torch

from src.utils import count_trainable_parameters, set_seed


def test_set_seed_reproducible():
    set_seed(123)
    a = torch.rand(3)
    set_seed(123)
    b = torch.rand(3)
    assert torch.equal(a, b)


def test_count_trainable_parameters():
    model = torch.nn.Linear(4, 2)
    assert count_trainable_parameters(model) == sum(p.numel() for p in model.parameters())
