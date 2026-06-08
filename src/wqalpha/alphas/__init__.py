from wqalpha.alphas.worldquant101 import *  # noqa: F403
from wqalpha.alphas.worldquant101 import register_all_alphas

__all__ = ["register_all_alphas"] + [f"alpha_{i:03d}" for i in range(1, 102)]
