from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from wqalpha.data import align_like
from wqalpha.features import FeatureEngine
from wqalpha.types import Panel

AlphaFunction = Callable[[FeatureEngine], pd.DataFrame]


@dataclass(frozen=True)
class Alpha:
    name: str
    function: AlphaFunction
    description: str = ""
    horizon: int = 1

    def compute(self, panel: Panel) -> pd.Series:
        matrix = self.function(FeatureEngine(panel))
        return align_like(panel, matrix, self.name)


class AlphaRegistry:
    def __init__(self) -> None:
        self._alphas: dict[str, Alpha] = {}

    def register(self, alpha: Alpha) -> None:
        if alpha.name in self._alphas:
            raise ValueError(f"Alpha already registered: {alpha.name}")
        self._alphas[alpha.name] = alpha

    def get(self, name: str) -> Alpha:
        return self._alphas[name]

    def names(self) -> list[str]:
        return sorted(self._alphas)

    def compute(self, name: str, panel: Panel) -> pd.Series:
        return self.get(name).compute(panel)

    def compute_all(self, panel: Panel) -> pd.DataFrame:
        return pd.concat([self.compute(name, panel) for name in self.names()], axis=1)
