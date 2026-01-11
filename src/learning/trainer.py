"""
Training orchestration for TTT learning.

Phase 6 builds a thin trainer that wires together:
- TTTModel (TinyLlama with TTTLinear layers)
- LaCTUpdater (chunk-wise gradient accumulation + update)
- MetricsTracker (loss history + LearningMetrics)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.config import LearningConfig
from src.learning.metrics import MetricsTracker

if TYPE_CHECKING:
    from src.models.ttt_model import TTTModel


@dataclass
class TTTTrainer:
    """
    Orchestrates TTT learning from documents.

    Step 6.4 only requires that __init__ works.
    """

    model: "TTTModel"
    config: LearningConfig

    def __post_init__(self) -> None:
        # Import here to avoid importing torch on module import
        # (some local dev envs may not have CUDA libs available).
        from src.models.lact import LaCTUpdater

        self.metrics = MetricsTracker()
        self.updater = LaCTUpdater(
            self.model,
            inner_lr=self.config.inner_lr,
            max_grad_norm=self.config.max_grad_norm,
        )

