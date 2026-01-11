"""
Learning metrics utilities.

Tracks loss over LaCT chunks and produces LearningMetrics (Step 6.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.config import LearningMetrics


@dataclass
class MetricsTracker:
    """
    Minimal metrics tracker for Phase 6.

    Step 6.2 requires only loss_history recording.
    Step 6.3 will add get_metrics() -> LearningMetrics.
    """

    loss_history: List[float] = field(default_factory=list)

    def record_loss(self, loss: float) -> None:
        self.loss_history.append(float(loss))

    def get_metrics(
        self,
        *,
        tokens_processed: int = 0,
        learning_time_seconds: float = 0.0,
        weight_delta_norm: float = 0.0,
    ) -> LearningMetrics:
        """
        Build a LearningMetrics object from tracked state.

        Step 6.3 only asserts initial_loss/final_loss correctness; the other
        required fields are filled from args / derived values.
        """
        if len(self.loss_history) == 0:
            initial_loss = 0.0
            final_loss = 0.0
        else:
            initial_loss = float(self.loss_history[0])
            final_loss = float(self.loss_history[-1])

        return LearningMetrics(
            initial_loss=initial_loss,
            final_loss=final_loss,
            loss_history=list(self.loss_history),
            chunks_processed=len(self.loss_history),
            tokens_processed=int(tokens_processed),
            learning_time_seconds=float(learning_time_seconds),
            weight_delta_norm=float(weight_delta_norm),
        )

