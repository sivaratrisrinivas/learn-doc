"""
LaCT - Large Chunk TTT.

Efficient TTT learning by processing documents in chunks:
1. Forward pass on chunk (2048 tokens)
2. Accumulate gradients
3. Single weight update per chunk

This makes TTT viable on free-tier GPUs (T4) by reducing memory overhead.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from .ttt_model import TTTModel
from .ttt_linear import TTTLinear


class LaCTUpdater:
    """
    Large Chunk TTT - efficient document learning.
    
    Usage:
        updater = LaCTUpdater(model)
        for chunk in chunks:
            loss = updater.process_chunk(chunk.token_ids)
            updater.apply_update()
        # Or simply:
        metrics = updater.process_document(chunks)
    """
    
    def __init__(
        self,
        model: TTTModel,
        inner_lr: float = 0.01,
        max_grad_norm: float = 1.0
    ):
        """
        Args:
            model: TTTModel with TTT-Linear layers
            inner_lr: Learning rate for weight updates
            max_grad_norm: Gradient clipping threshold
        """
        self.model = model
        self.inner_lr = inner_lr
        self.max_grad_norm = max_grad_norm
        
        # Accumulated gradients for each TTT layer
        self._accumulated_grads: List[Optional[torch.Tensor]] = [
            None for _ in model.ttt_layers
        ]
        self._loss_history: List[float] = []
    
    def process_chunk(self, token_ids: List[int]) -> float:
        """
        Forward pass on chunk, compute self-supervised loss.
        
        Args:
            token_ids: List of token IDs (e.g., 2048 tokens)
            
        Returns:
            Loss value for this chunk
        """
        # Step 5.2: Implement forward + loss
        raise NotImplementedError("Step 5.2")
    
    def apply_update(self) -> None:
        """
        Apply accumulated gradients to update W_h in all TTT layers.
        """
        # Step 5.4: Implement weight update
        raise NotImplementedError("Step 5.4")
    
    def process_document(
        self, 
        chunks: List,  # List[DocumentChunk]
        progress_callback=None
    ) -> dict:
        """
        Process entire document chunk by chunk.
        
        Args:
            chunks: List of DocumentChunk objects
            progress_callback: Optional callback(chunk_idx, total, loss)
            
        Returns:
            Dict with metrics: initial_loss, final_loss, total_chunks
        """
        # Step 5.5: Implement document processing loop
        raise NotImplementedError("Step 5.5")
    
    def get_loss_history(self) -> List[float]:
        """Return loss values from all processed chunks."""
        return self._loss_history.copy()
    
    def reset(self) -> None:
        """Reset accumulated gradients and loss history."""
        self._accumulated_grads = [None for _ in self.model.ttt_layers]
        self._loss_history = []
