"""Pydantic data models for TTT Playground"""
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class DocumentStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    LEARNING = "learning"
    READY = "ready"
    ERROR = "error"


class DocumentChunk(BaseModel):
    """Single chunk of document text"""
    index: int                          # Chunk position (0-indexed)
    text: str                           # Raw text content
    token_ids: List[int]                # Tokenized form
    token_count: int                    # Number of tokens
    start_page: Optional[int] = None    # Source page start
    end_page: Optional[int] = None      # Source page end


class Document(BaseModel):
    """Uploaded document with extracted content"""
    id: str                             # UUID
    filename: str                       # Original filename
    page_count: int                     # Total pages
    total_tokens: int                   # Total token count
    chunks: List[DocumentChunk]         # Chunked content
    status: DocumentStatus              # Processing status
    error_message: Optional[str] = None # Error if status=ERROR


class DocumentConstraints(BaseModel):
    """Validation constraints"""
    max_pages: int = 100
    max_file_size_mb: int = 50
    max_tokens: int = 100_000
    min_tokens: int = 500
    chunk_size: int = 2048              # LaCT chunk size


class LearningConfig(BaseModel):
    """TTT hyperparameters"""
    inner_lr: float = 0.01              # Inner loop learning rate
    chunk_size: int = 2048              # LaCT chunk size
    max_grad_norm: float = 1.0          # Gradient clipping
    loss_type: str = "next_token"       # "next_token" or "masked"


class LearningMetrics(BaseModel):
    """Metrics from TTT learning pass"""
    initial_loss: float                 # Loss before learning
    final_loss: float                   # Loss after learning
    loss_history: List[float]           # Per-chunk losses
    chunks_processed: int               # Number of chunks
    tokens_processed: int               # Total tokens
    learning_time_seconds: float        # Wall clock time
    weight_delta_norm: float            # L2 norm of weight change


class Question(BaseModel):
    """User question"""
    text: str                           # Question text
    max_tokens: int = 256               # Max response length
    temperature: float = 0.7            # Sampling temperature


class Answer(BaseModel):
    """Model response"""
    text: str                           # Generated answer
    tokens_generated: int               # Response length
    generation_time_seconds: float      # Latency


class ComparisonResult(BaseModel):
    """Side-by-side comparison output"""
    question: Question
    ttt_answer: Answer                  # TTT model (learned)
    base_answer: Answer                 # Base model (no learning)
    document_id: str                    # Source document
