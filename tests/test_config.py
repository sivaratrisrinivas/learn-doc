"""Tests for src/config.py Pydantic models"""
import pytest
from pydantic import ValidationError
from src.config import (
    DocumentStatus,
    DocumentChunk,
    Document,
    DocumentConstraints,
    LearningConfig,
    LearningMetrics,
    Question,
    Answer,
    ComparisonResult
)


class TestDocumentChunk:
    """Test DocumentChunk model"""
    
    def test_valid_chunk(self):
        """Test creating a valid DocumentChunk"""
        chunk = DocumentChunk(
            index=0,
            text="Sample text content",
            token_ids=[1, 2, 3, 4, 5],
            token_count=5,
            start_page=1,
            end_page=1
        )
        assert chunk.index == 0
        assert chunk.text == "Sample text content"
        assert chunk.token_count == 5
        assert chunk.start_page == 1
        assert chunk.end_page == 1
    
    def test_chunk_optional_pages(self):
        """Test chunk without page info"""
        chunk = DocumentChunk(
            index=1,
            text="Another chunk",
            token_ids=[10, 20, 30],
            token_count=3
        )
        assert chunk.start_page is None
        assert chunk.end_page is None
    
    def test_chunk_token_count_mismatch(self):
        """Test that token_count doesn't need to match token_ids length (flexibility)"""
        # This should still work - token_count is just metadata
        chunk = DocumentChunk(
            index=0,
            text="test",
            token_ids=[1, 2, 3],
            token_count=3  # Matches, but not enforced by Pydantic
        )
        assert chunk.token_count == 3


class TestDocument:
    """Test Document model"""
    
    def test_valid_document(self):
        """Test creating a valid Document"""
        chunks = [
            DocumentChunk(
                index=0,
                text="Chunk 1",
                token_ids=[1, 2],
                token_count=2
            ),
            DocumentChunk(
                index=1,
                text="Chunk 2",
                token_ids=[3, 4],
                token_count=2
            )
        ]
        doc = Document(
            id="test-uuid-123",
            filename="test.pdf",
            page_count=5,
            total_tokens=4,
            chunks=chunks,
            status=DocumentStatus.READY
        )
        assert doc.id == "test-uuid-123"
        assert doc.filename == "test.pdf"
        assert len(doc.chunks) == 2
        assert doc.status == DocumentStatus.READY
        assert doc.error_message is None
    
    def test_document_with_error(self):
        """Test Document with error status"""
        doc = Document(
            id="error-doc",
            filename="corrupt.pdf",
            page_count=0,
            total_tokens=0,
            chunks=[],
            status=DocumentStatus.ERROR,
            error_message="PDF extraction failed"
        )
        assert doc.status == DocumentStatus.ERROR
        assert doc.error_message == "PDF extraction failed"


class TestDocumentConstraints:
    """Test DocumentConstraints model"""
    
    def test_default_constraints(self):
        """Test default constraint values match spec"""
        constraints = DocumentConstraints()
        assert constraints.max_pages == 100
        assert constraints.max_file_size_mb == 50
        assert constraints.max_tokens == 100_000
        assert constraints.min_tokens == 500
        assert constraints.chunk_size == 2048
    
    def test_custom_constraints(self):
        """Test custom constraint values"""
        constraints = DocumentConstraints(
            max_pages=50,
            max_file_size_mb=25,
            chunk_size=1024
        )
        assert constraints.max_pages == 50
        assert constraints.max_file_size_mb == 25
        assert constraints.chunk_size == 1024
        # Unspecified should keep defaults
        assert constraints.max_tokens == 100_000
        assert constraints.min_tokens == 500


class TestDocumentStatus:
    """Test DocumentStatus enum"""
    
    def test_all_statuses(self):
        """Test all enum values exist"""
        assert DocumentStatus.PENDING == "pending"
        assert DocumentStatus.EXTRACTING == "extracting"
        assert DocumentStatus.CHUNKING == "chunking"
        assert DocumentStatus.LEARNING == "learning"
        assert DocumentStatus.READY == "ready"
        assert DocumentStatus.ERROR == "error"


class TestLearningConfig:
    """Test LearningConfig model - Step 1.2"""
    
    def test_default_values_match_spec(self):
        """Test default values match spec.md exactly"""
        config = LearningConfig()
        assert config.inner_lr == 0.01
        assert config.chunk_size == 2048
        assert config.max_grad_norm == 1.0
        assert config.loss_type == "next_token"
    
    def test_custom_config(self):
        """Test custom LearningConfig values"""
        config = LearningConfig(
            inner_lr=0.005,
            chunk_size=1024,
            max_grad_norm=0.5,
            loss_type="masked"
        )
        assert config.inner_lr == 0.005
        assert config.chunk_size == 1024
        assert config.max_grad_norm == 0.5
        assert config.loss_type == "masked"


class TestLearningMetrics:
    """Test LearningMetrics model - Step 1.2"""
    
    def test_valid_metrics(self):
        """Test creating valid LearningMetrics"""
        metrics = LearningMetrics(
            initial_loss=2.5,
            final_loss=1.8,
            loss_history=[2.5, 2.2, 2.0, 1.8],
            chunks_processed=4,
            tokens_processed=8192,
            learning_time_seconds=12.5,
            weight_delta_norm=0.15
        )
        assert metrics.initial_loss == 2.5
        assert metrics.final_loss == 1.8
        assert len(metrics.loss_history) == 4
        assert metrics.chunks_processed == 4
        assert metrics.tokens_processed == 8192
        assert metrics.learning_time_seconds == 12.5
        assert metrics.weight_delta_norm == 0.15
    
    def test_metrics_loss_decrease(self):
        """Test that metrics can track loss decrease"""
        metrics = LearningMetrics(
            initial_loss=3.0,
            final_loss=1.5,
            loss_history=[3.0, 2.5, 2.0, 1.5],
            chunks_processed=4,
            tokens_processed=8192,
            learning_time_seconds=10.0,
            weight_delta_norm=0.2
        )
        assert metrics.final_loss < metrics.initial_loss
        assert metrics.loss_history[0] == metrics.initial_loss
        assert metrics.loss_history[-1] == metrics.final_loss


class TestQuestion:
    """Test Question model - Step 1.3"""
    
    def test_question_defaults(self):
        """Test Question with default values"""
        question = Question(text="What is TTT?")
        assert question.text == "What is TTT?"
        assert question.max_tokens == 256
        assert question.temperature == 0.7
    
    def test_question_custom_params(self):
        """Test Question with custom parameters"""
        question = Question(
            text="Explain test-time training",
            max_tokens=512,
            temperature=0.9
        )
        assert question.text == "Explain test-time training"
        assert question.max_tokens == 512
        assert question.temperature == 0.9


class TestAnswer:
    """Test Answer model - Step 1.3"""
    
    def test_answer_creation(self):
        """Test creating an Answer"""
        answer = Answer(
            text="TTT is test-time training.",
            tokens_generated=5,
            generation_time_seconds=0.5
        )
        assert answer.text == "TTT is test-time training."
        assert answer.tokens_generated == 5
        assert answer.generation_time_seconds == 0.5


class TestComparisonResult:
    """Test ComparisonResult model - Step 1.3"""
    
    def test_comparison_result_creation(self):
        """Test creating a ComparisonResult"""
        question = Question(text="What is TTT?")
        ttt_answer = Answer(
            text="TTT learns from documents at inference time.",
            tokens_generated=8,
            generation_time_seconds=1.2
        )
        base_answer = Answer(
            text="I don't have information about TTT.",
            tokens_generated=7,
            generation_time_seconds=0.8
        )
        result = ComparisonResult(
            question=question,
            ttt_answer=ttt_answer,
            base_answer=base_answer,
            document_id="doc-123"
        )
        assert result.question.text == "What is TTT?"
        assert result.ttt_answer.text == "TTT learns from documents at inference time."
        assert result.base_answer.text == "I don't have information about TTT."
        assert result.document_id == "doc-123"
    
    def test_comparison_result_serialize_json(self):
        """Test serialization to JSON and deserialization back"""
        import json
        question = Question(text="Test question")
        ttt_answer = Answer(text="TTT answer", tokens_generated=2, generation_time_seconds=0.5)
        base_answer = Answer(text="Base answer", tokens_generated=2, generation_time_seconds=0.4)
        result = ComparisonResult(
            question=question,
            ttt_answer=ttt_answer,
            base_answer=base_answer,
            document_id="test-doc"
        )
        # Serialize to JSON
        json_str = result.model_dump_json()
        assert isinstance(json_str, str)
        # Deserialize back
        data = json.loads(json_str)
        result2 = ComparisonResult(**data)
        assert result2.question.text == "Test question"
        assert result2.ttt_answer.text == "TTT answer"
        assert result2.base_answer.text == "Base answer"
        assert result2.document_id == "test-doc"
