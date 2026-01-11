"""Document validation against constraints"""
from typing import Tuple, Optional
from src.document.pdf_parser import PDFParser, PDFExtractionError
from src.config import DocumentConstraints


class DocumentValidator:
    """Validate documents against constraints"""
    
    def __init__(self):
        """Initialize validator with PDF parser"""
        self.pdf_parser = PDFParser()
    
    def validate(
        self,
        file_bytes: bytes,
        constraints: DocumentConstraints
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if document meets constraints.
        
        Args:
            file_bytes: Raw PDF file content
            constraints: DocumentConstraints to validate against
            
        Returns:
            (is_valid, error_message_if_invalid)
        """
        # Check file size
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > constraints.max_file_size_mb:
            return False, f"File size ({file_size_mb:.2f} MB) exceeds maximum ({constraints.max_file_size_mb} MB)"
        
        # Try to parse PDF
        try:
            text, page_count = self.pdf_parser.parse(file_bytes)
        except PDFExtractionError as e:
            return False, f"Invalid PDF: {str(e)}"
        
        # Check page count
        if page_count > constraints.max_pages:
            return False, f"Page count ({page_count}) exceeds maximum ({constraints.max_pages})"
        
        # Estimate token count (rough: ~4 chars per token)
        # For accurate count, would need tokenizer, but that's expensive
        # This is a quick validation check
        estimated_tokens = len(text) // 4
        
        if estimated_tokens > constraints.max_tokens:
            return False, f"Estimated token count ({estimated_tokens}) exceeds maximum ({constraints.max_tokens})"
        
        if estimated_tokens < constraints.min_tokens:
            return False, f"Estimated token count ({estimated_tokens}) below minimum ({constraints.min_tokens})"
        
        # All checks passed
        return True, None
