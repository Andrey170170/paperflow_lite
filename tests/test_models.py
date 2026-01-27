"""Tests for paperflow data models."""

import pytest
from pydantic import ValidationError

from paperflow.models import (
    Classification,
    PaperSummary,
    PaperType,
    ParsedPaper,
    ProcessingResult,
    ProcessingStatus,
    ZoteroItem,
)


class TestPaperType:
    """Tests for PaperType enum."""

    def test_valid_paper_types(self) -> None:
        assert PaperType.EMPIRICAL.value == "empirical"
        assert PaperType.THEORETICAL.value == "theoretical"
        assert PaperType.REVIEW.value == "review"
        assert PaperType.METHODS.value == "methods"
        assert PaperType.COMMENTARY.value == "commentary"


class TestPaperSummary:
    """Tests for PaperSummary model."""

    def test_valid_summary(self) -> None:
        summary = PaperSummary(
            summary="This paper explores neural networks.",
            key_points=["Point 1", "Point 2", "Point 3"],
            methods="Used deep learning with transformers.",
            paper_type=PaperType.EMPIRICAL,
        )
        assert summary.summary == "This paper explores neural networks."
        assert len(summary.key_points) == 3
        assert summary.paper_type == PaperType.EMPIRICAL

    def test_paper_type_from_string(self) -> None:
        summary = PaperSummary(
            summary="Test",
            key_points=["Point"],
            methods="Test method",
            paper_type="empirical",  # type: ignore[arg-type]
        )
        assert summary.paper_type == PaperType.EMPIRICAL

    def test_minimum_key_points(self) -> None:
        # Should work with 1 key point
        summary = PaperSummary(
            summary="Test",
            key_points=["Single point"],
            methods="Method",
            paper_type=PaperType.REVIEW,
        )
        assert len(summary.key_points) == 1

    def test_empty_key_points_fails(self) -> None:
        with pytest.raises(ValidationError):
            PaperSummary(
                summary="Test",
                key_points=[],
                methods="Method",
                paper_type=PaperType.REVIEW,
            )


class TestClassification:
    """Tests for Classification model."""

    def test_valid_classification(self) -> None:
        classification = Classification(
            collections=["ML / Deep Learning"],
            tags=["methods-focused", "empirical"],
            confidence=0.85,
            reasoning="Paper focuses on ML techniques.",
        )
        assert classification.collections == ["ML / Deep Learning"]
        assert len(classification.tags) == 2
        assert classification.confidence == 0.85

    def test_confidence_bounds(self) -> None:
        # Valid range 0-1
        c1 = Classification(
            collections=["Test"],
            tags=[],
            confidence=0.0,
            reasoning="Low confidence",
        )
        assert c1.confidence == 0.0

        c2 = Classification(
            collections=["Test"],
            tags=[],
            confidence=1.0,
            reasoning="High confidence",
        )
        assert c2.confidence == 1.0

    def test_confidence_out_of_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Classification(
                collections=["Test"],
                tags=[],
                confidence=1.5,
                reasoning="Invalid",
            )

        with pytest.raises(ValidationError):
            Classification(
                collections=["Test"],
                tags=[],
                confidence=-0.1,
                reasoning="Invalid",
            )

    def test_empty_collections_fails(self) -> None:
        with pytest.raises(ValidationError):
            Classification(
                collections=[],
                tags=["tag"],
                confidence=0.5,
                reasoning="No collections",
            )


class TestParsedPaper:
    """Tests for ParsedPaper model."""

    def test_valid_parsed_paper(self) -> None:
        paper = ParsedPaper(
            title="Test Paper Title",
            abstract="This is an abstract.",
            full_text="Full text content here.",
            page_count=5,
            truncated=False,
        )
        assert paper.title == "Test Paper Title"
        assert paper.page_count == 5
        assert not paper.truncated

    def test_optional_fields(self) -> None:
        paper = ParsedPaper(
            title=None,
            abstract=None,
            full_text="Only full text available.",
            page_count=1,
            truncated=False,
        )
        assert paper.title is None
        assert paper.abstract is None

    def test_truncated_paper(self) -> None:
        paper = ParsedPaper(
            title="Long Paper",
            abstract=None,
            full_text="Truncated content...",
            page_count=10,
            truncated=True,
        )
        assert paper.truncated
        assert paper.page_count == 10


class TestZoteroItem:
    """Tests for ZoteroItem model."""

    def test_valid_zotero_item(self) -> None:
        item = ZoteroItem(
            key="ABC123",
            title="Research Paper",
            creators=["Smith, J.", "Doe, A."],
            item_type="journalArticle",
            collections=["COLL1"],
            tags=["neural-networks"],
            has_pdf=True,
            pdf_attachment_key="PDF456",
        )
        assert item.key == "ABC123"
        assert item.title == "Research Paper"
        assert len(item.creators) == 2
        assert item.has_pdf
        assert item.pdf_attachment_key == "PDF456"

    def test_item_without_pdf(self) -> None:
        item = ZoteroItem(
            key="XYZ789",
            title="No PDF Item",
            creators=[],
            item_type="webpage",
            collections=[],
            tags=[],
            has_pdf=False,
            pdf_attachment_key=None,
        )
        assert not item.has_pdf
        assert item.pdf_attachment_key is None


class TestProcessingStatus:
    """Tests for ProcessingStatus enum."""

    def test_status_values(self) -> None:
        assert ProcessingStatus.PENDING.value == "pending"
        assert ProcessingStatus.PROCESSING.value == "processing"
        assert ProcessingStatus.COMPLETED.value == "completed"
        assert ProcessingStatus.FAILED.value == "failed"
        assert ProcessingStatus.SKIPPED.value == "skipped"


class TestProcessingResult:
    """Tests for ProcessingResult model."""

    def test_successful_result(self) -> None:
        summary = PaperSummary(
            summary="Test",
            key_points=["Point"],
            methods="Method",
            paper_type=PaperType.EMPIRICAL,
        )
        classification = Classification(
            collections=["Test"],
            tags=["tag"],
            confidence=0.9,
            reasoning="Test",
        )
        result = ProcessingResult(
            item_key="ABC123",
            status=ProcessingStatus.COMPLETED,
            summary=summary,
            classification=classification,
            error=None,
        )
        assert result.status == ProcessingStatus.COMPLETED
        assert result.summary is not None
        assert result.classification is not None
        assert result.error is None

    def test_failed_result(self) -> None:
        result = ProcessingResult(
            item_key="XYZ789",
            status=ProcessingStatus.FAILED,
            summary=None,
            classification=None,
            error="PDF parsing failed: corrupt file",
        )
        assert result.status == ProcessingStatus.FAILED
        assert result.summary is None
        assert result.error == "PDF parsing failed: corrupt file"

    def test_skipped_result(self) -> None:
        result = ProcessingResult(
            item_key="SKP001",
            status=ProcessingStatus.SKIPPED,
            summary=None,
            classification=None,
            error="No PDF attachment found",
        )
        assert result.status == ProcessingStatus.SKIPPED
