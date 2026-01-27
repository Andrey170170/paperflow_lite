"""Data models for paperflow."""

from enum import Enum

from pydantic import BaseModel, Field


class PaperType(str, Enum):
    """Types of academic papers."""

    EMPIRICAL = "empirical"
    THEORETICAL = "theoretical"
    REVIEW = "review"
    METHODS = "methods"
    COMMENTARY = "commentary"


class PaperSummary(BaseModel):
    """Summary extracted from a paper by the LLM."""

    summary: str = Field(description="2-3 sentence summary of the paper")
    key_points: list[str] = Field(
        description="3-5 main findings or contributions",
        min_length=1,
    )
    methods: str = Field(description="Methodology or approach used")
    paper_type: PaperType = Field(description="Type of paper")


class Classification(BaseModel):
    """Classification result from the LLM."""

    collections: list[str] = Field(
        description="Collection names to assign (1-2)",
        min_length=1,
    )
    tags: list[str] = Field(
        description="Tags to apply",
        default_factory=list,
    )
    confidence: float = Field(
        description="Classification confidence (0-1)",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(description="Brief explanation for classification")


class ParsedPaper(BaseModel):
    """Parsed content from a PDF."""

    title: str | None = Field(description="Paper title if extracted")
    abstract: str | None = Field(description="Abstract if found")
    full_text: str = Field(description="Full extracted text")
    page_count: int = Field(description="Number of pages parsed", ge=1)
    truncated: bool = Field(description="Whether text was truncated due to page limit")


class ZoteroItem(BaseModel):
    """Representation of a Zotero library item."""

    key: str = Field(description="Zotero item key")
    title: str = Field(description="Item title")
    creators: list[str] = Field(description="Author names", default_factory=list)
    item_type: str = Field(description="Zotero item type (journalArticle, etc)")
    collections: list[str] = Field(description="Collection keys", default_factory=list)
    tags: list[str] = Field(description="Existing tags", default_factory=list)
    has_pdf: bool = Field(description="Whether item has PDF attachment")
    pdf_attachment_key: str | None = Field(description="Key of PDF attachment if exists")


class ProcessingStatus(str, Enum):
    """Status of paper processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProcessingResult(BaseModel):
    """Result of processing a single paper."""

    item_key: str = Field(description="Zotero item key that was processed")
    status: ProcessingStatus = Field(description="Processing status")
    summary: PaperSummary | None = Field(description="Generated summary", default=None)
    classification: Classification | None = Field(
        description="Classification result", default=None
    )
    error: str | None = Field(description="Error message if failed", default=None)
