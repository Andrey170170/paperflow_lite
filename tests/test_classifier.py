"""Tests for LLM classifier."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from paperflow.classifier import Classifier, ClassifierError
from paperflow.config import CollectionDef, LLMConfig, TagDef
from paperflow.models import Classification, PaperSummary, PaperType, ParsedPaper


@pytest.fixture
def llm_config() -> LLMConfig:
    """Create a test LLM configuration."""
    return LLMConfig(
        provider="openrouter",
        api_key="test_api_key",
        model="openai/gpt-4.1-mini",
        max_tokens=2000,
        temperature=0.3,
    )


@pytest.fixture
def collections() -> list[CollectionDef]:
    """Create test collection definitions."""
    return [
        CollectionDef(
            name="ML / Deep Learning",
            description="Machine learning, neural networks, deep learning",
            keywords=["neural", "transformer", "deep learning"],
        ),
        CollectionDef(
            name="Methods / Statistics",
            description="Research methods, statistical analysis",
            keywords=["regression", "bayesian", "statistics"],
        ),
        CollectionDef(
            name="Review Later",
            description="Unclear category",
            keywords=[],
        ),
    ]


@pytest.fixture
def tags() -> list[TagDef]:
    """Create test tag definitions."""
    return [
        TagDef(name="foundational", description="Seminal/classic paper"),
        TagDef(name="methods-focused", description="About methodology"),
        TagDef(name="empirical", description="Reports original data"),
    ]


@pytest.fixture
def sample_paper() -> ParsedPaper:
    """Create a sample parsed paper."""
    return ParsedPaper(
        title="Attention Is All You Need",
        abstract="We propose a new simple network architecture based on attention.",
        full_text="# Attention Is All You Need\n\n## Abstract\n\nThe dominant sequence...",
        page_count=15,
        truncated=True,
    )


@pytest.fixture
def classifier(
    llm_config: LLMConfig,
    collections: list[CollectionDef],
    tags: list[TagDef],
) -> Classifier:
    """Create a Classifier instance."""
    return Classifier(llm_config, collections, tags)


class TestClassifier:
    """Tests for Classifier class."""

    def test_init(
        self,
        llm_config: LLMConfig,
        collections: list[CollectionDef],
        tags: list[TagDef],
    ) -> None:
        """Test classifier initialization."""
        classifier = Classifier(llm_config, collections, tags)
        assert classifier.llm_config == llm_config
        assert len(classifier.collections) == 3
        assert len(classifier.tags) == 3

    @pytest.mark.asyncio
    async def test_summarize_success(
        self, classifier: Classifier, sample_paper: ParsedPaper
    ) -> None:
        """Test successful paper summarization."""
        mock_response = {
            "summary": "This paper introduces the Transformer architecture.",
            "key_points": [
                "Introduces self-attention mechanism",
                "Eliminates recurrence entirely",
                "Achieves state-of-the-art translation",
            ],
            "methods": "Encoder-decoder with multi-head attention.",
            "paper_type": "empirical",
        }

        with patch.object(
            classifier, "_call_llm", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = json.dumps(mock_response)
            result = await classifier.summarize(sample_paper)

        assert isinstance(result, PaperSummary)
        assert "Transformer" in result.summary
        assert len(result.key_points) == 3
        assert result.paper_type == PaperType.EMPIRICAL

    @pytest.mark.asyncio
    async def test_classify_success(
        self, classifier: Classifier
    ) -> None:
        """Test successful paper classification."""
        summary = PaperSummary(
            summary="This paper introduces neural network architecture.",
            key_points=["Point 1", "Point 2"],
            methods="Deep learning methods",
            paper_type=PaperType.EMPIRICAL,
        )

        mock_response = {
            "collections": ["ML / Deep Learning"],
            "tags": ["foundational", "empirical"],
            "confidence": 0.92,
            "reasoning": "Paper is clearly about deep learning architectures.",
        }

        with patch.object(
            classifier, "_call_llm", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = json.dumps(mock_response)
            result = await classifier.classify(summary)

        assert isinstance(result, Classification)
        assert "ML / Deep Learning" in result.collections
        assert "foundational" in result.tags
        assert result.confidence == 0.92

    @pytest.mark.asyncio
    async def test_process_full_pipeline(
        self, classifier: Classifier, sample_paper: ParsedPaper
    ) -> None:
        """Test full processing pipeline."""
        summary_response = {
            "summary": "A paper about transformers.",
            "key_points": ["Self-attention"],
            "methods": "Neural networks",
            "paper_type": "methods",
        }
        classify_response = {
            "collections": ["ML / Deep Learning"],
            "tags": ["methods-focused"],
            "confidence": 0.85,
            "reasoning": "Focuses on architecture.",
        }

        call_count = 0

        async def mock_llm_calls(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps(summary_response)
            return json.dumps(classify_response)

        with patch.object(
            classifier, "_call_llm", side_effect=mock_llm_calls
        ):
            summary, classification = await classifier.process(sample_paper)

        assert summary.paper_type == PaperType.METHODS
        assert "ML / Deep Learning" in classification.collections

    @pytest.mark.asyncio
    async def test_malformed_json_response(
        self, classifier: Classifier, sample_paper: ParsedPaper
    ) -> None:
        """Test handling of malformed LLM response."""
        with patch.object(
            classifier, "_call_llm", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = "This is not valid JSON at all"

            with pytest.raises(ClassifierError, match="Failed to parse"):
                await classifier.summarize(sample_paper)

    @pytest.mark.asyncio
    async def test_json_with_markdown_fences(
        self, classifier: Classifier, sample_paper: ParsedPaper
    ) -> None:
        """Test extraction of JSON from markdown code fences."""
        mock_response = """Here's the analysis:

```json
{
    "summary": "A paper about neural networks.",
    "key_points": ["Point 1"],
    "methods": "Deep learning",
    "paper_type": "empirical"
}
```

Hope this helps!"""

        with patch.object(
            classifier, "_call_llm", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_response
            result = await classifier.summarize(sample_paper)

        assert result.summary == "A paper about neural networks."

    @pytest.mark.asyncio
    async def test_invalid_collection_in_response(
        self, classifier: Classifier
    ) -> None:
        """Test handling of unknown collection in LLM response."""
        summary = PaperSummary(
            summary="Test",
            key_points=["Point"],
            methods="Method",
            paper_type=PaperType.EMPIRICAL,
        )

        mock_response = {
            "collections": ["Nonexistent Collection"],
            "tags": [],
            "confidence": 0.5,
            "reasoning": "Unclear",
        }

        with patch.object(
            classifier, "_call_llm", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = json.dumps(mock_response)
            result = await classifier.classify(summary)

        # Should fall back to "Review Later" or handle gracefully
        assert "Review Later" in result.collections or len(result.collections) > 0


class TestPromptFormatting:
    """Tests for prompt formatting."""

    def test_format_summarize_prompt(
        self, classifier: Classifier, sample_paper: ParsedPaper
    ) -> None:
        """Test summarize prompt includes paper content."""
        prompt = classifier._format_summarize_prompt(sample_paper)

        assert "Attention Is All You Need" in prompt
        assert "attention" in prompt.lower()

    def test_format_classify_prompt(self, classifier: Classifier) -> None:
        """Test classify prompt includes collections and tags."""
        summary = PaperSummary(
            summary="Neural network paper",
            key_points=["Point"],
            methods="DL",
            paper_type=PaperType.EMPIRICAL,
        )

        prompt = classifier._format_classify_prompt(summary)

        # Should include collection names and descriptions
        assert "ML / Deep Learning" in prompt
        assert "Methods / Statistics" in prompt
        # Should include tag names
        assert "foundational" in prompt
        assert "methods-focused" in prompt


class TestLLMCall:
    """Tests for LLM API calls."""

    @pytest.mark.asyncio
    async def test_call_llm_openrouter(self, classifier: Classifier) -> None:
        """Test OpenRouter API call."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"test": "response"}'}}]
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await classifier._call_llm("Test prompt")

        assert result == '{"test": "response"}'
        mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_llm_api_error(self, classifier: Classifier) -> None:
        """Test handling of API errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = Exception("API Error")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            with pytest.raises(ClassifierError, match="LLM API call failed"):
                await classifier._call_llm("Test prompt")
