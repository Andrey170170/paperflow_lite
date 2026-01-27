"""LLM classifier for paper summarization and classification."""

import json
import re
from pathlib import Path

import httpx

from paperflow.config import CollectionDef, LLMConfig, TagDef
from paperflow.models import Classification, PaperSummary, ParsedPaper

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class ClassifierError(Exception):
    """Error raised for classification issues."""

    pass


class Classifier:
    """LLM-based classifier for paper summarization and categorization."""

    def __init__(
        self,
        llm_config: LLMConfig,
        collections: list[CollectionDef],
        tags: list[TagDef],
    ) -> None:
        """Initialize the classifier.

        Args:
            llm_config: LLM provider configuration.
            collections: Available collections for classification.
            tags: Available tags for classification.
        """
        self.llm_config = llm_config
        self.collections = collections
        self.tags = tags
        self._prompts_dir = Path(__file__).parent.parent.parent / "prompts"

    async def summarize(self, paper: ParsedPaper) -> PaperSummary:
        """Generate a summary of the paper.

        Args:
            paper: Parsed paper content.

        Returns:
            PaperSummary with extracted information.

        Raises:
            ClassifierError: If summarization fails.
        """
        prompt = self._format_summarize_prompt(paper)
        response = await self._call_llm(prompt)
        return self._parse_response(response, PaperSummary)

    async def classify(self, summary: PaperSummary) -> Classification:
        """Classify a paper based on its summary.

        Args:
            summary: Paper summary.

        Returns:
            Classification with collections and tags.

        Raises:
            ClassifierError: If classification fails.
        """
        prompt = self._format_classify_prompt(summary)
        response = await self._call_llm(prompt)
        classification = self._parse_response(response, Classification)

        # Validate collections exist, fall back to "Review Later" if not
        valid_collection_names = {c.name for c in self.collections}
        validated_collections = [
            c for c in classification.collections if c in valid_collection_names
        ]

        if not validated_collections:
            # Fall back to "Review Later" if it exists
            review_later = next(
                (c.name for c in self.collections if "review" in c.name.lower()),
                self.collections[-1].name if self.collections else "Unknown",
            )
            validated_collections = [review_later]

        return Classification(
            collections=validated_collections,
            tags=classification.tags,
            confidence=classification.confidence,
            reasoning=classification.reasoning,
        )

    async def process(
        self, paper: ParsedPaper
    ) -> tuple[PaperSummary, Classification]:
        """Full processing pipeline: summarize then classify.

        Args:
            paper: Parsed paper content.

        Returns:
            Tuple of (PaperSummary, Classification).

        Raises:
            ClassifierError: If processing fails.
        """
        summary = await self.summarize(paper)
        classification = await self.classify(summary)
        return summary, classification

    def _format_summarize_prompt(self, paper: ParsedPaper) -> str:
        """Format the summarization prompt.

        Args:
            paper: Parsed paper.

        Returns:
            Formatted prompt string.
        """
        template = self._load_prompt("summarize")

        # Build content section
        content_parts = []
        if paper.title:
            content_parts.append(f"Title: {paper.title}")
        if paper.abstract:
            content_parts.append(f"Abstract: {paper.abstract}")
        content_parts.append(f"Full text:\n{paper.full_text[:10000]}")  # Limit text

        content = "\n\n".join(content_parts)
        return template.replace("{content}", content)

    def _format_classify_prompt(self, summary: PaperSummary) -> str:
        """Format the classification prompt.

        Args:
            summary: Paper summary.

        Returns:
            Formatted prompt string.
        """
        template = self._load_prompt("classify")

        # Format summary section
        summary_text = (
            f"Summary: {summary.summary}\n"
            f"Key Points:\n"
            + "\n".join(f"- {p}" for p in summary.key_points)
            + f"\nMethods: {summary.methods}\n"
            f"Paper Type: {summary.paper_type.value}"
        )

        # Format collections section
        collections_text = "\n".join(
            f"- **{c.name}**: {c.description}" for c in self.collections
        )

        # Format tags section
        tags_text = "\n".join(f"- **{t.name}**: {t.description}" for t in self.tags)

        prompt = template.replace("{summary}", summary_text)
        prompt = prompt.replace("{collections}", collections_text)
        prompt = prompt.replace("{tags}", tags_text)

        return prompt

    def _load_prompt(self, name: str) -> str:
        """Load a prompt template from file.

        Args:
            name: Prompt name (without extension).

        Returns:
            Prompt template string.
        """
        prompt_path = self._prompts_dir / f"{name}.md"
        if prompt_path.exists():
            return prompt_path.read_text()

        # Fallback prompts if files don't exist
        if name == "summarize":
            return self._default_summarize_prompt()
        elif name == "classify":
            return self._default_classify_prompt()
        return ""

    def _default_summarize_prompt(self) -> str:
        """Return default summarization prompt."""
        return """You are a research assistant helping organize an academic paper library.

Given the following paper content, extract:

1. **Summary** (2-3 sentences): What is this paper about? What problem does it address?
2. **Key Points** (3-5 bullets): Main findings, contributions, or arguments
3. **Methods** (1-2 sentences): What methodology or approach was used?
4. **Paper Type**: One of [empirical, theoretical, review, methods, commentary]

## Paper Content

{content}

## Output Format

Respond in JSON:
```json
{
  "summary": "...",
  "key_points": ["...", "..."],
  "methods": "...",
  "paper_type": "..."
}
```"""

    def _default_classify_prompt(self) -> str:
        """Return default classification prompt."""
        return """You are a research librarian classifying papers into a personal collection.

Given the paper summary and the available collections/tags, determine:
1. Which collection(s) this paper belongs to (1-2 max)
2. Which tags apply

## Paper Summary

{summary}

## Available Collections

{collections}

## Available Tags

{tags}

## Output Format

Respond in JSON:
```json
{
  "collections": ["Collection Name"],
  "tags": ["tag1", "tag2"],
  "confidence": 0.85,
  "reasoning": "Brief explanation"
}
```"""

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM API.

        Args:
            prompt: Prompt to send.

        Returns:
            LLM response content.

        Raises:
            ClassifierError: If API call fails.
        """
        headers = {
            "Authorization": f"Bearer {self.llm_config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.llm_config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.llm_config.max_tokens,
            "temperature": self.llm_config.temperature,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise ClassifierError(f"LLM API call failed: {e}") from e

    def _parse_response(
        self,
        response: str,
        model: type[PaperSummary] | type[Classification],
    ) -> PaperSummary | Classification:
        """Parse LLM response into a Pydantic model.

        Args:
            response: Raw LLM response.
            model: Target Pydantic model class.

        Returns:
            Parsed model instance.

        Raises:
            ClassifierError: If parsing fails.
        """
        # Try to extract JSON from markdown code fences
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        json_str = json_match.group(1) if json_match else response

        try:
            data = json.loads(json_str.strip())
            return model.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ClassifierError(f"Failed to parse LLM response: {e}") from e
