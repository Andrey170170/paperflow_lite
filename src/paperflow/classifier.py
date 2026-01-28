"""LLM classifier for paper summarization and classification."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import httpx

from paperflow.config import CollectionDef, LLMConfig, TagDef
from paperflow.logging_config import get_logger
from paperflow.models import Classification, PaperSummary, ParsedPaper

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

logger = get_logger("classifier")


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
            ClassifierError: If summarization fails after retries.
        """
        prompt = self._format_summarize_prompt(paper)
        return await self._call_llm_with_parse(prompt, PaperSummary)

    async def classify(self, summary: PaperSummary) -> Classification:
        """Classify a paper based on its summary.

        Args:
            summary: Paper summary.

        Returns:
            Classification with collections and tags.

        Raises:
            ClassifierError: If classification fails after retries.
        """
        prompt = self._format_classify_prompt(summary)
        classification = await self._call_llm_with_parse(prompt, Classification)

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

    async def _call_llm_with_parse(
        self,
        prompt: str,
        model: type[PaperSummary] | type[Classification],
    ) -> PaperSummary | Classification:
        """Call LLM and parse response, with retry on both API and parse failures.

        Args:
            prompt: Prompt to send.
            model: Target Pydantic model class for parsing.

        Returns:
            Parsed model instance.

        Raises:
            ClassifierError: If all retries exhausted.
        """
        max_retries = self.llm_config.max_retries
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                response = await self._call_llm_once(prompt)
                return self._parse_response(response, model)
            except ClassifierError as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = 2**attempt  # Exponential backoff: 2, 4, 8...
                    print(
                        f"  [Retry {attempt}/{max_retries}] {e} - retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)

        raise ClassifierError(f"Failed after {max_retries} attempts: {last_error}")

    async def _call_llm_once(self, prompt: str) -> str:
        """Make a single LLM API call (no retry).

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

        payload: dict[str, Any] = {
            "model": self.llm_config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.llm_config.max_tokens,
            "temperature": self.llm_config.temperature,
            "response_format": {"type": "json_object"},
        }

        # Add provider routing if configured
        provider_config: dict[str, Any] = {}
        if self.llm_config.routing:
            routing = self.llm_config.routing

            if routing.order:
                provider_config["order"] = routing.order
            if routing.allow_fallbacks is not None:
                provider_config["allow_fallbacks"] = routing.allow_fallbacks
            if routing.sort:
                provider_config["sort"] = routing.sort
            if routing.quantizations:
                provider_config["quantizations"] = routing.quantizations
            if routing.require_parameters is not None:
                provider_config["require_parameters"] = routing.require_parameters

        # Always require providers to support our parameters (especially response_format)
        if "require_parameters" not in provider_config:
            provider_config["require_parameters"] = True

        if provider_config:
            payload["provider"] = provider_config

        # Log request (without API key)
        log_payload = {**payload, "messages": [{"role": "user", "content": f"<prompt length={len(prompt)}>"}]}
        logger.info(f"LLM request: model={self.llm_config.model}")
        logger.debug(f"LLM request payload: {json.dumps(log_payload, indent=2)}")
        logger.debug(f"LLM request prompt:\n{prompt[:2000]}{'...' if len(prompt) > 2000 else ''}")

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

                # Log response
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                logger.info(
                    f"LLM response: status=200, "
                    f"prompt_tokens={usage.get('prompt_tokens', 'N/A')}, "
                    f"completion_tokens={usage.get('completion_tokens', 'N/A')}"
                )
                logger.debug(f"LLM response content:\n{content}")

                return content
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM API HTTP error: {e.response.status_code} - {e.response.text}")
            raise ClassifierError(f"LLM API call failed: {e}") from e
        except Exception as e:
            logger.error(f"LLM API error: {e}")
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
        json_str = self._extract_json(response)

        try:
            data = json.loads(json_str)
            return model.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ClassifierError(f"Failed to parse LLM response: {e}") from e

    def _extract_json(self, response: str) -> str:
        """Extract JSON from LLM response, handling common formatting issues.

        Args:
            response: Raw LLM response text.

        Returns:
            Cleaned JSON string.
        """
        # Try to extract from markdown code fences first (handle leading garbage like ".")
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            brace_match = re.search(r"\{.*\}", response, re.DOTALL)
            json_str = brace_match.group(0) if brace_match else response

        json_str = json_str.strip()

        # Fix common LLM JSON issues:

        # 0. Handle double opening braces: "{\n{" or "{ {" -> "{"
        # This happens when LLM wraps JSON in extra braces
        json_str = re.sub(r"^\{\s*\{", "{", json_str)
        # Also handle double closing braces at the end
        json_str = re.sub(r"\}\s*\}$", "}", json_str)

        # 1. Remove trailing commas before } or ]
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

        # 2. Fix unquoted keys (key: value -> "key": value)
        json_str = re.sub(r"(\{|,)\s*(\w+)\s*:", r'\1"\2":', json_str)

        # 3. Replace single quotes with double quotes (careful with apostrophes)
        # Only replace if it looks like a string delimiter
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
        json_str = re.sub(r"\[\s*'([^']*)'", r'["\1"', json_str)
        json_str = re.sub(r"',\s*'", '", "', json_str)
        json_str = re.sub(r"'\s*\]", '"]', json_str)

        # 4. Fix newlines inside string values (common LLM issue)
        # This regex finds strings and replaces newlines within them
        def fix_string_newlines(m: re.Match[str]) -> str:
            content = m.group(1)
            # Replace actual newlines with escaped newlines
            content = content.replace("\n", "\\n").replace("\r", "\\r")
            # Replace tabs
            content = content.replace("\t", "\\t")
            return f'"{content}"'

        # Match strings: "..." but not already escaped \"
        json_str = re.sub(r'"((?:[^"\\]|\\.)*)"', fix_string_newlines, json_str)

        # 5. Fix unescaped quotes within strings (try to detect and escape)
        # This is tricky - if we still can't parse, try a more aggressive approach
        try:
            json.loads(json_str)
        except json.JSONDecodeError:
            # Try removing any control characters
            json_str = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", json_str)
            # Normalize whitespace
            json_str = re.sub(r"\s+", " ", json_str)

        return json_str
