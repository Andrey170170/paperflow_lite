"""Configuration loading and validation for paperflow."""

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


def substitute_env_vars(value: str) -> str:
    """Substitute ${VAR_NAME} patterns with environment variables.

    Args:
        value: String potentially containing ${VAR_NAME} patterns.

    Returns:
        String with environment variables substituted.

    Raises:
        ValueError: If referenced environment variable is not set.
    """
    pattern = r"\$\{([^}]+)\}"

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            raise ValueError(f"Environment variable {var_name} not set")
        return env_value

    return re.sub(pattern, replacer, value)


def _substitute_in_dict(data: dict) -> dict:  # type: ignore[type-arg]
    """Recursively substitute env vars in a dictionary."""
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = substitute_env_vars(value)
        elif isinstance(value, dict):
            result[key] = _substitute_in_dict(value)
        elif isinstance(value, list):
            result[key] = [
                _substitute_in_dict(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


class ZoteroConfig(BaseModel):
    """Zotero API configuration."""

    library_id: str = Field(description="Zotero user or group library ID")
    library_type: Literal["user", "group"] = Field(description="Library type")
    api_key: str = Field(description="Zotero API key")
    inbox_collection: str | None = Field(
        description="Collection name to use as inbox (null for all unsorted)",
        default=None,
    )


class ProviderRouting(BaseModel):
    """OpenRouter provider routing configuration."""

    order: list[str] | None = Field(
        description="Provider order preference (e.g., ['google-vertex', 'groq'])",
        default=None,
    )
    allow_fallbacks: bool = Field(
        description="Allow fallback to other providers if preferred unavailable",
        default=True,
    )
    sort: str | None = Field(
        description="Sort providers by metric (e.g., 'throughput', 'price')",
        default=None,
    )
    quantizations: list[str] | None = Field(
        description="Allowed quantization levels (e.g., ['bf16', 'fp16', 'fp32'])",
        default=None,
    )
    require_parameters: bool | None = Field(
        description="Only route to providers that support all parameters (e.g., response_format)",
        default=None,
    )


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = Field(description="LLM provider (e.g., openrouter)")
    api_key: str = Field(description="API key for the provider")
    model: str = Field(description="Model identifier")
    max_tokens: int = Field(description="Maximum tokens in response", default=2000)
    temperature: float = Field(
        description="Temperature for generation",
        default=0.3,
        ge=0.0,
        le=2.0,
    )
    max_retries: int = Field(
        description="Maximum retry attempts for failed API calls",
        default=3,
        ge=1,
    )
    routing: ProviderRouting | None = Field(
        description="OpenRouter provider routing settings",
        default=None,
    )


class ParserConfig(BaseModel):
    """PDF parser configuration."""

    max_pages: int = Field(
        description="Maximum pages to parse per PDF",
        default=10,
        ge=1,
    )
    cache_dir: str = Field(
        description="Directory for caching parsed content",
        default=".cache/parsed",
    )


class ProcessingConfig(BaseModel):
    """Processing behavior configuration."""

    batch_size: int = Field(
        description="Number of papers to process per run",
        default=5,
        ge=1,
    )
    dry_run: bool = Field(
        description="Preview changes without applying them",
        default=False,
    )
    add_summary_note: bool = Field(
        description="Add summary as Zotero note",
        default=True,
    )


class CollectionDef(BaseModel):
    """Definition of a collection for classification."""

    name: str = Field(description="Collection name in Zotero")
    description: str = Field(description="Description for LLM classification")
    keywords: list[str] = Field(
        description="Keywords that suggest this collection",
        default_factory=list,
    )


class TagDef(BaseModel):
    """Definition of a tag for classification."""

    name: str = Field(description="Tag name")
    description: str = Field(description="When to apply this tag")


class AppConfig(BaseModel):
    """Root application configuration."""

    zotero: ZoteroConfig
    llm: LLMConfig
    parser: ParserConfig = Field(default_factory=ParserConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    collections: list[CollectionDef] = Field(
        description="Available collections for classification"
    )
    tags: list[TagDef] = Field(description="Available tags for classification")


def load_config(path: Path) -> AppConfig:
    """Load and validate configuration from a YAML file.

    Automatically loads environment variables from .env file if present.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated AppConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValidationError: If config is invalid.
    """
    # Load .env file if it exists (looks in current dir and parent dirs)
    load_dotenv()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw_data = yaml.safe_load(f)

    # Substitute environment variables
    data = _substitute_in_dict(raw_data)

    return AppConfig.model_validate(data)
