"""Tests for configuration loading."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from paperflow.config import (
    AppConfig,
    CollectionDef,
    LLMConfig,
    ParserConfig,
    ProcessingConfig,
    TagDef,
    ZoteroConfig,
    load_config,
    substitute_env_vars,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestSubstituteEnvVars:
    """Tests for environment variable substitution."""

    def test_basic_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_VAR", "test_value")
        result = substitute_env_vars("${TEST_VAR}")
        assert result == "test_value"

    def test_substitution_in_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("USER_ID", "12345")
        result = substitute_env_vars("prefix_${USER_ID}_suffix")
        assert result == "prefix_12345_suffix"

    def test_multiple_substitutions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VAR1", "one")
        monkeypatch.setenv("VAR2", "two")
        result = substitute_env_vars("${VAR1}-${VAR2}")
        assert result == "one-two"

    def test_missing_env_var_raises(self) -> None:
        # Ensure the variable doesn't exist
        os.environ.pop("NONEXISTENT_VAR_12345", None)
        with pytest.raises(ValueError, match="Environment variable .* not set"):
            substitute_env_vars("${NONEXISTENT_VAR_12345}")

    def test_no_substitution_needed(self) -> None:
        result = substitute_env_vars("plain_string")
        assert result == "plain_string"


class TestZoteroConfig:
    """Tests for ZoteroConfig model."""

    def test_valid_config(self) -> None:
        config = ZoteroConfig(
            library_id="12345",
            library_type="user",
            api_key="secret_key",
            inbox_collection="Inbox",
        )
        assert config.library_id == "12345"
        assert config.library_type == "user"

    def test_library_type_validation(self) -> None:
        # Valid types
        ZoteroConfig(
            library_id="1", library_type="user", api_key="key", inbox_collection=None
        )
        ZoteroConfig(
            library_id="1", library_type="group", api_key="key", inbox_collection=None
        )

        # Invalid type
        with pytest.raises(ValidationError):
            ZoteroConfig(
                library_id="1",
                library_type="invalid",
                api_key="key",
                inbox_collection=None,
            )


class TestLLMConfig:
    """Tests for LLMConfig model."""

    def test_valid_config(self) -> None:
        config = LLMConfig(
            provider="openrouter",
            api_key="secret",
            model="openai/gpt-4.1-mini",
            max_tokens=2000,
            temperature=0.3,
        )
        assert config.model == "openai/gpt-4.1-mini"
        assert config.temperature == 0.3

    def test_temperature_bounds(self) -> None:
        # Valid range
        LLMConfig(
            provider="openrouter",
            api_key="key",
            model="model",
            max_tokens=1000,
            temperature=0.0,
        )
        LLMConfig(
            provider="openrouter",
            api_key="key",
            model="model",
            max_tokens=1000,
            temperature=2.0,
        )

        # Invalid
        with pytest.raises(ValidationError):
            LLMConfig(
                provider="openrouter",
                api_key="key",
                model="model",
                max_tokens=1000,
                temperature=2.5,
            )

    def test_defaults(self) -> None:
        config = LLMConfig(
            provider="openrouter",
            api_key="key",
            model="openai/gpt-4.1-mini",
        )
        assert config.max_tokens == 2000
        assert config.temperature == 0.3


class TestParserConfig:
    """Tests for ParserConfig model."""

    def test_valid_config(self) -> None:
        config = ParserConfig(max_pages=10, cache_dir=".cache/parsed")
        assert config.max_pages == 10
        assert config.cache_dir == ".cache/parsed"

    def test_defaults(self) -> None:
        config = ParserConfig()
        assert config.max_pages == 10
        assert config.cache_dir == ".cache/parsed"


class TestProcessingConfig:
    """Tests for ProcessingConfig model."""

    def test_valid_config(self) -> None:
        config = ProcessingConfig(
            batch_size=5,
            dry_run=False,
            add_summary_note=True,
        )
        assert config.batch_size == 5
        assert not config.dry_run

    def test_defaults(self) -> None:
        config = ProcessingConfig()
        assert config.batch_size == 5
        assert not config.dry_run
        assert config.add_summary_note


class TestCollectionDef:
    """Tests for CollectionDef model."""

    def test_valid_collection(self) -> None:
        coll = CollectionDef(
            name="ML / Deep Learning",
            description="Machine learning papers",
            keywords=["neural", "deep"],
        )
        assert coll.name == "ML / Deep Learning"
        assert len(coll.keywords) == 2

    def test_empty_keywords(self) -> None:
        coll = CollectionDef(
            name="Misc",
            description="Other papers",
            keywords=[],
        )
        assert coll.keywords == []


class TestTagDef:
    """Tests for TagDef model."""

    def test_valid_tag(self) -> None:
        tag = TagDef(name="foundational", description="Seminal papers")
        assert tag.name == "foundational"


class TestLoadConfig:
    """Tests for loading config from file."""

    def test_load_valid_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ZOTERO_API_KEY", "zotero_test_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter_test_key")

        config = load_config(FIXTURES_DIR / "config_valid.yaml")

        assert config.zotero.library_id == "12345"
        assert config.zotero.api_key == "zotero_test_key"
        assert config.llm.api_key == "openrouter_test_key"
        assert config.llm.model == "openai/gpt-4.1-mini"
        assert len(config.collections) == 2
        assert len(config.tags) == 2

    def test_load_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))

    def test_load_invalid_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Even with env vars set, validation should fail
        monkeypatch.setenv("ZOTERO_API_KEY", "key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "key")

        with pytest.raises(ValidationError):
            load_config(FIXTURES_DIR / "config_invalid.yaml")


class TestAppConfig:
    """Tests for full AppConfig model."""

    def test_full_config(self) -> None:
        config = AppConfig(
            zotero=ZoteroConfig(
                library_id="123",
                library_type="user",
                api_key="key",
                inbox_collection="Inbox",
            ),
            llm=LLMConfig(
                provider="openrouter",
                api_key="key",
                model="model",
            ),
            parser=ParserConfig(),
            processing=ProcessingConfig(),
            collections=[
                CollectionDef(name="Test", description="Test collection", keywords=[])
            ],
            tags=[TagDef(name="test-tag", description="A test tag")],
        )
        assert config.zotero.library_id == "123"
        assert len(config.collections) == 1
        assert len(config.tags) == 1
