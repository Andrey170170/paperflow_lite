"""Tests for Zotero client."""

from unittest.mock import MagicMock, patch

import pytest

from paperflow.config import ZoteroConfig
from paperflow.models import ZoteroItem
from paperflow.zotero import ZoteroClient, ZoteroError


@pytest.fixture
def zotero_config() -> ZoteroConfig:
    """Create a test Zotero configuration."""
    return ZoteroConfig(
        library_id="12345",
        library_type="user",
        api_key="test_api_key",
        inbox_collection="Inbox",
    )


@pytest.fixture
def mock_pyzotero():
    """Create a mock pyzotero.Zotero instance."""
    with patch("paperflow.zotero.zotero.Zotero") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


class TestZoteroClient:
    """Tests for ZoteroClient class."""

    def test_init(self, zotero_config: ZoteroConfig, mock_pyzotero: MagicMock) -> None:
        """Test client initialization."""
        client = ZoteroClient(zotero_config)
        assert client.config == zotero_config

    def test_get_inbox_items_with_collection(
        self, zotero_config: ZoteroConfig, mock_pyzotero: MagicMock
    ) -> None:
        """Test fetching items from inbox collection."""
        # Mock collections response
        mock_pyzotero.collections.return_value = [
            {"key": "INBOX123", "data": {"name": "Inbox"}},
            {"key": "OTHER456", "data": {"name": "Other"}},
        ]

        # Mock collection items response
        mock_pyzotero.collection_items.return_value = [
            {
                "key": "ITEM001",
                "data": {
                    "itemType": "journalArticle",
                    "title": "Test Paper 1",
                    "creators": [{"lastName": "Smith", "firstName": "John"}],
                    "collections": ["INBOX123"],
                    "tags": [{"tag": "ai"}],
                },
            },
            {
                "key": "ITEM002",
                "data": {
                    "itemType": "journalArticle",
                    "title": "Test Paper 2",
                    "creators": [],
                    "collections": ["INBOX123"],
                    "tags": [],
                },
            },
        ]

        # Mock children (attachments) for each item
        def mock_children(item_key: str) -> list:
            if item_key == "ITEM001":
                return [
                    {
                        "key": "PDF001",
                        "data": {"itemType": "attachment", "contentType": "application/pdf"},
                    }
                ]
            return []

        mock_pyzotero.children.side_effect = mock_children

        client = ZoteroClient(zotero_config)
        items = client.get_inbox_items()

        assert len(items) == 2
        assert items[0].key == "ITEM001"
        assert items[0].title == "Test Paper 1"
        assert items[0].has_pdf
        assert items[0].pdf_attachment_key == "PDF001"
        assert items[1].key == "ITEM002"
        assert not items[1].has_pdf

    def test_get_inbox_items_no_collection(
        self, mock_pyzotero: MagicMock
    ) -> None:
        """Test fetching unfiled items when no inbox collection specified."""
        config = ZoteroConfig(
            library_id="12345",
            library_type="user",
            api_key="key",
            inbox_collection=None,
        )

        mock_pyzotero.items.return_value = [
            {
                "key": "ITEM003",
                "data": {
                    "itemType": "journalArticle",
                    "title": "Unfiled Paper",
                    "creators": [],
                    "collections": [],
                    "tags": [],
                },
            }
        ]
        mock_pyzotero.children.return_value = []

        client = ZoteroClient(config)
        items = client.get_inbox_items()

        assert len(items) == 1
        assert items[0].key == "ITEM003"

    def test_get_inbox_items_collection_not_found(
        self, zotero_config: ZoteroConfig, mock_pyzotero: MagicMock
    ) -> None:
        """Test error when inbox collection doesn't exist."""
        mock_pyzotero.collections.return_value = [
            {"key": "OTHER", "data": {"name": "Other"}}
        ]

        client = ZoteroClient(zotero_config)
        with pytest.raises(ZoteroError, match="Collection .* not found"):
            client.get_inbox_items()

    def test_get_item_pdf(
        self, zotero_config: ZoteroConfig, mock_pyzotero: MagicMock
    ) -> None:
        """Test downloading PDF attachment."""
        mock_pyzotero.file.return_value = b"PDF content bytes"

        client = ZoteroClient(zotero_config)
        pdf_bytes = client.get_item_pdf("PDF001")

        assert pdf_bytes == b"PDF content bytes"
        mock_pyzotero.file.assert_called_once_with("PDF001")

    def test_get_item_pdf_not_found(
        self, zotero_config: ZoteroConfig, mock_pyzotero: MagicMock
    ) -> None:
        """Test handling missing PDF."""
        mock_pyzotero.file.side_effect = Exception("Not found")

        client = ZoteroClient(zotero_config)
        result = client.get_item_pdf("NONEXISTENT")

        assert result is None

    def test_add_to_collection(
        self, zotero_config: ZoteroConfig, mock_pyzotero: MagicMock
    ) -> None:
        """Test adding item to a collection."""
        mock_pyzotero.item.return_value = {
            "key": "ITEM001",
            "version": 1,
            "data": {
                "collections": ["OLD_COLL"],
            },
        }

        client = ZoteroClient(zotero_config)
        client.add_to_collection("ITEM001", "NEW_COLL")

        # Verify update was called with new collection added
        mock_pyzotero.update_item.assert_called_once()
        call_args = mock_pyzotero.update_item.call_args[0][0]
        assert "NEW_COLL" in call_args["data"]["collections"]
        assert "OLD_COLL" in call_args["data"]["collections"]

    def test_add_tags(
        self, zotero_config: ZoteroConfig, mock_pyzotero: MagicMock
    ) -> None:
        """Test adding tags to an item."""
        mock_pyzotero.item.return_value = {
            "key": "ITEM001",
            "version": 1,
            "data": {
                "tags": [{"tag": "existing"}],
            },
        }

        client = ZoteroClient(zotero_config)
        client.add_tags("ITEM001", ["new-tag-1", "new-tag-2"])

        mock_pyzotero.update_item.assert_called_once()
        call_args = mock_pyzotero.update_item.call_args[0][0]
        tags = [t["tag"] for t in call_args["data"]["tags"]]
        assert "existing" in tags
        assert "new-tag-1" in tags
        assert "new-tag-2" in tags

    def test_add_note(
        self, zotero_config: ZoteroConfig, mock_pyzotero: MagicMock
    ) -> None:
        """Test adding a note to an item."""
        client = ZoteroClient(zotero_config)
        client.add_note("ITEM001", "<p>Summary content</p>")

        mock_pyzotero.create_items.assert_called_once()
        call_args = mock_pyzotero.create_items.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0]["itemType"] == "note"
        assert call_args[0]["parentItem"] == "ITEM001"
        assert call_args[0]["note"] == "<p>Summary content</p>"

    def test_remove_from_collection(
        self, zotero_config: ZoteroConfig, mock_pyzotero: MagicMock
    ) -> None:
        """Test removing item from a collection."""
        mock_pyzotero.item.return_value = {
            "key": "ITEM001",
            "version": 1,
            "data": {
                "collections": ["INBOX123", "OTHER456"],
            },
        }

        client = ZoteroClient(zotero_config)
        client.remove_from_collection("ITEM001", "INBOX123")

        mock_pyzotero.update_item.assert_called_once()
        call_args = mock_pyzotero.update_item.call_args[0][0]
        assert "INBOX123" not in call_args["data"]["collections"]
        assert "OTHER456" in call_args["data"]["collections"]

    def test_get_collection_key(
        self, zotero_config: ZoteroConfig, mock_pyzotero: MagicMock
    ) -> None:
        """Test finding collection key by name."""
        mock_pyzotero.collections.return_value = [
            {"key": "ABC123", "data": {"name": "ML Papers"}},
            {"key": "DEF456", "data": {"name": "Review Later"}},
        ]

        client = ZoteroClient(zotero_config)

        assert client.get_collection_key("ML Papers") == "ABC123"
        assert client.get_collection_key("Review Later") == "DEF456"
        assert client.get_collection_key("Nonexistent") is None

    def test_mark_as_processed(
        self, zotero_config: ZoteroConfig, mock_pyzotero: MagicMock
    ) -> None:
        """Test marking item as processed with a tag."""
        mock_pyzotero.item.return_value = {
            "key": "ITEM001",
            "version": 1,
            "data": {
                "tags": [],
            },
        }

        client = ZoteroClient(zotero_config)
        client.mark_as_processed("ITEM001")

        mock_pyzotero.update_item.assert_called_once()
        call_args = mock_pyzotero.update_item.call_args[0][0]
        tags = [t["tag"] for t in call_args["data"]["tags"]]
        assert "_paperflow_processed" in tags

    def test_is_processed(
        self, zotero_config: ZoteroConfig, mock_pyzotero: MagicMock
    ) -> None:
        """Test checking if item is already processed."""
        client = ZoteroClient(zotero_config)

        item_processed = ZoteroItem(
            key="ITEM001",
            title="Processed",
            creators=[],
            item_type="journalArticle",
            collections=[],
            tags=["_paperflow_processed"],
            has_pdf=True,
            pdf_attachment_key="PDF001",
        )
        assert client.is_processed(item_processed)

        item_not_processed = ZoteroItem(
            key="ITEM002",
            title="Not Processed",
            creators=[],
            item_type="journalArticle",
            collections=[],
            tags=["other-tag"],
            has_pdf=True,
            pdf_attachment_key="PDF002",
        )
        assert not client.is_processed(item_not_processed)


class TestZoteroItemParsing:
    """Tests for parsing Zotero API responses."""

    def test_format_creators(
        self, zotero_config: ZoteroConfig, mock_pyzotero: MagicMock
    ) -> None:
        """Test creator name formatting."""
        mock_pyzotero.collections.return_value = [
            {"key": "INBOX123", "data": {"name": "Inbox"}}
        ]
        mock_pyzotero.collection_items.return_value = [
            {
                "key": "ITEM001",
                "data": {
                    "itemType": "journalArticle",
                    "title": "Test",
                    "creators": [
                        {"lastName": "Smith", "firstName": "John"},
                        {"lastName": "Doe", "firstName": "Jane"},
                        {"name": "Organization Name"},
                    ],
                    "collections": ["INBOX123"],
                    "tags": [],
                },
            }
        ]
        mock_pyzotero.children.return_value = []

        client = ZoteroClient(zotero_config)
        items = client.get_inbox_items()

        assert len(items) == 1
        assert "Smith, John" in items[0].creators
        assert "Doe, Jane" in items[0].creators
        assert "Organization Name" in items[0].creators
