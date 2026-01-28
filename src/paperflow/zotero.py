"""Zotero API client for paperflow."""

from pyzotero import zotero

from paperflow.config import ZoteroConfig
from paperflow.models import ZoteroItem

PROCESSED_TAG = "_paperflow_processed"


class ZoteroError(Exception):
    """Error raised for Zotero API issues."""

    pass


class ZoteroClient:
    """Client for interacting with Zotero API."""

    def __init__(self, config: ZoteroConfig) -> None:
        """Initialize the Zotero client.

        Args:
            config: Zotero configuration.
        """
        self.config = config
        self._client = zotero.Zotero(
            config.library_id,
            config.library_type,
            config.api_key,
        )
        self._collections_cache: dict[str, str] | None = None

    def get_inbox_items(self) -> list[ZoteroItem]:
        """Fetch items from the inbox collection.

        If no inbox collection is configured, fetches unfiled items.
        Filters out attachments and notes (only returns top-level items).

        Returns:
            List of ZoteroItem objects.

        Raises:
            ZoteroError: If inbox collection not found.
        """
        if self.config.inbox_collection:
            collection_key = self._find_collection_key(self.config.inbox_collection)
            if collection_key is None:
                raise ZoteroError(
                    f"Collection '{self.config.inbox_collection}' not found"
                )
            raw_items = self._client.collection_items(collection_key)
        else:
            # Fetch all items (unfiled)
            raw_items = self._client.items()

        # Filter out attachments and notes - we only want top-level items (papers)
        top_level_items = [
            item for item in raw_items
            if item.get("data", {}).get("itemType") not in ("attachment", "note")
        ]

        return [self._parse_item(item) for item in top_level_items]

    def get_item_pdf(self, attachment_key: str) -> bytes | None:
        """Download PDF content for an attachment.

        Args:
            attachment_key: Key of the PDF attachment.

        Returns:
            PDF bytes or None if download fails.
        """
        try:
            return self._client.file(attachment_key)
        except Exception:
            return None

    def add_to_collection(self, item_key: str, collection_key: str) -> None:
        """Add an item to a collection.

        Args:
            item_key: Key of the item.
            collection_key: Key of the collection.
        """
        item = self._client.item(item_key)
        collections = item["data"].get("collections", [])
        if collection_key not in collections:
            collections.append(collection_key)
            item["data"]["collections"] = collections
            self._client.update_item(item)

    def add_tags(self, item_key: str, tags: list[str]) -> None:
        """Add tags to an item.

        Args:
            item_key: Key of the item.
            tags: List of tag names to add.
        """
        item = self._client.item(item_key)
        existing_tags = item["data"].get("tags", [])
        existing_tag_names = {t["tag"] for t in existing_tags}

        for tag in tags:
            if tag not in existing_tag_names:
                existing_tags.append({"tag": tag})

        item["data"]["tags"] = existing_tags
        self._client.update_item(item)

    def add_note(self, item_key: str, html_content: str) -> None:
        """Add a note to an item.

        Args:
            item_key: Key of the parent item.
            html_content: HTML content for the note.
        """
        note = {
            "itemType": "note",
            "parentItem": item_key,
            "note": html_content,
        }
        self._client.create_items([note])

    def remove_from_collection(self, item_key: str, collection_key: str) -> None:
        """Remove an item from a collection.

        Args:
            item_key: Key of the item.
            collection_key: Key of the collection to remove from.
        """
        item = self._client.item(item_key)
        collections = item["data"].get("collections", [])
        if collection_key in collections:
            collections.remove(collection_key)
            item["data"]["collections"] = collections
            self._client.update_item(item)

    def get_collection_key(self, name: str) -> str | None:
        """Find a collection's key by its name.

        Args:
            name: Collection name to find.

        Returns:
            Collection key or None if not found.
        """
        return self._find_collection_key(name)

    def mark_as_processed(self, item_key: str) -> None:
        """Mark an item as processed by adding a tag.

        Args:
            item_key: Key of the item.
        """
        self.add_tags(item_key, [PROCESSED_TAG])

    def is_processed(self, item: ZoteroItem) -> bool:
        """Check if an item has already been processed.

        Args:
            item: ZoteroItem to check.

        Returns:
            True if already processed.
        """
        return PROCESSED_TAG in item.tags

    def _find_collection_key(self, name: str) -> str | None:
        """Find collection key by name.

        Args:
            name: Collection name.

        Returns:
            Collection key or None.
        """
        if self._collections_cache is None:
            self._load_collections_cache()

        return self._collections_cache.get(name)  # type: ignore[union-attr]

    def _load_collections_cache(self) -> None:
        """Load all collections into cache."""
        collections = self._client.collections()
        self._collections_cache = {
            coll["data"]["name"]: coll["key"] for coll in collections
        }

    def _parse_item(self, raw: dict) -> ZoteroItem:  # type: ignore[type-arg]
        """Parse a raw Zotero API response into ZoteroItem.

        Args:
            raw: Raw API response dict.

        Returns:
            Parsed ZoteroItem.
        """
        data = raw["data"]

        # Format creator names
        creators = []
        for creator in data.get("creators", []):
            if "name" in creator:
                creators.append(creator["name"])
            else:
                last = creator.get("lastName", "")
                first = creator.get("firstName", "")
                if first:
                    creators.append(f"{last}, {first}")
                else:
                    creators.append(last)

        # Check for PDF attachment
        has_pdf = False
        pdf_key: str | None = None
        try:
            children = self._client.children(raw["key"])
            for child in children:
                child_data = child.get("data", {})
                if child_data.get("contentType") == "application/pdf":
                    has_pdf = True
                    pdf_key = child["key"]
                    break
        except Exception:
            pass

        return ZoteroItem(
            key=raw["key"],
            title=data.get("title", "Untitled"),
            creators=creators,
            item_type=data.get("itemType", "unknown"),
            collections=data.get("collections", []),
            tags=[t["tag"] for t in data.get("tags", [])],
            has_pdf=has_pdf,
            pdf_attachment_key=pdf_key,
        )
