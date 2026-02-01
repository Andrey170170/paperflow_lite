"""WebDAV client for downloading PDFs from Zotero WebDAV storage."""

import io
import zipfile

import httpx

from paperflow.config import WebDAVConfig
from paperflow.logging_config import get_logger

logger = get_logger("webdav")


class WebDAVError(Exception):
    """Error raised for WebDAV operations."""

    pass


class WebDAVClient:
    """Client for downloading files from Zotero's WebDAV storage.

    Zotero stores attachments on WebDAV as ZIP files named <attachment_key>.zip.
    Each ZIP contains the actual file (e.g., the PDF).
    """

    def __init__(self, config: WebDAVConfig) -> None:
        """Initialize the WebDAV client.

        Args:
            config: WebDAV configuration with URL and credentials.
        """
        self.config = config
        # Ensure URL doesn't have trailing slash for consistent path joining
        self._base_url = config.url.rstrip("/")
        self._auth = (config.username, config.password)

    def get_file(self, attachment_key: str) -> bytes | None:
        """Download and extract a file from Zotero's WebDAV storage.

        Args:
            attachment_key: The Zotero attachment key (e.g., 'ABC12345').

        Returns:
            File bytes if successful, None if download or extraction fails.
        """
        zip_url = f"{self._base_url}/{attachment_key}.zip"
        logger.debug(f"Downloading from WebDAV: {zip_url}")

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.get(zip_url, auth=self._auth)
                response.raise_for_status()

            zip_bytes = response.content
            logger.debug(f"Downloaded {len(zip_bytes)} bytes")

            # Extract the file from the ZIP
            return self._extract_from_zip(zip_bytes)

        except httpx.HTTPStatusError as e:
            logger.error(f"WebDAV HTTP error: {e.response.status_code} for {zip_url}")
            return None
        except httpx.RequestError as e:
            logger.error(f"WebDAV request error: {e}")
            return None
        except Exception as e:
            logger.error(f"WebDAV unexpected error: {e}")
            return None

    def _extract_from_zip(self, zip_bytes: bytes) -> bytes | None:
        """Extract the first file from a ZIP archive.

        Zotero's WebDAV ZIPs typically contain a single file.

        Args:
            zip_bytes: Raw ZIP file bytes.

        Returns:
            Extracted file bytes, or None if extraction fails.
        """
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                # Get list of files in the archive
                names = zf.namelist()
                if not names:
                    logger.error("ZIP archive is empty")
                    return None

                # Zotero ZIPs contain a single file
                filename = names[0]
                logger.debug(f"Extracting '{filename}' from ZIP")
                return zf.read(filename)

        except zipfile.BadZipFile:
            logger.error("Invalid ZIP file received from WebDAV")
            return None
        except Exception as e:
            logger.error(f"ZIP extraction error: {e}")
            return None
