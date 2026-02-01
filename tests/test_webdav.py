"""Tests for WebDAV client."""

import io
import zipfile

import httpx
import pytest
import respx

from paperflow.config import WebDAVConfig
from paperflow.webdav import WebDAVClient


@pytest.fixture
def webdav_config() -> WebDAVConfig:
    """Create a test WebDAV configuration."""
    return WebDAVConfig(
        url="https://nextcloud.example.com/remote.php/webdav/zotero",
        username="testuser",
        password="testpass",
    )


@pytest.fixture
def sample_zip_with_pdf() -> bytes:
    """Create a sample ZIP file containing a PDF."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Minimal PDF header (not a real PDF, but enough for testing)
        zf.writestr("document.pdf", b"%PDF-1.4 fake pdf content")
    return buffer.getvalue()


class TestWebDAVClient:
    """Tests for WebDAVClient class."""

    def test_init(self, webdav_config: WebDAVConfig) -> None:
        """Test client initialization."""
        client = WebDAVClient(webdav_config)
        assert client.config == webdav_config
        assert client._base_url == "https://nextcloud.example.com/remote.php/webdav/zotero"

    def test_init_strips_trailing_slash(self) -> None:
        """Test that trailing slash is stripped from URL."""
        config = WebDAVConfig(
            url="https://example.com/webdav/",
            username="user",
            password="pass",
        )
        client = WebDAVClient(config)
        assert client._base_url == "https://example.com/webdav"

    @respx.mock
    def test_get_file_success(
        self,
        webdav_config: WebDAVConfig,
        sample_zip_with_pdf: bytes,
    ) -> None:
        """Test successful file download and extraction."""
        respx.get(
            "https://nextcloud.example.com/remote.php/webdav/zotero/ABC12345.zip"
        ).mock(return_value=httpx.Response(200, content=sample_zip_with_pdf))

        client = WebDAVClient(webdav_config)
        result = client.get_file("ABC12345")

        assert result is not None
        assert b"%PDF-1.4 fake pdf content" in result

    @respx.mock
    def test_get_file_not_found(self, webdav_config: WebDAVConfig) -> None:
        """Test handling of 404 response."""
        respx.get(
            "https://nextcloud.example.com/remote.php/webdav/zotero/NOTFOUND.zip"
        ).mock(return_value=httpx.Response(404))

        client = WebDAVClient(webdav_config)
        result = client.get_file("NOTFOUND")

        assert result is None

    @respx.mock
    def test_get_file_unauthorized(self, webdav_config: WebDAVConfig) -> None:
        """Test handling of 401 response."""
        respx.get(
            "https://nextcloud.example.com/remote.php/webdav/zotero/NOAUTH.zip"
        ).mock(return_value=httpx.Response(401))

        client = WebDAVClient(webdav_config)
        result = client.get_file("NOAUTH")

        assert result is None

    @respx.mock
    def test_get_file_network_error(self, webdav_config: WebDAVConfig) -> None:
        """Test handling of network errors."""
        respx.get(
            "https://nextcloud.example.com/remote.php/webdav/zotero/ERROR.zip"
        ).mock(side_effect=httpx.ConnectError("Connection refused"))

        client = WebDAVClient(webdav_config)
        result = client.get_file("ERROR")

        assert result is None

    def test_extract_from_zip_invalid(self, webdav_config: WebDAVConfig) -> None:
        """Test handling of invalid ZIP data."""
        client = WebDAVClient(webdav_config)
        result = client._extract_from_zip(b"not a zip file")

        assert result is None

    def test_extract_from_zip_empty(self, webdav_config: WebDAVConfig) -> None:
        """Test handling of empty ZIP archive."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w"):
            pass  # Create empty ZIP
        empty_zip = buffer.getvalue()

        client = WebDAVClient(webdav_config)
        result = client._extract_from_zip(empty_zip)

        assert result is None

    @respx.mock
    def test_get_file_uses_basic_auth(self, webdav_config: WebDAVConfig) -> None:
        """Test that Basic Auth credentials are sent."""
        route = respx.get(
            "https://nextcloud.example.com/remote.php/webdav/zotero/AUTH123.zip"
        ).mock(return_value=httpx.Response(404))

        client = WebDAVClient(webdav_config)
        client.get_file("AUTH123")

        # Check that the request was made with auth
        assert route.called
        request = route.calls[0].request
        assert "authorization" in request.headers
        # Basic auth header should be present
        assert request.headers["authorization"].startswith("Basic ")
