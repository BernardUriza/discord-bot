"""Tests for insult.core.attachments — file classification and processing."""

from unittest.mock import AsyncMock

from insult.core.attachments import (
    MAX_ATTACHMENT_SIZE,
    AttachmentType,
    classify_attachment,
    process_attachment,
    process_attachments,
)

# --- Classification ---


class TestClassifyAttachment:
    def test_png_image(self):
        att_type, error = classify_attachment("screenshot.png", "image/png", 1000)
        assert att_type == AttachmentType.IMAGE
        assert error is None

    def test_jpg_image(self):
        att_type, _ = classify_attachment("photo.jpg", "image/jpeg", 1000)
        assert att_type == AttachmentType.IMAGE

    def test_webp_image(self):
        att_type, _ = classify_attachment("sticker.webp", "image/webp", 1000)
        assert att_type == AttachmentType.IMAGE

    def test_python_file(self):
        att_type, _ = classify_attachment("main.py", "text/plain", 500)
        assert att_type == AttachmentType.TEXT

    def test_javascript_file(self):
        att_type, _ = classify_attachment("app.js", "application/javascript", 500)
        assert att_type == AttachmentType.TEXT

    def test_json_file(self):
        att_type, _ = classify_attachment("config.json", "application/json", 500)
        assert att_type == AttachmentType.TEXT

    def test_txt_file(self):
        att_type, _ = classify_attachment("notes.txt", "text/plain", 500)
        assert att_type == AttachmentType.TEXT

    def test_csv_file(self):
        att_type, _ = classify_attachment("data.csv", "text/csv", 500)
        assert att_type == AttachmentType.TEXT

    def test_pdf_file(self):
        att_type, _ = classify_attachment("document.pdf", "application/pdf", 1000)
        assert att_type == AttachmentType.PDF

    def test_unsupported_mp3(self):
        att_type, error = classify_attachment("song.mp3", "audio/mpeg", 1000)
        assert att_type == AttachmentType.UNSUPPORTED
        assert error is not None

    def test_unsupported_zip(self):
        att_type, error = classify_attachment("archive.zip", "application/zip", 1000)
        assert att_type == AttachmentType.UNSUPPORTED
        assert error is not None

    def test_too_large(self):
        att_type, error = classify_attachment("big.png", "image/png", MAX_ATTACHMENT_SIZE + 1)
        assert att_type == AttachmentType.UNSUPPORTED
        assert "5MB" in error

    def test_exactly_at_limit(self):
        att_type, error = classify_attachment("ok.png", "image/png", MAX_ATTACHMENT_SIZE)
        assert att_type == AttachmentType.IMAGE
        assert error is None

    def test_text_content_type_fallback(self):
        """Files with text/ content type should be treated as text even with unknown extension."""
        att_type, _ = classify_attachment("data.unknown", "text/plain", 500)
        assert att_type == AttachmentType.TEXT

    def test_no_extension(self):
        att_type, _error = classify_attachment("Makefile", None, 500)
        assert att_type == AttachmentType.UNSUPPORTED


# --- Processing ---


def _mock_attachment(filename: str, content_type: str, size: int, data: bytes):
    att = AsyncMock()
    att.filename = filename
    att.content_type = content_type
    att.size = size
    att.read = AsyncMock(return_value=data)
    return att


class TestProcessAttachment:
    async def test_process_image(self):
        att = _mock_attachment("test.png", "image/png", 100, b"\x89PNG fake image data")
        result = await process_attachment(att)
        assert result.attachment_type == AttachmentType.IMAGE
        assert result.content_block is not None
        assert result.content_block["type"] == "image"
        assert result.content_block["source"]["media_type"] == "image/png"
        assert result.error is None

    async def test_process_text(self):
        att = _mock_attachment("hello.py", "text/plain", 20, b"print('hello')")
        result = await process_attachment(att)
        assert result.attachment_type == AttachmentType.TEXT
        assert result.content_block is not None
        assert result.content_block["type"] == "text"
        assert "hello.py" in result.content_block["text"]
        assert "print('hello')" in result.content_block["text"]

    async def test_process_pdf(self):
        att = _mock_attachment("doc.pdf", "application/pdf", 100, b"%PDF-1.4 fake")
        result = await process_attachment(att)
        assert result.attachment_type == AttachmentType.PDF
        assert result.content_block is not None
        assert result.content_block["type"] == "document"
        assert result.content_block["source"]["media_type"] == "application/pdf"

    async def test_process_unsupported(self):
        att = _mock_attachment("song.mp3", "audio/mpeg", 100, b"fake audio")
        result = await process_attachment(att)
        assert result.content_block is None
        assert result.error is not None

    async def test_process_too_large(self):
        att = _mock_attachment("huge.png", "image/png", MAX_ATTACHMENT_SIZE + 1, b"")
        result = await process_attachment(att)
        assert result.content_block is None
        assert "5MB" in result.error

    async def test_process_download_failure(self):
        att = _mock_attachment("fail.png", "image/png", 100, b"")
        att.read = AsyncMock(side_effect=Exception("Network error"))
        result = await process_attachment(att)
        assert result.content_block is None
        assert result.error is not None

    async def test_text_latin1_fallback(self):
        att = _mock_attachment("legacy.txt", "text/plain", 10, "café".encode("latin-1"))
        result = await process_attachment(att)
        assert result.content_block is not None
        assert "café" in result.content_block["text"]


class TestProcessAttachments:
    async def test_multiple_attachments(self):
        atts = [
            _mock_attachment("code.py", "text/plain", 20, b"x = 1"),
            _mock_attachment("screenshot.png", "image/png", 100, b"\x89PNG data"),
        ]
        blocks, errors = await process_attachments(atts)
        assert len(blocks) == 2
        assert len(errors) == 0

    async def test_mixed_valid_and_invalid(self):
        atts = [
            _mock_attachment("code.py", "text/plain", 20, b"x = 1"),
            _mock_attachment("song.mp3", "audio/mpeg", 100, b"audio"),
        ]
        blocks, errors = await process_attachments(atts)
        assert len(blocks) == 1  # only code.py
        assert len(errors) == 1  # mp3 rejected

    async def test_empty_list(self):
        blocks, errors = await process_attachments([])
        assert blocks == []
        assert errors == []
