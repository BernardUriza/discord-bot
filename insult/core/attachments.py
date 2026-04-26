"""Discord attachment processing for Claude API multimodal messages.

Classifies attachments by type, downloads content, and converts
to Claude API content blocks (text, image, document).

Each supported type has its own handler. Adding a new type means
adding a handler class and registering it once — no edits to
`process_attachment` or to the classifier.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import structlog

log = structlog.get_logger()

MAX_ATTACHMENT_SIZE = 5 * 1024 * 1024  # 5MB


class AttachmentType(Enum):
    IMAGE = "image"
    TEXT = "text"
    PDF = "pdf"
    UNSUPPORTED = "unsupported"


@dataclass
class ProcessedAttachment:
    """A Discord attachment processed and ready for Claude API."""

    attachment_type: AttachmentType
    filename: str
    content_block: dict | None  # Claude API content block, None if unsupported
    error: str | None = None  # Error message if processing failed


# ---------------------------------------------------------------------------
# Type handlers
# ---------------------------------------------------------------------------


class _Handler:
    """Base for type-specific handlers.

    Subclasses declare which extensions they own and how to turn raw bytes
    into a Claude API content block. Handlers don't deal with download
    failures, size limits, or logging — those live in `process_attachment`
    so each handler stays a pure transform.
    """

    attachment_type: ClassVar[AttachmentType]
    extensions: ClassVar[set[str]]

    def build_block(self, data: bytes, filename: str) -> tuple[dict | None, str | None]:
        """Convert downloaded bytes to a content block.

        Returns ``(block, error)``. ``block`` is None when ``data`` can't be
        rendered (e.g. text that's neither UTF-8 nor latin-1).
        """
        raise NotImplementedError


class ImageHandler(_Handler):
    attachment_type: ClassVar[AttachmentType] = AttachmentType.IMAGE
    extensions: ClassVar[set[str]] = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    # Claude API supported image media types
    media_types: ClassVar[dict[str, str]] = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }

    def build_block(self, data: bytes, filename: str) -> tuple[dict | None, str | None]:
        ext = _get_extension(filename)
        media_type = self.media_types.get(ext, "image/png")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.standard_b64encode(data).decode("ascii"),
            },
        }, None


class TextHandler(_Handler):
    attachment_type: ClassVar[AttachmentType] = AttachmentType.TEXT
    extensions: ClassVar[set[str]] = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".html",
        ".css",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".md",
        ".txt",
        ".csv",
        ".log",
        ".sh",
        ".bash",
        ".zsh",
        ".sql",
        ".xml",
        ".ini",
        ".cfg",
        ".env",
        ".rs",
        ".go",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".r",
        ".lua",
        ".dockerfile",
    }

    def build_block(self, data: bytes, filename: str) -> tuple[dict | None, str | None]:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = data.decode("latin-1")
            except Exception:
                return None, "No pude leer el archivo. Parece que no es texto."
        return {"type": "text", "text": f"[Archivo: {filename}]\n```\n{text}\n```"}, None


class PDFHandler(_Handler):
    attachment_type: ClassVar[AttachmentType] = AttachmentType.PDF
    extensions: ClassVar[set[str]] = {".pdf"}

    def build_block(self, data: bytes, filename: str) -> tuple[dict | None, str | None]:
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64.standard_b64encode(data).decode("ascii"),
            },
        }, None


# Registry — extension lookup walks this in order. New types: append a
# handler instance here and that's it.
_HANDLERS: list[_Handler] = [ImageHandler(), TextHandler(), PDFHandler()]
_TEXT_HANDLER: TextHandler = next(h for h in _HANDLERS if isinstance(h, TextHandler))

# Backwards-compatible exports — older callers and tests import these
# constants directly. Keep them in sync with the handler registry.
IMAGE_EXTENSIONS = ImageHandler.extensions
TEXT_EXTENSIONS = TextHandler.extensions
PDF_EXTENSIONS = PDFHandler.extensions
IMAGE_MEDIA_TYPES = ImageHandler.media_types


def _handler_for(filename: str, content_type: str | None) -> _Handler | None:
    """Pick the handler that owns this file, or None if unsupported.

    Extension match wins; the ``text/*`` content-type fallback rescues
    text files with unknown extensions (e.g. ``data.unknown`` served
    with ``Content-Type: text/plain``) — see test_text_content_type_fallback.
    """
    ext = _get_extension(filename)
    for h in _HANDLERS:
        if ext in h.extensions:
            return h
    if content_type and content_type.startswith("text/"):
        return _TEXT_HANDLER
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_attachment(filename: str, content_type: str | None, size: int) -> tuple[AttachmentType, str | None]:
    """Classify a Discord attachment by type. Returns (type, error_or_none)."""
    if size > MAX_ATTACHMENT_SIZE:
        return AttachmentType.UNSUPPORTED, f"Archivo muy pesado ({size / 1024 / 1024:.1f}MB). Maximo 5MB."

    handler = _handler_for(filename, content_type)
    if handler is not None:
        return handler.attachment_type, None

    ext = _get_extension(filename)
    return AttachmentType.UNSUPPORTED, f"No puedo leer archivos .{ext}. Mandame texto, codigo, imagenes o PDFs."


async def process_attachment(attachment) -> ProcessedAttachment:
    """Process a single discord.Attachment into a Claude API content block."""
    filename = attachment.filename
    att_type, error = classify_attachment(filename, attachment.content_type, attachment.size)

    if error:
        log.warning("attachment_rejected", filename=filename, reason=error)
        return ProcessedAttachment(
            attachment_type=att_type,
            filename=filename,
            content_block=None,
            error=error,
        )

    try:
        data = await attachment.read()
    except Exception as e:
        log.error("attachment_download_failed", filename=filename, error=str(e))
        return ProcessedAttachment(
            attachment_type=att_type,
            filename=filename,
            content_block=None,
            error="No pude descargar el archivo. Intentale de nuevo.",
        )

    handler = _handler_for(filename, attachment.content_type)
    if handler is None:
        return ProcessedAttachment(
            attachment_type=att_type,
            filename=filename,
            content_block=None,
            error="Tipo de archivo no soportado.",
        )

    block, build_error = handler.build_block(data, filename)
    if build_error or block is None:
        return ProcessedAttachment(
            attachment_type=att_type,
            filename=filename,
            content_block=None,
            error=build_error,
        )

    log.info("attachment_processed", filename=filename, type=att_type.value, size=len(data))
    return ProcessedAttachment(attachment_type=att_type, filename=filename, content_block=block)


async def process_attachments(attachments: list) -> tuple[list[dict], list[str]]:
    """Process multiple Discord attachments.

    Returns:
        (content_blocks, errors) — content blocks for Claude API and in-character error messages
    """
    blocks = []
    errors = []

    for att in attachments:
        result = await process_attachment(att)
        if result.content_block:
            blocks.append(result.content_block)
        if result.error:
            errors.append(f"**{result.filename}**: {result.error}")

    return blocks, errors


def _get_extension(filename: str) -> str:
    """Get lowercase file extension including the dot."""
    dot_idx = filename.rfind(".")
    if dot_idx == -1:
        return ""
    return filename[dot_idx:].lower()
