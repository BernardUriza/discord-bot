"""Discord attachment processing for Claude API multimodal messages.

Classifies attachments by type, downloads content, and converts
to Claude API content blocks (text, image, document).
"""

import base64
from dataclasses import dataclass
from enum import Enum

import structlog

log = structlog.get_logger()

MAX_ATTACHMENT_SIZE = 5 * 1024 * 1024  # 5MB

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
TEXT_EXTENSIONS = {
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
PDF_EXTENSIONS = {".pdf"}

# Claude API supported image media types
IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


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


def classify_attachment(filename: str, content_type: str | None, size: int) -> tuple[AttachmentType, str | None]:
    """Classify a Discord attachment by type. Returns (type, error_or_none)."""
    if size > MAX_ATTACHMENT_SIZE:
        return AttachmentType.UNSUPPORTED, f"Archivo muy pesado ({size / 1024 / 1024:.1f}MB). Maximo 5MB."

    ext = _get_extension(filename)

    if ext in IMAGE_EXTENSIONS:
        return AttachmentType.IMAGE, None
    if ext in TEXT_EXTENSIONS or (content_type and content_type.startswith("text/")):
        return AttachmentType.TEXT, None
    if ext in PDF_EXTENSIONS:
        return AttachmentType.PDF, None

    return AttachmentType.UNSUPPORTED, f"No puedo leer archivos .{ext}. Mandame texto, codigo, imagenes o PDFs."


async def process_attachment(attachment) -> ProcessedAttachment:
    """Process a single discord.Attachment into a Claude API content block.

    Args:
        attachment: discord.Attachment object
    """
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

    if att_type == AttachmentType.IMAGE:
        ext = _get_extension(filename)
        media_type = IMAGE_MEDIA_TYPES.get(ext, "image/png")
        block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.standard_b64encode(data).decode("ascii"),
            },
        }
        log.info("attachment_processed", filename=filename, type="image", size=len(data))
        return ProcessedAttachment(attachment_type=att_type, filename=filename, content_block=block)

    if att_type == AttachmentType.TEXT:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = data.decode("latin-1")
            except Exception:
                return ProcessedAttachment(
                    attachment_type=att_type,
                    filename=filename,
                    content_block=None,
                    error="No pude leer el archivo. Parece que no es texto.",
                )
        block = {"type": "text", "text": f"[Archivo: {filename}]\n```\n{text}\n```"}
        log.info("attachment_processed", filename=filename, type="text", size=len(data))
        return ProcessedAttachment(attachment_type=att_type, filename=filename, content_block=block)

    if att_type == AttachmentType.PDF:
        block = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64.standard_b64encode(data).decode("ascii"),
            },
        }
        log.info("attachment_processed", filename=filename, type="pdf", size=len(data))
        return ProcessedAttachment(attachment_type=att_type, filename=filename, content_block=block)

    return ProcessedAttachment(
        attachment_type=att_type,
        filename=filename,
        content_block=None,
        error="Tipo de archivo no soportado.",
    )


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
