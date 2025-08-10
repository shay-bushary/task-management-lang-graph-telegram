"""PDF processing utilities with safe file operations and validation."""

import os
import shutil
from pathlib import Path
from typing import Optional, Tuple
import logging

from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFValidationError(Exception):
    """Exception raised when PDF validation fails."""

    pass


def validate_pdf_file(file_path: Path) -> Tuple[bool, Optional[str]]:
    """Validate that a file is a valid PDF.

    Args:
        file_path: Path to the PDF file to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not file_path.exists():
            return False, "File does not exist"

        if not file_path.is_file():
            return False, "Path is not a file"

        # Check file extension
        if file_path.suffix.lower() != ".pdf":
            return False, "File does not have .pdf extension"

        # Check file size (max 50MB)
        file_size = file_path.stat().st_size
        max_size = 50 * 1024 * 1024  # 50MB
        if file_size > max_size:
            return (
                False,
                f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum allowed size (50MB)",
            )

        if file_size == 0:
            return False, "File is empty"

        # Try to read the PDF
        try:
            reader = PdfReader(str(file_path))
            num_pages = len(reader.pages)

            if num_pages == 0:
                return False, "PDF has no pages"

            # Try to extract text from first page to ensure it's readable
            first_page = reader.pages[0]
            text = first_page.extract_text()

            logger.info(
                f"PDF validation successful: {num_pages} pages, {len(text)} characters on first page"
            )
            return True, None

        except Exception as e:
            return False, f"Failed to read PDF: {str(e)}"

    except Exception as e:
        logger.error(f"Error validating PDF {file_path}: {str(e)}")
        return False, f"Validation error: {str(e)}"


def safe_save_uploaded_file(
    file_content: bytes, filename: str, upload_dir: Path
) -> Path:
    """Safely save an uploaded file to the uploads directory.

    Args:
        file_content: The file content as bytes
        filename: Original filename
        upload_dir: Directory to save the file

    Returns:
        Path to the saved file

    Raises:
        PDFValidationError: If the file is not a valid PDF
        OSError: If file operations fail
    """
    try:
        # Ensure upload directory exists
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize filename
        safe_filename = sanitize_filename(filename)

        # Create unique filename if file already exists
        file_path = upload_dir / safe_filename
        counter = 1
        while file_path.exists():
            name_parts = safe_filename.rsplit(".", 1)
            if len(name_parts) == 2:
                name, ext = name_parts
                file_path = upload_dir / f"{name}_{counter}.{ext}"
            else:
                file_path = upload_dir / f"{safe_filename}_{counter}"
            counter += 1

        # Write file content
        with open(file_path, "wb") as f:
            f.write(file_content)

        logger.info(f"File saved to {file_path}")

        # Validate the saved PDF
        is_valid, error_msg = validate_pdf_file(file_path)
        if not is_valid:
            # Clean up invalid file
            try:
                file_path.unlink()
            except Exception:
                pass
            raise PDFValidationError(f"Invalid PDF file: {error_msg}")

        return file_path

    except Exception as e:
        logger.error(f"Error saving uploaded file {filename}: {str(e)}")
        raise


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to make it safe for filesystem operations.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace dangerous characters
    dangerous_chars = '<>:"/\\|?*'
    for char in dangerous_chars:
        filename = filename.replace(char, "_")

    # Remove leading/trailing whitespace and dots
    filename = filename.strip(" .")

    # Ensure filename is not empty
    if not filename:
        filename = "unnamed_file.pdf"

    # Ensure .pdf extension
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    # Limit filename length
    if len(filename) > 255:
        name_part = filename[:-4]  # Remove .pdf
        filename = name_part[:251] + ".pdf"  # Keep .pdf extension

    return filename


def get_pdf_metadata(file_path: Path) -> dict:
    """Extract metadata from a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        Dictionary containing PDF metadata
    """
    try:
        reader = PdfReader(str(file_path))
        metadata = {
            "num_pages": len(reader.pages),
            "file_size": file_path.stat().st_size,
            "filename": file_path.name,
        }

        # Add PDF metadata if available
        if reader.metadata:
            pdf_meta = reader.metadata
            metadata.update(
                {
                    "title": pdf_meta.get("/Title", ""),
                    "author": pdf_meta.get("/Author", ""),
                    "subject": pdf_meta.get("/Subject", ""),
                    "creator": pdf_meta.get("/Creator", ""),
                    "producer": pdf_meta.get("/Producer", ""),
                    "creation_date": str(pdf_meta.get("/CreationDate", "")),
                    "modification_date": str(pdf_meta.get("/ModDate", "")),
                }
            )

        return metadata

    except Exception as e:
        logger.error(f"Error extracting PDF metadata from {file_path}: {str(e)}")
        return {
            "num_pages": 0,
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "filename": file_path.name,
            "error": str(e),
        }


def cleanup_temp_files(file_path: Path) -> None:
    """Clean up temporary files.

    Args:
        file_path: Path to the file to clean up
    """
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary file {file_path}: {str(e)}")
