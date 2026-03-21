"""
Tests for src/helpers/process_raw_texts.py.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.helpers.process_raw_texts import Processor  # pylint: disable=import-error


# pylint: disable=protected-access
class TestProcessRawTexts:
    """
    Class for testing Processor class of src/helpers/process_raw_texts.py.
    """

    @pytest.fixture
    def processor(self):
        """
        Fixture for Processor object.
        """
        return Processor(overwrite=True)

    def test_init_and_repr(self):
        """
        Test Processor initialization with different overwrite parameters and string representation.
        """
        processor_overwrite_true_default = Processor()
        processor_overwrite_true = Processor(overwrite=True)
        processor_overwrite_false = Processor(overwrite=False)
        assert processor_overwrite_true_default.overwrite is True
        assert processor_overwrite_true.overwrite is True
        assert processor_overwrite_false.overwrite is False
        assert (
            str(processor_overwrite_true_default) == "'Processor'(self.overwrite=True)"
        )
        assert str(processor_overwrite_true) == "'Processor'(self.overwrite=True)"
        assert str(processor_overwrite_false) == "'Processor'(self.overwrite=False)"

    def test_process_corpus(self, tmp_path):
        """
        Test process_corpus only processes PDFs and creates output dir.
        """
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        (input_dir / "test1.pdf").touch()
        (input_dir / "test2.pdf").touch()
        (input_dir / "test3.txt").touch()
        (input_dir / "test4.md").touch()

        processor = Processor(overwrite=True)

        with patch.object(processor, "output_md_collection_path", output_dir):
            with patch(
                "glob.glob", return_value=[str(f) for f in input_dir.glob("*.pdf")]
            ):
                with patch.object(processor, "_pdf_to_md") as mock_convert:
                    processor.process_corpus()

        assert output_dir.exists()
        assert mock_convert.call_count == 2

    def test_pdf_to_md_raises_file_not_found_error(self, processor):
        """
        Test _pdf_to_md raises a FileNotFoundError when a file does not exist.
        """
        pdf_path = Path("non-existent.pdf")
        md_path = Path("non-existent.md")

        with patch("pymupdf.open", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                processor._pdf_to_md(pdf_path, md_path)

    def test_handles_special_characters(self, tmp_path, processor):
        """
        Test _pdf_to_md handles special characters correctly.
        """
        pdf_path = tmp_path / "/path/to/pdf.pdf"
        md_path = tmp_path / "path/to/md.md"

        test_markdown = "# Title\n\nCafé, naïve, 你好, emojis 🎉"

        with patch("pymupdf.open"):
            with patch("pymupdf4llm.to_markdown", return_value=test_markdown):
                with patch.object(Path, "write_bytes") as mock_write:
                    processor._pdf_to_md(pdf_path, md_path)

        written_bytes = mock_write.call_args[0][0]
        assert written_bytes.decode("utf-8") == test_markdown

    def test_handles_empty_pdf(self, processor):
        """
        Test _pdf_to_md handles empty PDFs correctly.
        """
        pdf_path = Path("/path/to/empty.pdf")
        md_path = MagicMock(spec=Path)

        with patch("pymupdf.open"):
            with patch("pymupdf4llm.to_markdown", return_value=""):
                processor._pdf_to_md(pdf_path, md_path)

        md_path.write_bytes.assert_called_once_with(b"")
