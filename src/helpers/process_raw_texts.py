"""
Processor that converts PDFs to .MD for better chunking
"""

import glob
from pathlib import Path

import pymupdf
import pymupdf4llm

from src.config.constants import PathsStorage


class Processor:
    """
    Processor of raw PDFs
    """

    pdf_collection_path = PathsStorage.RAW_PDF_COLLECTION.value
    output_md_collection_path = PathsStorage.RAW_MD_COLLECTION.value
    input_path_pattern = f"{pdf_collection_path}/*.pdf"

    def __init__(self, overwrite: bool = True) -> None:
        """
        Initialize an instance of class

        Args:
            overwrite (bool): Flag to overwrite files on every run
        """
        self.overwrite = overwrite

    def __repr__(self) -> str:
        """
        Method that returns string representation of the class

        Returns:
            str: String representation
        """
        return f"{self.__class__.__name__!r}({self.overwrite=!r})"

    def process_corpus(self) -> None:
        """
        Method that processes converting whole corpus of PDFs into .MD files
        """
        self.output_md_collection_path.mkdir(parents=True, exist_ok=True)

        for pdf_path in map(Path, glob.glob(self.input_path_pattern)):
            md_path = (self.output_md_collection_path / pdf_path.stem).with_suffix(
                ".md"
            )
            if self.overwrite or not md_path.exists():
                self._pdf_to_md(pdf_path, md_path)

    def _pdf_to_md(self, pdf_path: Path, md_path: Path) -> None:
        """
        Method that converts single PDF file into .MD

        Args:
            pdf_path (Path): path to PDF file
            md_path (Path): path to .MD file
        """
        with pymupdf.open(pdf_path) as doc:
            md_data = pymupdf4llm.to_markdown(doc)
            clean_md = md_data.encode(encoding="utf-8", errors="surrogatepass").decode(
                encoding="utf-8", errors="ignore"
            )
            md_path.write_bytes(clean_md.encode(encoding="utf-8"))
