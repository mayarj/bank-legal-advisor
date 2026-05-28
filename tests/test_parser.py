import pytest
import fitz

from src.rag.parser import parse_pdf


@pytest.fixture
def sample_pdf(tmp_path) -> str:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "LAW-88-2003: Banking Loan Collateral Requirements")
    page.insert_text((50, 100), "Article 1: This law governs collateral requirements for all banking loans.")
    page.insert_text((50, 120), "Article 2: All loans exceeding 50,000 units must be secured by real estate.")
    path = tmp_path / "test_law.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def multi_page_pdf(tmp_path) -> str:
    doc = fitz.open()
    for i in range(1, 4):
        page = doc.new_page()
        page.insert_text((50, 72), f"Page {i}: Article {i} content about banking regulation.")
    path = tmp_path / "multi_page.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def empty_pdf(tmp_path) -> str:
    doc = fitz.open()
    doc.new_page()  # blank page, no text
    path = tmp_path / "empty.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


# ── parse_pdf ─────────────────────────────────────────────────────────────────

class TestParsePdf:

    def test_returns_string(self, sample_pdf):
        result = parse_pdf(sample_pdf)

        assert isinstance(result, str)

    def test_extracts_text_content(self, sample_pdf):
        result = parse_pdf(sample_pdf)

        assert "LAW-88-2003" in result
        assert "Article 1" in result
        assert "Article 2" in result

    def test_extracts_all_pages(self, multi_page_pdf):
        result = parse_pdf(multi_page_pdf)

        assert "Page 1" in result
        assert "Page 2" in result
        assert "Page 3" in result

    def test_pages_separated_by_newlines(self, multi_page_pdf):
        result = parse_pdf(multi_page_pdf)

        assert "\n\n" in result

    def test_empty_pdf_returns_empty_string(self, empty_pdf):
        result = parse_pdf(empty_pdf)

        assert result == ""

    def test_raises_on_missing_file(self):
        with pytest.raises(Exception):
            parse_pdf("/non/existent/file.pdf")

    def test_result_is_stripped(self, sample_pdf):
        result = parse_pdf(sample_pdf)

        assert result == result.strip()