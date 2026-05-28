import fitz


def parse_pdf(file_path: str) -> str:
    with fitz.open(file_path) as doc:
        if doc.is_encrypted:
            raise ValueError(f"PDF is encrypted and cannot be parsed: {file_path}")
        return "\n\n".join(page.get_text() for page in doc).strip()