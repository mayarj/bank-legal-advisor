import os
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.rag.pipeline import run_pipeline

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/", summary="Upload a PDF and ingest the legislation it contains")
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF file containing legislation text"),
    session: AsyncSession = Depends(get_db),
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        legislation = await run_pipeline(tmp_path, session)
    finally:
        os.unlink(tmp_path)

    if legislation is None:
        raise HTTPException(
            status_code=422,
            detail="Could not extract legislation from the uploaded file. "
                   "Ensure the PDF contains structured legislative text.",
        )

    return {
        "message": "Legislation ingested successfully.",
        "code": legislation.code,
        "subject": legislation.subject,
        "status": legislation.status.value,
        "articles_count": len(legislation.articles),
        "relationships_count": len(legislation.relationships),
    }