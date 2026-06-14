import os
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.rag.pipeline import run_pipeline, run_pipeline_from_text

router = APIRouter(prefix="/ingest", tags=["ingest"])

_SUPPORTED_EXTENSIONS = (".pdf", ".txt")


@router.post("/", summary="Upload a PDF or text file and ingest the legislation it contains")
async def ingest_document(
    file: UploadFile = File(..., description="PDF or .txt file containing legislation text"),
    session: AsyncSession = Depends(get_db),
):
    filename = (file.filename or "").lower()
    if not filename.endswith(_SUPPORTED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Only PDF or .txt files are accepted.")

    content = await file.read()

    if filename.endswith(".txt"):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Text file must be UTF-8 encoded.")
        legislation = await run_pipeline_from_text(text, session)
    else:
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
                   "Ensure it contains structured legislative text.",
        )

    return {
        "message": "Legislation ingested successfully.",
        "code": legislation.code,
        "subject": legislation.subject,
        "status": legislation.status.value,
        "articles_count": len(legislation.articles),
        "relationships_count": len(legislation.relationships),
    }