from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from core import analyzer

app = FastAPI(
    title="AUTOSAR HLD Analysis API",
    description="Local prototype API for HLD ingestion, cited search, inventory, and comparison.",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    document_id: str
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)

class CompareRequest(BaseModel):
    document_a: str
    document_b: str

@app.get("/health")
def health():
    return {"status":"ok","service":"autosar-hld-analysis-api","version":"1.0.0"}

@app.get("/documents")
def list_documents():
    return {"documents": analyzer.list_documents()}

@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
    try:
        data = await file.read()
        result = analyzer.ingest_pdf(data, file.filename)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/query")
def query(req: QueryRequest):
    try:
        return analyzer.query(req.document_id, req.question, req.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@app.get("/documents/{document_id}/inventory")
def inventory(document_id: str):
    try:
        report = analyzer.report(document_id)
        return {"document": report["document"], "inventory": report["inventory"]}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@app.get("/documents/{document_id}/report")
def report(document_id: str):
    try:
        return analyzer.report(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@app.post("/compare")
def compare(req: CompareRequest):
    try:
        return analyzer.compare(req.document_a, req.document_b)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
