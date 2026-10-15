# AUTOSAR HLD Document Analysis Assistant
**Recruitment case study implementation — Case Study 1**

A runnable prototype for extracting, searching, comparing, and reviewing AUTOSAR High-Level Design (HLD) documents. It follows the case-study brief's emphasis on evidence-based outputs, citations, traceability, controlled ingestion, and human review.

## What is implemented
- PDF upload and text extraction (PyMuPDF; OCR is an extension point)
- Context-aware chunking with page references
- Heuristic extraction of components, interfaces/ports, signals, and dependencies
- Evidence-grounded search / Q&A with page citations and an explicit "not found" response
- Potential inconsistency checks (duplicate identifiers, unresolved dependencies, terminology variants)
- Two-document comparison
- Component inventory and dependency views
- CSV / JSON report export
- Demo HLD included under `sample_data/`
- Streamlit UI and FastAPI endpoints
- Local TF-IDF retrieval; no API key or external LLM required for demo
- Optional local sentence-transformer embeddings documented as a future enhancement

## Quick start
Python 3.10+ recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. Upload a real HLD PDF or use the included demo document.

## Run the API (optional)
In a second terminal:
```bash
uvicorn api:app --reload
```
Open `http://127.0.0.1:8000/docs` for interactive API documentation.

Endpoints:
- `GET /health`
- `POST /documents` — upload a PDF
- `GET /documents` — list indexed documents
- `POST /query` — ask a question against a document
- `GET /documents/{document_id}/inventory` — extracted architecture entities
- `POST /compare` — compare two indexed documents

## Demo walkthrough for evaluator
1. Open the Streamlit app.
2. Select **Load sample HLD** in the sidebar.
3. Review the dashboard and extracted architecture inventory.
4. Ask: `Which components depend on the VehicleSpeedSignal?`
5. Ask: `What does BrakeManager do?`
6. Open **Findings & Quality Checks** to see potential issues and evidence.
7. Use **Document Comparison** after loading a second PDF.
8. Export the report as JSON or CSV.

## Architecture
```text
PDF HLD
  -> Validation + PyMuPDF extraction
  -> Page-aware, section-aware chunks
  -> Local lexical retrieval (TF-IDF)
  -> Entity extraction + deterministic checks
  -> Evidence-grounded answer with page citations
  -> Human review + CSV/JSON export

Streamlit UI -------------------- FastAPI
        \                          /
         Shared analysis service
```

## Important engineering limitations
- This is a recruitment-ready **prototype**, not a certified AUTOSAR compliance checker.
- Entity extraction and inconsistency checks are heuristic and must be validated against real, approved HLDs.
- PDF text extraction works best for text-based PDFs. OCR for scanned documents is an explicit future extension.
- The local retrieval engine uses TF-IDF to keep the demo reproducible and dependency-light. For production, evaluate BGE/E5 or another approved embedding model with FAISS/ChromaDB and an approved local/private LLM.
- AI outputs must be reviewed by authorized engineering specialists before design approval, release, or vehicle validation. Do not upload confidential company documents unless authorized.

## Suggested next steps for production
1. Add OCR using Tesseract for scanned PDFs.
2. Replace/augment TF-IDF with evaluated BGE/E5 embeddings and persistent ChromaDB/FAISS.
3. Integrate an approved local/private LLM for richer grounded summaries, with strict context-only prompting.
4. Add role-based access, project isolation, version approval, audit logs, and PostgreSQL.
5. Benchmark extraction precision/recall and answer groundedness with architecture engineers.

## Project structure
```text
AUTOSAR_HLD_Analysis_Assistant/
├── app.py
├── api.py
├── core.py
├── requirements.txt
├── README.md
├── Dockerfile
├── .gitignore
├── sample_data/demo_autosar_hld.pdf
└── tests/test_core.py
```


## Retrieval fix (v1.1)
The query pipeline now routes explicit `Case Study N` questions toward the matching section pages and down-ranks table-of-contents mentions. Confidence is based on retrieval evidence for ordinary questions. This improves section-specific questions without claiming the prototype is a full generative RAG system.
