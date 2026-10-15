from pathlib import Path
from core import HLDAnalyzer

def test_sample_pdf_ingestion_and_query():
    path = Path(__file__).parents[1] / "sample_data" / "demo_autosar_hld.pdf"
    service = HLDAnalyzer()
    result = service.ingest_pdf(path.read_bytes(), path.name)
    assert result["page_count"] >= 1
    assert result["document_id"]
    assert isinstance(result["inventory"]["components"], list)
    q = service.query(result["document_id"], "BrakeManager VehicleSpeedSignal")
    assert "answer" in q
    assert "citations" in q
