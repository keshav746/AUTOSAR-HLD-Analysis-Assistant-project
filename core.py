from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import io
import re
import json
from collections import Counter, defaultdict

import fitz
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class PageText:
    page: int
    text: str

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    page: int
    section: str
    text: str

@dataclass
class Finding:
    severity: str
    category: str
    title: str
    detail: str
    evidence: str
    page: int
    status: str = "Needs engineering review"


class HLDAnalyzer:
    """Small, deterministic local HLD analysis service for a reproducible pilot."""
    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}

    def ingest_pdf(self, pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        if not pdf_bytes:
            raise ValueError("The uploaded file is empty.")
        if not (filename.lower().endswith(".pdf") or pdf_bytes[:4] == b"%PDF"):
            raise ValueError("Please upload a PDF file.")
        if len(pdf_bytes) > 30 * 1024 * 1024:
            raise ValueError("File is larger than the 30 MB pilot limit.")
        try:
            pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"Could not read PDF: {exc}") from exc
        pages: List[PageText] = []
        for i, page in enumerate(pdf):
            text = page.get_text("text") or ""
            pages.append(PageText(i + 1, text.strip()))
        pdf.close()
        combined = "\n".join(p.text for p in pages)
        if len(combined.strip()) < 20:
            raise ValueError("No readable text was found. This pilot supports text-based PDFs; scanned PDFs need OCR.")
        doc_id = hashlib.sha256(pdf_bytes).hexdigest()[:12]
        chunks = self._make_chunks(doc_id, filename, pages)
        doc = {
            "document_id": doc_id,
            "filename": filename,
            "page_count": len(pages),
            "pages": [asdict(p) for p in pages],
            "chunks": [asdict(c) for c in chunks],
            "text": combined,
            "inventory": self._extract_entities(combined, pages),
            "findings": [],
        }
        doc["findings"] = [asdict(f) for f in self._quality_checks(doc)]
        self.documents[doc_id] = doc
        return self.public_document(doc)

    def public_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        return {k: doc[k] for k in ("document_id", "filename", "page_count", "inventory", "findings")}

    def _make_chunks(self, doc_id: str, filename: str, pages: List[PageText]) -> List[Chunk]:
        chunks = []
        for p in pages:
            text = p.text
            if not text:
                continue
            # Preserve paragraph context and cap chunk size to keep retrieval focused.
            paragraphs = [re.sub(r"\s+", " ", x).strip() for x in re.split(r"\n\s*\n", text) if x.strip()]
            if not paragraphs:
                paragraphs = [re.sub(r"\s+", " ", text).strip()]
            buf = ""
            section = "Unclassified section"
            for para in paragraphs:
                heading = re.match(r"^(?:\d+(?:\.\d+)*\.?\s+)?([A-Z][A-Za-z0-9 /()_-]{2,90})$", para)
                if heading and len(para) < 90:
                    section = para
                if buf and len(buf) + len(para) > 900:
                    chunks.append(Chunk(f"{doc_id}-p{p.page}-c{len(chunks)+1}", doc_id, filename, p.page, section, buf))
                    buf = ""
                buf = (buf + "\n" + para).strip()
            if buf:
                chunks.append(Chunk(f"{doc_id}-p{p.page}-c{len(chunks)+1}", doc_id, filename, p.page, section, buf))
        return chunks

    def _extract_entities(self, text: str, pages: List[PageText]) -> Dict[str, Any]:
        # Deliberately conservative, explainable extraction. It does not claim semantic certainty.
        comp_pattern = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*(?:Manager|Controller|Handler|Service|Gateway|Provider|Monitor|Coordinator|Component|SWC)\b")
        interface_pattern = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:Port|Interface|If|API)\b")
        signal_pattern = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:Signal|Msg|Message|Frame|PDU|DID)\b")
        dep_pattern = re.compile(
            r"\b([A-Z][A-Za-z0-9]*(?:Manager|Controller|Handler|Service|Gateway|Provider|Monitor|Coordinator|Component|SWC))\b"
            r".{0,100}?\b(?:depends on|uses|consumes|requires|calls|receives from|communicates with)\b.{0,50}?"
            r"\b([A-Z][A-Za-z0-9]*(?:Manager|Controller|Handler|Service|Gateway|Provider|Monitor|Coordinator|Component|SWC))\b",
            re.I
        )
        components = sorted(set(comp_pattern.findall(text)))
        interfaces = sorted(set(interface_pattern.findall(text)))
        signals = sorted(set(signal_pattern.findall(text)))
        dependencies = []
        for match in dep_pattern.finditer(text):
            dependencies.append({"source": match.group(1), "target": match.group(2), "relationship": "depends on / uses"})
        # Add common arrow notation if present.
        for line in text.splitlines():
            m = re.search(r"\b([A-Z][A-Za-z0-9]*(?:Manager|Controller|Handler|Service|Gateway|Provider|Monitor|Coordinator|Component|SWC))\s*[-=]+>\s*([A-Z][A-Za-z0-9]*(?:Manager|Controller|Handler|Service|Gateway|Provider|Monitor|Coordinator|Component|SWC))", line)
            if m:
                dependencies.append({"source": m.group(1), "target": m.group(2), "relationship": "declared flow"})
        # Deduplicate while preserving order.
        seen, unique_deps = set(), []
        for d in dependencies:
            key = (d["source"], d["target"], d["relationship"])
            if key not in seen:
                seen.add(key)
                unique_deps.append(d)
        page_map = {}
        for entity_type, values in (("component", components), ("interface", interfaces), ("signal", signals)):
            page_map[entity_type] = {}
            for value in values:
                page_map[entity_type][value] = next((p.page for p in pages if value in p.text), 1)
        return {
            "components": [{"name": n, "page": page_map["component"].get(n, 1)} for n in components],
            "interfaces": [{"name": n, "page": page_map["interface"].get(n, 1)} for n in interfaces],
            "signals": [{"name": n, "page": page_map["signal"].get(n, 1)} for n in signals],
            "dependencies": unique_deps,
            "extraction_note": "Heuristic extraction from readable PDF text; validate all entities and relationships with an engineer."
        }

    def _quality_checks(self, doc: Dict[str, Any]) -> List[Finding]:
        findings: List[Finding] = []
        text = doc["text"]
        pages = doc["pages"]
        inventory = doc["inventory"]
        # Repeated identifier patterns: identifiers repeated in different contexts may be legitimate; flag for review only.
        id_matches = re.findall(r"\b(?:IF|SIG|PORT|SWC|COMP)[_-]?\d{2,6}\b", text, flags=re.I)
        counts = Counter(x.upper() for x in id_matches)
        for ident, count in counts.items():
            if count > 3:
                p = next((x["page"] for x in pages if ident.lower() in x["text"].lower()), 1)
                findings.append(Finding("Low", "Repeated identifier", f"Identifier {ident} appears {count} times",
                    "Repeated references may be normal. Confirm that the identifier is consistently defined and not duplicated across unrelated entities.",
                    self._evidence(text, ident), p))
        all_components = {x["name"] for x in inventory["components"]}
        for dep in inventory["dependencies"]:
            if dep["source"] not in all_components or dep["target"] not in all_components:
                continue
            # no warning when both exist; relationships remain listed as evidence.
        # Potential unresolved dependencies from prose patterns.
        for m in re.finditer(r"\b([A-Z][A-Za-z0-9]*(?:Manager|Controller|Handler|Service|Gateway|Provider|Monitor|Coordinator|Component|SWC))\b.{0,70}\b(?:depends on|requires|uses)\b.{0,50}\b([A-Z][A-Za-z0-9]*(?:Manager|Controller|Handler|Service|Gateway|Provider|Monitor|Coordinator|Component|SWC))\b", text, re.I):
            source, target = m.group(1), m.group(2)
            if target not in all_components:
                p = next((x["page"] for x in pages if m.group(0) in x["text"]), 1)
                findings.append(Finding("Medium", "Potential unresolved dependency", f"{source} references {target}",
                    "The referenced target was not found in the extracted component inventory. It may be named differently or omitted from the document.",
                    m.group(0), p))
        if len(text.strip()) < 500:
            findings.append(Finding("Medium", "Limited document text", "Document contains little extractable text",
                "The document may be incomplete or scanned. Confirm extraction quality before relying on analysis.",
                text[:300], 1))
        if not inventory["components"]:
            findings.append(Finding("Medium", "No components detected", "No recognizable architecture components were extracted",
                "The HLD may use a different naming convention. Review the source and configure extraction patterns if needed.",
                (text[:250] or "No readable text"), 1))
        if not findings:
            findings.append(Finding("Info", "No automatic flags", "No potential issues were flagged by basic checks",
                "This does not establish correctness or completeness. Review extracted entities and dependencies manually.",
                "Automated checks completed; no heuristic flags produced.", 1))
        return findings

    @staticmethod
    def _evidence(text: str, needle: str, radius: int = 180) -> str:
        i = text.lower().find(needle.lower())
        if i < 0:
            return text[:radius]
        return text[max(0, i-radius):min(len(text), i+len(needle)+radius)].strip()

    def query(self, document_id: str, question: str, top_k: int = 4) -> Dict[str, Any]:
        doc = self._get_doc(document_id)
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        chunks = doc["chunks"]
        if not chunks:
            return {"answer": "No searchable text is available in this document.", "citations": [], "confidence": "Low"}

        corpus = [c["text"] for c in chunks]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
        try:
            matrix = vectorizer.fit_transform(corpus + [question])
            scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        except ValueError:
            scores = [0.0] * len(chunks)

        # Intent-aware section routing: a question about "Case Study 2" should
        # retrieve that section, not the table of contents where the phrase also appears.
        case_match = re.search(r"\bcase\s+study\s+(\d+)\b", question, flags=re.I)
        routed_pages = set()
        if case_match:
            target_num = int(case_match.group(1))
            pages = doc["pages"]
            start_page = None
            end_page = len(pages) + 1
            header_pattern = re.compile(rf"\bCase\s+Study\s+{target_num}\b", re.I)
            next_header_pattern = re.compile(r"\bCase\s+Study\s+\d+\b", re.I)
            for page in pages:
                # Prefer a standalone section heading, which avoids matching the contents list.
                if re.search(rf"(?m)^\s*Case\s+Study\s+{target_num}\s*$", page["text"], re.I):
                    start_page = page["page"]
                    break
            if start_page is not None:
                for page in pages:
                    if page["page"] > start_page and next_header_pattern.search(page["text"]):
                        # Ignore contents pages before the routed section starts.
                        if page["page"] > start_page + 1:
                            end_page = page["page"]
                            break
                routed_pages = set(range(start_page, end_page))

        # Boost chunks on the routed section pages and penalize front-matter/contents
        # mentions. Exact query phrase matches also receive a modest boost.
        adjusted = list(scores)
        for i, chunk in enumerate(chunks):
            text = chunk["text"]
            page = int(chunk["page"])
            if routed_pages:
                if page in routed_pages:
                    adjusted[i] += 0.75
                else:
                    adjusted[i] *= 0.20
            normalized_q = re.sub(r"\s+", " ", question.lower())
            normalized_text = re.sub(r"\s+", " ", text.lower())
            if len(normalized_q) > 5 and normalized_q in normalized_text:
                adjusted[i] += 0.25
            if case_match and re.search(rf"\bCase\s+Study\s+{int(case_match.group(1))}\b", text, re.I):
                adjusted[i] += 0.35

        ranked = sorted(range(len(adjusted)), key=lambda i: adjusted[i], reverse=True)[:max(top_k, 6 if routed_pages else top_k)]
        selected = [(chunks[i], float(adjusted[i]), float(scores[i])) for i in ranked if adjusted[i] > 0.015]
        if not selected:
            return {
                "answer": "I could not find supporting evidence for this question in the uploaded HLD. Try different terminology or verify that the relevant section is present. No answer has been inferred.",
                "citations": [], "confidence": "Low",
                "note": "Evidence-first behavior: the assistant does not invent architecture details when retrieval returns no relevant passage."
            }

        # For section-title questions, include the actual section introduction and the
        # next few relevant chunks in document order to provide a coherent overview.
        if case_match and routed_pages:
            selected = sorted(selected, key=lambda item: (item[0]["page"], -item[1]))[:max(top_k, 4)]
        else:
            selected = selected[:top_k]

        citations = []
        for c, adjusted_score, lexical_score in selected:
            citations.append({
                "filename": c["filename"], "page": c["page"], "section": c["section"],
                "excerpt": c["text"][:900], "relevance": round(adjusted_score, 3)
            })

        if case_match and routed_pages:
            answer = (
                f"Case Study {int(case_match.group(1))} section evidence from the uploaded document is shown below. "
                "The passages are presented in source order; review the citations to confirm the full scope and details."
            )
            for c, _, _ in selected[:4]:
                excerpt = re.sub(r"\s+", " ", c["text"]).strip()
                if len(excerpt) > 650:
                    excerpt = excerpt[:647].rsplit(" ", 1)[0] + "..."
                answer += f"\n\n{excerpt} [Source: {c['filename']}, p. {c['page']}]"
            confidence = "High" if selected else "Low"
        else:
            answer_parts = []
            for c, _, _ in selected[:3]:
                excerpt = re.sub(r"\s+", " ", c["text"]).strip()
                if len(excerpt) > 430:
                    excerpt = excerpt[:427].rsplit(" ", 1)[0] + "..."
                answer_parts.append(f"{excerpt} [Source: {c['filename']}, p. {c['page']}]")
            answer = "\n\n".join(answer_parts)
            best = max(x[2] for x in selected)
            confidence = "High" if best >= 0.18 else "Medium" if best >= 0.06 else "Low"

        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "note": "Retrieved source passages are evidence, not an approval. Confirm the cited HLD text before using it for engineering decisions."
        }

    def compare(self, document_a: str, document_b: str) -> Dict[str, Any]:
        a, b = self._get_doc(document_a), self._get_doc(document_b)
        inv_a, inv_b = a["inventory"], b["inventory"]
        def names(inv, key): return {x["name"] for x in inv[key]}
        comps_a, comps_b = names(inv_a, "components"), names(inv_b, "components")
        interfaces_a, interfaces_b = names(inv_a, "interfaces"), names(inv_b, "interfaces")
        signals_a, signals_b = names(inv_a, "signals"), names(inv_b, "signals")
        deps_a = {(x["source"], x["target"]) for x in inv_a["dependencies"]}
        deps_b = {(x["source"], x["target"]) for x in inv_b["dependencies"]}
        return {
            "document_a": a["filename"], "document_b": b["filename"],
            "components": {"added_in_b": sorted(comps_b-comps_a), "removed_in_b": sorted(comps_a-comps_b), "unchanged": sorted(comps_a & comps_b)},
            "interfaces": {"added_in_b": sorted(interfaces_b-interfaces_a), "removed_in_b": sorted(interfaces_a-interfaces_b), "unchanged": sorted(interfaces_a & interfaces_b)},
            "signals": {"added_in_b": sorted(signals_b-signals_a), "removed_in_b": sorted(signals_a-signals_b), "unchanged": sorted(signals_a & signals_b)},
            "dependencies": {"added_in_b": [list(x) for x in sorted(deps_b-deps_a)], "removed_in_b": [list(x) for x in sorted(deps_a-deps_b)]},
            "warning": "Comparison is based on extracted names and relationships, not a formal semantic diff. Review against source pages."
        }

    def report(self, document_id: str) -> Dict[str, Any]:
        doc = self._get_doc(document_id)
        return {
            "report_version": "1.0",
            "document": {"document_id": doc["document_id"], "filename": doc["filename"], "page_count": doc["page_count"]},
            "inventory": doc["inventory"],
            "findings": doc["findings"],
            "governance": "AI-assisted findings are advisory and require authorized engineering review."
        }

    def _get_doc(self, document_id: str) -> Dict[str, Any]:
        if document_id not in self.documents:
            raise ValueError(f"Document '{document_id}' is not indexed in this session.")
        return self.documents[document_id]

    def list_documents(self) -> List[Dict[str, Any]]:
        return [self.public_document(d) for d in self.documents.values()]


analyzer = HLDAnalyzer()
