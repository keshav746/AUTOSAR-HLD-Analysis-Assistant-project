import json
import pandas as pd
import streamlit as st
from pathlib import Path
from core import analyzer

st.set_page_config(page_title="AUTOSAR HLD Assistant", page_icon="🚘", layout="wide")

st.markdown("""
<style>
.main .block-container {padding-top: 1.6rem; max-width: 1440px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; background: linear-gradient(115deg,#102a43,#176b87); color: white; margin-bottom: 1rem;}
.hero h1 {color:white; margin-bottom:.3rem; font-size:2rem;}
.hero p {color:#e5f3fa; margin-bottom:0;}
.small-muted {color:#667085;font-size:.88rem}
.metric-card {border:1px solid #e3e8ef;border-radius:12px;padding:14px;background:#fff;}
</style>
<div class="hero">
<h1>🚘 AUTOSAR HLD Document Analysis Assistant</h1>
<p>Evidence-first architecture exploration · traceable findings · engineer-in-the-loop review</p>
</div>
""", unsafe_allow_html=True)

if "active_doc" not in st.session_state:
    st.session_state.active_doc = None

with st.sidebar:
    st.header("Workspace")
    st.caption("Local prototype — documents stay in this running process.")
    sample_path = Path(__file__).parent / "sample_data" / "demo_autosar_hld.pdf"
    if st.button("Load sample HLD", use_container_width=True, type="primary"):
        try:
            doc = analyzer.ingest_pdf(sample_path.read_bytes(), sample_path.name)
            st.session_state.active_doc = doc["document_id"]
            st.success("Sample HLD indexed.")
        except Exception as e:
            st.error(str(e))
    uploaded = st.file_uploader("Upload an HLD PDF", type=["pdf"])
    if uploaded is not None and st.button("Index uploaded document", use_container_width=True):
        try:
            doc = analyzer.ingest_pdf(uploaded.getvalue(), uploaded.name)
            st.session_state.active_doc = doc["document_id"]
            st.success(f"Indexed {uploaded.name}")
        except Exception as e:
            st.error(str(e))
    docs = analyzer.list_documents()
    if docs:
        options = {f'{d["filename"]} · {d["document_id"]}': d["document_id"] for d in docs}
        selected_label = next((k for k,v in options.items() if v == st.session_state.active_doc), list(options.keys())[0])
        choice = st.selectbox("Active document", list(options.keys()), index=list(options.keys()).index(selected_label))
        st.session_state.active_doc = options[choice]
        st.divider()
        st.caption("Indexed documents")
        for d in docs:
            st.write(f'📄 **{d["filename"]}**')
            st.caption(f'{d["page_count"]} pages · ID `{d["document_id"]}`')

doc_id = st.session_state.active_doc
if not doc_id or doc_id not in analyzer.documents:
    st.info("Start by clicking **Load sample HLD** in the sidebar, or upload a text-based HLD PDF.")
    st.markdown("""
    ### What this pilot demonstrates
    - Parse PDF text and preserve page-level evidence
    - Extract likely components, interfaces, signals, and dependencies
    - Search the HLD and return source passages with page citations
    - Compare two revisions and flag potential quality issues
    - Export structured results for engineering review
    """)
    st.stop()

doc = analyzer.documents[doc_id]
inv = doc["inventory"]
findings = doc["findings"]
components = inv["components"]
interfaces = inv["interfaces"]
signals = inv["signals"]
dependencies = inv["dependencies"]

st.caption(f'Active source: **{doc["filename"]}** · {doc["page_count"]} pages · ID `{doc_id}`')
m1,m2,m3,m4 = st.columns(4)
m1.metric("Components detected", len(components))
m2.metric("Interfaces / ports", len(interfaces))
m3.metric("Signals / messages", len(signals))
m4.metric("Review flags", len([f for f in findings if f["severity"] in ("Medium","High")]))

tab_overview, tab_search, tab_inventory, tab_findings, tab_compare, tab_export = st.tabs([
    "Overview", "Ask the HLD", "Architecture Inventory", "Findings & Quality Checks", "Document Comparison", "Export"
])

with tab_overview:
    left,right = st.columns([1.25,1])
    with left:
        st.subheader("Document snapshot")
        st.write("This workspace turns an HLD PDF into a searchable, page-aware engineering reference. Extracted entities are candidates, not authoritative architecture facts.")
        st.markdown("**Analysis pipeline**")
        for n,step in enumerate(["Validate and extract PDF text","Split text into page-aware chunks","Retrieve relevant evidence for each question","Extract candidate architecture entities","Run deterministic review checks","Present findings for engineer disposition"],1):
            st.write(f"{n}. {step}")
    with right:
        st.subheader("Engineering guardrails")
        st.success("Source citations are attached to retrieved passages.")
        st.warning("Heuristic extraction may miss entities or infer false positives.")
        st.info("No automatic architecture approval or source-document modification.")
    st.subheader("Extracted components")
    if components:
        st.dataframe(pd.DataFrame(components), use_container_width=True, hide_index=True)
    else:
        st.info("No recognizable component names found. Review naming conventions and source extraction quality.")
    st.subheader("Dependency snapshot")
    if dependencies:
        st.dataframe(pd.DataFrame(dependencies), use_container_width=True, hide_index=True)
    else:
        st.info("No dependency relationships confidently matched by the current heuristic.")

with tab_search:
    st.subheader("Evidence-grounded question answering")
    st.write("Ask about a component, interface, signal, flow, or design decision. Answers use retrieved HLD passages rather than invented architecture facts.")
    with st.form("query_form"):
        question = st.text_input("Question", placeholder="e.g. Which components depend on VehicleSpeedSignal?")
        submitted = st.form_submit_button("Search HLD", type="primary")
    if submitted:
        try:
            result = analyzer.query(doc_id, question)
            st.markdown(f"**Confidence indicator:** `{result['confidence']}`")
            st.markdown("#### Retrieved answer / evidence")
            st.write(result["answer"])
            if result.get("note"): st.caption(result["note"])
            st.markdown("#### Sources")
            if result["citations"]:
                for i,c in enumerate(result["citations"],1):
                    with st.expander(f'{i}. {c["filename"]} · page {c["page"]} · relevance {c["relevance"]}'):
                        st.caption(f'Section hint: {c["section"]}')
                        st.write(c["excerpt"])
            else:
                st.info("No citations returned. The system intentionally avoids guessing.")
        except Exception as e:
            st.error(str(e))
    st.markdown("**Try these demo questions**")
    examples = [
        "What does BrakeManager do?",
        "Which components depend on VehicleSpeedSignal?",
        "What interfaces or ports are described?",
        "What is the communication path between VehicleStateProvider and BrakeManager?"
    ]
    cols = st.columns(2)
    for i,q in enumerate(examples):
        if cols[i%2].button(q, key=f"q{i}", use_container_width=True):
            result = analyzer.query(doc_id, q)
            st.markdown(f'**Confidence:** `{result["confidence"]}`')
            st.write(result["answer"])
            for c in result["citations"]:
                with st.expander(f'{c["filename"]} · page {c["page"]}'):
                    st.write(c["excerpt"])

with tab_inventory:
    st.subheader("Architecture entity inventory")
    entity_tab1,entity_tab2,entity_tab3,entity_tab4 = st.tabs(["Components","Interfaces / Ports","Signals / Messages","Dependencies"])
    with entity_tab1:
        st.dataframe(pd.DataFrame(components, columns=["name","page"]), use_container_width=True, hide_index=True)
    with entity_tab2:
        st.dataframe(pd.DataFrame(interfaces, columns=["name","page"]), use_container_width=True, hide_index=True)
    with entity_tab3:
        st.dataframe(pd.DataFrame(signals, columns=["name","page"]), use_container_width=True, hide_index=True)
    with entity_tab4:
        st.dataframe(pd.DataFrame(dependencies, columns=["source","target","relationship"]), use_container_width=True, hide_index=True)
    st.caption(inv.get("extraction_note",""))

with tab_findings:
    st.subheader("Potential mismatches and completeness checks")
    st.caption("Findings are advisory, generated by explainable rules, and must be confirmed by an engineer.")
    for i,f in enumerate(findings):
        color = {"High":"🔴","Medium":"🟠","Low":"🟡","Info":"🔵"}.get(f["severity"],"⚪")
        with st.expander(f'{color} {f["severity"]} · {f["category"]} · {f["title"]}', expanded=(i==0)):
            st.write(f["detail"])
            st.markdown("**Evidence**")
            st.code(f["evidence"] or "No text evidence available")
            st.caption(f'Page {f["page"]} · Status: {f["status"]}')
            status = st.selectbox("Review disposition", ["Needs engineering review","Confirmed issue","Not an issue","Deferred"], key=f"status_{i}")
            if st.button("Save disposition for this session", key=f"save_{i}"):
                f["status"] = status
                st.success("Disposition saved for this session.")

with tab_compare:
    st.subheader("Compare two HLD revisions")
    other_docs = [d for d in analyzer.list_documents() if d["document_id"] != doc_id]
    if not other_docs:
        st.info("Index a second HLD PDF from the sidebar to enable revision comparison.")
    else:
        other_map = {f'{d["filename"]} · {d["document_id"]}':d["document_id"] for d in other_docs}
        other_label = st.selectbox("Compare active document against", list(other_map.keys()))
        if st.button("Compare revisions", type="primary"):
            result = analyzer.compare(doc_id, other_map[other_label])
            st.warning(result["warning"])
            for group in ["components","interfaces","signals","dependencies"]:
                st.markdown(f"#### {group.title()}")
                g = result[group]
                if group != "dependencies":
                    a,b,c = st.columns(3)
                    a.markdown("**Added in B**")
                    a.write(g.get("added_in_b",[]))
                    b.markdown("**Removed in B**")
                    b.write(g.get("removed_in_b",[]))
                    c.markdown("**Unchanged**")
                    c.write(g.get("unchanged",[]))
                else:
                    a,b = st.columns(2)
                    a.markdown("**Added relationships**")
                    a.write(g["added_in_b"])
                    b.markdown("**Removed relationships**")
                    b.write(g["removed_in_b"])

with tab_export:
    st.subheader("Export analysis artifacts")
    report = analyzer.report(doc_id)
    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    st.download_button("Download full report (JSON)", report_json, file_name=f"{doc['filename'].rsplit('.',1)[0]}_analysis.json", mime="application/json", type="primary")
    entities_rows = []
    for typ in ["components","interfaces","signals"]:
        for x in inv[typ]:
            entities_rows.append({"entity_type":typ[:-1], "name":x["name"], "page":x["page"]})
    for d in dependencies:
        entities_rows.append({"entity_type":"dependency", "name":f'{d["source"]} -> {d["target"]}', "page":""})
    st.download_button("Download entity inventory (CSV)", pd.DataFrame(entities_rows).to_csv(index=False), file_name="architecture_inventory.csv", mime="text/csv")
    st.download_button("Download findings (CSV)", pd.DataFrame(findings).to_csv(index=False), file_name="hld_findings.csv", mime="text/csv")
    st.markdown("#### Report preview")
    st.json({"document":report["document"],"entity_counts":{"components":len(components),"interfaces":len(interfaces),"signals":len(signals),"dependencies":len(dependencies)},"finding_count":len(findings),"governance":report["governance"]})

st.divider()
st.caption("AUTOSAR HLD Assistant · Recruitment prototype · Evidence-based outputs require engineering validation.")
