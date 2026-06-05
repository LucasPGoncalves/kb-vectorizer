from __future__ import annotations

import textwrap
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

QDRANT_URL: str = "http://localhost:6333"
COLLECTION_NAME: str = "hybrid_rag_docs"

# We drop the dual-chunking strategy. One solid chunk size is enough when 
# a reranker is reading the final text.
CHUNK_SIZE: int = 500     
OVERLAP_RATIO: float = 0.10     

# Dense model for concepts, Sparse model for exact keywords, Cross-Encoder for final rank
DENSE_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL: str = "Qdrant/bm25"
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FETCH_K: int = 30         # Broad net: candidates retrieved from Qdrant
FINAL_TOP_N: int = 5      # Final context window size fed to the LLM

RAW_DOCUMENTS: list[dict[str, str]] = [
    {
        "doc_id": "doc-001",
        "title": "UPS System Architecture Overview",
        "text": (
            "An Uninterruptible Power Supply (UPS) system acts as a critical "
            "buffer between the utility grid and sensitive electronic equipment. "
            "Modern UPS units employ double-conversion topology, which completely "
            "isolates the load from raw mains power. During a utility outage, the "
            "inverter seamlessly draws from the battery bank without any transfer "
            "gap, protecting servers from brownouts and voltage spikes. Proper "
            "sizing requires calculating both kVA load and battery runtime "
            "requirements to meet tier-three data center uptime mandates."
        ),
    },
    {
        "doc_id": "doc-002",
        "title": "Causes of Unexpected Server Shutdown Events",
        "text": (
            "Unexpected server shutdowns are most frequently caused by three "
            "root categories: electrical supply disruptions (including mains "
            "failure, voltage sag, and harmonic distortion), thermal runaway "
            "events triggered by blocked airflow or CRAC unit malfunction, and "
            "firmware-level watchdog timeouts. Electrical disruptions account "
            "for approximately 43 percent of unplanned downtime incidents in "
            "enterprise environments. Best practice mandates deploying redundant "
            "PDUs paired with online UPS systems and generator failover within "
            "a maximum ten-second switchover window."
        ),
    },
    {
        "doc_id": "doc-003",
        "title": "Incident Report – Fault Unit SN-X2047-QR",
        "text": (
            "INCIDENT REPORT 2024-11-14. Affected hardware: APC Smart-UPS "
            "unit serial number SN-X2047-QR, rack position U22, cabinet "
            "CAB-NORTH-07. Fault code: E0xF3 — Battery Internal Resistance "
            "Exceeded Threshold. Unit SN-X2047-QR was flagged by the DCIM "
            "platform at 03:17 UTC. Maintenance action: replace battery "
            "cartridge BCT-4890 immediately. Until replacement, unit "
            "SN-X2047-QR must be treated as BYPASS MODE ONLY and all "
            "downstream loads must be migrated to SN-X2051-QR."
        ),
    },
    {
        "doc_id": "doc-004",
        "title": "Battery Chemistry: VRLA vs Lithium-Ion in Data Centres",
        "text": (
            "Valve-Regulated Lead Acid (VRLA) batteries have dominated UPS "
            "installations for decades due to low upfront cost, but their "
            "energy density ceiling, sensitivity to operating temperature, and "
            "mandatory 3-to-5 year replacement cycles are accelerating adoption "
            "of Lithium-Ion alternatives. Li-Ion cells offer two to three times "
            "greater cycle life, tolerate wider temperature ranges, and provide "
            "remote state-of-health monitoring via built-in battery management "
            "systems (BMS). Total cost of ownership over a 10-year horizon "
            "typically favours Li-Ion despite its higher capital expenditure."
        ),
    },
    {
        "doc_id": "doc-005",
        "title": "Generator Failover and ATS Configuration",
        "text": (
            "An Automatic Transfer Switch (ATS) is the critical relay that "
            "disconnects a facility from utility power and connects it to "
            "standby generator output when mains voltage drops below acceptable "
            "thresholds. NFPA 110 mandates a maximum 10-second transfer time "
            "for Level 1 systems (life-safety loads). Configuration best "
            "practices include setting undervoltage pickup at 90 percent of "
            "nominal and frequency deviation at plus or minus 3 Hz. "
            "Exercising the generator under 30 percent load monthly and "
            "under 100 percent load quarterly ensures readiness."
        ),
    },
    {
        "doc_id": "doc-006",
        "title": "DCIM Platform Integration Guide",
        "text": (
            "The Data Centre Infrastructure Management (DCIM) platform "
            "aggregates real-time telemetry from PDUs, UPS units, CRAC "
            "systems, and environmental sensors. Asset tagging follows the "
            "format <TYPE>-<ZONE>-<UNIT_SEQ>, e.g., UPS-NORTH-07 or "
            "PDU-SOUTH-12. Each physical device is cross-referenced by its "
            "manufacturer serial number so that fault events raised in SNMP "
            "traps can be automatically correlated to rack-level impact "
            "assessments. Alert thresholds are configurable per asset class "
            "and route to PagerDuty, Slack, or ServiceNow depending on "
            "severity tier."
        ),
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedCandidate:
    chunk_id: str
    parent_doc_id: str
    text: str

@dataclass
class RerankedResult:
    parent_doc_id: str
    best_text: str
    relevance_score: float

# ──────────────────────────────────────────────────────────────────────────────
# 2. CHUNKING UTILITY (Simplified)
# ──────────────────────────────────────────────────────────────────────────────

def chunk_document(doc_id: str, text: str, chunk_size: int, overlap: float) -> list[dict]:
    step = int(chunk_size * (1 - overlap))
    chunks = []
    start = 0
    i = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        fragment = text[start:end].strip()
        if fragment:
            human_id = f"{doc_id}::chunk::{i}"
            chunks.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, human_id)),
                "chunk_id_str": human_id,
                "document": fragment,
                "metadata": {"parent_doc_id": doc_id, "chunk_index": i}
            })
        if end == len(text):
            break
        start += step
        i += 1
    return chunks

# ──────────────────────────────────────────────────────────────────────────────
# 3. ENSEMBLE RAG PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

class HybridRAGPipeline:
    def __init__(self, raw_documents: list[dict[str, str]]) -> None:
        print("\n" + "═" * 70)
        print("  Initialising Hybrid Qdrant + Cross-Encoder RAG")
        print("═" * 70)

        # ── 1. Initialize Qdrant Client ──────────────────────────────────────
        self.client = QdrantClient(url=QDRANT_URL)
        self.client.set_model(DENSE_MODEL)
        self.client.set_sparse_model(SPARSE_MODEL)

        # ── 2. Initialize Reranker ───────────────────────────────────────────
        print(f"[Models] Loading Cross-Encoder: {RERANKER_MODEL} ...")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # ── 3. Chunk and Ingest ──────────────────────────────────────────────
        print("[Qdrant] Preparing collection and ingesting data ...")
        
        # Reset collection for idempotency
        if self.client.collection_exists(COLLECTION_NAME):
            self.client.delete_collection(COLLECTION_NAME)

        docs, metadatas, ids = [], [], []
        for doc in raw_documents:
            for chunk in chunk_document(doc["doc_id"], doc["text"], CHUNK_SIZE, OVERLAP_RATIO):
                docs.append(chunk["document"])
                # Payload carries everything needed to trace a chunk back to its origin:
                #   parent_doc_id  → which source document this chunk came from
                #   chunk_index    → position within that document (for ordering / context window)
                #   chunk_id_str   → human-readable ID before UUID conversion (for logging/debugging)
                metadatas.append({
                    **chunk["metadata"],
                    "chunk_id_str": chunk["chunk_id_str"],
                })
                ids.append(chunk["id"])

        # add() drives fastembed dense+sparse encoding internally.
        # IDs are now valid UUIDs so the earlier 400 Bad Request is gone.
        self.client.add(
            collection_name=COLLECTION_NAME,
            documents=docs,
            metadata=metadatas,
            ids=ids,
        )
        print(f"[Qdrant] ✓ Ingested {len(docs)} chunks successfully.\n")

    def query(self, user_query: str, verbose: bool = True) -> list[RerankedResult]:
        if verbose:
            print("─" * 70)
            print(f"  QUERY: \"{user_query}\"")
            print("─" * 70)

        # ── Step 1: Hybrid Retrieval (The Broad Net) ─────────────────────────
        # Qdrant natively runs Dense + BM25, fuses them, and returns the top FETCH_K
        search_results = self.client.query(
            collection_name=COLLECTION_NAME,
            query_text=user_query,
            limit=FETCH_K
        )
        
        candidates = [
            RetrievedCandidate(
                chunk_id=hit.id,
                parent_doc_id=hit.metadata["parent_doc_id"],
                text=hit.document
            )
            for hit in search_results
        ]

        if verbose:
            print(f"  ┌─ Qdrant Hybrid Fetch ─")
            print(f"  │  Retrieved {len(candidates)} raw candidate chunks for reranking.")
            print("  └" + "─" * 60)

        # ── Step 2: Cross-Encoder Reranking (The Precision Check) ────────────
        model_inputs = [[user_query, c.text] for c in candidates]
        scores = self.reranker.predict(model_inputs)

        # Attach scores and sort
        for candidate, score in zip(candidates, scores):
            candidate.relevance_score = float(score)
            
        candidates.sort(key=lambda x: x.relevance_score, reverse=True)

        # ── Step 3: Deduplication ────────────────────────────────────────────
        # Keep only the highest-scoring chunk per parent document
        final_results = []
        seen_docs = set()
        
        for c in candidates:
            if c.parent_doc_id not in seen_docs:
                seen_docs.add(c.parent_doc_id)
                final_results.append(
                    RerankedResult(
                        parent_doc_id=c.parent_doc_id,
                        best_text=c.text,
                        relevance_score=c.relevance_score
                    )
                )
            if len(final_results) == FINAL_TOP_N:
                break

        if verbose:
            self._print_results(final_results)

        return final_results

    def _print_results(self, results: list[RerankedResult]) -> None:
        print("\n" + "═" * 70)
        print("  CROSS-ENCODER RESULTS — Top 5 unique parent documents")
        print("═" * 70)
        for i, r in enumerate(results, start=1):
            snippet = textwrap.shorten(r.best_text, width=80, placeholder="…")
            print(f"  [{i}] {r.parent_doc_id:<12} Logit Score: {r.relevance_score:>6.2f}  {snippet}")
        print("═" * 70 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 4. ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # RAW_DOCUMENTS is the same list from your previous script
    pipeline = HybridRAGPipeline(raw_documents=RAW_DOCUMENTS)
    
    query = "What is the fault status of unit SN-X2047-QR and what causes power failure on servers?"
    pipeline.query(query)