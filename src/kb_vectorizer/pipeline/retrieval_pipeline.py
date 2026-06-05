"""
ensemble_rag.py
═══════════════════════════════════════════════════════════════════════════════
Three-Lane Ensemble RAG Retrieval Pipeline with Reciprocal Rank Fusion (RRF)
═══════════════════════════════════════════════════════════════════════════════

Architecture:
  Lane 1  ── BM25 Keyword Search        ──┐
  Lane 2  ── Chroma Dense (Small Chunks) ──┼──► RRF Junction ──► Top-5 Context
  Lane 3  ── Chroma Dense (Large Chunks) ──┘

Dependencies (install before running):
    pip install chromadb rank-bm25 sentence-transformers

Chroma server expected at http://chroma-db:8000  (Docker-compose default).
To run locally without Docker set CHROMA_HOST=localhost in your environment.
"""

from __future__ import annotations

import hashlib
import os
import re
import textwrap
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8000"))

SMALL_CHUNK_SIZE: int = 200     # characters (proxy for ~200 tokens)
LARGE_CHUNK_SIZE: int = 800     # characters (proxy for ~800 tokens)
OVERLAP_RATIO: float = 0.10     # 10 % overlap for both strategies

COLLECTION_SMALL: str = "rag_small_chunks"
COLLECTION_LARGE: str = "rag_large_chunks"

EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

TOP_K_PER_LANE: int = 20        # candidates retrieved per lane
RRF_K: int = 60                 # RRF penalty constant (standard)
FINAL_TOP_N: int = 5            # context window size fed to the LLM


# ──────────────────────────────────────────────────────────────────────────────
# 1. SAMPLE CORPUS
#    Designed so keyword search and vector search naturally DISAGREE:
#      • Doc-3 contains an exact serial number (SN-X2047-QR) that a pure
#        vector search will mis-rank (no semantic meaning), while BM25 nails it.
#      • Doc-1/2 are thematically close to a conceptual query about
#        "power failure", so vector search ranks them high but BM25 misses
#        them because the query uses different vocabulary.
#    RRF elegantly surfaces BOTH types of documents when the query combines
#    a technical code with a conceptual question.
# ──────────────────────────────────────────────────────────────────────────────

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
# 2. DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A text fragment produced by one of the two chunking strategies."""
    chunk_id: str
    parent_doc_id: str
    text: str
    strategy: str           # "small" | "large"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedItem:
    """A single result returned from any one retrieval lane."""
    chunk_id: str
    parent_doc_id: str
    text: str
    score: float            # raw lane score (not used after RRF)
    lane: str               # "bm25" | "chroma_small" | "chroma_large"


@dataclass
class FusedResult:
    """Post-RRF deduplicated result keyed on parent_doc_id."""
    parent_doc_id: str
    rrf_score: float
    best_text: str          # highest-scoring chunk text representing this doc


# ──────────────────────────────────────────────────────────────────────────────
# 3. CHUNKING UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _chunk_text(
    text: str,
    chunk_size: int,
    overlap_ratio: float,
) -> list[str]:
    """
    Splits *text* into overlapping windows of *chunk_size* characters.
    Overlap = floor(chunk_size * overlap_ratio) characters.
    """
    overlap = int(chunk_size * overlap_ratio)
    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start += step
    return [c for c in chunks if c]


def build_chunks(
    documents: list[dict[str, str]],
    small_size: int = SMALL_CHUNK_SIZE,
    large_size: int = LARGE_CHUNK_SIZE,
    overlap: float = OVERLAP_RATIO,
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Returns (small_chunks, large_chunks) for the full corpus.
    Each chunk carries parent_doc_id metadata for later deduplication.
    """
    small_chunks: list[Chunk] = []
    large_chunks: list[Chunk] = []

    for doc in documents:
        doc_id = doc["doc_id"]
        text = doc["text"]

        for i, fragment in enumerate(_chunk_text(text, small_size, overlap)):
            cid = f"{doc_id}::small::{i}"
            small_chunks.append(
                Chunk(
                    chunk_id=cid,
                    parent_doc_id=doc_id,
                    text=fragment,
                    strategy="small",
                    metadata={"parent_doc_id": doc_id, "title": doc["title"], "chunk_index": i},
                )
            )

        for i, fragment in enumerate(_chunk_text(text, large_size, overlap)):
            cid = f"{doc_id}::large::{i}"
            large_chunks.append(
                Chunk(
                    chunk_id=cid,
                    parent_doc_id=doc_id,
                    text=fragment,
                    strategy="large",
                    metadata={"parent_doc_id": doc_id, "title": doc["title"], "chunk_index": i},
                )
            )

    return small_chunks, large_chunks


# ──────────────────────────────────────────────────────────────────────────────
# 4. CHROMA CLIENT & INGESTION
# ──────────────────────────────────────────────────────────────────────────────

def get_chroma_client() -> chromadb.HttpClient:
    """
    Returns an HttpClient pointing at the standalone Chroma container.
    Uses chromadb.EphemeralClient() as a local fallback when the host
    'chroma-db' is not reachable (useful for unit-testing outside Docker).
    """
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()          # raises if server is unreachable
        print(f"[Chroma] Connected to remote server at {CHROMA_HOST}:{CHROMA_PORT}")
        return client
    except Exception as exc:
        print(
            f"[Chroma] Remote server unavailable ({exc}). "
            "Falling back to in-process EphemeralClient."
        )
        return chromadb.EphemeralClient()


def ingest_chunks(
    client: chromadb.ClientAPI,
    collection_name: str,
    chunks: list[Chunk],
    embedder: SentenceTransformer,
    batch_size: int = 64,
) -> chromadb.Collection:
    """
    Creates (or resets) a Chroma collection and upserts all chunks in batches.
    Embeddings are computed locally via SentenceTransformer.
    """
    # Delete if already exists so reruns stay idempotent
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [c.text for c in batch]
        ids = [c.chunk_id for c in batch]
        metadatas = [c.metadata for c in batch]
        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    print(f"[Chroma] '{collection_name}' — ingested {len(chunks)} chunks.")
    return collection


# ──────────────────────────────────────────────────────────────────────────────
# 5. RETRIEVAL LANES
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_accents(text: str) -> str:
    """
    Strips diacritic marks from a Unicode string so that accented and
    unaccented forms match during BM25 retrieval.

    Examples (pt-BR):
        'verificação' → 'verificacao'
        'ação'        → 'acao'
        'São Paulo'   → 'Sao Paulo'
    """
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


# Token character class covers:
#   a-z 0-9          — ASCII alphanumerics
#   \u00c0-\u017e    — Latin Extended A/B (all pt-BR accented chars + other
#                      Romance / European scripts that may appear in documents)
_TOKEN_RE = re.compile(
    r"[a-z0-9\u00c0-\u017e]+"
    r"(?:[-_./][a-z0-9\u00c0-\u017e]+)*",
    re.UNICODE,
)
_SEPARATOR_RE = re.compile(r"[-_./]")


def _tokenize(text: str) -> list[str]:
    """
    Fuzzy multi-lingual tokenizer optimised for Brazilian Portuguese with
    graceful support for English and other Latin-script languages.

    Three expansion layers per raw token
    ──────────────────────────────────────
    Given the raw token 'SN-X2047-QR' extracted from lowercased text:

      1. Original token          → 'sn-x2047-qr'
         Preserves the canonical form so exact-match queries score highest.

      2. Flattened (no separator) → 'snx2047qr'
         Handles user typos / copy-paste that drops hyphens, dots, etc.
         e.g. user types "SNX2047QR" and the document contains "SN-X2047-QR".

      3. Accent-stripped variants  (applied to both forms above if they
         contain non-ASCII characters)
         e.g. 'verificação' → also indexes 'verificacao' and 'verificaao'
         → allows unaccented queries to match accented indexed content and
           vice-versa, which is the most common BM25 failure mode in pt-BR.

    Why this beats a simple .lower() + split()
    ───────────────────────────────────────────
    BM25Okapi hashes token strings directly; without expansion, 'abc-123' and
    'abc123' are two completely disjoint vocabulary entries with zero IDF
    overlap.  Expansion tokens are added to BOTH the index and the query via
    the same function, so the symmetry is guaranteed.

    Stop-word removal is intentionally omitted: short function words ('de',
    'da', 'do', 'em', 'no') carry meaningful disambiguation weight in pt-BR
    technical text and removing them causes false negatives on short queries.
    """
    tokens: list[str] = []
    seen: set[str] = set()

    def _add(t: str) -> None:
        if t and t not in seen:
            seen.add(t)
            tokens.append(t)

    raw_text_lower = text.lower()
    # Capture accented + ASCII alphanumeric tokens (including separator chains)
    for match in _TOKEN_RE.finditer(raw_text_lower):
        original = match.group()                            # e.g. 'sn-x2047-qr'
        _add(original)

        has_separator = bool(_SEPARATOR_RE.search(original))

        # Layer 2 — flatten separators
        if has_separator:
            flat = _SEPARATOR_RE.sub("", original)          # 'snx2047qr'
            _add(flat)

        # Layer 3 — accent-stripped variants (only adds a new token when
        # the original actually contains non-ASCII characters)
        has_accents = any(ord(ch) > 127 for ch in original)
        if has_accents:
            stripped = _normalize_accents(original)         # 'verificacao'
            _add(stripped)
            if has_separator:
                _add(_SEPARATOR_RE.sub("", stripped))       # flat + stripped

    return tokens


# ── Lane 1: BM25 ──────────────────────────────────────────────────────────────

class BM25Lane:
    """
    Wraps rank_bm25.BM25Okapi for keyword-based retrieval.
    Built over standard-sized (small) chunks so the index vocabulary is rich.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        self.index = BM25Okapi(tokenized)
        print(f"[BM25]  Index built over {len(chunks)} documents.")

    def search(self, query: str, top_k: int = TOP_K_PER_LANE) -> list[RetrievedItem]:
        query_tokens = _tokenize(query)
        scores = self.index.get_scores(query_tokens)

        # pair (score, original_index) and sort descending
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[RetrievedItem] = []
        for rank_idx, (chunk_idx, score) in enumerate(ranked):
            chunk = self.chunks[chunk_idx]
            results.append(
                RetrievedItem(
                    chunk_id=chunk.chunk_id,
                    parent_doc_id=chunk.parent_doc_id,
                    text=chunk.text,
                    score=float(score),
                    lane="bm25",
                )
            )
        return results


# ── Lanes 2 & 3: Chroma Dense Vector ─────────────────────────────────────────

class ChromaDenseLane:
    """
    Wraps a Chroma collection for dense vector retrieval.
    Embedding is computed locally (CPU-bound SentenceTransformer).
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        embedder: SentenceTransformer,
        lane_name: str,
    ) -> None:
        self.collection = collection
        self.embedder = embedder
        self.lane_name = lane_name

    def search(self, query: str, top_k: int = TOP_K_PER_LANE) -> list[RetrievedItem]:
        query_embedding = self.embedder.encode([query], show_progress_bar=False).tolist()
        response = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        results: list[RetrievedItem] = []
        for chunk_id, doc_text, meta, dist in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            # Chroma cosine distance → similarity score
            similarity = 1.0 - dist
            results.append(
                RetrievedItem(
                    chunk_id=chunk_id,
                    parent_doc_id=meta["parent_doc_id"],
                    text=doc_text,
                    score=similarity,
                    lane=self.lane_name,
                )
            )
        return results


# ──────────────────────────────────────────────────────────────────────────────
# 6. RRF FUSION JUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    lane_results: list[list[RetrievedItem]],
    k: int = RRF_K,
    top_n: int = FINAL_TOP_N,
) -> list[FusedResult]:
    """
    Applies Reciprocal Rank Fusion across N retrieval lane result lists.
    
    FIXED: Implements in-lane deduplication to prevent chunk accumulation bias.
    A parent document now receives a maximum of ONE vote per lane, based solely 
    on its highest-ranking chunk in that lane.
    """
    # doc_id → {"rrf_score": float, "best_text": str, "best_chunk_score": float}
    fused: dict[str, dict] = {}

    for lane in lane_results:
        # Track documents we've already scored in THIS specific lane
        seen_docs_in_lane = set()
        
        for rank_0based, item in enumerate(lane):
            doc_id = item.parent_doc_id
            
            # If we've already scored this document in this lane, skip it.
            # This prevents long documents with many mediocre chunks from outscoring 
            # short documents with highly relevant chunks.
            if doc_id in seen_docs_in_lane:
                continue
                
            # Mark as seen
            seen_docs_in_lane.add(doc_id)
            
            # Standard RRF calculation using its top rank in this lane
            rank = rank_0based + 1                          
            contribution = 1.0 / (k + rank)

            if doc_id not in fused:
                fused[doc_id] = {
                    "rrf_score": 0.0,
                    "best_text": item.text,
                    "best_chunk_score": item.score,
                }

            fused[doc_id]["rrf_score"] += contribution

            # Keep track of the absolute highest-scoring chunk text across all lanes
            # to serve as the representative payload
            if item.score > fused[doc_id]["best_chunk_score"]:
                fused[doc_id]["best_text"] = item.text
                fused[doc_id]["best_chunk_score"] = item.score

    # Sort final results by the accumulated RRF score descending
    sorted_items = sorted(fused.items(), key=lambda x: x[1]["rrf_score"], reverse=True)

    return [
        FusedResult(
            parent_doc_id=doc_id,
            rrf_score=data["rrf_score"],
            best_text=data["best_text"],
        )
        for doc_id, data in sorted_items[:top_n]
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 7. ENSEMBLE PIPELINE ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

class EnsembleRAGPipeline:
    """
    Wires together the ingestion phase and the three-lane retrieval + RRF.

    Usage
    ─────
        pipeline = EnsembleRAGPipeline(documents=RAW_DOCUMENTS)
        results  = pipeline.query("SN-X2047-QR battery fault causes power loss")
    """

    def __init__(self, documents: list[dict[str, str]]) -> None:
        print("\n" + "═" * 70)
        print("  Initialising Ensemble RAG Pipeline")
        print("═" * 70)

        # ── Embedding model (shared across both Chroma lanes) ──────────────
        print(f"\n[Embed]  Loading '{EMBEDDING_MODEL}' ...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # ── Chunk corpus ───────────────────────────────────────────────────
        print("[Chunk]  Building small and large chunk sets ...")
        self.small_chunks, self.large_chunks = build_chunks(documents)
        print(
            f"[Chunk]  Small chunks: {len(self.small_chunks)} | "
            f"Large chunks: {len(self.large_chunks)}"
        )

        # ── Chroma client & collections ────────────────────────────────────
        chroma_client = get_chroma_client()
        small_col = ingest_chunks(chroma_client, COLLECTION_SMALL, self.small_chunks, self.embedder)
        large_col = ingest_chunks(chroma_client, COLLECTION_LARGE, self.large_chunks, self.embedder)

        # ── Lane instances ─────────────────────────────────────────────────
        self.lane_bm25 = BM25Lane(self.small_chunks)
        self.lane_small = ChromaDenseLane(small_col, self.embedder, "chroma_small")
        self.lane_large = ChromaDenseLane(large_col, self.embedder, "chroma_large")

        print("\n[Pipeline] ✓ Ready.\n")

    # ──────────────────────────────────────────────────────────────────────────

    def query(
        self,
        user_query: str,
        top_k: int = TOP_K_PER_LANE,
        top_n: int = FINAL_TOP_N,
        verbose: bool = True,
    ) -> list[FusedResult]:
        """
        Full retrieval pipeline:
          1. Run all three lanes (top_k candidates each).
          2. Apply RRF fusion & deduplication by parent_doc_id.
          3. Return top_n FusedResults.
        """
        if verbose:
            print("─" * 70)
            print(f"  QUERY: \"{user_query}\"")
            print("─" * 70)

        # ── Parallel (here sequential; trivially parallelisable via ThreadPool)
        bm25_results   = self.lane_bm25.search(user_query, top_k)
        small_results  = self.lane_small.search(user_query, top_k)
        large_results  = self.lane_large.search(user_query, top_k)

        if verbose:
            _print_lane("LANE 1 — BM25 Keyword", bm25_results[:5])
            _print_lane("LANE 2 — Chroma Dense (Small Chunks)", small_results[:5])
            _print_lane("LANE 3 — Chroma Dense (Large Chunks)", large_results[:5])

        # ── RRF Fusion ─────────────────────────────────────────────────────
        fused = reciprocal_rank_fusion(
            [bm25_results, small_results, large_results],
            k=RRF_K,
            top_n=top_n,
        )

        if verbose:
            _print_fused_results(fused)

        return fused

    def build_context_window(self, fused_results: list[FusedResult]) -> str:
        """
        Formats the top-N deduplicated results into a clean LLM context string.
        Positions most relevant document first (anti "Lost in the Middle").
        """
        lines: list[str] = ["=== RETRIEVED CONTEXT (Top-5, RRF-Ranked) ===\n"]
        for i, result in enumerate(fused_results, start=1):
            lines.append(
                f"[{i}] Source: {result.parent_doc_id}  |  "
                f"RRF Score: {result.rrf_score:.6f}\n"
                f"{result.best_text}\n"
            )
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 8. PRETTY-PRINT HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _print_lane(title: str, items: list[RetrievedItem]) -> None:
    print(f"\n  ┌─ {title} (top 5 shown) ─")
    for i, item in enumerate(items, start=1):
        snippet = textwrap.shorten(item.text, width=80, placeholder="…")
        print(f"  │  #{i:>2}  [{item.parent_doc_id}]  score={item.score:.4f}  {snippet}")
    print("  └" + "─" * 60)


def _print_fused_results(results: list[FusedResult]) -> None:
    print("\n" + "═" * 70)
    print("  RRF FUSION RESULTS — Top 5 unique parent documents")
    print("═" * 70)
    for i, r in enumerate(results, start=1):
        snippet = textwrap.shorten(r.best_text, width=90, placeholder="…")
        print(
            f"  [{i}] {r.parent_doc_id:<12}  RRF={r.rrf_score:.6f}  {snippet}"
        )
    print("═" * 70 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 9. ENTRY POINT — DEMO SCENARIOS
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    pipeline = EnsembleRAGPipeline(documents=RAW_DOCUMENTS)

    # ── Scenario A ─────────────────────────────────────────────────────────────
    # Hybrid query: specific serial number (keyword-friendly) + conceptual
    # question about power failure (vector-friendly).
    # Expected behaviour:
    #   • BM25 ranks doc-003 (SN-X2047-QR) very high; ranks doc-001/002 low.
    #   • Vector search ranks doc-001/002 high (power failure semantics); may
    #     rank doc-003 lower due to sparse semantic content in the incident log.
    #   • RRF fuses both signals → doc-003, doc-001, doc-002 all surface in Top-5.
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  SCENARIO A — Hybrid: Serial Number + Conceptual Power Failure  ║")
    print("╚" + "═" * 68 + "╝")

    query_a = "What is the fault status of unit SN-X2047-QR and what causes power failure on servers?"
    fused_a = pipeline.query(query_a, verbose=True)
    context_a = pipeline.build_context_window(fused_a)
    print("\n─── Context Window Ready for LLM ───────────────────────────────────")
    print(context_a)

    # ── Scenario B ─────────────────────────────────────────────────────────────
    # Pure conceptual query — validates that dense-vector lanes dominate when
    # no exact serial numbers are present.
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  SCENARIO B — Pure Conceptual: Battery Technology Comparison     ║")
    print("╚" + "═" * 68 + "╝")

    query_b = "Which battery chemistry offers better long-term value for data centre UPS systems?"
    fused_b = pipeline.query(query_b, verbose=True)
    context_b = pipeline.build_context_window(fused_b)
    print("\n─── Context Window Ready for LLM ───────────────────────────────────")
    print(context_b)

    # ── Scenario C ─────────────────────────────────────────────────────────────
    # Pure keyword / exact-match query — validates BM25 lane dominance.
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  SCENARIO C — Pure Keyword: Exact Fault Code Lookup              ║")
    print("╚" + "═" * 68 + "╝")

    query_c = "E0xF3 fault code BCT-4890 battery cartridge replacement SN-X2047-QR"
    fused_c = pipeline.query(query_c, verbose=True)
    context_c = pipeline.build_context_window(fused_c)
    print("\n─── Context Window Ready for LLM ───────────────────────────────────")
    print(context_c)