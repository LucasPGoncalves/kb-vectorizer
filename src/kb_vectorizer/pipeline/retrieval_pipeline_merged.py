"""
retrieval_pipeline_merged.py
═══════════════════════════════════════════════════════════════════════════════
Best-of-Both RAG Pipeline  —  async + metadata filtering edition
═══════════════════════════════════════════════════════════════════════════════

What changed in this revision
───────────────────────────────
1. Async concurrent Qdrant queries
     Both collections are queried simultaneously via asyncio + a thread-pool
     executor. Since QdrantClient.query() is a blocking call, wrapping it in
     run_in_executor() keeps the event loop free while both network round-trips
     are in flight. On a remote Qdrant instance this roughly halves retrieval
     latency.

     Public surface:
         await pipeline.query(...)          ← async, preferred in a service
         pipeline.query_sync(...)           ← sync wrapper for scripts / tests

2. Metadata filtering
     query() and query_sync() now accept an optional `filters` dict that maps
     payload field names to their required values. Filters are translated to a
     Qdrant Filter with must-match conditions and applied at the HNSW graph
     level — NOT as a post-processing step — so they don't expand latency.

     The LLM tool caller decides which filters to pass based on conversation
     context, e.g.:
         await pipeline.query(
             "fault status SN-X2047-QR",
             filters={"zone": "NORTH", "doc_type": "incident_report"},
         )

     Any metadata field stored in the Qdrant payload at ingest time is
     filterable. The build_chunks() function passes through an arbitrary
     `extra_metadata` dict so callers can attach domain fields (zone, severity,
     doc_type, date, asset_id, …) without modifying this file.

3. Reranker runs in a thread-pool executor
     CrossEncoder.predict() is synchronous PyTorch — running it on the event
     loop would block all concurrent requests. It is dispatched to a dedicated
     ThreadPoolExecutor (default 1 worker; raise for multi-GPU or high-QPS).

Architecture
─────────────
                    ┌─ Qdrant small ─┐   (concurrent, thread-pool)
  query() ──────────┤                ├──► merge & dedup ──► reranker (thread-pool)
                    └─ Qdrant large ─┘                    ──► Top-N context

Dependencies:
    pip install qdrant-client[fastembed] sentence-transformers
"""

from __future__ import annotations

import asyncio
import re
import textwrap
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models as qmodels
from sentence_transformers import CrossEncoder

# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

QDRANT_URL: str = "http://localhost:6333"

COLLECTION_SMALL: str = "merged_rag_small"   # ~200-token chunks  (precision)
COLLECTION_LARGE: str = "merged_rag_large"   # ~800-token chunks  (context depth)

SMALL_CHUNK_SIZE: int = 200
LARGE_CHUNK_SIZE: int = 800
OVERLAP_RATIO: float = 0.10

DENSE_MODEL:    str = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL:   str = "Qdrant/bm25"
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FETCH_K_PER_COLLECTION: int = 20   # candidates per collection before reranking
FINAL_TOP_N:            int = 5    # results returned to the LLM

# Number of threads for the reranker pool.
# 1 is safe on CPU. Raise to match GPU count if running on GPU.
RERANKER_WORKERS: int = 1

# ──────────────────────────────────────────────────────────────────────────────
# 1. SAMPLE CORPUS
# ──────────────────────────────────────────────────────────────────────────────

RAW_DOCUMENTS: list[dict[str, Any]] = [
    {
        "doc_id": "doc-001",
        "title":  "UPS System Architecture Overview",
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
        # Extra metadata fields — filterable at query time
        "doc_type": "architecture",
        "zone":     "ALL",
    },
    {
        "doc_id": "doc-002",
        "title":  "Causes of Unexpected Server Shutdown Events",
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
        "doc_type": "architecture",
        "zone":     "ALL",
    },
    {
        "doc_id": "doc-003",
        "title":  "Incident Report – Fault Unit SN-X2047-QR",
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
        "doc_type": "incident_report",
        "zone":     "NORTH",
    },
    {
        "doc_id": "doc-004",
        "title":  "Battery Chemistry: VRLA vs Lithium-Ion in Data Centres",
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
        "doc_type": "architecture",
        "zone":     "ALL",
    },
    {
        "doc_id": "doc-005",
        "title":  "Generator Failover and ATS Configuration",
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
        "doc_type": "architecture",
        "zone":     "ALL",
    },
    {
        "doc_id": "doc-006",
        "title":  "DCIM Platform Integration Guide",
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
        "doc_type": "integration_guide",
        "zone":     "ALL",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# 2. DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    """Single chunk returned from Qdrant before reranking."""
    chunk_id:       str
    chunk_id_str:   str     # human-readable: "doc-001::small::0"
    parent_doc_id:  str
    title:          str
    text:           str
    strategy:       str     # "small" | "large"
    relevance_score: float = 0.0


@dataclass
class RankedResult:
    """Final deduplicated result handed back to the LLM tool caller."""
    parent_doc_id:   str
    title:           str
    best_text:       str
    strategy:        str    # chunk size the reranker preferred
    relevance_score: float


# ──────────────────────────────────────────────────────────────────────────────
# 3. TOKENIZER  (ported from retrieval_pipeline.py — Ensemble)
# ──────────────────────────────────────────────────────────────────────────────
# Three-layer fuzzy tokenizer for pt-BR technical text.
# NOTE: Qdrant's built-in BM25 (fastembed) tokenizes independently.
# _tokenize() is retained here for an optional in-process BM25 lane.
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


_TOKEN_RE    = re.compile(r"[a-z0-9\u00c0-\u017e]+(?:[-_./][a-z0-9\u00c0-\u017e]+)*", re.UNICODE)
_SEPARATOR_RE = re.compile(r"[-_./]")


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    seen:   set[str]  = set()

    def _add(t: str) -> None:
        if t and t not in seen:
            seen.add(t)
            tokens.append(t)

    for match in _TOKEN_RE.finditer(text.lower()):
        original = match.group()
        _add(original)
        has_sep = bool(_SEPARATOR_RE.search(original))
        if has_sep:
            _add(_SEPARATOR_RE.sub("", original))
        has_accents = any(ord(ch) > 127 for ch in original)
        if has_accents:
            stripped = _normalize_accents(original)
            _add(stripped)
            if has_sep:
                _add(_SEPARATOR_RE.sub("", stripped))

    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# 4. CHUNKING
# ──────────────────────────────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int, overlap: float) -> list[str]:
    step = int(chunk_size * (1 - overlap))
    fragments: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        fragment = text[start:end].strip()
        if fragment:
            fragments.append(fragment)
        if end == len(text):
            break
        start += step
    return fragments


def build_chunks(
    documents:  list[dict[str, Any]],
    small_size: int   = SMALL_CHUNK_SIZE,
    large_size: int   = LARGE_CHUNK_SIZE,
    overlap:    float = OVERLAP_RATIO,
) -> tuple[list[dict], list[dict]]:
    """Returns (small_chunks, large_chunks).

    Any key in the source document beyond doc_id / title / text is treated as
    extra metadata and stored in the Qdrant payload, making it available for
    filtering at query time.  Example domain fields: zone, doc_type, severity,
    date, asset_id — add them to the document dict and they flow through
    automatically.
    """
    CORE_KEYS = {"doc_id", "title", "text"}

    small_chunks: list[dict] = []
    large_chunks: list[dict] = []

    for doc in documents:
        doc_id = doc["doc_id"]
        title  = doc["title"]
        text   = doc["text"]

        # Any key outside the three core fields becomes a filterable payload field
        extra = {k: v for k, v in doc.items() if k not in CORE_KEYS}

        for strategy, size, bucket in (
            ("small", small_size, small_chunks),
            ("large", large_size, large_chunks),
        ):
            for i, fragment in enumerate(_split_text(text, size, overlap)):
                human_id = f"{doc_id}::{strategy}::{i}"
                bucket.append({
                    "id":           str(uuid.uuid5(uuid.NAMESPACE_DNS, human_id)),
                    "chunk_id_str": human_id,
                    "document":     fragment,
                    "metadata": {
                        "parent_doc_id": doc_id,
                        "title":         title,
                        "strategy":      strategy,
                        "chunk_index":   i,
                        "chunk_id_str":  human_id,
                        **extra,            # ← zone, doc_type, etc. land here
                    },
                })

    return small_chunks, large_chunks


# ──────────────────────────────────────────────────────────────────────────────
# 5. FILTER TRANSLATION
# ──────────────────────────────────────────────────────────────────────────────

def _build_qdrant_filter(filters: dict[str, Any] | None) -> qmodels.Filter | None:
    """
    Converts a plain {field: value} dict into a Qdrant Filter applied at the
    HNSW graph level (not post-processing), so it adds negligible latency.

    Each key/value pair becomes a MatchValue condition; all conditions are
    combined with must (logical AND).

    Examples
    ────────
        {"zone": "NORTH"}
            → only chunks where payload.zone == "NORTH"

        {"zone": "NORTH", "doc_type": "incident_report"}
            → chunks in the NORTH zone that are incident reports

        None  →  no filter (search entire corpus)
    """
    if not filters:
        return None

    conditions = [
        qmodels.FieldCondition(
            key=field,
            match=qmodels.MatchValue(value=value),
        )
        for field, value in filters.items()
    ]
    return qmodels.Filter(must=conditions)


# ──────────────────────────────────────────────────────────────────────────────
# 6. PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

class MergedRAGPipeline:
    """
    Async-first retrieval pipeline for use as an LLM tool.

    Key behaviours
    ──────────────
    • Both Qdrant collections are queried concurrently (asyncio + thread-pool).
    • The cross-encoder reranker runs in a dedicated thread-pool executor so it
      never blocks the event loop.
    • Metadata filters (zone, doc_type, date, …) are pushed into Qdrant as
      HNSW-level pre-filters, not post-processing.
    • Results are deduplicated first by chunk UUID, then by parent document, so
      the LLM always receives at most one passage per source document.

    Usage as an LLM tool (async service)
    ──────────────────────────────────────
        pipeline = MergedRAGPipeline(raw_documents=RAW_DOCUMENTS)

        # unfiltered
        results = await pipeline.query("fault status SN-X2047-QR")

        # scoped to a zone and document type
        results = await pipeline.query(
            "fault status SN-X2047-QR",
            filters={"zone": "NORTH", "doc_type": "incident_report"},
        )

        context = pipeline.build_context_window(results)

    Usage in a script / test (sync)
    ────────────────────────────────
        results = pipeline.query_sync("fault status SN-X2047-QR")
    """

    def __init__(self, raw_documents: list[dict[str, Any]]) -> None:
        print("\n" + "═" * 70)
        print("  Initialising Merged RAG Pipeline  (async + metadata filters)")
        print("═" * 70)

        # ── Qdrant client ────────────────────────────────────────────────────
        self.client = QdrantClient(url=QDRANT_URL)
        self.client.set_model(DENSE_MODEL)
        self.client.set_sparse_model(SPARSE_MODEL)

        # ── Cross-Encoder in its own thread pool ─────────────────────────────
        # predict() is a blocking PyTorch call — it must never run on the event
        # loop. A dedicated pool avoids starving other thread-pool work.
        print(f"[Reranker] Loading '{RERANKER_MODEL}' ...")
        self.reranker       = CrossEncoder(RERANKER_MODEL)
        self._reranker_pool = ThreadPoolExecutor(
            max_workers=RERANKER_WORKERS,
            thread_name_prefix="reranker",
        )

        # ── Qdrant queries also block — use a separate pool ──────────────────
        # Two workers = the two concurrent collection queries can both be
        # in-flight simultaneously without queuing behind each other.
        self._qdrant_pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="qdrant",
        )

        # ── Ingest ───────────────────────────────────────────────────────────
        print("[Chunks]   Building small + large chunk sets ...")
        small_chunks, large_chunks = build_chunks(raw_documents)
        print(
            f"[Chunks]   Small: {len(small_chunks)}  |  "
            f"Large: {len(large_chunks)}"
        )
        self._ingest(COLLECTION_SMALL, small_chunks)
        self._ingest(COLLECTION_LARGE, large_chunks)
        print("[Qdrant]   ✓ Both collections ready.\n")

    # ── Ingestion ────────────────────────────────────────────────────────────

    def _ingest(self, collection_name: str, chunks: list[dict]) -> None:
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
        self.client.add(
            collection_name=collection_name,
            documents=[c["document"] for c in chunks],
            metadata=[c["metadata"] for c in chunks],
            ids=[c["id"] for c in chunks],
        )
        print(f"[Qdrant]   '{collection_name}' — ingested {len(chunks)} chunks.")

    # ── Core async query ─────────────────────────────────────────────────────

    async def query(
        self,
        user_query:  str,
        filters:     dict[str, Any] | None = None,
        fetch_k:     int  = FETCH_K_PER_COLLECTION,
        top_n:       int  = FINAL_TOP_N,
        verbose:     bool = True,
    ) -> list[RankedResult]:
        """
        Async entry point — preferred when running inside a service.

        Parameters
        ──────────
        user_query  Query string, typically formed by the LLM tool caller.
        filters     Optional {field: value} dict applied as Qdrant pre-filters.
                    Any metadata field stored at ingest time is valid here.
                    Pass None (default) to search the full corpus.
        fetch_k     Candidates retrieved per collection before reranking.
        top_n       Final results returned to the caller.
        verbose     Print retrieval diagnostics to stdout.
        """
        if verbose:
            filter_str = str(filters) if filters else "none"
            print("─" * 70)
            print(f"  QUERY:   \"{user_query}\"")
            print(f"  FILTERS: {filter_str}")
            print("─" * 70)

        qdrant_filter = _build_qdrant_filter(filters)
        loop          = asyncio.get_event_loop()

        # ── Step 1: Both collections queried concurrently ────────────────────
        # QdrantClient.query() is synchronous — run each in the thread pool so
        # both network round-trips are in-flight at the same time.
        def _fetch(collection: str):
            return self.client.query(
                collection_name=collection,
                query_text=user_query,
                limit=fetch_k,
                query_filter=qdrant_filter,
            )

        small_hits, large_hits = await asyncio.gather(
            loop.run_in_executor(self._qdrant_pool, _fetch, COLLECTION_SMALL),
            loop.run_in_executor(self._qdrant_pool, _fetch, COLLECTION_LARGE),
        )

        # ── Step 2: Merge + deduplicate by chunk UUID ────────────────────────
        seen_ids:   set[str]       = set()
        candidates: list[Candidate] = []

        for hit in (*small_hits, *large_hits):
            if hit.id in seen_ids:
                continue
            seen_ids.add(hit.id)
            candidates.append(Candidate(
                chunk_id      = hit.id,
                chunk_id_str  = hit.metadata.get("chunk_id_str", str(hit.id)),
                parent_doc_id = hit.metadata["parent_doc_id"],
                title         = hit.metadata.get("title", ""),
                text          = hit.document,
                strategy      = hit.metadata.get("strategy", "unknown"),
            ))

        if verbose:
            small_n = sum(1 for c in candidates if c.strategy == "small")
            large_n = sum(1 for c in candidates if c.strategy == "large")
            print(f"  ┌─ Qdrant dual-collection fetch (concurrent) ─")
            print(f"  │  {len(candidates)} unique candidates  "
                  f"({small_n} small, {large_n} large)")
            print("  └" + "─" * 60)

        # ── Step 3: Cross-Encoder reranking (off the event loop) ─────────────
        # predict() is synchronous PyTorch — dispatched to the reranker pool
        # so it never blocks concurrent requests on the event loop.
        model_inputs = [[user_query, c.text] for c in candidates]
        scores = await loop.run_in_executor(
            self._reranker_pool,
            self.reranker.predict,
            model_inputs,
        )
        for candidate, score in zip(candidates, scores, strict=False):
            candidate.relevance_score = float(score)

        candidates.sort(key=lambda c: c.relevance_score, reverse=True)

        # ── Step 4: Deduplicate by parent document ───────────────────────────
        # Candidates are sorted — first hit per doc_id is always the best chunk.
        final_results: list[RankedResult] = []
        seen_docs:     set[str]           = set()

        for c in candidates:
            if c.parent_doc_id in seen_docs:
                continue
            seen_docs.add(c.parent_doc_id)
            final_results.append(RankedResult(
                parent_doc_id   = c.parent_doc_id,
                title           = c.title,
                best_text       = c.text,
                strategy        = c.strategy,
                relevance_score = c.relevance_score,
            ))
            if len(final_results) == top_n:
                break

        if verbose:
            self._print_results(final_results)

        return final_results

    # ── Sync convenience wrapper ─────────────────────────────────────────────

    def query_sync(
        self,
        user_query: str,
        filters:    dict[str, Any] | None = None,
        fetch_k:    int  = FETCH_K_PER_COLLECTION,
        top_n:      int  = FINAL_TOP_N,
        verbose:    bool = True,
    ) -> list[RankedResult]:
        """
        Blocking wrapper around query() for scripts and tests.
        Do NOT call this from inside a running event loop — use await query()
        directly instead.
        """
        return asyncio.run(
            self.query(user_query, filters=filters, fetch_k=fetch_k,
                       top_n=top_n, verbose=verbose)
        )

    # ── Context window ───────────────────────────────────────────────────────

    def build_context_window(self, results: list[RankedResult]) -> str:
        """
        Formats reranked results into a structured context string for the LLM.

        Most relevant document is placed first — LLMs attend more strongly to
        content at the start of the context (anti "lost in the middle").
        """
        lines: list[str] = [
            f"=== RETRIEVED CONTEXT  (top {len(results)}, reranker-ranked) ===\n"
        ]
        for i, r in enumerate(results, start=1):
            lines.append(
                f"[{i}] Source: {r.parent_doc_id}  |  "
                f"Title: {r.title}  |  "
                f"Chunk: {r.strategy}  |  "
                f"Score: {r.relevance_score:.4f}\n"
                f"{r.best_text}\n"
            )
        return "\n".join(lines)

    # ── Pretty-print ─────────────────────────────────────────────────────────

    def _print_results(self, results: list[RankedResult]) -> None:
        print("\n" + "═" * 70)
        print(f"  RERANKED RESULTS — Top {len(results)} unique source documents")
        print("═" * 70)
        for i, r in enumerate(results, start=1):
            snippet = textwrap.shorten(r.best_text, width=72, placeholder="…")
            print(
                f"  [{i}] {r.parent_doc_id:<10}  "
                f"chunk={r.strategy:<5}  "
                f"score={r.relevance_score:>7.3f}  "
                f"{snippet}"
            )
        print("═" * 70 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 7. ENTRY POINT — DEMO SCENARIOS
# ──────────────────────────────────────────────────────────────────────────────

async def _main() -> None:
    pipeline = MergedRAGPipeline(raw_documents=RAW_DOCUMENTS)

    async def _run(label: str, query: str, filters: dict | None = None) -> None:
        print("\n" + "╔" + "═" * 68 + "╗")
        print(f"║  {label:<66}  ║")
        print("╚" + "═" * 68 + "╝")
        results = await pipeline.query(query, filters=filters, verbose=True)
        context = pipeline.build_context_window(results)
        print("─── Context Window Ready for LLM " + "─" * 37)
        print(context)

    # Scenario A — mixed query, no filter (full corpus)
    await _run(
        "SCENARIO A — Hybrid: Serial Number + Conceptual Power Failure",
        "What is the fault status of unit SN-X2047-QR and what causes power failure on servers?",
    )

    # Scenario B — pure conceptual, no filter
    await _run(
        "SCENARIO B — Pure Conceptual: Battery Technology Comparison",
        "Which battery chemistry offers better long-term value for data centre UPS systems?",
    )

    # Scenario C — pure keyword, no filter
    await _run(
        "SCENARIO C — Pure Keyword: Exact Fault Code Lookup",
        "E0xF3 fault code BCT-4890 battery cartridge replacement SN-X2047-QR",
    )

    # Scenario D — same keyword query, filtered to NORTH zone incident reports only
    # This is the pattern the LLM tool caller will use in production.
    await _run(
        "SCENARIO D — Filtered: NORTH zone incident reports only",
        "E0xF3 fault code BCT-4890 battery cartridge replacement SN-X2047-QR",
        filters={"zone": "NORTH", "doc_type": "incident_report"},
    )


if __name__ == "__main__":
    asyncio.run(_main())