from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalConfig:
    return_mode: str = "docs"          # Options: 'docs', 'parent', 'chunks'
    unique_per_doc: bool = True
    use_mmr: bool = False
    mmr_lambda: float = 0.5
    mmr_top_n: int = 8

@dataclass
class HybridConfig(RetrievalConfig):
    use_hybrid: bool = False           # enable keyword channel
    rrf_k: int = 60                    # smoothing param for RRF
    k_vec: int = 12
    k_kw: int = 50
