"""
pipeline.py – Production-grade RAG pipeline.

Wires together:
  load_and_chunk  →  index_chunks  →  hybrid_search  →  rerank
  →  build_context  →  generate_answer

Public API
----------
    initialize(pdf_path)          – one-time index build
    ask(question)                 – returns structured PipelineResult dict
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import TypedDict

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag.pipeline")


# ---------------------------------------------------------------------------
# Structured output type
# ---------------------------------------------------------------------------

class PipelineResult(TypedDict):
    answer: str
    sources: list[dict]
    retrieval_time: float
    generation_time: float
    total_time: float


# ---------------------------------------------------------------------------
# Lazy imports (keeps startup fast; errors surface at call-time)
# ---------------------------------------------------------------------------

def _import_components():
    from context_assembly import build_context
    from generation import generate_answer
    from hybrid_retrieval import hybrid_search
    from hybrid_retrieval import index_chunks
    from ingestion import load_and_chunk
    from reranking import rerank
    return (
        load_and_chunk,
        index_chunks,
        hybrid_search,
        rerank,
        build_context,
        generate_answer,
    )


# ---------------------------------------------------------------------------
# Index state (module-level singleton)
# ---------------------------------------------------------------------------

_indexed: bool = False


def initialize(pdf_path: str, *, rerank_top_k: int = 3) -> None:
    """
    Load a PDF, chunk it, and build BM25 + vector indexes.
    Must be called once before ask().

    Parameters
    ----------
    pdf_path    : path to the source PDF document.
    rerank_top_k: stored as a module default for subsequent ask() calls.
    """
    global _indexed, _default_top_k

    load_and_chunk, index_chunks, *_ = _import_components()

    logger.info("Initializing pipeline with document: %s", pdf_path)
    t0 = time.perf_counter()

    try:
        chunks = load_and_chunk(pdf_path)
        logger.info("Loaded %d chunks from '%s'", len(chunks), pdf_path)
        index_chunks(chunks)
        _indexed = True
        _default_top_k = rerank_top_k
        logger.info(
            "Indexes built in %.3fs (rerank_top_k=%d)",
            time.perf_counter() - t0,
            rerank_top_k,
        )
    except FileNotFoundError:
        logger.error("Document not found: %s", pdf_path)
        raise
    except Exception:
        logger.exception("Failed to initialize pipeline")
        raise


_default_top_k: int = 3


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def ask(
    question: str,
    *,
    hybrid_top_k: int = 20,
    rerank_top_k: int | None = None,
) -> PipelineResult:
    """
    Run the full RAG pipeline for *question*.

    Parameters
    ----------
    question     : Natural-language question to answer.
    hybrid_top_k : Number of candidates returned by hybrid_search().
    rerank_top_k : Number of chunks kept after reranking.
                   Defaults to the value set in initialize().

    Returns
    -------
    PipelineResult dict with keys:
        answer, sources, retrieval_time, generation_time, total_time
    """
    if not _indexed:
        raise RuntimeError(
            "Pipeline not initialized. Call initialize(pdf_path) first."
        )

    _, _, hybrid_search, rerank, build_context, generate_answer = (
        _import_components()
    )

    top_k = rerank_top_k if rerank_top_k is not None else _default_top_k

    logger.info("Question received: %r", question)
    pipeline_start = time.perf_counter()

    # ── Stage 1: Retrieval ─────────────────────────────────────────────────
    retrieval_start = time.perf_counter()
    try:
        logger.debug("Running hybrid_search (top_k=%d)…", hybrid_top_k)
        candidates = hybrid_search(question, top_k=hybrid_top_k)
        logger.info("hybrid_search returned %d candidates", len(candidates))

        logger.debug("Running rerank (top_k=%d)…", top_k)
        reranked = rerank(question, candidates, top_k=top_k)
        logger.info("rerank kept %d chunks", len(reranked))

        logger.debug("Building context…")
        context = build_context(reranked)
    except RuntimeError as exc:
        logger.error("Retrieval stage failed: %s", exc)
        raise
    except Exception:
        logger.exception("Unexpected error during retrieval")
        raise

    retrieval_time = time.perf_counter() - retrieval_start
    logger.info("Retrieval stage completed in %.3fs", retrieval_time)

    # ── Stage 2: Generation ────────────────────────────────────────────────
    generation_start = time.perf_counter()
    try:
        logger.debug("Calling generate_answer…")
        answer = generate_answer(question, context)
    except Exception:
        logger.exception("Generation stage failed")
        raise

    generation_time = time.perf_counter() - generation_start
    logger.info("Generation stage completed in %.3fs", generation_time)

    # ── Assemble result ────────────────────────────────────────────────────
    total_time = time.perf_counter() - pipeline_start

    sources: list[dict] = [
        {
            "page": chunk["page"],
            "rerank_score": chunk.get("rerank_score"),
            "snippet": chunk["text"][:120].replace("\n", " "),
        }
        for chunk in reranked
    ]

    result: PipelineResult = {
        "answer": answer,
        "sources": sources,
        "retrieval_time": round(retrieval_time, 4),
        "generation_time": round(generation_time, 4),
        "total_time": round(total_time, 4),
    }

    logger.info(
        "Pipeline finished | retrieval=%.3fs | generation=%.3fs | total=%.3fs",
        retrieval_time,
        generation_time,
        total_time,
    )
    return result


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def _build_cli_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Hybrid RAG pipeline – interactive CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pdf",
        help="Path to the PDF document to index.",
    )
    parser.add_argument(
        "--question", "-q",
        default=None,
        help=(
            "Single question to answer (non-interactive mode). "
            "If omitted, enters an interactive REPL."
        ),
    )
    parser.add_argument(
        "--hybrid-top-k",
        type=int,
        default=20,
        metavar="N",
        help="Number of hybrid search candidates.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=3,
        metavar="N",
        help="Number of chunks to keep after reranking.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Print the full JSON result instead of the formatted answer.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def _print_result(result: PipelineResult, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("\n" + "═" * 60)
    print("ANSWER")
    print("─" * 60)
    print(result["answer"])
    print("\nSOURCES")
    print("─" * 60)
    for i, src in enumerate(result["sources"], 1):
        score_str = (
            f"  score={src['rerank_score']:.4f}"
            if src.get("rerank_score") is not None
            else ""
        )
        print(f"  [{i}] page {src['page']}{score_str}")
        print(f"       \"{src['snippet']}...\"")
    print("\nTIMINGS")
    print("─" * 60)
    print(f"  Retrieval  : {result['retrieval_time']:.3f}s")
    print(f"  Generation : {result['generation_time']:.3f}s")
    print(f"  Total      : {result['total_time']:.3f}s")
    print("═" * 60 + "\n")


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    # Apply requested log level
    logging.getLogger().setLevel(args.log_level)

    # ── Initialize ──────────────────────────────────────────────────────────
    try:
        initialize(args.pdf, rerank_top_k=args.rerank_top_k)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] Initialization failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Single-question mode ────────────────────────────────────────────────
    if args.question:
        try:
            result = ask(
                args.question,
                hybrid_top_k=args.hybrid_top_k,
                rerank_top_k=args.rerank_top_k,
            )
            _print_result(result, as_json=args.output_json)
        except Exception as exc:
            print(f"[ERROR] Pipeline error: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    # ── Interactive REPL ────────────────────────────────────────────────────
    print("\nHybrid RAG – interactive mode  (type 'exit' or Ctrl-C to quit)\n")
    while True:
        try:
            question = input("Question> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        try:
            result = ask(
                question,
                hybrid_top_k=args.hybrid_top_k,
                rerank_top_k=args.rerank_top_k,
            )
            _print_result(result, as_json=args.output_json)
        except Exception as exc:
            print(f"[ERROR] {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
