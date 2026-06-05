"""
evaluation.py – Production-grade Ragas evaluation suite.

Evaluates the Hybrid RAG pipeline on four core metrics:
    • Faithfulness         – Is the answer grounded in the retrieved context?
    • Answer Relevancy     – Does the answer address the question?
    • Context Recall       – Did retrieval find the right evidence?
    • Context Precision    – Are the top-ranked contexts actually relevant?

Usage
-----
    # Evaluate against the default dataset
    python evaluation.py data/sample.pdf

    # Point to a custom ground truth file
    python evaluation.py data/sample.pdf --dataset my_dataset.json

    # Emit raw JSON instead of the formatted report
    python evaluation.py data/sample.pdf --json

    # Evaluate a single question by index
    python evaluation.py data/sample.pdf --index 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag.evaluation")


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

class GroundTruthSample(TypedDict):
    """Schema for a single entry in the evaluation dataset JSON file."""
    question: str
    ground_truth: str
    ground_truth_context: list[str]


class SampleResult(TypedDict):
    """Per-sample result after evaluation."""
    question: str
    answer: str
    retrieved_contexts: list[str]
    ground_truth: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_recall: float | None
    context_precision: float | None


class EvaluationReport(TypedDict):
    """Full evaluation report returned by run_evaluation()."""
    summary: dict[str, float | None]
    per_sample: list[SampleResult]
    total_time: float
    num_samples: int


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

DEFAULT_DATASET_PATH = Path(__file__).parent / "evaluation_dataset.json"


def load_evaluation_dataset(
    path: str | Path = DEFAULT_DATASET_PATH,
) -> list[GroundTruthSample]:
    """
    Load the ground truth evaluation dataset from a JSON file.

    Expected schema (list of objects):
        [
          {
            "question": str,
            "ground_truth": str,
            "ground_truth_context": [str, ...]
          },
          ...
        ]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    with open(path, encoding="utf-8") as fh:
        data: list[dict[str, Any]] = json.load(fh)

    # ── Validate ───────────────────────────────────────────────────────────
    required_keys = {"question", "ground_truth", "ground_truth_context"}
    validated: list[GroundTruthSample] = []

    for i, entry in enumerate(data):
        missing = required_keys - entry.keys()
        if missing:
            raise ValueError(
                f"Sample {i} is missing required keys: {missing}"
            )
        validated.append(
            GroundTruthSample(
                question=entry["question"],
                ground_truth=entry["ground_truth"],
                ground_truth_context=entry["ground_truth_context"],
            )
        )

    logger.info("Loaded %d evaluation samples from %s", len(validated), path)
    return validated


# ---------------------------------------------------------------------------
# Pipeline runner  (collects retrieved contexts for Ragas)
# ---------------------------------------------------------------------------

def _collect_pipeline_outputs(
    samples: list[GroundTruthSample],
) -> list[dict[str, Any]]:
    """
    Run each sample through the RAG pipeline, capturing the answer and the
    raw retrieved context strings that Ragas needs.
    """
    from context_assembly import build_context
    from hybrid_retrieval import hybrid_search
    from generation import generate_answer
    from reranking import rerank

    outputs: list[dict[str, Any]] = []

    for i, sample in enumerate(samples):
        question = sample["question"]
        logger.info("[%d/%d] Running pipeline for: %r", i + 1, len(samples), question)

        t0 = time.perf_counter()

        try:
            # Retrieval
            candidates = hybrid_search(question, top_k=20)
            reranked = rerank(question, candidates, top_k=3)
            context = build_context(reranked)

            # Collect individual context strings for Ragas
            retrieved_contexts = [chunk["text"] for chunk in reranked]

            # Generation
            answer = generate_answer(question, context)
        except Exception:
            logger.exception("Pipeline failed for sample %d", i)
            answer = ""
            retrieved_contexts = []

        elapsed = time.perf_counter() - t0
        logger.info(
            "[%d/%d] Completed in %.2fs | answer length=%d | contexts=%d",
            i + 1, len(samples), elapsed, len(answer), len(retrieved_contexts),
        )

        outputs.append({
            "question": question,
            "answer": answer,
            "retrieved_contexts": retrieved_contexts,
            "ground_truth": sample["ground_truth"],
            "ground_truth_context": sample["ground_truth_context"],
        })

    return outputs


# ---------------------------------------------------------------------------
# Ragas evaluation
# ---------------------------------------------------------------------------

def _run_ragas_evaluation(
    outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build a Ragas EvaluationDataset from pipeline outputs and evaluate it
    against the four core metrics.

    Returns the ragas Result object.
    """
    from ragas import SingleTurnSample, EvaluationDataset, evaluate
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        ContextRecall,
        ContextPrecisionWithReference,
    )

    # Build Ragas samples
    ragas_samples: list[SingleTurnSample] = []
    for out in outputs:
        ragas_samples.append(
            SingleTurnSample(
                user_input=out["question"],
                response=out["answer"],
                retrieved_contexts=out["retrieved_contexts"],
                reference=out["ground_truth"],
                reference_contexts=out["ground_truth_context"],
            )
        )

    eval_dataset = EvaluationDataset(samples=ragas_samples)

    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    import os

    llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), api_key=os.getenv("GROQ_API_KEY"))
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Instantiate metrics
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextRecall(llm=llm),
        ContextPrecisionWithReference(llm=llm),
    ]

    logger.info("Starting Ragas evaluation with %d samples…", len(ragas_samples))
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
    )
    logger.info("Ragas evaluation complete")
    return result


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

def run_evaluation(
    pdf_path: str,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    *,
    sample_indices: list[int] | None = None,
) -> EvaluationReport:
    """
    End-to-end evaluation: load data → run pipeline → score with Ragas.

    Parameters
    ----------
    pdf_path       : Path to the PDF document to index.
    dataset_path   : Path to the ground truth JSON file.
    sample_indices : Optional list of 0-based indices to evaluate a subset.

    Returns
    -------
    EvaluationReport with per-sample scores and aggregate summary.
    """
    from pipeline import initialize

    total_start = time.perf_counter()

    # ── 1. Load ground truth ───────────────────────────────────────────────
    all_samples = load_evaluation_dataset(dataset_path)
    if sample_indices is not None:
        all_samples = [all_samples[i] for i in sample_indices]
        logger.info("Evaluating subset: indices %s", sample_indices)

    # ── 2. Initialize pipeline ─────────────────────────────────────────────
    logger.info("Initializing pipeline with: %s", pdf_path)
    initialize(pdf_path)

    # ── 3. Collect pipeline outputs ────────────────────────────────────────
    outputs = _collect_pipeline_outputs(all_samples)

    # ── 4. Run Ragas evaluation ────────────────────────────────────────────
    ragas_result = _run_ragas_evaluation(outputs)

    # ── 5. Assemble report ─────────────────────────────────────────────────
    # ragas_result is a Result object; .to_pandas() gives per-row scores
    df = ragas_result.to_pandas()

    # Map our metric keys to candidate Ragas column names (checked in order)
    metric_candidates = {
        "faithfulness": ["faithfulness"],
        "answer_relevancy": ["answer_relevancy"],
        "context_recall": ["context_recall"],
        "context_precision": [
            "context_precision_with_reference",
            "llm_context_precision_with_reference",
            "context_precision",
        ],
    }

    per_sample: list[SampleResult] = []
    for i, out in enumerate(outputs):
        row = df.iloc[i] if i < len(df) else {}
        
        # Resolve metrics dynamically checking all candidates
        metrics_scores = {}
        for metric_name, candidates in metric_candidates.items():
            val = None
            for c in candidates:
                if c in row:
                    val = row[c]
                    break
            metrics_scores[metric_name] = _safe_float(val)

        sample_result = SampleResult(
            question=out["question"],
            answer=out["answer"],
            retrieved_contexts=out["retrieved_contexts"],
            ground_truth=out["ground_truth"],
            faithfulness=metrics_scores["faithfulness"],
            answer_relevancy=metrics_scores["answer_relevancy"],
            context_recall=metrics_scores["context_recall"],
            context_precision=metrics_scores["context_precision"],
        )
        per_sample.append(sample_result)

    # Aggregate means
    summary: dict[str, float | None] = {}
    for metric_name in metric_candidates:
        values = [
            s[metric_name]  # type: ignore[literal-required]
            for s in per_sample
            if s.get(metric_name) is not None  # type: ignore[arg-type]
        ]
        summary[metric_name] = round(sum(values) / len(values), 4) if values else None

    total_time = time.perf_counter() - total_start

    report = EvaluationReport(
        summary=summary,
        per_sample=per_sample,
        total_time=round(total_time, 4),
        num_samples=len(per_sample),
    )

    logger.info(
        "Evaluation finished | samples=%d | total=%.1fs | summary=%s",
        len(per_sample),
        total_time,
        json.dumps(summary),
    )
    return report


def _safe_float(value: Any) -> float | None:
    """Convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        import math
        f = float(value)
        return None if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(report: EvaluationReport, *, as_json: bool) -> None:
    """Pretty-print the evaluation report."""
    if as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return

    print("\n" + "═" * 70)
    print("  RAGAS EVALUATION REPORT")
    print("═" * 70)
    print(f"  Samples evaluated : {report['num_samples']}")
    print(f"  Total time        : {report['total_time']:.1f}s")
    print("─" * 70)

    # Summary table
    print("\n  AGGREGATE SCORES")
    print("  " + "─" * 40)
    for metric, score in report["summary"].items():
        bar = ""
        if score is not None:
            filled = int(score * 20)
            bar = "█" * filled + "░" * (20 - filled)
            print(f"  {metric:<25s} {score:.4f}  {bar}")
        else:
            print(f"  {metric:<25s} N/A")

    # Per-sample breakdown
    print("\n" + "─" * 70)
    print("  PER-SAMPLE BREAKDOWN")
    print("─" * 70)

    for i, sample in enumerate(report["per_sample"], 1):
        print(f"\n  [{i}] {sample['question']}")
        print(f"      Answer   : {sample['answer'][:100]}…" if len(sample["answer"]) > 100 else f"      Answer   : {sample['answer']}")
        print(f"      Contexts : {len(sample['retrieved_contexts'])} retrieved")

        scores_str = "      Scores   : "
        parts = []
        for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
            val = sample.get(metric)
            if val is not None:
                parts.append(f"{metric}={val:.4f}")
            else:
                parts.append(f"{metric}=N/A")
        print(scores_str + " | ".join(parts))

    print("\n" + "═" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="evaluation",
        description="Ragas evaluation suite for the Hybrid RAG pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pdf",
        help="Path to the PDF document to index.",
    )
    parser.add_argument(
        "--dataset", "-d",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the ground truth JSON evaluation dataset.",
    )
    parser.add_argument(
        "--index", "-i",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Evaluate only specific sample indices (0-based).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output raw JSON report.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="Save the JSON report to a file.",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    # Validate GROQ_API_KEY early
    if not os.getenv("GROQ_API_KEY"):
        print(
            "[ERROR] GROQ_API_KEY is not set. Ragas metrics require an LLM.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        report = run_evaluation(
            pdf_path=args.pdf,
            dataset_path=args.dataset,
            sample_indices=args.index,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Evaluation failed")
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    _print_report(report, as_json=args.output_json)

    # Optionally persist to file
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("Report saved to %s", out_path)


if __name__ == "__main__":
    main()
