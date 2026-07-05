"""
RAGAS Evaluation Pipeline for Document Portal.

What is RAGAS?
    RAGAS (Retrieval Augmented Generation Assessment) is a framework that
    measures the quality of a RAG pipeline using four key metrics:

    1. Faithfulness       — Is the answer grounded in the retrieved context?
                            High score = no hallucination.
    2. Answer Relevancy   — Does the answer actually address the question?
                            High score = on-topic responses.
    3. Context Precision  — Are the retrieved chunks relevant to the question?
                            High score = retriever isn't pulling junk.
    4. Context Recall     — Did the retriever find all the information needed?
                            High score = nothing important was missed.

Why this matters for interviews:
    "I built a formal evaluation pipeline using RAGAS so I can objectively
    measure and improve retrieval quality — not just eyeball the answers."
    This shows production thinking, not just getting something to work.

Usage:
    # Run the full evaluation suite:
    python evaluation/ragas_eval.py --pdf path/to/doc.pdf

    # Or run programmatically:
    from evaluation.ragas_eval import RAGASEvaluator
    evaluator = RAGASEvaluator()
    results = evaluator.evaluate(pdf_path="doc.pdf", questions=my_questions)
    evaluator.print_report(results)

Output:
    Saves a JSON report to evaluation/reports/ragas_report_{timestamp}.json
    Prints a formatted table to stdout.
"""

from __future__ import annotations

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from logger import GLOBAL_LOGGER as log


# ── Test question bank ────────────────────────────────────────────────────────
# These are the default questions used when no custom questions are provided.
# In a real evaluation you'd tailor these per document.

DEFAULT_QUESTIONS = [
    "What is the main topic of this document?",
    "What are the key conclusions or findings?",
    "Who are the main stakeholders mentioned?",
    "What recommendations are made?",
    "What methodology or approach is described?",
]


class RAGASEvaluator:
    """
    Runs RAGAS evaluation against the Document Portal RAG pipeline.

    The evaluator:
    1. Ingests a PDF through the same pipeline the app uses (ChatIngestor)
    2. For each test question, retrieves context + generates an answer
    3. Packages (question, answer, contexts, ground_truth) into a RAGAS dataset
    4. Runs all four RAGAS metrics
    5. Saves and returns a structured report
    """

    REPORTS_DIR = Path(__file__).parent / "reports"

    def __init__(self):
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        pdf_path: str,
        questions: Optional[List[str]] = None,
        ground_truths: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Run RAGAS evaluation on a PDF with given test questions.

        Args:
            pdf_path:      Path to the PDF to evaluate against.
            questions:     List of test questions. Defaults to DEFAULT_QUESTIONS.
            ground_truths: Optional reference answers. If not provided,
                           RAGAS uses LLM-based evaluation (no ground truth needed
                           for faithfulness + answer_relevancy).
            session_id:    Optional session ID. A new one is generated if not given.

        Returns:
            dict with metrics, per-question scores, and metadata.
        """
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset
        except ImportError:
            log.error("RAGAS not installed. Run: pip install ragas datasets")
            raise

        questions = questions or DEFAULT_QUESTIONS
        ground_truths = ground_truths or [""] * len(questions)

        log.info("Starting RAGAS evaluation", pdf=pdf_path, num_questions=len(questions))

        # ── Step 1: Ingest the PDF ─────────────────────────────────────────
        session_id = session_id or f"ragas-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        index_dir = self._ingest_pdf(pdf_path, session_id)

        # ── Step 2: Load RAG chain ─────────────────────────────────────────
        from src.document_chat.retrieval import ConversationalRAG
        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=3)

        # ── Step 3: Collect answers + contexts for each question ───────────
        import asyncio

        eval_rows = []
        for i, question in enumerate(questions):
            log.info(f"Evaluating question {i+1}/{len(questions)}", question=question[:60])
            try:
                # Get the retrieved context chunks
                retrieved_docs = rag.retriever.invoke(question)
                contexts = [doc.page_content for doc in retrieved_docs]

                # Get the LLM answer
                answer = asyncio.run(rag.invoke(question, chat_history=[]))

                eval_rows.append({
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": ground_truths[i],
                })
                log.info(f"Question {i+1} answered", answer_preview=answer[:80])

            except Exception as e:
                log.warning(f"Question {i+1} failed", error=str(e))
                eval_rows.append({
                    "question": question,
                    "answer": "ERROR: " + str(e),
                    "contexts": [],
                    "ground_truth": ground_truths[i],
                })

        # ── Step 4: Run RAGAS ─────────────────────────────────────────────
        dataset = Dataset.from_list(eval_rows)

        metrics = [faithfulness, answer_relevancy]
        # context_precision and context_recall need ground_truth
        if any(gt.strip() for gt in ground_truths):
            metrics += [context_precision, context_recall]
            log.info("Ground truths provided — running all 4 RAGAS metrics")
        else:
            log.info("No ground truths — running faithfulness + answer_relevancy only")

        log.info("Running RAGAS scoring...")
        ragas_result = evaluate(dataset, metrics=metrics)

        # ── Step 5: Build + save report ────────────────────────────────────
        report = self._build_report(
            pdf_path=pdf_path,
            session_id=session_id,
            eval_rows=eval_rows,
            ragas_result=ragas_result,
        )
        self._save_report(report)

        return report

    def print_report(self, report: dict) -> None:
        """Print a human-readable summary of the RAGAS evaluation report."""
        print("\n" + "=" * 60)
        print("  RAGAS EVALUATION REPORT")
        print("=" * 60)
        print(f"  PDF:        {report['metadata']['pdf_path']}")
        print(f"  Session:    {report['metadata']['session_id']}")
        print(f"  Timestamp:  {report['metadata']['timestamp']}")
        print(f"  Questions:  {report['metadata']['num_questions']}")
        print("-" * 60)
        print("  METRIC SCORES (0.0 = worst, 1.0 = best)")
        print("-" * 60)

        scores = report.get("scores", {})
        metric_labels = {
            "faithfulness": "Faithfulness      (no hallucination)",
            "answer_relevancy": "Answer Relevancy  (on-topic answers)",
            "context_precision": "Context Precision (retriever quality)",
            "context_recall": "Context Recall    (nothing missed)",
        }
        for key, label in metric_labels.items():
            if key in scores:
                score = scores[key]
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                status = "✅" if score >= 0.7 else "⚠️ " if score >= 0.5 else "❌"
                print(f"  {status} {label}: {score:.3f}  [{bar}]")

        print("-" * 60)
        print(f"  Report saved to: {report['metadata'].get('report_path', 'N/A')}")
        print("=" * 60 + "\n")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ingest_pdf(self, pdf_path: str, session_id: str) -> str:
        """Ingest PDF through the same ChatIngestor the app uses. Returns index_dir."""
        from src.document_ingestion.data_ingestion import ChatIngestor

        class _LocalFile:
            """Minimal file adapter for local paths (mirrors FastAPIFileAdapter)."""
            def __init__(self, path: str):
                self.filename = Path(path).name
                self._path = path
            def read(self) -> bytes:
                with open(self._path, "rb") as f:
                    return f.read()

        faiss_base = os.getenv("FAISS_BASE", "faiss_index")
        ci = ChatIngestor(
            temp_base="data",
            faiss_base=faiss_base,
            use_session_dirs=True,
            session_id=session_id,
        )
        ci.built_retriver([_LocalFile(pdf_path)], chunk_size=400, chunk_overlap=50, k=3)
        return os.path.join(faiss_base, session_id)

    def _build_report(
        self,
        pdf_path: str,
        session_id: str,
        eval_rows: list,
        ragas_result,
    ) -> dict:
        """Package RAGAS results into a structured report dict."""
        scores = {}
        try:
            # ragas_result is a dict-like object
            for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if key in ragas_result:
                    scores[key] = round(float(ragas_result[key]), 4)
        except Exception as e:
            log.warning("Could not extract scores", error=str(e))

        return {
            "metadata": {
                "pdf_path": pdf_path,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(eval_rows),
            },
            "scores": scores,
            "per_question": [
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "num_context_chunks": len(row["contexts"]),
                    "ground_truth": row["ground_truth"] or None,
                }
                for row in eval_rows
            ],
        }

    def _save_report(self, report: dict) -> None:
        """Save report as JSON. Adds the file path into the report metadata."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.REPORTS_DIR / f"ragas_report_{ts}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        report["metadata"]["report_path"] = str(path)
        log.info("RAGAS report saved", path=str(path))


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on a PDF")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file to evaluate")
    parser.add_argument(
        "--questions", nargs="*",
        help="Custom test questions (space-separated). Uses defaults if not provided."
    )
    parser.add_argument(
        "--ground-truths", nargs="*",
        help="Reference answers for context_precision and context_recall metrics."
    )
    args = parser.parse_args()

    evaluator = RAGASEvaluator()
    results = evaluator.evaluate(
        pdf_path=args.pdf,
        questions=args.questions,
        ground_truths=args.ground_truths,
    )
    evaluator.print_report(results)


if __name__ == "__main__":
    main()
