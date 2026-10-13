from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from config import JUDGE_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, EMBEDDING_MODEL
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI

class SingleCandidateChatOpenAI(ChatOpenAI):
    """Force n=1 on every call — Gemini compat layer rejects n>1."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs["n"] = 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs["n"] = 1
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
    
@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    empty = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "per_question": [],
    }

    if not questions:
        return empty

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        evaluator_llm = LangchainLLMWrapper(SingleCandidateChatOpenAI(
            model=JUDGE_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0.0,
        ))
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)   # local BAAI/bge-m3, no API
        )

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy,
                    context_precision, context_recall],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=RunConfig(max_workers=1),
        )

        df = result.to_pandas()

        def _clamp01(x) -> float:
            try:
                v = float(x)
            except (TypeError, ValueError):
                return 0.0
            if v != v:            # NaN check
                return 0.0
            return max(0.0, min(1.0, v))

        per_question = [
            EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=list(row.get("contexts", [])),
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=_clamp01(row.get("faithfulness", 0.0)),
                answer_relevancy=_clamp01(row.get("answer_relevancy", 0.0)),
                context_precision=_clamp01(row.get("context_precision", 0.0)),
                context_recall=_clamp01(row.get("context_recall", 0.0)),
            )
            for _, row in df.iterrows()
        ]

        def _avg(metric: str) -> float:
            vals = [getattr(r, metric) for r in per_question]
            return round(_clamp01(sum(vals) / len(vals)), 4) if vals else 0.0

        return {
            "faithfulness": _avg("faithfulness"),
            "answer_relevancy": _avg("answer_relevancy"),
            "context_precision": _avg("context_precision"),
            "context_recall": _avg("context_recall"),
            "per_question": per_question,
            "metadata": {
                "evaluator_models": [JUDGE_MODEL, EMBEDDING_MODEL],
            },
        }

    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return empty


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "LLM hallucinating",
            "Tighten prompt, lower temperature",
        ),
        "context_recall": (
            "Missing relevant chunks",
            "Improve chunking or add BM25",
        ),
        "context_precision": (
            "Too many irrelevant chunks",
            "Add reranking or metadata filter",
        ),
        "answer_relevancy": (
            "Answer doesn't match question",
            "Improve prompt template",
        ),
    }

    metrics = ["faithfulness", "answer_relevancy",
               "context_precision", "context_recall"]

    scored: list[tuple[float, str, float, EvalResult]] = []
    for r in eval_results:
        vals = {m: getattr(r, m) for m in metrics}
        avg = sum(vals.values()) / len(vals)
        worst_metric = min(vals, key=vals.get)
        scored.append((avg, worst_metric, vals[worst_metric], r))

    scored.sort(key=lambda x: x[0])  # ascending — worst first

    failures = []
    for avg, worst_metric, worst_score, r in scored[:bottom_n]:
        diagnosis, fix = diagnostic_tree.get(
            worst_metric, ("Unknown issue", "Manual inspection required")
        )
        failures.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "worst_metric": worst_metric,
            "score": round(worst_score, 4),
            "avg_score": round(avg, 4),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    return failures


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
