from __future__ import annotations

"""Phase A: RAGAS Production Evaluation — 50q, 3 distributions, cluster analysis."""

import json
import os
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, ANSWERS_PATH, ARTIFACTS_DIR
from src.experiment_support import write_json

Distribution = str  # "factual" | "multi_hop" | "adversarial"

DIAGNOSTIC_TREE = {
    "faithfulness":      ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall":    ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy":  ("Answer doesn't match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }
        return min(scores, key=scores.get)


# ─── Đã implement sẵn ────────────────────────────────────────────────────────

@dataclass
class RagasEvaluation:
    """Task 2 output: measured rows plus model identifiers reported by m4."""
    results: list[RagasResult]
    evaluator_models: list[str]

def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load 50q test set với 3 distributions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers từ setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json không tìm thấy tại {path}\n"
            "→ Chạy trước: python scripts/setup_answers.py --live"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_phase_a_report(results: list[RagasResult], clusters: dict, bottom: list[dict]) -> dict:
    """Provided exporter: preserve the student-produced Task 3 rows verbatim."""
    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness":      sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy":  sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall":    sum(r.context_recall for r in subset) / len(subset),
                "avg_score":         sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_question": [asdict(r) for r in results],
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": bottom,
    }
    return report


def save_phase_a_report(results: list[RagasResult], clusters: dict, bottom: list[dict],
                        path: str = os.path.join(ARTIFACTS_DIR, "ragas_50q.json")) -> None:
    write_json(path, build_phase_a_report(results, clusters, bottom))



# ─── Tasks 1-4: Sinh viên implement ──────────────────────────────────────────

def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Task 1: Nhóm 50 câu hỏi theo 3 distributions."""
    groups = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        dist = item["distribution"]
        if dist not in groups:
            raise ValueError(f"Unknown distribution: {dist!r}")
        groups[dist].append(item)
    return groups


def run_ragas_50q(answers: list[dict]) -> RagasEvaluation:
    """Task 2: Chạy RAGAS 4 metrics trên toàn bộ 50 câu hỏi."""
    try:
        from src.m4_eval import evaluate_ragas
    except ImportError:
        print("⚠️  Không tìm thấy src/m4_eval.py — đã copy từ Day 18 chưa?")
        raise ValueError("Missing m4 evaluator; cannot declare its actual models")

    questions     = [a["question"]     for a in answers]
    ans_texts     = [a["answer"]       for a in answers]
    contexts      = [a["contexts"]     for a in answers]
    ground_truths = [a["ground_truth"] for a in answers]

    raw = evaluate_ragas(questions, ans_texts, contexts, ground_truths)
    per_q = raw.get("per_question", [])

    if len(per_q) != len(answers):
        raise ValueError(
            f"RAGAS returned {len(per_q)} rows for {len(answers)} answers"
        )

    results: list[RagasResult] = []
    for a, pq in zip(answers, per_q):
        # pq may be a dict or an object; handle both
        def _g(key):
            return pq[key] if isinstance(pq, dict) else getattr(pq, key)
        results.append(RagasResult(
            question_id        = a["id"],
            distribution       = a["distribution"],
            question           = a["question"],
            answer             = a["answer"],
            contexts           = a["contexts"],
            ground_truth       = a["ground_truth"],
            faithfulness       = float(_g("faithfulness")),
            answer_relevancy   = float(_g("answer_relevancy")),
            context_precision  = float(_g("context_precision")),
            context_recall     = float(_g("context_recall")),
        ))

    evaluator_models = raw.get("metadata", {}).get("evaluator_models", [])
    if not evaluator_models:
        raise ValueError(
            "m4 evaluator did not report evaluator_models — required by Task 2."
        )
    return RagasEvaluation(results, evaluator_models)


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Task 3: Lấy 10 câu hỏi có avg_score thấp nhất."""
    sorted_asc = sorted(results, key=lambda r: r.avg_score)
    worst = sorted_asc[:10]

    output = []
    for i, r in enumerate(worst, start=1):
        diag, fix = DIAGNOSTIC_TREE[r.worst_metric]
        output.append({
            "rank":          i,
            "question_id":   r.question_id,
            "distribution":  r.distribution,
            "question":      r.question,
            "avg_score":     round(r.avg_score, 4),
            "worst_metric":  r.worst_metric,
            "diagnosis":     diag,
            "suggested_fix": fix,
        })
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    """Task 4: Phân tích failure clusters theo (worst_metric × distribution)."""
    dists = ("factual", "multi_hop", "adversarial")
    matrix = {
        metric: {d: 0 for d in dists}
        for metric in DIAGNOSTIC_TREE
    }
    for r in results:
        if r.worst_metric not in matrix:
            raise ValueError(f"Unknown worst_metric: {r.worst_metric!r}")
        matrix[r.worst_metric][r.distribution] += 1

    # Per-distribution average score, ignoring empty groups
    means: dict[str, float] = {}
    for d in dists:
        subset = [r for r in results if r.distribution == d]
        if subset:
            means[d] = sum(r.avg_score for r in subset) / len(subset)

    if not means:
        return {
            "matrix": matrix,
            "distribution_avg_scores": {},
            "dominant_failure_distribution": None,
            "dominant_failure_metric": None,
            "insight": "No results to analyze.",
        }

    # Weakest distribution = LOWEST average quality (not the largest group)
    dominant_dist = min(means, key=means.get)

    # Dominant failure metric = most common worst_metric across ALL rows
    dominant_metric = max(matrix, key=lambda m: sum(matrix[m].values()))

    insight = (
        f"Distribution '{dominant_dist}' has the lowest mean score "
        f"({means[dominant_dist]:.3f}). "
        f"Metric '{dominant_metric}' is the most frequent weakest metric. "
        f"Suggested fix: {DIAGNOSTIC_TREE[dominant_metric][1]}"
    )

    return {
        "matrix":                         matrix,
        "distribution_avg_scores":        {k: round(v, 4) for k, v in means.items()},
        "dominant_failure_distribution":  dominant_dist,
        "dominant_failure_metric":        dominant_metric,
        "insight":                        insight,
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scripts.run_experiments import main
    raise SystemExit(main(["--phase", "a", *sys.argv[1:]]))
