from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str, reference: str = "") -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B)."""
    PROMPT_TEMPLATE = '''Bạn là một expert đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}
Chính sách/đáp án tham chiếu: {reference}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí: độ chính xác, đầy đủ, súc tích.
Trả lời JSON (chỉ JSON, không text khác):
{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}
'''
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system",
             "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
            {"role": "user",
             "content": PROMPT_TEMPLATE.format(
                 question=question,
                 answer_a=answer_a,
                 answer_b=answer_b,
                 reference=reference,
             )},
        ],
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    # Normalize / guardrails
    winner = str(data.get("winner", "tie")).strip()
    if winner not in ("A", "B", "tie"):
        winner = "tie"
    scores = data.get("scores") or {"A": 0.0, "B": 0.0}
    return {
        "winner":    winner,
        "reasoning": str(data.get("reasoning", "")),
        "scores": {
            "A": float(scores.get("A", 0.0)),
            "B": float(scores.get("B", 0.0)),
        },
    }


def swap_and_average(question: str, answer_a: str, answer_b: str, reference: str = "") -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi), lấy kết quả nhất quán."""
    pass1     = pairwise_judge(question, answer_a, answer_b, reference=reference)
    pass2_raw = pairwise_judge(question, answer_b, answer_a, reference=reference)

    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw["winner"], "tie")

    if pass1["winner"] == winner_pass2:
        final = pass1["winner"]
    else:
        final = "tie"

    position_consistent = (pass1["winner"] == winner_pass2)

    scores_pass2 = {
        "A": float(pass2_raw["scores"].get("B", 0.0)),
        "B": float(pass2_raw["scores"].get("A", 0.0)),
    }

    return JudgeResult(
        question            = question,
        answer_a            = answer_a,
        answer_b            = answer_b,
        winner_pass1        = pass1["winner"],
        winner_pass2        = winner_pass2,
        final_winner        = final,
        reasoning_pass1     = pass1["reasoning"],
        reasoning_pass2     = pass2_raw["reasoning"],
        position_consistent = position_consistent,
        scores_pass1        = pass1["scores"],
        scores_pass2        = scores_pass2,
    )


def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels."""
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have same length")
    if not judge_labels:
        return 0.0

    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(human_labels, judge_labels))
    except ImportError:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (
            (judge_labels.count(1) / n) * (human_labels.count(1) / n) +
            (judge_labels.count(0) / n) * (human_labels.count(0) / n)
        )
        if p_e == 1:
            return 0.0
        return (p_o - p_e) / (1 - p_e)


def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias."""
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged":        0,
            "position_bias_rate":  0.0,
            "position_bias_count": 0,
            "verbosity_bias":      0.0,
            "verbosity_details":   {"a_wins_a_longer": 0, "b_wins_b_longer": 0,
                                     "total_decisive": 0},
            "interpretation":      "No judge results to analyze.",
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate  = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive else 0.0

    parts = []
    parts.append(
        "Position bias cao — nên dùng swap-and-average."
        if position_bias_rate > 0.3
        else "Position bias thấp — judge ổn định."
    )
    parts.append(
        "Verbosity bias đáng lo ngại (>0.6)."
        if verbosity_bias > 0.6
        else "Verbosity bias trong ngưỡng chấp nhận."
    )
    interpretation = " ".join(parts)

    return {
        "total_judged":        total,
        "position_bias_rate":  round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias":      round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive":  decisive,
        },
        "interpretation": interpretation,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scripts.run_experiments import main
    raise SystemExit(main(["--phase", "b", *sys.argv[1:]]))
