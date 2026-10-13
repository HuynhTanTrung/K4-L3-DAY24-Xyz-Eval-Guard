from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Dùng Google Gemini qua SDK mới `google-genai`.
API key đọc từ biến môi trường OPENAI_API_KEY (giữ nguyên tên cho tương thích).

Cài:  pip uninstall google-generativeai -y && pip install google-genai

Test: pytest tests/test_m5.py
"""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field
import json as _json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


# Model mặc định — override qua env GEMINI_MODEL
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Helpers ─────────────────────────────────────────────

_client = None


def _get_gemini_client():
    """Lazy init genai.Client — dùng OPENAI_API_KEY làm Gemini key."""
    global _client
    if not OPENAI_API_KEY:
        return None
    if _client is None:
        try:
            from google import genai
            _client = genai.Client(api_key=OPENAI_API_KEY)
        except Exception as e:
            print(f"  ⚠️  Gemini init failed: {e}")
            return None
    return _client


def _gemini_generate(client, prompt: str, max_tokens: int = 2000) -> str:
    """Gọi Gemini (SDK mới) và trả về text. Trả '' nếu lỗi/cắt output."""
    try:
        from google.genai import types
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=0.3,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
    except Exception:
        config = None

    resp = None
    try:
        if config is not None:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
        else:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
    except Exception:
        # Retry không thinking_config (model cũ không hỗ trợ)
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
        except Exception as e2:
            print(f"  ⚠️  Gemini generate failed: {e2}")
            return ""

    # Kiểm tra finish_reason trước khi truy cập .text
    try:
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            fr = getattr(candidates[0], "finish_reason", None)
            fr_name = getattr(fr, "name", str(fr))
            if fr_name in ("MAX_TOKENS", "2"):
                print("  ⚠️  Gemini output truncated (MAX_TOKENS). "
                      "Tăng max_output_tokens.")
        return (resp.text or "").strip()
    except Exception as e:
        print(f"  ⚠️  Gemini response has no valid text: {e}")
        return ""


def _safe_json_parse(content: str) -> dict:
    """Parse JSON từ LLM output, xử lý markdown code fence ```json ... ```."""
    if not content:
        return {}
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        return _json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if m:
            try:
                return _json.loads(m.group(0))
            except Exception:
                return {}
        return {}


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    client = _get_gemini_client()
    if client:
        prompt = (
            "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. "
            "Chỉ trả về phần tóm tắt, không thêm tiêu đề hay giải thích.\n\n"
            f"Đoạn văn:\n{text}"
        )
        out = _gemini_generate(client, prompt, max_tokens=500)
        if out:
            return out

    # Extractive fallback (không cần API)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    if not sentences:
        return text
    return ". ".join(sentences[:2]).rstrip(".") + "."


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    client = _get_gemini_client()
    if client:
        prompt = (
            f"Dựa trên đoạn văn dưới đây, tạo {n_questions} câu hỏi mà đoạn văn "
            "có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng, KHÔNG đánh số, "
            "KHÔNG thêm tiêu đề.\n\n"
            f"Đoạn văn:\n{text}"
        )
        raw = _gemini_generate(client, prompt, max_tokens=600)
        if raw:
            lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
            cleaned = [re.sub(r"^[\d\-\*\.\)\s]+", "", ln).strip() for ln in lines]
            cleaned = [q for q in cleaned if q]
            if cleaned:
                return cleaned[:n_questions]

    # Extractive fallback
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    client = _get_gemini_client()
    if client:
        prompt = (
            "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu "
            "và nói về chủ đề gì. Chỉ trả về 1 câu duy nhất.\n\n"
            f"Tài liệu: {document_title}\n\n"
            f"Đoạn văn:\n{text}"
        )
        context = _gemini_generate(client, prompt, max_tokens=300)
        if context:
            return f"{context}\n\n{text}"

    # Simple fallback — vẫn khác original_text nếu có title
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}" if prefix else text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    client = _get_gemini_client()
    if client:
        prompt = (
            "Trích xuất metadata từ đoạn văn sau. "
            "CHỈ trả về JSON hợp lệ, không thêm giải thích hay markdown:\n"
            '{"topic": "...", "entities": ["..."], '
            '"category": "policy|hr|it|finance", "language": "vi|en"}\n\n'
            f"Đoạn văn:\n{text}"
        )
        raw = _gemini_generate(client, prompt, max_tokens=600)
        parsed = _safe_json_parse(raw)
        if parsed:
            return parsed

    # Rule-based fallback
    text_lower = text.lower()
    if any(k in text_lower for k in ("nghỉ phép", "nhân viên", "thử việc", "lương")):
        category = "hr"
    elif any(k in text_lower for k in ("mật khẩu", "vpn", "server", "hệ thống")):
        category = "it"
    elif any(k in text_lower for k in ("ngân sách", "chi phí", "hóa đơn")):
        category = "finance"
    else:
        category = "policy"

    has_vietnamese = any(
        c in text for c in "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
                           "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    )

    return {
        "topic": "general",
        "entities": [],
        "category": category,
        "language": "vi" if has_vietnamese else "en",
    }


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single Gemini call to get summary + questions + context + metadata."""
    client = _get_gemini_client()
    if client:
        prompt = (
            "Phân tích đoạn văn sau và trả về DUY NHẤT một JSON hợp lệ, "
            "không thêm giải thích hay markdown:\n"
            "{\n"
            '  "summary": "tóm tắt 2-3 câu",\n'
            '  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],\n'
            '  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",\n'
            '  "metadata": {"topic": "...", "entities": ["..."], '
            '"category": "policy|hr|it|finance", "language": "vi|en"}\n'
            "}\n\n"
            f"Tài liệu: {source}\n\n"
            f"Đoạn văn:\n{text}"
        )
        raw = _gemini_generate(client, prompt, max_tokens=2000)
        parsed = _safe_json_parse(raw)
        if parsed:
            return parsed

    # Fallback: ghép từ 4 hàm riêng
    ctx_full = contextual_prepend(text, source)
    context_line = ctx_full.split("\n\n")[0] if source else ""
    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": context_line,
        "metadata": extract_metadata(text),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo (Gemini SDK mới) ===\n")
    print(f"Model: {GEMINI_MODEL}")
    print(f"API key set: {bool(OPENAI_API_KEY)}\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")