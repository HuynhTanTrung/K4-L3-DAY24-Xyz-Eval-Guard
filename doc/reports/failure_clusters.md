# Phân tích lỗi Phase A

Nhóm: Xyz
Người phụ trách: Huỳnh Tấn Trung - 2A202602742 (điền khi nộp)

Nguồn: `artifacts/ragas_50q.json`; run timestamp: `2026-10-13T04:02:47.304755+00:00`.
Generation mode: `llm`; model: `gemini-3.8-flash`.
Evaluator models: `["BAAI/bge-m3", "gemini-3.8-flash"]` (LLM = gemini-3.8-flash, embedding = BAAI/bge-m3 local).
Chạy ở chế độ `--live`; không có fallback extractive cho 50 answers.

## Điểm theo distribution

| Metric            | factual (20) | multi_hop (20) | adversarial (10) |
| ----------------- | ------------ | -------------- | ---------------- |
| faithfulness      | 1.0000       | 0.7400         | 1.0000           |
| answer_relevancy  | 0.9382       | 0.7511         | 0.7762           |
| context_precision | 0.9500       | 0.9250         | 0.9000           |
| context_recall    | 0.9000       | 0.6208         | 0.6417           |
| avg_score         | 0.9470       | 0.7592         | 0.8295           |

## Bottom 10 và nguyên nhân

| Rank | ID  | Distribution | Question                                                                                              | avg_score | worst_metric      | Diagnosis                     | Suggested fix                            |
| ---- | --- | ------------ | ----------------------------------------------------------------------------------------------------- | --------- | ----------------- | ----------------------------- | ---------------------------------------- |
| 1    | 24  | multi_hop    | Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?                            | 0.3750    | faithfulness      | LLM hallucinating             | Tighten system prompt, lower temperature |
| 2    | 39  | multi_hop    | So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0 về độ dài tối thiểu, thời hạn đổi và MFA.           | 0.3750    | answer_relevancy  | Answer doesn't match question | Improve prompt template                  |
| 3    | 40  | multi_hop    | Nhân viên cấp Manager thử việc 90 ngày, tháng thứ 3, phát hiện vi phạm bảo mật. Nên/không nên làm gì? | 0.4822    | faithfulness      | LLM hallucinating             | Tighten system prompt, lower temperature |
| 4    | 37  | multi_hop    | Nhân viên tự ý xóa malware và chia sẻ thông tin sự cố trên Slack nội bộ. Vi phạm chính sách nào?      | 0.4925    | context_recall    | Missing relevant chunks       | Improve chunking or add BM25             |
| 5    | 30  | multi_hop    | So sánh quyền lợi bảo hiểm giữa nhân viên thử việc và nhân viên chính thức.                           | 0.6250    | answer_relevancy  | Answer doesn't match question | Improve prompt template                  |
| 6    | 50  | adversarial  | Nhân viên Manager có thể dùng VPN cá nhân (NordVPN) khi WFH không?                                    | 0.6250    | answer_relevancy  | Answer doesn't match question | Improve prompt template                  |
| 7    | 35  | multi_hop    | Nhân viên Junior P1 lương cơ bản 12 triệu thử việc: nhận lương bao nhiêu và phụ cấp gì?               | 0.6403    | context_recall    | Missing relevant chunks       | Improve chunking or add BM25             |
| 8    | 11  | factual      | VPN có bắt buộc không khi làm việc từ xa (WFH)?                                                       | 0.7266    | context_recall    | Missing relevant chunks       | Improve chunking or add BM25             |
| 9    | 7   | factual      | Nghỉ phép không lương 20 ngày cần ai phê duyệt?                                                       | 0.7285    | context_precision | Too many irrelevant chunks    | Add reranking or metadata filter         |
| 10   | 6   | factual      | Thông tin lương thuộc cấp độ phân loại dữ liệu nào?                                                   | 0.7442    | context_precision | Too many irrelevant chunks    | Add reranking or metadata filter         |

**Đối chiếu bằng chứng:**

- **ID 24 (faithfulness = 0.0)**: Answer trả về "Không tìm thấy." dù context có đủ điều kiện tính phạt (2%/tháng trên số tiền chưa hoàn ứng, sau 15 ngày). Đây là **generation failure** — LLM bỏ qua context — chứ không phải thiếu retrieval. Context recall = 0.5, context precision ≈ 1.0.
- **ID 39 (answer_relevancy = 0.0)**: Answer "Không tìm thấy." mặc dù có 3 chunks liên quan (v1 status, v2 MFA, v1 header). Lỗi generation: model không chịu tổng hợp từ nhiều chunks.
- **ID 37 (context_recall = 0.0)**: Retrieval chỉ trả về 3 chunks, thiếu điều khoản "báo cáo trong vòng 1 giờ" và "không chia sẻ rộng rãi" — thiếu chunk → hallucination hợp lý nhưng không đầy đủ.
- **ID 50 (answer_relevancy = 0.0)**: Pipeline trả "Không tìm thấy." dù context có quy định cấm VPN cá nhân rõ ràng. Đây là **policy contradiction trap** — model không nhận diện được phủ định.

**Nhóm lỗi chính:** 6/10 câu trong bottom 10 là multi_hop; lỗi phổ biến nhất là trả lời "Không tìm thấy." khi context có dữ liệu nhưng cần tổng hợp từ 2+ chunks.

## Matrix và nhóm yếu nhất

**Matrix worst_metric × distribution** (mỗi câu đếm một lần theo worst_metric):

| worst_metric      | factual | multi_hop | adversarial |
| ----------------- | ------- | --------- | ----------- |
| faithfulness      | 0       | 6         | 0           |
| answer_relevancy  | 16      | 5         | 1           |
| context_precision | 3       | 0         | 2           |
| context_recall    | 1       | 9         | 7           |

Nhóm yếu theo mean thấp nhất: **multi_hop**; mean: **0.7592**.
Worst metric xuất hiện nhiều nhất: **answer_relevancy**; số câu: **22/50**.

**Giải thích:** multi_hop có mean thấp nhất vì đòi hỏi tổng hợp từ 2+ tài liệu (retrieval + suy luận số học), trong khi pipeline đôi khi trả về "Không tìm thấy." khi top-k không phủ hết mọi mảnh cần thiết. answer_relevancy bị nhiều câu hỏi có ngữ cảnh thừa hoặc câu trả lời lệch trọng tâm.

**Đề xuất sửa:** Tăng `RERANK_TOP_K` lên 5 cho câu multi_hop; thêm few-shot ví dụ dạng "nếu chỉ thấy một phần dữ liệu, trả lời phần thấy được thay vì từ chối"; sau mỗi thay đổi chạy lại `python scripts/run_experiments.py --phase a --live` và so sánh per_distribution.avg_score.

## Bẫy adversarial và giới hạn

Adversarial mean = 0.8295, cao hơn multi_hop (0.7592) nhưng thấp hơn factual (0.9470). Điểm thấp đến từ 4 câu context_recall ≤ 0.5 và 1 câu answer_relevancy = 0.0 (ID 50).

Ba bẫy adversarial nổi bật:

- **Version conflict** (ID 41, 42, 44, 49): pipeline thường retrieve cả v2023 và v2024; đôi khi trả lời theo cả hai thay vì ưu tiên bản hiện hành → recall thấp.
- **Negation** (ID 46, 48): câu hỏi phủ định ("có được... không?") — pipeline trả lời đúng là phủ định nhưng context_recall vẫn thấp vì chunk reference không khớp câu hỏi.
- **Policy contradiction** (ID 50): câu hỏi nghe hợp lý nhưng vi phạm chính sách; pipeline trả "Không tìm thấy." dù context có dữ liệu → lỗi generation rõ ràng.

**Giới hạn:** 50 câu là mẫu nhỏ; RAGAS với gemini-3.8-flash có thể có variance giữa các lần chạy; điểm adversarial thấp không tự chứng minh pipeline "nhận ra" bẫy — chỉ cho thấy nó không xử lý tốt.
