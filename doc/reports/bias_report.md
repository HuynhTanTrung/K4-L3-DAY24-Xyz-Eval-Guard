# Đánh giá LLM Judge Phase B

Nhóm: Xyz
Người phụ trách: Huỳnh Tấn Trung - 2A202602742

Nguồn: `artifacts/judge_results.json`; run timestamp: `2026-10-13T04:05:19.706678+00:00`.
Models generation/judge: generation = `["gemini-3.8-flash"]`; judge = `["gemini-3.8-flash"]`.

## Năm cặp A/B và hai lượt

Nguồn cố định: A từ `artifacts/answers_50q.json:answer`; B từ `fixtures/human_labels_10q.json:model_answer`; reference từ `fixtures/test_set_50q.json:ground_truth`. Ids: 1, 5, 12, 21, 41.

| ID  | Winner pass1 | Winner pass2 (đã quy về A/B gốc) | Final | Position consistent | Scores pass1 (A/B) | Scores pass2 (A/B) |
| --- | ------------ | -------------------------------- | ----- | ------------------- | ------------------ | ------------------ |
| 1   | B            | B                                | B     | True                | 0.75 / 0.90        | 0.75 / 0.95        |
| 5   | A            | A                                | A     | True                | 1.00 / 0.00        | 1.00 / 0.00        |
| 12  | A            | A                                | A     | True                | 1.00 / 0.60        | 1.00 / 0.60        |
| 21  | B            | B                                | B     | True                | 0.50 / 1.00        | 0.50 / 1.00        |
| 41  | A            | A                                | A     | True                | 0.90 / 0.10        | 0.85 / 0.20        |

Position bias count/rate: **0 / 0.0**.
Verbosity win rate: **0.80** (4/5 decisive cases có winner dài hơn); total_decisive = 5.

**Giải thích:**

- Không có case nào lệch giữa hai lượt swap → position bias thấp ở mẫu 5 cặp.
- Verbosity: ID 1, 12, 21, 41 winner đều dài hơn loser. Chỉ ID 5 winner (A) ngắn hơn (B). Đây là dấu hiệu thống kê đáng lưu ý, **không** phải kết luận nhân quả — reasoning cho thấy judge chấm theo đúng nội dung chính sách, không chỉ theo độ dài.

## Calibration đúng model_answer

Helper đánh giá đúng `question`/`model_answer` có trong `fixtures/human_labels_10q.json`; không lấy answer mới. Prompt chỉ nhận question, answer, reference.

| ID  | model_answer (rút gọn)                             | Judge label | Human label | Agree | Reasoning (rút gọn)                         |
| --- | -------------------------------------------------- | ----------- | ----------- | ----- | ------------------------------------------- |
| 1   | NV được nghỉ 3 ngày có lương khi kết hôn           | 1           | 1           | ✓     | Đúng số ngày, đúng chính sách               |
| 5   | Cần Giám đốc phòng ban phê duyệt                   | 0           | 0           | ✓     | Sai người phê duyệt (>50tr cần CEO)         |
| 12  | NV được thưởng Tết tối thiểu 1 tháng lương         | 1           | 1           | ✓     | Đúng mức thưởng                             |
| 21  | 18 ngày phép, Senior 20-35tr                       | 1           | 1           | ✓     | Đúng và đầy đủ                              |
| 23  | Hoàn trả 25 triệu vì chưa đủ 1 năm cam kết         | 1           | 1           | ✓     | Đúng số tiền và lý do                       |
| 29  | Trưởng phòng phê duyệt, phạt 2%/tháng              | 0           | 0           | ✓     | Thiếu Kế toán trưởng + thiếu số tiền cụ thể |
| 33  | Phép 19 ngày, phụ cấp 1.500.000                    | 1           | 1           | ✓     | Đúng và đầy đủ                              |
| 41  | 12 ngày phép năm                                   | 0           | 0           | ✓     | Dùng chính sách cũ (v2023)                  |
| 46  | Thử việc không được nghỉ phép năm, xin không lương | 1           | 1           | ✓     | Đúng phủ định                               |
| 50  | Được, miễn là đảm bảo an toàn                      | 0           | 0           | ✓     | Sai: chính sách cấm VPN cá nhân             |

Cohen's κ: **1.000**.
Observed agreement = 10/10 = 1.00; chance agreement = (7/10 × 7/10) + (3/10 × 3/10) = 0.49 + 0.09 = 0.58.
κ = (1.00 − 0.58) / (1 − 0.58) = 1.00 → theo Landis–Koch: **almost perfect agreement**.

## Giới hạn và cách dùng judge

- **Model/prompt:** gemini-3.8-flash, prompt tiếng Việt có reference + 3 tiêu chí (chính xác, đầy đủ, súc tích).
- **Số cặp rất nhỏ:** chỉ 5 cặp pairwise và 10 nhãn calibration. κ = 1.0 trên 10 mẫu không có nghĩa judge hoàn hảo — chỉ có nghĩa không có case lệch trong mẫu này.
- **Không xem judge là đáp án tuyệt đối:** với 10 nhãn human cố định trong fixtures, pipeline có thể vô tình "học" đúng phân bố. Cần mở rộng lên ≥50 nhãn để tin cậy hơn.
- **Cách kiểm chứng:** so sánh reasoning của judge với reference; cả 10 case đều có lý do khớp với ground_truth → đáng tin cho calibration này.
- **Không đạt κ > 0.6?** Thực tế đạt 1.0, nhưng do số mẫu nhỏ, chỉ nên dùng judge làm bộ lọc nhiễu, không thay human review khi đưa ra quyết định về chất lượng RAG.
