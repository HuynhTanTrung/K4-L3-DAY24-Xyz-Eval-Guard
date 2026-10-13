# Nhóm thực hiện K4 Track 3B

Tên nhóm: Xyz

Repo chung: https://github.com/HuynhTanTrung/K4-L3-DAY24-Xyz-Eval-Guard

Trưởng nhóm/người nộp: Huỳnh Tấn Trung - 2A202602742

Ngày lab/deadline áp dụng: {{Ngày giờ theo thông báo lớp}} — Asia/Ho_Chi_Minh (UTC+07:00)

Commit dùng để chấm: Lab24

Mỗi nhóm nộp một repo. Điền đủ thành viên và một mục INDIVIDUAL cho từng người.
Thêm/xóa rows để khớp nhóm thật; mẫu ba rows không ép nhóm phải có ba người.

## Thành viên và phân công

| STT | Họ và tên       | MSSV        | Vai trò / công việc thật | Bằng chứng           |
| --- | --------------- | ----------- | ------------------------ | -------------------- |
| 1   | Huỳnh Tấn Trung | 2A202602742 | All                      | All                  |
| 2   | {{Họ tên}}      | {{MSSV}}    | {{Phase/file/hàm}}       | {{Commit/PR/report}} |
| 3   | {{Họ tên}}      | {{MSSV}}    | {{Phase/file/hàm}}       | {{Commit/PR/report}} |

Người tích hợp/chạy batch/nộp link: {{Họ tên, công việc và bằng chứng}}.

## Môi trường và chuẩn bị trước lớp

- Windows 11 + Python 3.11 venv + MSVC Build Tools 2022.
- Deps: `requirements.txt`, `requirements-test.lock.txt`, `pipwin install annoy`, `google-genai`, `langchain-openai`, `langchain-huggingface`.
- Qdrant `qdrant/qdrant:v1.9.7` chạy qua `docker compose`; `en_core_web_lg`, `BAAI/bge-m3` đã tải.
- 50 answers sinh bằng `setup_answers.py --live`, `generation_mode="llm"`, `generation_model="gemini-3.8-flash"`.

## INDIVIDUAL

Sao chép mẫu cho mỗi thành viên; tiêu đề họ tên không dấu-MSSV.

### Huỳnh Tấn Trung-2A202602742

- **Đã làm:** tích hợp m1–m5 vào `src/`; sửa `config.py` + `setup_answers.py` để dùng Gemini qua `OPENAI_BASE_URL`; implement TODO Phase A/B/C; viết 3 báo cáo.
- **Bằng chứng:** `artifacts/answers_50q.json`, `ragas_50q.json`, `judge_results.json`, `guard_results.json`; `doc/reports/failure_clusters.md`, `bias_report.md`, `blueprint.md`.
- **Cách kiểm tra:** `pytest tests/ -q`; các lệnh `run_experiments.py --phase a/b/c --live`; `scripts/check_lab.py`.
- **Đã học:** phân biệt `base_url` / `api_base` / `OPENAI_API_BASE`; Gemini compat endpoint chỉ hỗ trợ chat + `n=1`; RAGAS mặc định dùng OpenAI nếu không truyền `llm=`/`embeddings=`.
- **Giới hạn:** Phase C clean false-positive rate = 1.0 (8/8 clean bị block nhầm); input-guard P95 = 4911 ms (vượt 500 ms budget). Đã ghi rõ trong `blueprint.md` kèm hướng cải thiện.
- **AI hỗ trợ:** dùng AI để đọc stack trace và gợi ý cấu trúc báo cáo; tự kiểm chứng bằng cách chạy lại script và đối chiếu JSON với schema.

## Cập nhật sau khi nộp

{{Ghi phiên bản nộp; nếu sửa sau deadline, ghi commit/thời điểm/lý do và thông báo
coach cho phép áp dụng bản cập nhật. Nếu không có cập nhật, ghi rõ không có.}}
