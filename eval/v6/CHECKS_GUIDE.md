# Hướng dẫn kiểm tra sau Step 2 (ViTool v1.1)

Hai tác giả làm **độc lập**, mỗi người một file `_annotator1.json` / `_annotator2.json`.
Chỉ sửa các trường đang là `null` (và `note` nếu cần). Không sửa trường khác.

## 1. rewrite_check (50 mục): query cũ → query mới

Mỗi mục có `old_query`, `new_query` và `added_values` (các giá trị đã được thêm vào).

- `meaning_preserved`: `"y"` nếu query mới giữ đúng ý định và mọi thông tin của query cũ,
  chỉ **thêm** các giá trị trong `added_values`; `"n"` nếu làm mất, đổi nghĩa, hoặc thêm
  thông tin ngoài danh sách.
- `natural`: mức tự nhiên của query mới như người Việt gõ cho trợ lý:
  - `3` = tự nhiên, không có gì gượng;
  - `2` = hiểu được nhưng hơi gượng (liệt kê kiểu điền form, lặp từ, trật tự lạ);
  - `1` = không tự nhiên (lộ mã kỹ thuật, tên tham số tiếng Anh, câu sai ngữ pháp).

## 2. v11_spotcheck (30 trajectory v1.1 ngẫu nhiên)

Mỗi mục có `query`, và với từng bước: lời gọi gold + observation của các bước trước.
Câu hỏi: **còn giá trị nào trong lời gọi gold mà người dùng chưa nói, không suy ra được
từ query, không lấy từ observation trước, và không phải lựa chọn enum hiển nhiên?**
(Cùng tiêu chí với `ANNOTATION_GUIDE.md` của Step 0; số lượng không nêu trong query
tính là chưa grounded.)

- `fabricated_value_found`: `"y"` hoặc `"n"`. Từ v1.2: lựa chọn enum (thanh toán, sàn, cách hoàn tiền, hạng ghế...) mà người dùng không nói cũng tính là bịa.
- `fabricated_values`: nếu `"y"`, ghi danh sách dạng `["add_to_cart.quantity=2", ...]`.
- `note`: tuỳ chọn (vd query mới có vấn đề khác).
