# Hướng dẫn gán nhãn — kiểm tra audit grounding (Step 0, v6)

## Mục đích
Bộ dữ liệu ViTool có các giá trị tham số "gold" (đáp án) mà model phải sinh ra.
Một script tự động (audit) đã đánh dấu những giá trị **không thể suy ra** từ
thông tin model được nhìn thấy. Ta cần biết audit đó đúng đến đâu trước khi sửa
dữ liệu dựa trên nó. Bạn gán nhãn 60 giá trị; một phần được audit đánh dấu, một
phần không — **bạn không được biết mục nào thuộc loại nào** (đừng mở
`audit_sample_key.json`).

## Mỗi mục cho bạn xem
- `query`: câu yêu cầu của người dùng (model chỉ thấy câu này + các bước trước).
- `previous_steps`: các lời gọi tool ở bước trước **và observation** (kết quả trả
  về) của chúng. Đây là thông tin model có tại bước này.
- `tool`, `tool_description`, `param_schema`: tool đang gọi và mô tả tham số
  (kiểu, `enum` = danh sách giá trị cho phép, `required`).
- `gold_call`: toàn bộ lời gọi gold của bước này (để hiểu ngữ cảnh).
- `param` và `gold_value`: **giá trị cần gán nhãn**. Nếu là phần tử của một danh
  sách (vd 1 tên hành khách), `gold_value` là phần tử đó.

## Câu hỏi
"Một người cẩn thận, chỉ đọc `query` và `previous_steps`, có viết ra **đúng giá
trị này** (cho phép khác định dạng) mà không phải đoán không?"

## Nhãn — chọn đúng 1, xét theo thứ tự ưu tiên từ trên xuống
1. `inferable_from_query` — giá trị có trong query, nguyên văn **hoặc** sau khi
   chuẩn hoá/suy luận chắc chắn:
   - khác định dạng: `25/12/2026` ↔ `2026-12-25`; `1,5 triệu` ↔ `1500000`;
     `500k` ↔ `500000`; có/không dấu; viết hoa/thường; `HN` ↔ `Hà Nội`;
   - dịch/chuẩn hoá tên: `tiếng Pháp` → `fr`; `đô la Mỹ` → `USD`;
   - ngày tương đối khi query có mốc rõ (vd "ngày mai" và query có ngày hôm nay);
   - số lượng 1 khi query nói rõ một món/một người ("mua **một** chiếc", "đặt vé
     cho tôi") và không có con số nào khác.
   Nếu phải chọn giữa nhiều giá trị hợp lý → KHÔNG phải nhãn này.
2. `from_prior_observation` — giá trị xuất hiện trong observation của một bước
   trước (vd `flight_id`, `cart_id`, `product_id` do bước search/add trả về).
   Chỉ tính observation, không tính params của bước trước (nếu param bước trước
   cũng là giá trị bịa thì vẫn là bịa).
3. `tool_default_or_enum` — giá trị nằm trong `enum` của tham số (hoặc là
   `default` khai báo trong `param_schema`). Điền thêm `enum_justified`:
   `y` nếu ý nghĩa query ủng hộ đúng lựa chọn này (vd "mất thẻ" → `lost`),
   `n` nếu query không nói gì và đây chỉ là một lựa chọn tuỳ ý.
4. `ungrounded` — không thuộc 3 loại trên: giá trị phải bịa/đoán (số tài khoản,
   mã khách hàng, mã đơn, mã voucher, địa chỉ, tên người, giá tiền, ngày… không
   có trong query/observation; số lượng khi query không nói gì).

## Quy tắc thêm
- Chấm **giá trị**, không chấm việc tham số có nên tồn tại hay không.
- Không dùng kiến thức bên ngoài để "đoán đúng" (vd tự biết mã sân bay) — trừ
  các ánh xạ chuẩn hoá phổ biến ở mục 1.
- Phân vân giữa `inferable_from_query` và `ungrounded` → chọn cái bạn tin hơn và
  ghi lý do vào `note`. Mọi mục đều phải có đúng 1 nhãn.
- Làm **độc lập**, không trao đổi với người kia cho tới khi cả hai nộp xong.

## Cách điền
- File JSON riêng của bạn (`audit_sample_annotator1.json` hoặc `..._annotator2.json`):
  với mỗi mục, sửa `"label": null` thành một trong 4 chuỗi trên (giữ dấu ngoặc
  kép), `"enum_justified"` thành `"y"`/`"n"` khi nhãn là `tool_default_or_enum`,
  và `"note"` nếu cần. Không sửa field khác.
- Hoặc dùng `audit_sample.csv` (cột `label_annotator_1` / `label_annotator_2`).
- Gửi lại file; script `eval/v6/score_audit_v6.py` sẽ kiểm tra định dạng và tính
  precision/recall của audit + Cohen's kappa.

## Ví dụ
| query (rút gọn) | param = gold_value | nhãn |
|---|---|---|
| "Chuyển 2 triệu cho tài khoản 0123456789" | amount = 2000000 | inferable_from_query |
| "Khoá thẻ vì bị mất" | reason = lost (enum) | tool_default_or_enum, enum_justified=y |
| "Thanh toán giỏ hàng" (không nói phương thức) | payment_method = cod (enum) | tool_default_or_enum, enum_justified=n |
| bước trước trả `{"cart_id": "CART25E4F4"}` | cart_id = CART25E4F4 | from_prior_observation |
| "Xem lịch sử giao dịch tháng này" | account_number = 1234567890 | ungrounded |
| "Mua laptop Dell và theo dõi đơn" | shipping_address = 123 Nguyễn Trãi | ungrounded |
