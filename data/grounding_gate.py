"""
data/grounding_gate.py — Cổng G3 (grounding) của pipeline ViTool, chạy sau Cổng 2.

Mỗi giá trị gold param (phần tử list / giá trị lồng được xét riêng) phải lấy được từ:
  query     : có trong query sau chuẩn hoá (NFC, lowercase, gộp khoảng trắng; dạng
              bỏ dấu và bỏ khoảng trắng; ngày dd/mm/yyyy <-> yyyy-mm-dd; số dạng
              "1.000.000", "1,5 triệu", "500k", "2 tỷ"), hoặc qua từ điển nhỏ
              (tiền tệ, ngôn ngữ, viết tắt thành phố: "đô" -> USD, "Sài Gòn" -> TP.HCM)
  prev_obs  : có trong observation THẬT của một step trước (chạy gold prefix qua env)
  enum      : nằm trong enum của schema, hoặc bằng default của schema
  boolean   : param boolean (có/không suy ra từ câu chữ)
Không thuộc loại nào -> UNGROUNDED -> trajectory fail G3.

Quy tắc đã chốt sau Step 0 (eval/v6/audit_validation.json):
  - Đây là bản "lenient" của audit v5 (precision 0.97-1.00 theo nhãn tay),
    TRỪ quy tắc "quantity/passengers = 1 ngầm định": schema không khai báo default,
    nên số lượng không xuất hiện trong query là chưa grounded -> query phải nêu số lượng.
"""

from __future__ import annotations

import json
import re
import unicodedata

from env.mock_tools import WorldState, run_step

MULT = {"triệu": 1e6, "tr": 1e6, "trieu": 1e6, "k": 1e3, "nghìn": 1e3, "ngàn": 1e3,
        "nghin": 1e3, "ngan": 1e3, "tỷ": 1e9, "tỉ": 1e9, "ty": 1e9, "m": 1e6}

# Viết số bằng chữ (một, hai, ...) để "mua hai chiếc" ground được quantity=2.
WORD_NUM = {"một": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "sáu": 6, "bảy": 7,
            "tám": 8, "chín": 9, "mười": 10}
_COUNTERS = ("chiếc|cái|món|vé|người|khách|phòng|đêm|ngày|hộp|chai|đôi|bộ|gói|sản phẩm|suất|"
             "tấm|quyển|cuốn|kg|lon|thùng|chỗ|ghế|lọ|tuýp|túi|đơn|lần|tháng|năm|cặp")

LEXICON = {
    "usd": ["đô", "dollar", "usd", "$"], "vnd": ["việt nam đồng", "tiền việt", "vnđ", "vnd", "đồng"],
    "eur": ["euro", "eur"], "jpy": ["yên", "yen", "jpy"], "krw": ["won", "krw"],
    "cny": ["nhân dân tệ", "tệ", "cny"], "gbp": ["bảng anh", "gbp"], "thb": ["baht", "bạt"],
    "fr": ["pháp"], "en": ["anh", "english"], "vi": ["việt"], "ja": ["nhật"], "ko": ["hàn"],
    "zh": ["trung"], "de": ["đức"], "es": ["tây ban nha"], "th": ["thái"],
    "hà nội": ["hn", "hà nội", "ha noi"], "hn": ["hà nội"],
    "tp.hcm": ["hcm", "sài gòn", "hồ chí minh", "sg"], "hcm": ["tp.hcm", "sài gòn", "hồ chí minh"],
    "hồ chí minh": ["hcm", "sài gòn"], "đà nẵng": ["đn", "da nang"],
}


def nfc(s) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s)).strip().lower())


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def text_forms(s) -> set[str]:
    a = nfc(s)
    b = strip_accents(a)
    return {a, b, a.replace(" ", ""), b.replace(" ", "")}


# unit (triệu, k, ...) then an optional currency suffix glued or spaced ("500.000đ", "2 triệu đồng")
_NUM_RE = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)*)\s*(triệu|trieu|tr|tỷ|tỉ|ty|nghìn|nghin|ngàn|ngan|k|m)?"
                     r"(?:\s*(?:vnđ|vnd|đồng|dong|đ|d))?(?!\w)")


def query_numbers(q: str) -> set[float]:
    out: set[float] = set()
    qn = nfc(q)
    for m in _NUM_RE.finditer(qn):
        raw, unit = m.group(1), m.group(2)
        cands = set()
        if re.fullmatch(r"\d{1,3}([.,]\d{3})+", raw):
            # Vietnamese thousands grouping: "500.000" = 500000 (not the decimal 500.0)
            cands.add(float(re.sub(r"[.,]", "", raw)))
        else:
            try:
                cands.add(float(raw.replace(",", ".")))
            except ValueError:
                pass
        for c in cands:
            out.add(c)
            if unit:
                out.add(c * MULT[unit])
    # chỉ nhận số bằng chữ khi có lượng từ theo sau ("hai chiếc"), tránh "năm 2024", "ba mẹ"
    for w, n in WORD_NUM.items():
        if re.search(rf"(?<!\w){w}\s+(?:{_COUNTERS})(?!\w)", qn):
            out.add(float(n))
    return out


def date_forms(v) -> set[str]:
    s = nfc(v)
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
    else:
        m = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", s)
        if not m:
            return set()
        d, mo, y = int(m[1]), int(m[2]), int(m[3])
    return {f"{d:02d}/{mo:02d}/{y}", f"{d}/{mo}/{y}", f"{y}-{mo:02d}-{d:02d}",
            f"{d:02d}-{mo:02d}-{y}", f"{d}-{mo}-{y}", f"{d:02d}/{mo:02d}", f"{d}/{mo}",
            f"ngày {d} tháng {mo}", f"{d} tháng {mo}"}


def flatten(v):
    if isinstance(v, dict):
        for x in v.values():
            yield from flatten(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from flatten(x)
    else:
        yield v


def as_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and re.fullmatch(r"\s*\d+([.,]\d+)?\s*", v):
        return float(v.strip().replace(",", "."))
    return None


# ---------------------------------------------------------------------------
# G3 v1.2: an enum value is grounded only if the query states it (literally or
# in Vietnamese words), comes from an earlier observation, or equals a schema
# default. Phrase table: (tool.param, enum value) -> regex list, matched on the
# NFC-lowercased query and on its accent-free form. Printed in REPORT_v6.
# ---------------------------------------------------------------------------
ENUM_PHRASES: dict[str, dict[str, list[str]]] = {
    "block_card.reason": {
        "lost": [r"\bmất\b", r"đánh rơi", r"thất lạc", r"\blost\b"],
        "stolen": [r"bị trộm", r"bị cắp", r"bị móc", r"\btrộm\b", r"\bstolen\b"],
        "damaged": [r"\bhỏng\b", r"\bhư\b", r"\bgãy\b", r"không (dùng|sử dụng|quẹt) được", r"\bdamaged\b"],
        "fraud": [r"gian lận", r"lừa đảo", r"giao dịch lạ", r"bị hack", r"đáng ngờ", r"\bfraud\b"]},
    "loan_inquiry.loan_type": {
        "home": [r"mua nhà", r"\bnhà\b", r"bất động sản", r"nhà ở"],
        "car": [r"ô tô", r"xe hơi", r"mua xe", r"\boto\b"],
        "personal": [r"tiêu dùng", r"cá nhân", r"tín chấp"],
        "business": [r"kinh doanh", r"doanh nghiệp"]},
    "get_interest_rates.product_type": {
        "savings": [r"tiết kiệm", r"gửi tiền", r"tiền gửi"],
        "loan": [r"\bvay\b", r"cho vay", r"khoản vay"],
        "credit_card": [r"thẻ tín dụng", r"credit card"]},
    "find_branch.type": {
        "branch": [r"chi nhánh", r"phòng giao dịch"],
        "atm": [r"\batm\b", r"cây rút tiền", r"máy rút tiền"]},
    "civil_status_lookup.document_type": {
        "birth": [r"khai sinh"], "marriage": [r"kết hôn", r"hôn nhân"],
        "death": [r"khai tử"], "residence": [r"cư trú", r"tạm trú", r"thường trú", r"hộ khẩu"]},
    "utility_bill_inquiry.utility_type": {
        "electricity": [r"\bđiện\b(?!\s*(thoại|tử|máy|ảnh))", r"\bevn\b"],
        "water": [r"(tiền|hóa đơn|hoá đơn|cấp|cước|điện)\s*nước", r"điện\s*(và|,)\s*nước", r"nước\s*(sạch|sinh hoạt|máy)"]},
    "search_product.platform": {
        "shopee": [r"shopee"], "lazada": [r"lazada"], "tiki": [r"\btiki\b"], "sendo": [r"sendo"]},
    "checkout.payment_method": {
        "cod": [r"\bcod\b", r"khi nhận hàng", r"lúc nhận hàng", r"tiền mặt", r"nhận hàng (rồi|mới) (trả|thanh toán)"],
        "momo": [r"momo"], "vnpay": [r"vn\s?pay"], "bank_transfer": [r"chuyển khoản"]},
    "return_request.refund_method": {
        "original": [r"phương thức (thanh toán )?ban đầu", r"cách thanh toán ban đầu", r"hoàn (tiền )?về (nguồn|phương thức)",
                     r"như lúc thanh toán"],
        "wallet": [r"\bví\b"], "bank": [r"ngân hàng", r"tài khoản"]},
    "book_flight.seat_class": {
        "economy": [r"phổ thông", r"\beconomy\b", r"hạng thường"],
        "business": [r"thương gia", r"\bbusiness\b"], "first": [r"hạng nhất", r"first class"]},
    "news_search.category": {
        "politics": [r"chính trị", r"thời sự"], "economy": [r"kinh tế", r"tài chính", r"cổ phiếu", r"chứng khoán"],
        "sports": [r"thể thao", r"bóng đá"], "tech": [r"công nghệ"], "entertainment": [r"giải trí", r"showbiz"]},
    "open_savings_account.term_months": {  # numbers ("12 tháng") are read by query_numbers
        12: [r"(1|một) năm"], 24: [r"(2|hai) năm"], 36: [r"(3|ba) năm"]},
}


def enum_stated(tool: str, param: str, v, query: str) -> bool:
    qn = nfc(query)
    qa = strip_accents(qn)
    pats = ENUM_PHRASES.get(f"{tool}.{param}", {}).get(v, [])
    if isinstance(v, str):
        pats = pats + [re.escape(nfc(v)), re.escape(nfc(v).replace("_", " "))]
    for pat in pats:
        if re.search(pat, qn) or re.search(strip_accents(pat), qa):
            return True
    n = as_number(v)
    return n is not None and n in query_numbers(query)


def ground_value(v, pschema: dict, query: str, prev_vals: list, prev_text: str,
                 tool: str = "", param: str = "", enum_rule: str = "v1.2") -> str | None:
    """Trả nguồn grounding ('query'|'lexicon'|'prev_obs'|'enum'|'enum_stated'|'boolean') hoặc None.
    enum_rule: "v1.2" (mặc định, pipeline hiện tại) = enum phải được query nêu (ENUM_PHRASES);
    "v1.1" = mọi giá trị enum đều grounded (quy tắc dùng để tạo v1.1)."""
    if isinstance(v, bool):
        return "boolean"
    enum = pschema.get("enum") or (pschema.get("items") or {}).get("enum") or []
    if "default" in pschema and v == pschema["default"]:
        return "enum"
    if v in enum:
        if enum_rule == "v1.1" or enum_stated(tool, param, v, query):
            return "enum" if enum_rule == "v1.1" else "enum_stated"
        if v in prev_vals:
            return "prev_obs"
        return None
    qforms = text_forms(query)
    n = as_number(v)
    if n is not None and n in query_numbers(query):
        return "query"
    if isinstance(v, str):
        if any(f and f in q for f in text_forms(v) for q in qforms):
            return "query"
        if any(df in q for df in date_forms(v) for q in qforms):
            return "query"
        qn = nfc(query)
        for key in (nfc(v), strip_accents(nfc(v))):
            if key in LEXICON and any(w in qn for w in LEXICON[key]):
                return "lexicon"
    if v in prev_vals or (n is not None and n in {as_number(x) for x in prev_vals}):
        return "prev_obs"
    if isinstance(v, str) and len(v) >= 3 and nfc(v) in prev_text:
        return "prev_obs"
    return None


def grounding_report(query: str, steps: list[tuple[str, dict]], catalog: dict,
                     enum_rule: str = "v1.2") -> list[dict]:
    """Danh sách giá trị UNGROUNDED: [{step, tool, param, value}] (rỗng = pass)."""
    world = WorldState()
    obs_hist: list = []
    bad = []
    for i, (tool, params) in enumerate(steps):
        prev_vals = [x for o in obs_hist for x in flatten(o)]
        prev_text = nfc(json.dumps(obs_hist, ensure_ascii=False))
        props = catalog.get(tool, {}).get("parameters", {}).get("properties", {})
        for k, val in params.items():
            for v in (flatten(val) if isinstance(val, (list, dict)) else [val]):
                if ground_value(v, props.get(k, {}), query, prev_vals, prev_text,
                                tool=tool, param=k, enum_rule=enum_rule) is None:
                    bad.append({"step": i, "tool": tool, "param": k, "value": v})
        try:
            obs_hist.append(run_step(tool, params, world))
        except Exception as e:  # cổng 2 đã đảm bảo chạy được; phòng hờ
            obs_hist.append({"error": str(e)})
    return bad


def gate_grounding(query: str, steps: list[tuple[str, dict]], catalog: dict,
                   enum_rule: str = "v1.2") -> tuple[bool, str]:
    """Cổng G3. Trả (ok, reason)."""
    bad = grounding_report(query, steps, catalog, enum_rule=enum_rule)
    if not bad:
        return True, "ok"
    b = bad[0]
    more = f" (+{len(bad) - 1} giá trị khác)" if len(bad) > 1 else ""
    return False, f"giá trị '{b['value']}' ở {b['tool']}.{b['param']} không có trong query/observation{more}"
