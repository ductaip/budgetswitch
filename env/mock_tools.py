"""
ViTool — Executable Mock Environment (NEW, v10.0).

36 deterministic mock tools across 5 domains (banking, public_services,
ecommerce, travel, info_retrieval), matching data/vitool/tool_catalog.json.

WHY THIS EXISTS
---------------
1. Cổng 2 (executable gate) khi build data: chạy gold trajectory; loại nếu
   bất kỳ step nào fail (tool không có / thiếu param / referenced-id không tồn
   tại). Đây là thứ chặn được đống invalid của W5 (book_train không trong
   catalog; book_hotel thiếu hotel_id).
2. Cho phép multi-step có data-dependency THẬT: search_* tạo id deterministic,
   book_*/checkout tiêu thụ id đó. Gold lưu giá trị đã resolve; env verify
   tính nhất quán.
3. Pilot/eval: chạy tool-call của agent → so final-state với gold → TSR
   cho task multi-step/action (verify_final_outcome thật, không còn stub).

NGUYÊN TẮC: DETERMINISTIC tuyệt đối — cùng input luôn cùng output. Không random,
không wall-clock. Đây là yêu cầu reproducibility của benchmark.

CRITICAL RULE: module này KHÔNG reference oracle_mode/expected_mode (R4).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable


# ============================================================
# Errors
# ============================================================
class ToolError(Exception):
    """Raised khi một step không thực thi được (executable failure)."""


# ============================================================
# Deterministic helpers
# ============================================================
def _det(prefix: str, *parts: Any) -> str:
    """ID deterministic, ổn định theo input. KHÔNG random/time."""
    raw = "|".join(json.dumps(p, ensure_ascii=False, sort_keys=True) if not isinstance(p, str) else p
                   for p in parts)
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:6].upper()
    return f"{prefix}{h}"


def _det_int(lo: int, hi: int, *parts: Any) -> int:
    """Số nguyên deterministic trong [lo, hi]."""
    raw = "|".join(str(p) for p in parts)
    h = int(hashlib.sha1(raw.encode("utf-8")).hexdigest(), 16)
    return lo + (h % (hi - lo + 1))


# ============================================================
# World state (chỉ snapshot các store CÓ mutation để so final-state)
# ============================================================
@dataclass
class WorldState:
    # registries từ search/lookup (read) — để dependent step tiêu thụ
    flights: dict[str, dict] = field(default_factory=dict)
    hotels: dict[str, dict] = field(default_factory=dict)
    trains: dict[str, dict] = field(default_factory=dict)
    products: dict[str, dict] = field(default_factory=dict)
    # mutation stores — định nghĩa "task accomplished"
    carts: dict[str, dict] = field(default_factory=dict)
    orders: dict[str, dict] = field(default_factory=dict)
    bookings: dict[str, dict] = field(default_factory=dict)
    transactions: list = field(default_factory=list)
    blocked_cards: list = field(default_factory=list)
    savings: list = field(default_factory=list)
    returns: list = field(default_factory=list)
    appointments: list = field(default_factory=list)

    def snapshot(self) -> dict:
        """State chuẩn hoá để so sánh gold vs agent. Chỉ gồm mutation effects."""
        return {
            "transactions": self.transactions,
            "blocked_cards": sorted(self.blocked_cards),
            "savings": self.savings,
            "returns": self.returns,
            "appointments": self.appointments,
            "carts": {k: self.carts[k] for k in sorted(self.carts)},
            "orders": sorted(self.orders.keys()),
            "bookings": sorted(self.bookings.keys()),
        }


# ============================================================
# Tool implementations (mỗi tool: (params, world) -> observation)
# Read tools: trả info deterministic. Action/dependent tools: mutate + check.
# ============================================================

# ---- banking ----
def t_check_balance(p, w):
    bal = _det_int(50_000, 500_000_000, "bal", p["account_number"])
    return {"account_number": p["account_number"], "balance_vnd": bal, "currency": "VND"}

def t_transfer_money(p, w):
    amt = p["amount"]
    if not isinstance(amt, (int, float)) or amt <= 0:
        raise ToolError("amount phải > 0")
    tx_id = _det("TX", p["from_account"], p["to_account"], amt, p.get("note", ""))
    rec = {"tx_id": tx_id, "from": p["from_account"], "to": p["to_account"],
           "amount": amt, "note": p.get("note", "")}
    w.transactions.append(rec)
    return {"tx_id": tx_id, "status": "success", **rec}

def t_transaction_history(p, w):
    return {"account_number": p["account_number"],
            "transactions": [{"tx_id": _det("TX", p["account_number"], i), "amount": _det_int(10_000, 5_000_000, p["account_number"], i)} for i in range(3)]}

def t_open_savings_account(p, w):
    sid = _det("SV", p["customer_id"], p["initial_deposit"], p.get("term_months", 0))
    rec = {"savings_id": sid, "customer_id": p["customer_id"],
           "initial_deposit": p["initial_deposit"], "term_months": p.get("term_months", 0)}
    w.savings.append(rec)
    return {"savings_id": sid, "status": "opened", **rec}

def t_block_card(p, w):
    w.blocked_cards.append(p["card_number"])
    return {"card_number": p["card_number"], "status": "blocked", "reason": p["reason"]}

def t_loan_inquiry(p, w):
    rate = round(6.0 + _det_int(0, 60, p["loan_type"], p["amount"]) / 10.0, 1)
    return {"loan_type": p["loan_type"], "amount": p["amount"],
            "interest_rate_pct": rate, "term_months": p.get("term_months", 12)}

def t_get_interest_rates(p, w):
    rate = round(3.0 + _det_int(0, 50, p["product_type"]) / 10.0, 1)
    return {"product_type": p["product_type"], "rate_pct": rate, "term_months": p.get("term_months")}

def t_find_branch(p, w):
    return {"city": p["city"], "branches": [{"name": f"CN {p['city']}", "branch_id": _det("BR", p["city"], p.get("district", ""))}]}

# ---- public_services ----
def t_lookup_bhxh(p, w):
    return {"bhxh_number": p["bhxh_number"], "status": "active",
            "months_paid": _det_int(1, 300, p["bhxh_number"])}

def t_tax_inquiry(p, w):
    return {"tax_code": p["tax_code"], "year": p.get("year", 2025),
            "amount_due_vnd": _det_int(0, 50_000_000, p["tax_code"], p.get("year", 2025))}

def t_civil_status_lookup(p, w):
    return {"citizen_id": p["citizen_id"], "document_type": p["document_type"],
            "status": "valid", "ref": _det("DOC", p["citizen_id"], p["document_type"])}

def t_vehicle_registration(p, w):
    return {"plate_number": p["plate_number"], "status": "registered",
            "owner_ref": _det("VR", p["plate_number"])}

def t_utility_bill_inquiry(p, w):
    return {"customer_code": p["customer_code"], "utility_type": p["utility_type"],
            "amount_due_vnd": _det_int(0, 3_000_000, p["customer_code"], p["utility_type"])}

def t_book_public_appointment(p, w):
    aid = _det("AP", p["office"], p["service"], p["date"])
    rec = {"appointment_id": aid, "office": p["office"], "service": p["service"], "date": p["date"]}
    w.appointments.append(rec)
    return {"appointment_id": aid, "status": "booked", **rec}

def t_document_status(p, w):
    return {"receipt_number": p["receipt_number"], "status": "processing"}

def t_fee_lookup(p, w):
    return {"service_name": p["service_name"], "fee_vnd": _det_int(0, 1_000_000, p["service_name"])}

# ---- ecommerce ----
def t_search_product(p, w):
    pid = _det("PROD", p["keyword"])
    price = _det_int(50_000, 30_000_000, p["keyword"])
    w.products[pid] = {"product_id": pid, "name": p["keyword"], "price_vnd": price}
    return {"keyword": p["keyword"], "results": [w.products[pid]]}

def t_add_to_cart(p, w):
    pid = p["product_id"]
    if pid not in w.products:
        # cho phép pid hợp lệ-pattern nếu đến từ ngoài? Không — yêu cầu đến từ search.
        raise ToolError(f"product_id '{pid}' chưa tồn tại (phải đến từ search_product)")
    cid = _det("CART", pid)
    cart = w.carts.setdefault(cid, {"cart_id": cid, "items": [], "vouchers": []})
    cart["items"].append({"product_id": pid, "quantity": p["quantity"]})
    return {"cart_id": cid, "items": cart["items"]}

def t_checkout(p, w):
    cid = p["cart_id"]
    if cid not in w.carts:
        raise ToolError(f"cart_id '{cid}' không tồn tại (phải đến từ add_to_cart)")
    oid = _det("ORD", cid, p["payment_method"])
    w.orders[oid] = {"order_id": oid, "cart_id": cid, "payment_method": p["payment_method"],
                     "shipping_address": p.get("shipping_address", "")}
    return {"order_id": oid, "status": "placed"}

def t_track_order(p, w):
    return {"order_id": p["order_id"], "status": "in_transit",
            "eta_days": _det_int(1, 7, p["order_id"])}

def t_return_request(p, w):
    rid = _det("RET", p["order_id"], p["reason"])
    rec = {"return_id": rid, "order_id": p["order_id"], "reason": p["reason"],
           "refund_method": p.get("refund_method", "original")}
    w.returns.append(rec)
    return {"return_id": rid, "status": "requested", **rec}

def t_compare_prices(p, w):
    return {"product_name": p["product_name"],
            "offers": [{"platform": plat, "price_vnd": _det_int(50_000, 30_000_000, p["product_name"], plat)}
                       for plat in ("Shopee", "Lazada", "Tiki")]}

def t_apply_voucher(p, w):
    cid = p["cart_id"]
    if cid not in w.carts:
        raise ToolError(f"cart_id '{cid}' không tồn tại (phải đến từ add_to_cart)")
    w.carts[cid]["vouchers"].append(p["voucher_code"])
    return {"cart_id": cid, "voucher_code": p["voucher_code"], "status": "applied"}

# ---- travel ----
def t_search_flight(p, w):
    fid = _det("FL", p["origin"], p["destination"], p["date"])
    w.flights[fid] = {"flight_id": fid, "origin": p["origin"], "destination": p["destination"],
                      "date": p["date"], "price_vnd": _det_int(500_000, 8_000_000, fid)}
    return {"flights": [w.flights[fid]]}

def t_book_flight(p, w):
    fid = p["flight_id"]
    if fid not in w.flights:
        raise ToolError(f"flight_id '{fid}' không tồn tại (phải đến từ search_flight)")
    bid = _det("BK", fid, json.dumps(p["passenger_names"], ensure_ascii=False))
    w.bookings[bid] = {"booking_id": bid, "type": "flight", "flight_id": fid,
                       "passengers": p["passenger_names"], "seat_class": p.get("seat_class", "economy")}
    return {"booking_id": bid, "status": "confirmed"}

def t_search_hotel(p, w):
    hid = _det("HT", p["city"], p["checkin"], p["checkout"])
    w.hotels[hid] = {"hotel_id": hid, "city": p["city"], "checkin": p["checkin"],
                     "checkout": p["checkout"], "price_per_night_vnd": _det_int(300_000, 5_000_000, hid)}
    return {"hotels": [w.hotels[hid]]}

def t_book_hotel(p, w):
    hid = p["hotel_id"]
    if hid not in w.hotels:
        raise ToolError(f"hotel_id '{hid}' không tồn tại (phải đến từ search_hotel)")
    bid = _det("BK", hid, p["checkin"], p["checkout"])
    w.bookings[bid] = {"booking_id": bid, "type": "hotel", "hotel_id": hid,
                       "room_type": p.get("room_type", "standard"),
                       "checkin": p["checkin"], "checkout": p["checkout"]}
    return {"booking_id": bid, "status": "confirmed"}

def t_search_train(p, w):
    tid = _det("TR", p["origin"], p["destination"], p["date"])
    w.trains[tid] = {"train_id": tid, "origin": p["origin"], "destination": p["destination"], "date": p["date"]}
    return {"trains": [w.trains[tid]]}

def t_cancel_booking(p, w):
    bid = p["booking_id"]
    if bid not in w.bookings:
        raise ToolError(f"booking_id '{bid}' không tồn tại (phải đến từ book_flight/book_hotel)")
    w.bookings[bid]["status"] = "cancelled"
    return {"booking_id": bid, "status": "cancelled", "reason": p.get("reason", "")}

def t_baggage_info(p, w):
    return {"flight_id": p["flight_id"], "allowance_kg": 20 if p.get("seat_class") != "business" else 30}

# ---- info_retrieval ----
def t_get_weather(p, w):
    temp = _det_int(18, 36, "w", p["city"])
    return {"city": p["city"], "temp_c": temp, "condition": "sunny" if temp > 28 else "cloudy"}

def t_news_search(p, w):
    return {"keyword": p["keyword"], "articles": [{"title": f"Tin về {p['keyword']}", "id": _det("NW", p["keyword"], p.get("category", ""))}]}

def t_translate_text(p, w):
    return {"text": p["text"], "target_lang": p["target_lang"],
            "translation": f"[{p['target_lang']}] {p['text']}"}

def t_currency_convert(p, w):
    rate = {"USD": 25_000, "EUR": 27_000, "JPY": 170, "VND": 1}.get(p["to_currency"].upper(), 1000)
    base = {"USD": 25_000, "EUR": 27_000, "JPY": 170, "VND": 1}.get(p["from_currency"].upper(), 1000)
    out = round(p["amount"] * base / rate, 2)
    return {"amount": p["amount"], "from": p["from_currency"], "to": p["to_currency"], "result": out}

def t_calculate(p, w):
    expr = str(p["expression"])
    if not all(ch in "0123456789+-*/(). " for ch in expr):
        raise ToolError("expression chỉ cho phép số và + - * / ( )")
    try:
        val = eval(expr, {"__builtins__": {}}, {})  # safe: chars whitelisted ở trên
    except Exception as e:
        raise ToolError(f"expression không hợp lệ: {e}")
    return {"expression": expr, "result": val}

def t_wiki_lookup(p, w):
    return {"query": p["query"], "summary": f"Tóm tắt về {p['query']}.", "ref": _det("WK", p["query"])}


# ============================================================
# Registry
# ============================================================
TOOLS: dict[str, Callable[[dict, WorldState], dict]] = {
    "check_balance": t_check_balance, "transfer_money": t_transfer_money,
    "transaction_history": t_transaction_history, "open_savings_account": t_open_savings_account,
    "block_card": t_block_card, "loan_inquiry": t_loan_inquiry,
    "get_interest_rates": t_get_interest_rates, "find_branch": t_find_branch,
    "lookup_bhxh": t_lookup_bhxh, "tax_inquiry": t_tax_inquiry,
    "civil_status_lookup": t_civil_status_lookup, "vehicle_registration": t_vehicle_registration,
    "utility_bill_inquiry": t_utility_bill_inquiry, "book_public_appointment": t_book_public_appointment,
    "document_status": t_document_status, "fee_lookup": t_fee_lookup,
    "search_product": t_search_product, "add_to_cart": t_add_to_cart, "checkout": t_checkout,
    "track_order": t_track_order, "return_request": t_return_request,
    "compare_prices": t_compare_prices, "apply_voucher": t_apply_voucher,
    "search_flight": t_search_flight, "book_flight": t_book_flight,
    "search_hotel": t_search_hotel, "book_hotel": t_book_hotel, "search_train": t_search_train,
    "cancel_booking": t_cancel_booking, "baggage_info": t_baggage_info,
    "get_weather": t_get_weather, "news_search": t_news_search, "translate_text": t_translate_text,
    "currency_convert": t_currency_convert, "calculate": t_calculate, "wiki_lookup": t_wiki_lookup,
}


# ============================================================
# Execution API
# ============================================================
def run_step(tool_name: str, params: dict, world: WorldState) -> dict:
    """Chạy 1 tool call. Raise ToolError nếu tool không có / dependency thiếu."""
    if tool_name not in TOOLS:
        raise ToolError(f"tool '{tool_name}' không có trong catalog")
    if not isinstance(params, dict):
        raise ToolError(f"params phải là dict, nhận {type(params).__name__}")
    return TOOLS[tool_name](params, world)


def _as_steps(steps: list) -> list[tuple[str, dict]]:
    """Chấp nhận cả Step objects lẫn dict {tool_name/expected_tool, params/expected_params}."""
    out = []
    for s in steps:
        if hasattr(s, "expected_tool"):
            out.append((s.expected_tool, dict(s.expected_params)))
        elif isinstance(s, dict):
            name = s.get("tool_name") or s.get("expected_tool")
            params = s.get("params") or s.get("expected_params") or {}
            out.append((name, dict(params)))
        else:
            raise ToolError(f"step không nhận dạng được: {type(s).__name__}")
    return out


def run_trajectory(steps: list) -> tuple[list[dict], WorldState]:
    """Chạy toàn bộ trajectory trên 1 world (thứ tự = data dependency)."""
    world = WorldState()
    obs = []
    for tool_name, params in _as_steps(steps):
        obs.append(run_step(tool_name, params, world))
    return obs, world


def is_executable(steps: list) -> tuple[bool, str]:
    """
    Cổng 2 (executable gate). Trả (ok, reason).
    Dùng khi build data: gold hợp lệ CHỈ KHI chạy hết không lỗi.
    """
    try:
        run_trajectory(steps)
        return True, "ok"
    except ToolError as e:
        return False, str(e)
    except KeyError as e:
        return False, f"thiếu required param {e}"


def compute_final_state(steps: list) -> dict:
    """Chạy gold → trả expected_final_state (deterministic) để lưu vào dataset."""
    _, world = run_trajectory(steps)
    return world.snapshot()


def verify_final_outcome(agent_steps: list, expected_final_state: dict) -> bool:
    """
    Phiên bản EXECUTABLE thật (thay stub trong schema_verifier).
    Chạy trajectory của agent → so snapshot với expected_final_state.
    """
    try:
        _, world = run_trajectory(agent_steps)
    except (ToolError, KeyError):
        return False
    return world.snapshot() == expected_final_state


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    print(f"Registry: {len(TOOLS)} tools")
    assert len(TOOLS) == 36, f"Cần đúng 36 tool, có {len(TOOLS)}"

    # 1) Multi-step HỢP LỆ: search_flight -> book_flight (data dependency)
    obs, world = run_trajectory([
        {"tool_name": "search_flight", "params": {"origin": "HCM", "destination": "Đà Nẵng", "date": "20/07/2026"}},
    ])
    fid = obs[0]["flights"][0]["flight_id"]
    good_traj = [
        {"tool_name": "search_flight", "params": {"origin": "HCM", "destination": "Đà Nẵng", "date": "20/07/2026"}},
        {"tool_name": "book_flight", "params": {"flight_id": fid, "passenger_names": ["Nguyen Van A"]}},
    ]
    ok, reason = is_executable(good_traj)
    assert ok, f"Trajectory hợp lệ phải pass, lỗi: {reason}"
    print(f"✅ Multi-step hợp lệ PASS gate (flight_id={fid})")

    # 2) Case W5 invalid #1: tool không trong catalog (book_train)
    ok, reason = is_executable([{"tool_name": "book_train", "params": {"x": 1}}])
    assert not ok, "book_train phải bị loại"
    print(f"✅ Gate CHẶN 'book_train': {reason}")

    # 3) Case W5 invalid #2: book_hotel với hotel_id không từ search (thiếu dependency)
    ok, reason = is_executable([{"tool_name": "book_hotel", "params": {"hotel_id": "HT_FAKE", "checkin": "1/1", "checkout": "2/1"}}])
    assert not ok, "book_hotel với hotel_id fake phải bị loại"
    print(f"✅ Gate CHẶN book_hotel hotel_id fake: {reason}")

    # 4) Case W5 invalid #3: thiếu required param (book_flight không có flight_id)
    ok, reason = is_executable([{"tool_name": "book_flight", "params": {"passenger_names": ["A"]}}])
    assert not ok, "book_flight thiếu flight_id phải bị loại"
    print(f"✅ Gate CHẶN book_flight thiếu flight_id: {reason}")

    # 5) Determinism: chạy 2 lần cùng input → cùng final-state
    s1 = compute_final_state(good_traj)
    s2 = compute_final_state(good_traj)
    assert s1 == s2, "Phải deterministic"
    print("✅ Deterministic: 2 lần chạy cho final-state giống nhau")

    # 6) verify_final_outcome: agent đi đúng path → True; sai tool → False
    assert verify_final_outcome(good_traj, s1) is True
    bad = [{"tool_name": "get_weather", "params": {"city": "Hà Nội"}}]
    assert verify_final_outcome(bad, s1) is False
    print("✅ verify_final_outcome phân biệt đúng/sai")

    # 7) Multi-step ecommerce 3 bước: search_product -> add_to_cart -> checkout
    o, w = run_trajectory([{"tool_name": "search_product", "params": {"keyword": "tai nghe bluetooth"}}])
    pid = o[0]["results"][0]["product_id"]
    o2, w2 = run_trajectory([{"tool_name": "search_product", "params": {"keyword": "tai nghe bluetooth"}}])
    pid2 = o2[0]["results"][0]["product_id"]
    cid = _det("CART", pid)
    ecom = [
        {"tool_name": "search_product", "params": {"keyword": "tai nghe bluetooth"}},
        {"tool_name": "add_to_cart", "params": {"product_id": pid, "quantity": 1}},
        {"tool_name": "checkout", "params": {"cart_id": cid, "payment_method": "cod"}},
    ]
    ok, reason = is_executable(ecom)
    assert ok, f"Ecommerce 3-step phải pass: {reason}"
    print(f"✅ Ecommerce 3-step (search→cart→checkout) PASS gate")

    print("\n✅ Tất cả self-test mock env PASS.")