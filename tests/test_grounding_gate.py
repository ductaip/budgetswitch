"""
Unit tests for Gate G3 (data/grounding_gate.py).

Run: venv/bin/python -m pytest tests/test_grounding_gate.py -q
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.grounding_gate import gate_grounding, grounding_report, query_numbers  # noqa: E402

CATALOG = {k: v for k, v in json.loads((REPO_ROOT / "data/vitool/tool_catalog.json").read_text()).items()
           if not k.startswith("_")}


@pytest.mark.parametrize("query,value", [
    ("giá không quá 500.000đ", 500000),        # glued "đ" (bug fixed in v6)
    ("tối đa 10.000.000đ", 10000000),
    ("tối đa 2000000đ", 2000000),
    ("ngân sách 500.000 đồng", 500000),
    ("khoảng 500.000 VNĐ", 500000),
    ("2 triệu đồng", 2000000),
    ("1,5 triệu", 1500000),
    ("500k", 500000),
    ("2 tỷ", 2000000000),
    ("mua hai chiếc", 2),                     # number word + counter
])
def test_money_and_counts_are_read(query, value):
    assert float(value) in query_numbers(query)


@pytest.mark.parametrize("query,value", [
    ("Q1 2024", 1),                           # digit inside a token
    ("năm 2024", 5),                          # "năm" = year, not five
    ("ba mẹ tôi muốn đi Huế", 3),             # "ba" = father
    ("giá 500.000đ", 500),                    # no partial match of "500.000"
])
def test_no_false_numbers(query, value):
    assert float(value) not in query_numbers(query)


def test_max_price_with_glued_dong_passes_g3():
    steps = [("search_product", {"keyword": "quần jeans nữ", "max_price": 500000})]
    ok, reason = gate_grounding("Tìm quần jeans nữ giá không quá 500.000đ", steps, CATALOG)
    assert ok, reason


def test_quantity_must_be_stated():
    """Decision after Step 0: quantity absent from the query is NOT grounded (no schema default)."""
    steps = [("search_product", {"keyword": "điện thoại"}),
             ("add_to_cart", {"product_id": "PRODX", "quantity": 1})]
    bad = grounding_report("Mua điện thoại", steps, CATALOG)
    assert any(b["param"] == "quantity" for b in bad)


def test_id_from_previous_observation_is_grounded():
    steps = [("search_flight", {"origin": "Hà Nội", "destination": "Huế", "date": "15/05/2024"})]
    from env.mock_tools import WorldState, run_step
    obs = run_step(*steps[0], WorldState())
    fid = obs["flights"][0]["flight_id"]
    steps.append(("book_flight", {"flight_id": fid, "passenger_names": ["Nguyễn Văn A"]}))
    q = "Đặt vé từ Hà Nội đến Huế ngày 15/05/2024 cho Nguyễn Văn A"
    assert grounding_report(q, steps, CATALOG) == []
    bad = grounding_report("Đặt vé từ Hà Nội đến Huế ngày 15/05/2024", steps, CATALOG)
    assert [(b["param"], b["value"]) for b in bad] == [("passenger_names", "Nguyễn Văn A")]


def test_lexicon_currency():
    steps = [("currency_convert", {"amount": 100, "from_currency": "USD", "to_currency": "VND"})] \
        if "currency_convert" in CATALOG else None
    if steps is None:
        pytest.skip("no currency tool in catalog")
    props = CATALOG["currency_convert"]["parameters"]["properties"]
    steps = [("currency_convert", {k: v for k, v in steps[0][1].items() if k in props})]
    assert grounding_report("Đổi 100 đô sang tiền việt", steps, CATALOG) == []


# ---- G3 v1.2: enum values must be stated -------------------------------------
@pytest.mark.parametrize("query,value,ok", [
    ("Mua áo trên Shopee rồi thanh toán khi nhận hàng", "cod", True),
    ("Mua áo trên Shopee rồi thanh toán bằng chuyển khoản", "bank_transfer", True),
    ("Mua áo trên Shopee rồi thanh toán qua VNPay", "vnpay", True),
    ("Mua áo trên Shopee rồi thanh toán", "cod", False),          # choice never made
])
def test_payment_method_must_be_stated(query, value, ok):
    steps = [("checkout", {"cart_id": "CARTX", "payment_method": value})]
    bad = grounding_report(query, steps, CATALOG)
    assert (not any(b["param"] == "payment_method" for b in bad)) == ok


def test_enum_rule_v11_still_available():
    steps = [("checkout", {"cart_id": "CARTX", "payment_method": "cod"})]
    assert not any(b["param"] == "payment_method"
                   for b in grounding_report("thanh toán", steps, CATALOG, enum_rule="v1.1"))


@pytest.mark.parametrize("query,value,ok", [
    ("Tra cứu hóa đơn điện tháng này", "electricity", True),
    ("Tra cứu hóa đơn điện và nước", "water", True),
    ("Tra cứu tiền điện thoại", "electricity", False),            # "điện thoại" is not electricity
])
def test_utility_type_phrases(query, value, ok):
    steps = [("utility_bill_inquiry", {"customer_code": "KH001", "utility_type": value})]
    props = CATALOG["utility_bill_inquiry"]["parameters"]["properties"]
    steps = [("utility_bill_inquiry", {k: v for k, v in steps[0][1].items() if k in props})]
    bad = grounding_report(query, steps, CATALOG)
    assert (not any(b["param"] == "utility_type" for b in bad)) == ok
