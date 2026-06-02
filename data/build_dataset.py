"""
data/build_dataset.py — ViTool dataset builder (v10.0).

PIPELINE:
  LLM draft  →  resolve_ids  →  Cổng 1 (schema)  →  Cổng 2 (executable)  →  stratify  →  save
                                                                                          (+ expected_final_state)
  Cổng 3 (human verify) là THỦ CÔNG sau bước này — không tự động hoá được.

resolve_ids: chạy search/add step → lấy id THẬT (deterministic) từ observation →
tự điền vào dependent step sau (book_flight.flight_id, book_hotel.hotel_id,
add_to_cart.product_id, checkout.cart_id, ...). Đây là thứ sửa GỐC lỗi W5:
gold multi-step không cần human đoán hotel_id — env sinh ra, pipeline điền vào.

CRITICAL RULES (xem README §10):
  R1: generator (Llama-70B/Groq) ≠ defer/baseline model (GPT-4o-mini). Module này
      CHỈ draft data, không phải nơi chạy M3.
  R2: gold hợp lệ chỉ khi qua cổng 2 (chạy thật tới expected_final_state).
  R5: chất lượng từ 3 cổng + human, không từ "gen kỹ hơn".

CHẠY:
  export GROQ_API_KEY=...        # để draft
  python data/build_dataset.py --domain travel --difficulty hard --n 20 \
         --out data/vitool/drafts_hard_travel.jsonl
  # hoặc gộp + stratify nhiều domain (xem --help)
Không có key vẫn chạy được self-test logic: `python data/build_dataset.py --selftest`
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.vitool.schema import (  # noqa: E402
    Trajectory, Step, Difficulty, Language, save_trajectories, load_trajectories,
)
from verifiers.universal.schema_verifier import verify_schema  # noqa: E402
from env.mock_tools import run_step, is_executable, compute_final_state, WorldState, ToolError  # noqa: E402


# ============================================================
# Dependency injection map (resolve_ids)
# tool -> {param_name: id_type}. id_type khớp key trong observation.
# ============================================================
DEP_FIELDS: dict[str, dict[str, str]] = {
    "book_flight": {"flight_id": "flight_id"},
    "book_hotel": {"hotel_id": "hotel_id"},
    "book_train": {"train_id": "train_id"},
    "add_to_cart": {"product_id": "product_id"},
    "checkout": {"cart_id": "cart_id"},
    "apply_voucher": {"cart_id": "cart_id"},
    "cancel_booking": {"booking_id": "booking_id"},
    "return_request": {"order_id": "order_id"},
    "track_order": {"order_id": "order_id"},
    "baggage_info": {"flight_id": "flight_id"},
}
_ID_KEYS = {"flight_id", "hotel_id", "train_id", "product_id", "cart_id", "order_id", "booking_id"}


def _harvest_ids(obs, produced: dict) -> None:
    """Đi đệ quy obs, ghi nhận id mới nhất cho mỗi id_type."""
    if isinstance(obs, dict):
        for k, v in obs.items():
            if k in _ID_KEYS and isinstance(v, (str, int)):
                produced[k] = v
            else:
                _harvest_ids(v, produced)
    elif isinstance(obs, list):
        for item in obs:
            _harvest_ids(item, produced)


def resolve_ids(steps: list[tuple[str, dict]]) -> tuple[list[tuple[str, dict]], str | None]:
    """
    Chạy trajectory tăng dần; điền id thật từ observation vào dependent step.
    Trả (patched_steps, err). err=None nếu resolve OK.
    """
    produced: dict = {}
    world = WorldState()
    patched: list[tuple[str, dict]] = []
    for tool, params in steps:
        params = dict(params)
        for pname, idtype in DEP_FIELDS.get(tool, {}).items():
            if idtype in produced:
                params[pname] = produced[idtype]  # ghi đè placeholder bằng id thật
        try:
            obs = run_step(tool, params, world)
        except (ToolError, KeyError) as e:
            return patched, f"{tool}: {e}"
        _harvest_ids(obs, produced)
        patched.append((tool, params))
    return patched, None


# ============================================================
# Cổng 1 (schema) + Cổng 2 (executable)
# ============================================================
def gate_schema(steps: list[tuple[str, dict]], catalog: dict) -> tuple[bool, str]:
    for tool, params in steps:
        if tool not in catalog:
            return False, f"tool '{tool}' không trong catalog"
        sch = catalog.get(tool, {}).get("parameters")
        if sch is not None and not verify_schema(params, sch):
            return False, f"step '{tool}' không pass schema"
    return True, "ok"


def accept_trajectory(steps: list[tuple[str, dict]], catalog: dict):
    """
    Áp dụng resolve_ids → cổng 1 → cổng 2.
    Trả (ok, patched_steps, final_state, reason).
    """
    patched, err = resolve_ids(steps)
    if err is not None:
        return False, patched, None, f"[resolve] {err}"
    ok1, r1 = gate_schema(patched, catalog)
    if not ok1:
        return False, patched, None, f"[cổng1] {r1}"
    ok2, r2 = is_executable([{"tool_name": t, "params": p} for t, p in patched])
    if not ok2:
        return False, patched, None, f"[cổng2] {r2}"
    final = compute_final_state([{"tool_name": t, "params": p} for t, p in patched])
    return True, patched, final, "ok"


# ============================================================
# LLM drafting (generator = Groq Llama-70B — R1: KHÔNG dùng cho M3/baseline)
# ============================================================
_HARD_SYS = """Bạn là chuyên gia tạo dữ liệu tool-using agent tiếng Việt.
Tạo task {difficulty}. Quy tắc:
- Query tự nhiên như người Việt gõ, KHÔNG nhắc tên tool.
- Mỗi step: {{expected_tool, expected_params, difficulty:"{difficulty}"}}.
- Với task HARD: 3+ tool call có data-dependency (step sau dùng kết quả step trước).
  Với param phụ thuộc (flight_id, hotel_id, cart_id, order_id, booking_id) cứ để
  giá trị bất kỳ — pipeline sẽ tự điền id thật. KHÔNG để chuỗi placeholder kiểu
  "id_sản_phẩm".
- Dùng format Việt: ngày dd/mm/yyyy, tiền VNĐ, SĐT 0xxx.
- Tools hỗ trợ sequential: search_flight→book_flight, search_hotel→book_hotel,
  search_product→add_to_cart→checkout(→apply_voucher), book_*→cancel_booking.
Trả về JSON array, KHÔNG markdown, KHÔNG giải thích."""


def _parse_json_array(text: str) -> list:
    """Bóc JSON array từ output LLM (chịu được ```json fences / rác đầu cuối)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []


def draft_trajectories(client, model: str, domain: str, difficulty: str, n: int, catalog: dict) -> list:
    """Gọi LLM draft n task. Trả list raw {query, steps}. (Cần openai-compatible client.)"""
    tools_for_domain = {k: v for k, v in catalog.items() if v.get("domain") == domain}
    tools_desc = "\n".join(f"- {k}: {v.get('description_vi', v.get('description', ''))}"
                           for k in tools_for_domain)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _HARD_SYS.format(difficulty=difficulty)},
            {"role": "user", "content": (
                f"Tạo {n} task '{difficulty}' cho domain '{domain}' với tools:\n{tools_desc}\n\n"
                f"Trả JSON array, mỗi item: {{\"query\":..., \"steps\":[...]}}."
            )},
        ],
        temperature=0.8,
        max_tokens=3500,
    )
    return _parse_json_array(resp.choices[0].message.content)


def _raw_to_steps(raw_traj: dict) -> list[tuple[str, dict]]:
    steps = []
    for s in raw_traj.get("steps", []):
        tool = s.get("expected_tool") or s.get("tool_name")
        params = s.get("expected_params") or s.get("params") or {}
        if tool:
            steps.append((tool, dict(params)))
    return steps


def _to_trajectory(raw, steps, final_state, traj_id, domain, difficulty) -> Trajectory:
    return Trajectory(
        traj_id=traj_id,
        query=raw.get("query", ""),
        language=Language.VI,
        steps=[Step(expected_tool=t, expected_params=p,
                    difficulty=Difficulty(difficulty)) for t, p in steps],
        source="llama_draft+executable_gate",
        domain=domain,
        expected_final_state=final_state,
    )


# ============================================================
# Stratify
# ============================================================
def stratify_report(trajs: list[Trajectory], min_per_bin: int = 50) -> dict:
    by_bin = Counter()
    for t in trajs:
        # độ khó của trajectory = độ khó cao nhất trong các step
        order = {"easy": 0, "medium": 1, "hard": 2}
        lvl = max((s.difficulty.value for s in t.steps), key=lambda d: order[d])
        by_bin[lvl] += 1
    report = {"counts": dict(by_bin),
              "meets_min": {b: by_bin.get(b, 0) >= min_per_bin for b in ("easy", "medium", "hard")},
              "min_per_bin": min_per_bin}
    return report


# ============================================================
# Orchestrate (cần API)
# ============================================================
def build(client, model: str, catalog: dict, plan: dict, out_path: str,
          min_per_bin: int = 50, max_attempts_factor: int = 3) -> None:
    """plan = {(domain, difficulty): n_target}. Draft, gate, dedup, save."""
    kept: list[Trajectory] = []
    seen_queries = set()
    counters = defaultdict(int)
    for (domain, difficulty), target in plan.items():
        attempts = 0
        while counters[(domain, difficulty)] < target and attempts < target * max_attempts_factor:
            attempts += 1
            batch = draft_trajectories(client, model, domain, difficulty,
                                       n=min(10, target), catalog=catalog)
            for raw in batch:
                q = (raw.get("query") or "").strip()
                if not q or q in seen_queries:
                    continue
                steps = _raw_to_steps(raw)
                if not steps:
                    continue
                ok, patched, final, reason = accept_trajectory(steps, catalog)
                if not ok:
                    print(f"  [reject] {domain}/{difficulty}: {reason}")
                    continue
                tid = f"{domain}_{difficulty}_{counters[(domain, difficulty)]:04d}"
                kept.append(_to_trajectory(raw, patched, final, tid, domain, difficulty))
                seen_queries.add(q)
                counters[(domain, difficulty)] += 1
                print(f"  ✅ {tid} ({len(patched)} steps)")
    save_trajectories(kept, out_path)
    rep = stratify_report(kept, min_per_bin)
    print(f"\n✅ Saved {len(kept)} trajectories → {out_path}")
    print(f"   Stratify: {rep}")


# ============================================================
# Self-test (KHÔNG cần API) — chứng minh resolve_ids + gate + stratify
# ============================================================
def _selftest():
    catalog = {k: v for k, v in json.load(open(Path(__file__).resolve().parent.parent /
               "data/vitool/tool_catalog.json")).items() if not k.startswith("_")}

    # Draft A: multi-step với flight_id PLACEHOLDER → resolve_ids phải sửa thành id thật
    draftA = [("search_flight", {"origin": "HCM", "destination": "Đà Nẵng", "date": "20/07/2026"}),
              ("book_flight", {"flight_id": "PLACEHOLDER_SAI", "passenger_names": ["Nguyen Van A"]})]
    ok, patched, final, reason = accept_trajectory(draftA, catalog)
    assert ok, f"Draft A phải pass sau resolve, lỗi: {reason}"
    assert patched[1][1]["flight_id"].startswith("FL"), "flight_id phải được điền id thật"
    print(f"✅ A: resolve_ids sửa flight_id PLACEHOLDER → {patched[1][1]['flight_id']}, pass cả 2 cổng")

    # Draft B: ecommerce 3-step, product_id + cart_id để placeholder
    draftB = [("search_product", {"keyword": "laptop gaming"}),
              ("add_to_cart", {"product_id": "id_sản_phẩm", "quantity": 1}),
              ("checkout", {"cart_id": "cart_bừa", "payment_method": "cod"})]
    ok, patched, final, reason = accept_trajectory(draftB, catalog)
    assert ok, f"Draft B phải pass sau resolve, lỗi: {reason}"
    assert patched[1][1]["product_id"].startswith("PROD")
    assert patched[2][1]["cart_id"].startswith("CART")
    print(f"✅ B: resolve product_id→{patched[1][1]['product_id']}, cart_id→{patched[2][1]['cart_id']}, pass")
    assert final["orders"], "final_state phải có order"
    print(f"   final_state.orders = {final['orders']}")

    # Draft C: tool không trong catalog → reject (cổng/resolve)
    draftC = [("book_train", {"foo": 1})]
    ok, _, _, reason = accept_trajectory(draftC, catalog)
    assert not ok, "book_train phải bị loại"
    print(f"✅ C: loại 'book_train' → {reason}")

    # Draft D: thiếu required param → cổng 1 schema loại
    draftD = [("check_balance", {})]  # thiếu account_number
    ok, _, _, reason = accept_trajectory(draftD, catalog)
    assert not ok, "check_balance thiếu account_number phải bị loại"
    print(f"✅ D: loại check_balance thiếu param → {reason}")

    # Stratify
    trajs = [
        _to_trajectory({"query": "q1"}, draftA, None, "t1", "travel", "hard"),
        _to_trajectory({"query": "q2"}, [("get_weather", {"city": "Hà Nội"})], None, "t2", "info_retrieval", "easy"),
    ]
    rep = stratify_report(trajs, min_per_bin=1)
    assert rep["counts"].get("hard", 0) == 1 and rep["counts"].get("easy", 0) == 1
    print(f"✅ Stratify report: {rep}")

    print("\n✅ build_dataset self-test PASS (không cần API).")


# ============================================================
# CLI
# ============================================================
def _make_client():
    """openai-compatible client. Ưu tiên GROQ_API_KEY (Groq) cho DRAFT."""
    from openai import OpenAI
    if os.getenv("GROQ_API_KEY"):
        return OpenAI(api_key=os.environ["GROQ_API_KEY"],
                      base_url="https://api.groq.com/openai/v1"), "llama-3.3-70b-versatile"
    if os.getenv("OPENAI_API_KEY"):
        # Lưu ý R1: dùng OpenAI để DRAFT thì M3/baseline phải đổi sang model KHÁC.
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"]), "gpt-4o-mini"
    raise SystemExit("Cần GROQ_API_KEY (khuyến nghị cho draft) hoặc OPENAI_API_KEY.")


def main():
    ap = argparse.ArgumentParser(description="ViTool dataset builder (v10.0)")
    ap.add_argument("--selftest", action="store_true", help="chạy self-test logic, không cần API")
    ap.add_argument("--domain", help="banking|public_services|ecommerce|travel|info_retrieval")
    ap.add_argument("--difficulty", default="hard", choices=["easy", "medium", "hard"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="data/vitool/drafts_new.jsonl")
    ap.add_argument("--min-per-bin", type=int, default=50)
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if not args.domain:
        raise SystemExit("Cần --domain (hoặc --selftest).")

    catalog = {k: v for k, v in json.load(open(
        Path(__file__).resolve().parent.parent / "data/vitool/tool_catalog.json")).items()
        if not k.startswith("_")}
    client, model = _make_client()
    print(f"Draft bằng model: {model} (generator). Nhớ R1: M3/baseline phải dùng model KHÁC.")
    build(client, model, catalog, {(args.domain, args.difficulty): args.n},
          args.out, min_per_bin=args.min_per_bin)


if __name__ == "__main__":
    main()