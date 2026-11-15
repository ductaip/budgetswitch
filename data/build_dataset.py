"""
data/build_dataset.py — ViTool dataset builder (v10.0).

PIPELINE:
  LLM draft  →  resolve_ids  →  Cổng 1 (schema)  →  Cổng 1.5 (placeholder)  →  Cổng 2 (executable)
            →  Cổng G3 (grounding, data/grounding_gate.py)  →  stratify  →  save
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
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.vitool.schema import (  # noqa: E402
    Trajectory, Step, Difficulty, Language, save_trajectories, load_trajectories,
)
from verifiers.universal.schema_verifier import verify_schema  # noqa: E402
from env.mock_tools import run_step, is_executable, compute_final_state, WorldState, ToolError  # noqa: E402
from data.grounding_gate import gate_grounding, grounding_report  # noqa: E402


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


def _has_real_dependency(steps: list[tuple[str, dict]]) -> bool:
    """True nếu có ít nhất 1 dependent step (book/checkout/cancel/...) sau 1 search/add."""
    dependent = set(DEP_FIELDS.keys())
    producers_seen = False
    for tool, _ in steps:
        if tool in ("search_flight", "search_hotel", "search_train", "search_product", "add_to_cart"):
            producers_seen = True
        if tool in dependent and producers_seen:
            return True
    return False


def _difficulty_structure_ok(steps: list[tuple[str, dict]], difficulty: str) -> tuple[bool, str]:
    """Ép cấu trúc khớp nhãn độ khó (chống Llama gán 'hard' cho task tầm thường)."""
    n = len(steps)
    if difficulty == "hard":
        if n < 3:
            return False, f"hard cần ≥3 step, chỉ có {n}"
        if not _has_real_dependency(steps):
            return False, "hard cần data-dependency thật (vd search→book), không phải các step rời"
    elif difficulty == "medium":
        if n < 2:
            return False, f"medium cần ≥2 step, chỉ có {n}"
    return True, "ok"


import re as _re

_PLACEHOLDER_CHECKS = [
    (r'\b(tên|họ tên|người|passenger)\s*\d+\b', True),    # 'Tên 1'
    (r'[A-ZÀ-Ỹ]{2,}_[A-ZÀ-Ỹ_]{2,}', False),              # ĐỊA_CHỈ_GIAO_HÀNG (KHÔNG ignorecase)
    (r'(placeholder|id_sản|không xác định|chưa xác định|cần điền|\bn/a\b)', True),
    (r'^x{4,}$', True),                                    # xxxxxxxxxxxx
    (r'^(khach_hang|customer|user|client)_?\d+$', True),  # id giả generic
    (r'^id_\w+$', True),                                  # 'id_đơn_hàng' (v1.1)
]


def _is_placeholder(v) -> str | None:
    """Phát hiện placeholder trong free-text param (cổng executable KHÔNG bắt được)."""
    if isinstance(v, list):
        for x in v:
            r = _is_placeholder(x)
            if r:
                return r
        return None
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return "<empty string>"  # gold value '' (v1.1)
    for pat, ic in _PLACEHOLDER_CHECKS:
        if _re.search(pat, s, _re.I if ic else 0):
            return s
    return None


def gate_placeholder(steps: list[tuple[str, dict]]) -> tuple[bool, str]:
    """Cổng 1.5: loại trajectory có placeholder free-text (Tên 1, ĐỊA_CHỈ..., không xác định)."""
    for tool, params in steps:
        for k, v in params.items():
            r = _is_placeholder(v)
            if r is not None:
                return False, f"placeholder '{r}' ở {tool}.{k}"
    return True, "ok"


def accept_trajectory(steps: list[tuple[str, dict]], catalog: dict, difficulty: str | None = None,
                      query: str | None = None, enum_rule: str = "v1.2"):
    """
    Áp dụng: cấu trúc độ khó → resolve_ids → cổng 1 (schema) → cổng 1.5 (placeholder) → cổng 2 (executable)
    → cổng G3 (grounding: mọi giá trị gold có trong query / observation trước / enum).
    G3 chỉ chạy khi có `query` (luôn truyền query khi build/revalidate; v1.1 trở đi).
    Trả (ok, patched_steps, final_state, reason).
    """
    if difficulty is not None:
        ok_s, r_s = _difficulty_structure_ok(steps, difficulty)
        if not ok_s:
            return False, steps, None, f"[độ khó] {r_s}"
    patched, err = resolve_ids(steps)
    if err is not None:
        return False, patched, None, f"[resolve] {err}"
    ok1, r1 = gate_schema(patched, catalog)
    if not ok1:
        return False, patched, None, f"[cổng1] {r1}"
    okp, rp = gate_placeholder(patched)
    if not okp:
        return False, patched, None, f"[placeholder] {rp}"
    ok2, r2 = is_executable([{"tool_name": t, "params": p} for t, p in patched])
    if not ok2:
        return False, patched, None, f"[cổng2] {r2}"
    if query is not None:
        ok3, r3 = gate_grounding(query, patched, catalog, enum_rule=enum_rule)
        if not ok3:
            return False, patched, None, f"[G3] {r3}"
    final = compute_final_state([{"tool_name": t, "params": p} for t, p in patched])
    return True, patched, final, "ok"


# ============================================================
# LLM drafting (generator = Groq Llama-70B — R1: KHÔNG dùng cho M3/baseline)
# ============================================================
_HARD_SYS = """Bạn là chuyên gia tạo dữ liệu tool-using agent tiếng Việt.
Tạo task độ khó "{difficulty}". QUY TẮC BẮT BUỘC:

[TOOL] CHỈ dùng tool có trong DANH SÁCH được cung cấp. TUYỆT ĐỐI KHÔNG bịa tool mới.
  Điền ĐỦ MỌI param BẮT BUỘC (xem "BẮT BUỘC params"). Thiếu/sai kiểu → loại.
  amount/tiền PHẢI > 0. Enum phải đúng giá trị cho phép.

[ĐỘ KHÓ] Hiểu đúng:
  - easy  = 1 tool call đơn giản.
  - medium= 2 tool call (có thể độc lập hoặc nối tiếp).
  - hard  = 3+ tool call CÓ DATA-DEPENDENCY THẬT: step sau DÙNG output step trước
    (search_flight→book_flight→...). KHÔNG phải 2 việc rời nhau. Task hard mà chỉ
    2 step độc lập sẽ bị LOẠI.

[ĐA DẠNG — RẤT QUAN TRỌNG] Mỗi task PHẢI khác hẳn nhau về kịch bản, KHÔNG lặp ý:
  - Đổi tool, đổi tình huống, đổi số liệu, đổi địa danh, đổi ý định.
  - VÍ DỤ tình huống đa dạng (đừng chỉ làm 1 kiểu):
    travel: đặt vé+khách sạn cho chuyến công tác / hủy vé cũ rồi đặt vé mới /
            so sánh rồi đặt / đặt nhiều chặng / đặt vé+hỏi hành lý.
    ecommerce: tìm+thêm giỏ+áp voucher+thanh toán / mua nhiều món / mua rồi
            theo dõi đơn / mua rồi trả hàng / so giá nhiều sàn rồi mua.
    banking: mở tiết kiệm sau khi xem lãi suất+kiểm tra số dư / chuyển tiền nhiều
            bước / khóa thẻ khẩn cấp+mở thẻ / vay+tính lãi.
  - KHÔNG lặp lại cùng một cặp tool cho nhiều task.

[FORMAT VIỆT] ngày dd/mm/yyyy, tiền VNĐ, SĐT 0xxx, địa danh VN thật, query tự
  nhiên như người Việt gõ, KHÔNG nhắc tên tool, KHÔNG để placeholder "id_sản_phẩm".
  Với param phụ thuộc (flight_id/hotel_id/cart_id/order_id/booking_id) điền giá trị
  hợp lệ bất kỳ — pipeline tự thay id thật; nhưng step TẠO id phải đứng TRƯỚC.

Trả về JSON array thuần, KHÔNG markdown, KHÔNG giải thích."""


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


def _retry_seconds_from_error(msg: str) -> float | None:
    """Bóc thời gian chờ từ message 429. Trả số giây, hoặc None."""
    m = re.search(r"try again in (\d+)m([\d.]+)s", msg)  # '36m2.3s' — kiểm tra TRƯỚC
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r"try again in ([\d.]+)\s*s", msg)
    if m:
        return float(m.group(1))
    return None


def _is_daily_limit(msg: str) -> bool:
    """429 do cạn quota NGÀY (TPD/RPD) — chờ vô ích, phải dừng."""
    m = msg.lower()
    return ("per day" in m or "tpd" in m or "rpd" in m
            or "requests per day" in m or "tokens per day" in m)


def _chat_with_backoff(client, *, max_retries: int = 6, base_pause: float = 3.0,
                       max_wait: float = 90.0, **kwargs):
    """
    Gọi chat.completions.create với chống-429 THÔNG MINH:
    - 429 do TPM (phút): chờ đúng số giây server bảo (cap ở max_wait), retry.
    - 429 do TPD/RPD (ngày): DỪNG NGAY — chờ không giải quyết được (reset theo ngày).
    - Nếu server bảo chờ > max_wait: coi như cạn quota → dừng, báo rõ.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_err = e
            msg = str(e)
            is_429 = "429" in msg or "rate_limit" in msg.lower() or "rate limit" in msg.lower()

            if is_429 and _is_daily_limit(msg):
                raise RuntimeError(
                    "❌ CẠN QUOTA NGÀY của Groq free tier (TPD/RPD). Chờ không giải quyết được.\n"
                    "   → Sang ngày mai (reset), HOẶC nâng Dev tier: https://console.groq.com/settings/billing\n"
                    "   → Hoặc đổi sang key/tổ chức khác.\n"
                    f"   Lỗi gốc: {msg[:200]}")

            wait = _retry_seconds_from_error(msg)
            if wait is None:
                wait = base_pause * (2 ** attempt) + random.uniform(0, 1.0)
            else:
                wait += 0.3

            if wait > max_wait:
                raise RuntimeError(
                    f"❌ Groq yêu cầu chờ {wait:.0f}s (>{max_wait:.0f}s) — gần như chắc đã cạn quota.\n"
                    "   → Sang ngày mai hoặc nâng Dev tier. Không chờ {wait:.0f}s.\n"
                    f"   Lỗi gốc: {msg[:200]}")

            tag = "429 rate-limit (phút)" if is_429 else "lỗi tạm"
            print(f"    ⏳ {tag}, chờ {wait:.1f}s rồi thử lại (lần {attempt + 1}/{max_retries})…")
            time.sleep(wait)
    raise RuntimeError(f"Hết retry sau {max_retries} lần. Lỗi cuối: {last_err}")


def _tool_signature(name: str, spec: dict) -> str:
    """Mô tả tool kèm required params + ENUM + kiểu để LLM không gen sai schema."""
    params = spec.get("parameters", {}) or {}
    req = params.get("required", []) if isinstance(params, dict) else []
    props = params.get("properties", {}) if isinstance(params, dict) else {}
    desc = spec.get("description_vi", spec.get("description", ""))

    def fmt(p):
        info = props.get(p, {})
        t = info.get("type", "")
        bits = [t] if t else []
        if "enum" in info:
            bits.append("CHỈ ∈ {" + ", ".join(map(str, info["enum"])) + "}")
        if "minimum" in info:
            bits.append(f"≥{info['minimum']}")
        if "pattern" in info:
            bits.append(f"khớp /{info['pattern']}/")
        return f"{p}({'; '.join(bits)})" if bits else p

    req_str = ", ".join(fmt(p) for p in req) if req else "(không bắt buộc param)"
    opt = [p for p in props if p not in req]
    opt_str = f" | tuỳ chọn: {', '.join(fmt(p) for p in opt)}" if opt else ""
    return f"- {name}: {desc}\n    BẮT BUỘC params: {req_str}{opt_str}"


def draft_trajectories(client, model: str, domain: str, difficulty: str, n: int, catalog: dict,
                       inter_batch_pause: float = 3.5, batch_seed: int = 0) -> list:
    """Gọi LLM draft n task (chống-429). Trả list raw {query, steps}."""
    tools_for_domain = {k: val for k, val in catalog.items() if val.get("domain") == domain}
    if not tools_for_domain:
        valid = sorted({v.get("domain") for v in catalog.values()})
        raise SystemExit(
            f"❌ Domain '{domain}' KHÔNG khớp domain nào trong catalog.\n"
            f"   Domain hợp lệ: {valid}\n"
            f"   (Nếu thấy nhiều domain dính làm 1 chuỗi → lỗi expand $DOMAINS trong bash; "
            f"chạy loop `for d in banking ecommerce ...` cho đúng.)")
    valid_names = list(tools_for_domain.keys())
    tools_desc = "\n".join(_tool_signature(k, val) for k, val in tools_for_domain.items())
    # xoay "góc" kịch bản mỗi batch để LLM không lặp cùng một ý
    angles = ["tình huống công việc/khẩn cấp", "người dùng cá nhân đời thường",
              "kết hợp nhiều tool theo chuỗi dài", "có hủy/đổi/trả rồi làm lại",
              "so sánh/tra cứu trước rồi mới hành động", "tình huống gia đình/nhóm nhiều người"]
    angle = angles[batch_seed % len(angles)]
    # gợi ý tham số đa dạng theo domain để giảm lặp địa danh/ý
    diversity_hint = {
        "travel": "Dùng NHIỀU cặp địa danh VN khác nhau (HN, HCM, Đà Nẵng, Nha Trang, Phú Quốc, "
                  "Đà Lạt, Huế, Cần Thơ, Hải Phòng, Sa Pa) và nhiều loại chuỗi (đặt vé+khách sạn / "
                  "hủy rồi đặt lại / đặt vé+hỏi hành lý / đặt nhiều chặng / đổi vé).",
        "ecommerce": "Dùng NHIỀU loại sản phẩm khác nhau (điện tử, thời trang, gia dụng, mỹ phẩm, sách, "
                     "thực phẩm) và nhiều chuỗi (mua+voucher / mua+theo dõi / mua+trả hàng / so giá+mua).",
        "banking": "Dùng nhiều loại giao dịch (chuyển tiền, mở tiết kiệm, vay, khóa thẻ, tra cứu lãi).",
        "public_services": "Dùng nhiều dịch vụ (BHXH, thuế, hộ tịch, đăng ký xe, hóa đơn điện nước, đặt lịch).",
        "info_retrieval": "Dùng nhiều loại truy vấn (thời tiết, tin tức, dịch, đổi tiền, tính toán, tra cứu).",
    }.get(domain, "Đa dạng tình huống và tham số.")
    resp = _chat_with_backoff(
        client,
        model=model,
        messages=[
            {"role": "system", "content": _HARD_SYS.format(difficulty=difficulty)},
            {"role": "user", "content": (
                f"Tạo {n} task '{difficulty}' cho domain '{domain}'.\n"
                f"LẦN NÀY nghiêng về: {angle}.\n"
                f"ĐA DẠNG: {diversity_hint}\n"
                f"Mỗi task PHẢI khác nhau rõ rệt về tham số/tình huống.\n\n"
                f"DANH SÁCH TOOL HỢP LỆ (CHỈ dùng đúng tên này, KHÔNG bịa thêm):\n"
                f"{', '.join(valid_names)}\n\n"
                f"CHI TIẾT (PHẢI điền ĐỦ param BẮT BUỘC, thiếu/sai → loại):\n{tools_desc}\n\n"
                f"Mỗi item: {{\"query\":..., \"steps\":[{{\"expected_tool\":..., "
                f"\"expected_params\":{{...đủ required...}}, \"difficulty\":\"{difficulty}\"}}]}}.\n"
                f"Trả JSON array thuần, KHÔNG markdown."
            )},
        ],
        temperature=1.0,
        max_tokens=2500,
    )
    # nghỉ chủ động giữa các batch để không dồn TPM
    time.sleep(inter_batch_pause)
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
        print(f"\n>>> Gen {domain}/{difficulty} (target {target})")
        attempts = 0
        while counters[(domain, difficulty)] < target and attempts < target * max_attempts_factor:
            attempts += 1
            batch = draft_trajectories(client, model, domain, difficulty,
                                       n=min(5, target), catalog=catalog, batch_seed=attempts)
            for raw in batch:
                q = (raw.get("query") or "").strip()
                if not q or q in seen_queries:
                    continue
                steps = _raw_to_steps(raw)
                if not steps:
                    continue
                ok, patched, final, reason = accept_trajectory(steps, catalog, difficulty=difficulty, query=q)
                if not ok:
                    print(f"  [reject] {domain}/{difficulty}: {reason}")
                    continue
                tid = f"{domain}_{difficulty}_{counters[(domain, difficulty)]:04d}"
                kept.append(_to_trajectory(raw, patched, final, tid, domain, difficulty))
                seen_queries.add(q)
                counters[(domain, difficulty)] += 1
                print(f"  ✅ {tid} ({len(patched)} steps)  [{counters[(domain, difficulty)]}/{target}]")
                print(f"  ✅ {tid} ({len(patched)} steps)")
    save_trajectories(kept, out_path)
    # dedup ngữ nghĩa sau khi gom (LLM hay lặp ý)
    try:
        from data.dedup import dedup_trajectories
        deduped = dedup_trajectories(kept, threshold=0.93)
        if len(deduped) < len(kept):
            save_trajectories(deduped, out_path)
            kept = deduped
    except Exception as e:
        print(f"  [build] bỏ qua dedup ({type(e).__name__}); chạy `python data/dedup.py` riêng sau.")
    rep = stratify_report(kept, min_per_bin)
    print(f"\n✅ Saved {len(kept)} trajectories → {out_path}")
    print(f"   Stratify: {rep}")


# ============================================================
# Revalidate pool (chạy data CŨ qua pipeline → giữ cái pass; KHÔNG xóa tay)
# ============================================================
def revalidate_pool(in_path: str, out_path: str, catalog: dict) -> None:
    """
    Đọc trajectory cũ, chạy qua accept_trajectory (resolve_ids + 2 cổng),
    GIỮ cái pass (đã patch id thật + có expected_final_state), BỎ cái fail.
    Bạn không cần biết/xoá cái nào — pipeline tự lọc.
    """
    trajs = load_trajectories(in_path)
    order = {"easy": 0, "medium": 1, "hard": 2}
    kept: list[Trajectory] = []
    rej = 0
    for t in trajs:
        steps = [(s.expected_tool, s.expected_params) for s in t.steps]
        ok, patched, final, reason = accept_trajectory(steps, catalog, query=t.query)
        if not ok:
            rej += 1
            continue
        lvl = max((s.difficulty.value for s in t.steps), key=lambda d: order[d])
        kept.append(Trajectory(
            traj_id=t.traj_id,
            query=t.query,
            language=t.language,
            steps=[Step(expected_tool=tt, expected_params=pp, difficulty=Difficulty(lvl))
                   for tt, pp in patched],
            source=(t.source or "") + "+revalidated",
            domain=t.domain,
            expected_final_state=final,
        ))
    save_trajectories(kept, out_path)
    print(f"✅ Revalidate: giữ {len(kept)}/{len(trajs)} (bỏ {rej} cái fail gate) → {out_path}")
    print(f"   Stratify: {stratify_report(kept, 50)}")



# ============================================================
# Gate report: chạy MỌI cổng trên file có sẵn, đếm pass/fail (không sửa data)
# ============================================================
def gate_report(paths: list[str], catalog: dict, out_path: str | None = None,
                split_names: list[str] | None = None) -> dict:
    """Đếm số trajectory fail ở từng cổng (cấu trúc, resolve, 1, 1.5, 2, G3) + chi tiết G3."""
    order = {"easy": 0, "medium": 1, "hard": 2}
    rows = []
    for pi, path in enumerate(paths):
        split = (split_names or [])[pi] if split_names and pi < len(split_names) else Path(path).stem
        for t in load_trajectories(path):
            steps = [(s.expected_tool, s.expected_params) for s in t.steps]
            tier = max((s.difficulty.value for s in t.steps), key=lambda d: order[d])
            ok, patched, _, reason = accept_trajectory(steps, catalog, difficulty=tier, query=t.query)
            first_fail = "pass" if ok else reason.split("]")[0].strip("[")
            bad = grounding_report(t.query, patched, catalog)
            rows.append({"traj_id": t.traj_id, "split": split, "domain": t.domain, "difficulty": tier,
                         "first_failing_gate": first_fail, "g3_pass": not bad,
                         "ids_changed_by_resolve": patched != steps, "g3_ungrounded": bad})
    def tab(key):
        d: dict = defaultdict(lambda: {"trajs": 0, "pass_all": 0, "g3_fail": 0})
        for r in rows:
            for k in ("all", key(r)):
                d[k]["trajs"] += 1
                d[k]["pass_all"] += r["first_failing_gate"] == "pass"
                d[k]["g3_fail"] += not r["g3_pass"]
        return dict(sorted(d.items()))
    rep = {
        "n_trajectories": len(rows),
        "first_failing_gate": dict(Counter(r["first_failing_gate"] for r in rows)),
        "by_split": tab(lambda r: r["split"]),
        "by_difficulty": tab(lambda r: r["difficulty"]),
        "by_domain": tab(lambda r: r["domain"]),
        "g3_ungrounded_values": sum(len(r["g3_ungrounded"]) for r in rows),
        "g3_top_params": Counter(f"{b['tool']}.{b['param']}" for r in rows
                                 for b in r["g3_ungrounded"]).most_common(25),
        "ids_changed_by_resolve": sum(r["ids_changed_by_resolve"] for r in rows),
        "trajectories": rows,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    return rep


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
    ap.add_argument("--revalidate", action="store_true",
                    help="chạy data CŨ qua pipeline, giữ cái pass (không cần API, không xóa tay)")
    ap.add_argument("--in", dest="in_path", default="data/vitool/vitool_all.jsonl",
                    help="file nguồn cho --revalidate")
    ap.add_argument("--domain", help="banking|public_services|ecommerce|travel|info_retrieval")
    ap.add_argument("--difficulty", default="hard", choices=["easy", "medium", "hard"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="data/vitool/drafts_new.jsonl")
    ap.add_argument("--min-per-bin", type=int, default=50)
    ap.add_argument("--gate-report", nargs="+", metavar="JSONL",
                    help="chạy mọi cổng (gồm G3) trên các file, in số pass/fail; --out = file JSON chi tiết")
    args = ap.parse_args()

    if args.gate_report:
        catalog = {k: v for k, v in json.load(open(
            Path(__file__).resolve().parent.parent / "data/vitool/tool_catalog.json")).items()
            if not k.startswith("_")}
        out = args.out if args.out.endswith(".json") else None
        rep = gate_report(args.gate_report, catalog, out)
        print(json.dumps({k: v for k, v in rep.items() if k != "trajectories"}, indent=1, ensure_ascii=False))
        return

    if args.selftest:
        _selftest()
        return

    if args.revalidate:
        catalog = {k: v for k, v in json.load(open(
            Path(__file__).resolve().parent.parent / "data/vitool/tool_catalog.json")).items()
            if not k.startswith("_")}
        revalidate_pool(args.in_path, args.out, catalog)
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