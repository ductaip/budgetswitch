# ROADMAP — ViTool → RIVF 2026

**Deadline:** RIVF 15/7/2026 (EasyChair). Notification 30/9, camera-ready 30/10, hội nghị 18–20/12.
**Khung:** ~6 tuần từ đầu tháng 6. **Aggressive** cho paper đầu tay + dataset mới → có **Go/No-Go gate** giữa đường.
**Đường găng (critical path):** **(1) mời advisor → (2) executable env → (3) human-verify data.** Ba thứ này quyết định kịp hay không, KHÔNG phải phần viết (6 trang viết nhanh).

---

## Team split

| Mảng | Member A | Member B | Advisor |
|---|---|---|---|
| Executable env (36 mock-tool) | **chủ trì** | hỗ trợ | — |
| Data gen + 3 cổng | hỗ trợ | **chủ trì** | sanity check |
| Human verify | banking, travel, ecommerce | info_retrieval, public_services + cross-check | spot-check |
| Pilot/oracle + figures | chạy pilot 4 mode | **oracle + Pareto + heatmap** | review hình |
| Baselines (8) | ReDAct, ATLAS | random, oracle, fixed, LLM-only | — |
| Viết | §4, §5 (+§6 nếu F5) | §1, §2, §3 | review toàn bộ (corresponding author) |
| (optional) F5 GRPO | train lại + eval đúng | reward executable | review novelty |

Họp đồng bộ 2 buổi/tuần (vd CN + thứ 4), code review chéo PR, shared Notion.

---

## Quyết định đã chốt
- **KHÔNG re-measure GRPO cũ** (kết quả không đổi, và data sắp đổi → adapter cũ vô nghĩa).
- **Tool-use SFT (`tool_use_lora`) GIỮ** — dùng lại bản W3.
- **GRPO mode-routing CẮT khỏi lõi** → chỉ làm ở F5 nếu còn giờ.
- **M3 + baseline LLM-only đổi sang GPT-4o-mini** (contamination fix; gen = Llama-70B).
- **HARD = executable multi-step** (cần mock env).

---

## Tuần 0 (~2–8/6) — Kickoff
- [ ] **[Both] Email 2–3 advisor NGAY** (corresponding author) — đường găng dài nhất, không chờ result. (Template: docs/ADVISOR_EMAIL_TEMPLATES.md cũ.)
- [ ] [Both] Repo restructure: thêm `env/`, `eval/`; **giữ nguyên code cũ**; cập nhật CRITICAL_RULES (R1–R5).
- [ ] [A] Bắt đầu `env/mock_tools.py` — 36 mock-tool deterministic (long pole).
- [ ] [B] Scaffold `data/build_dataset.py`: nối schema-gate (đã có) + chừa hook executable-gate; viết prompt gen hard multi-step (3+ step, data-dependency).
- **M0:** advisor emails sent · repo restructured · env + pipeline scaffolding chạy.

## Tuần 1 (~9–15/6) — Env + data draft
- [ ] [A] Hoàn thiện 36 mock-tool + **final-state checker**; nâng `verify_final_outcome` từ stub → executable thật.
- [ ] [A] Đổi `modes/m3_defer.py` → GPT-4o-mini; smoke test 4 mode chạy end-to-end trên vài step.
- [ ] [B] Gen draft trajectory (Llama-70B) → qua **Cổng 1 (schema)** + **Cổng 2 (executable)**; bắt đầu re-validate easy/medium cũ qua pipeline mới.
- [ ] [A] Pre-compute tool embeddings (e5) cho M2.
- **M1:** executable env xong · 4 mode chạy được · draft data đang chảy qua cổng 1+2 · VỨT 44 hard W5 invalid.

## Tuần 2 (~16–22/6) — Human verify *(BOTTLENECK — dồn lực ~4 ngày)*
- [ ] [B] Gen hard ~100–120/domain draft (5 domain) → qua dedup ngữ nghĩa + 2 cổng → ~150+/tier.
- [ ] [Both] **Dedup ngữ nghĩa** (embedding sim) TRƯỚC verify — LLM gen hay lặp ý, không lọc thì verify phí + dataset bị chê thiếu đa dạng.
- [ ] [Both] **Human-verify + sửa gold qua CLI verify tool** (`data/verify_cli.py`), chia domain; Cohen's kappa trên phần overlap. ~528 cái mới + spot-check 172 cũ ≈ **~12h công → ~4 ngày** (3h/ngày/người × 2).
- [ ] [B] Đảm bảo **≥150 mỗi tầng** easy/medium/hard (hard = executable multi-step). Tổng ~600–700.
- [ ] [Both] Viết datasheet ViTool.
- **M2:** ViTool ~600–700 traj human-verified · stratified ≥150/bin · executable-validated · deduped · **kappa report**.
- **⚠️ Rủi ro:** đây là tuần nặng nhất. Nếu đến 25/6 chưa verify xong → cân nhắc cắt còn ~400 cho RIVF, để full 600–700 cho SOICT.

---

## ⛔ GO / NO-GO GATE (~25/6)
Lao tiếp RIVF **chỉ khi đủ CẢ 3**:
1. ✅ Advisor đã đồng ý.
2. ✅ Data sạch, phân tầng ≥150/bin (hoặc ≥120 nếu sát giờ), executable-validated, deduped.
3. ✅ Pilot chạy ra **figure oracle sơ bộ** (dù số chưa final).

**Thiếu bất kỳ điều nào → LÙI SOICT** (deadline muộn hơn, 12 trang có chỗ thở). Đừng nhét paper mỏng vào RIVF — reject vừa không giúp hồ sơ vừa đốt công trình cho venue đó.

---

## Tuần 3 (~23–29/6) — Pilot / oracle *(headline)*
- [ ] [A] Chạy pilot: mỗi step × 4 mode offline → `success` (executable) + `cost` (token + $).
- [ ] [B] `eval/oracle.py`: oracle policy + **savings per-difficulty + overall**; `eval/figures.py`: heatmap (Fig1) + Pareto (Fig2).
- [ ] [Both] Chạy 8 baselines (GPT-4o-mini cho LLM-only/M3).
- **M3:** Fig1 + Fig2 + bảng per-mode token/$/success xong · **F1/F2/F3 có data**.

## Tuần 4 (~30/6–6/7) — Writing + (optional F5)
- [ ] [B] Draft §1 Intro · §2 Related Work · §3 ViTool.
- [ ] [A] Draft §4 Mode-success (F1) · §5 Oracle-gap (F2,F3); polish figures.
- [ ] [Both, IF còn giờ] **F5:** train GRPO trên ViTool + executable reward + **eval ĐÚNG** (chat-template + masked first-token argmax) → 1 mục chẩn đoán. Hết giờ → **cắt F5**, paper vẫn nguyên.
- **M4:** draft §1–§5 xong · figures đẹp.

## Tuần 5 (~7–13/7) — Advisor review + revise
- [ ] [Both] Gửi advisor review (đầu tuần) → sửa theo feedback.
- [ ] [Both] **Điền số thật vào abstract** (KHÔNG trước M3) · LaTeX IEEEtran **6 trang** check.
- **M5:** full 6-trang draft · abstract số thật · advisor signed off.

## Tuần 6 (~14–15/7) — Polish + submit
- [ ] [Both] Self-review cuối · references · **release GitHub/HF + datasheet** (link trong paper).
- [ ] [Both] **SUBMIT RIVF — 15/7** (EasyChair).
- Buffer ~0 → đó là lý do gate 25/6 quan trọng.

---

## Tổng quan tiến độ

```
W0 ⏳  Kickoff: advisor emails + repo restructure + env/data scaffolding
W1 ○  Executable env xong + 4 mode chạy + draft data qua cổng 1+2
W2 ○  Human-verify (BOTTLENECK) → ViTool stratified ≥50/bin + kappa
   ⛔ GATE 25/6: advisor + data + figure oracle sơ bộ → else SOICT
W3 ○  Pilot/oracle + 8 baselines → Fig1 heatmap + Fig2 Pareto (F1/F2/F3)
W4 ○  Viết §1–§5 (+ F5 optional)
W5 ○  Advisor review + điền số + LaTeX 6 trang
W6 ○  Polish + release + SUBMIT 15/7
```

---

## Fallback (nếu trượt gate hoặc RIVF reject)
- → **SOICT 2026** (Springer CCIS ~12 trang). Có chỗ cho: executable multi-step đầy đủ + cross-lingual (EN/Thai/ID, F4 mở rộng) + F5 diagnostic đầy đủ.
- Viết substance **venue-agnostic** để retarget không tốn công.
- Verify deadline SOICT 2026 trên trang chính thức (thường muộn hơn RIVF ~Aug–Sep).

---

*Lưu ý: số tuần là tương đối từ đầu tháng 6. Đường găng = advisor + executable env + human-verify. Mọi thứ phía sau chờ 3 cái đó.*