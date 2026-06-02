# ViTool — NCKH (Benchmark-First, RIVF 2026) — README v10.0

**Tên đề tài (đề xuất):** *ViTool: An Executable, Difficulty-Stratified Benchmark for Per-Step Compute-Mode Allocation in Vietnamese Tool-Using SLM Agents*

**Trạng thái:** v10.0 — **pivot từ v9.2 (method-first/GRPO) sang benchmark-first**. Headline là benchmark + oracle-gap analysis (robust, không phụ thuộc GRPO thắng); GRPO xuống mục chẩn đoán optional.

**Team:** 2 SV năm 3 (AI Engineering) + 1 PhD advisor (mời NGAY — corresponding author).
**Resource:** Kaggle T4×2 | Qwen2.5-3B-Instruct (tool-use SFT, đã có) | GPT-4o-mini (cho M3 + baseline) | Groq Llama-3.3-70B (chỉ để DRAFT data).
**Venue:** **RIVF 2026** — VinUniversity Hà Nội, 18–20/12/2026. **6 trang, IEEE format (IEEEtran), IEEE Xplore.** Deadline **15/7/2026** (tentative, đã confirm trên rivf2026.org). Notification 30/9, camera-ready 30/10.
**Fallback:** SOICT 2026 (Springer CCIS ~12 trang, deadline thường muộn hơn — verify).
**Xác suất accept (ước lượng thực tế, KHÔNG phải con số marketing):** ~55–70% nếu execute tốt + có advisor co-author. Base rate full paper các venue này ~38%. (Bản v9.2 ghi "87–92%" là kỳ vọng, đã loại bỏ.)

---

## Mục lục
1. [v10.0 là gì & vì sao pivot](#1-v100-là-gì--vì-sao-pivot)
2. [Thesis 1 câu + 3 contributions](#2-thesis--contributions)
3. [Findings F1–F5 (hypotheses)](#3-findings)
4. [4 mode — ai thực thi, cost](#4-4-mode)
5. [Data pipeline (cốt lõi value-add)](#5-data-pipeline)
6. [Executable environment (NEW)](#6-executable-environment)
7. [Pilot / oracle protocol](#7-pilot--oracle-protocol)
8. [Baselines + metrics](#8-baselines--metrics)
9. [Code: giữ gì / thêm gì](#9-code-reuse-map)
10. [Critical rules (updated)](#10-critical-rules)
11. [Paper structure 6 trang IEEE](#11-paper-structure)
12. [Risk & fallback](#12-risk--fallback)
13. [Honest assessment](#13-honest-assessment)

---

## 1. v10.0 là gì & vì sao pivot

v9.2 đẩy *"train SLM 3B học per-step compute-mode routing qua outcome-only GRPO"* làm headline. Kết quả W5 (full GRPO) cho thấy **không đứng được như headline**:

- **Routing collapse / không difficulty-aware.** Hai eval trong notebook mâu thuẫn nhau (M0=87% vs M3=36% trên cùng model/data) → đo không đáng tin; eval khớp cơ chế train cho thấy collapse về M0. Kể cả lấy con số 36% theo nghĩa đen thì M3 (15×) bắn vào 35% step easy và 0% step hard → routing ngược, và làm hệ thống đắt hơn always-think → bị fixed-M1 dominate trên Pareto.
- **Gốc rễ (đã confirm bằng code):** (a) reward geometry — β=0.05 khiến M3 phải đúng hơn M0 >70 điểm % mới được chọn (bất khả); (b) verifier không xử được multi-step phụ thuộc khi thiếu execution env (hard data W5 invalid: `book_train` không có trong catalog, thiếu `hotel_id`); (c) entropy bonus tắt ở step 500 nhưng train tới 570 → trôi về mode rẻ; (d) `expected_final_state` rỗng → γ·final_success thành hằng số.

**Pivot:** Đo *cơ hội* của routing (oracle-gap) thay vì *học* nó. Oracle là tính toán offline ("mode rẻ nhất mà success ở mỗi step"), không cần model học → headline robust với mọi kết quả GRPO. Đồng thời **nâng cấp benchmark thành executable + multi-step** (sửa luôn lỗ verifier). GRPO xuống F5 optional.

**Tinh thần:** *Đo cơ hội routing, không học nó.* Đây là benchmark + analysis paper, không phải method paper.

---

## 2. Thesis & contributions

**Thesis (1 câu):** *Chúng tôi release ViTool — benchmark VN tool-use executable, phân tầng độ khó — và dùng nó đặc tả per-step compute-mode allocation cho SLM 3B: reasoning phí ở easy step, cần escalation ở hard step, một oracle router per-step tiết kiệm ~X% compute ở chất lượng tương đương, và học chính policy đó từ outcome-only signal là khó.*

- **C1 — ViTool benchmark.** VN tool-use, 5 domain, catalog 36 tool có **mock implementation (executable)**, phân tầng easy/medium/hard, gold **human-verified + executable-validated**. Reusable, public (GitHub/HF + datasheet). Tiền lệ accept: Arabic Tool Bench, FunctionChat-Bench Korean.
- **C2 — Mode-success + oracle-gap analysis.** Heatmap mode×difficulty + Pareto cost-quality (fixed modes + oracle + random + baselines). Định lượng overthink/underthink CHO TOOL AGENT + headroom của oracle routing.
- **C3 — Diagnostic study of learned routing (optional, F5).** Thử train mode-routing trên ViTool + executable reward; báo cáo honest (collapse hoặc cải thiện nhẹ) + phân tích lý do. Motivate future work.

---

## 3. Findings

> **Kỷ luật (giữ từ v9.2 §16):** đây là *finding-shaped hypotheses*. Số **điền sau experiment**, KHÔNG pre-commit trong abstract.

- **F1 (lõi, headline) — Đo cấu trúc overthink/underthink cho tool agent.** Easy step: M0 ≈ M1 về success nhưng M0 dùng 3–5× ít token (reasoning dư thừa). Hard step: M0 sụp, M1/M2 phục hồi. Là người **đầu tiên định lượng cho tool-using agent** (BCAE 2604.19780 đã làm cho math reasoning). Pilot cũ đã confirm nửa F1 (M0≈88% easy).
- **F2 (cơ hội, figure trung tâm) — Oracle-gap.** Oracle per-step (mode rẻ nhất mà success) đạt ~X% TSR của always-think (M1) ở ~Y% cost → headroom Z%. Motivate hướng nghiên cứu **mà không cần learned router**.
- **F3 (sắc, phản trực giác) — Khi nào defer (M3) mới đáng.** M3 (15×) chỉ net-positive trên **lát mỏng** hard step nơi 3B fail nhưng frontier success; easy/medium là phí thuần. Escalation đắt → success gap phải lớn mới đáng. Đẩy lùi naive "cứ đẩy việc khó sang model to".
- **F4 (vì sao tiếng Việt) — VN làm cliff lộ rõ.** VN tool-use có M0 success thấp hơn EN trên task tương đương → chi phí phân bổ sai compute cao hơn ở VN → mode allocation quan trọng hơn ở low-resource. (+ cross-lingual EN/Thai/ID nếu kịp.)
- **F5 (chẩn đoán, optional) — Learned routing khó.** Outcome-only GRPO khó học policy mà oracle cho thấy: dưới cost pressure collapse về mode rẻ. Với executable reward signal mới, test xem có cải thiện không — cải thiện = positive contribution; không = negative result sạch. Không bao giờ đụng headline.

---

## 4. 4 mode

**Quan trọng — phân biệt: có 4 mode (M0/M1/M2/M3, KHÔNG có M4). Mode là CÁCH THỰC THI một step lúc pilot, KHÔNG phải data.**

| Mode | Mô tả | Ai thực thi | Cost tương đối | Có train? |
|---|---|---|---|---|
| **M0 fast** | sinh tool-call trực tiếp, no CoT | **Qwen-3B** (tool_use_lora) | ~1× | dùng lại SFT, không train mới |
| **M1 think** | CoT rồi tool-call | **Qwen-3B** (tool_use_lora) | ~3× | nt |
| **M2 think+search** | CoT + retrieval trên catalog (e5 precomputed) | **Qwen-3B** (tool_use_lora) | ~5× | nt |
| **M3 defer** | gửi step sang frontier LLM | **GPT-4o-mini (API)** — KHÔNG phải Qwen, KHÔNG phải Llama | ~15× ($) | **không train** |
| Oracle | chọn mode rẻ nhất mà success | tính **offline** | — | không train |

**⚠️ Contamination fix:** code `m3_defer.py` hiện gọi `llama-3.3-70b` — phải đổi sang **GPT-4o-mini**, vì Llama-70B là model DRAFT gold → nếu M3/baseline cũng là Llama thì "quality ceiling" = Llama tự chấm Llama ≈ giả. **Generator ≠ defer/baseline model.**

---

## 5. Data pipeline

**Cốt lõi value-add của paper. Mistake≈0 KHÔNG đến từ "gen kỹ hơn" mà đến từ 3 CỔNG CHẶN.**

```
Llama-70B DRAFT  →  [Cổng 1: schema gate]  →  [Cổng 2: executable gate]  →  [Cổng 3: human verify]  →  stratify
(query + gold      (jsonschema: tool/param   (chạy thật qua 36 mock-tool   (2 người sửa gold,
 tool-calls)        sai → auto-reject)         tới expected final-state)     Cohen's kappa)
```

- **Cổng 1 — schema-validate tự động** (`verifiers/universal/schema_verifier.py` đã có): tool không trong catalog / thiếu required param → reject TRƯỚC khi nhận. Lẽ ra đã chặn sạch đống invalid W5.
- **Cổng 2 — executable gate (NEW):** trajectory chỉ là gold hợp lệ **nếu chạy thật tới đúng `expected_final_state`**. Cổng mạnh nhất.
- **Cổng 3 — human verify:** bước biến output thành benchmark. Kappa trên phần overlap.
- **Phân tầng đủ N:** **≥150 mỗi tầng** easy/medium/hard (tổng ~600–700). 5 domain: banking, ecommerce, info_retrieval, public_services, travel.
- **HARD = executable multi-step (bản RIVF-best):** 3+ tool call có data-dependency, verify bằng final-state — KHÔNG còn dùng static param-match cho multi-step (cái đã giết reward signal W5).
- **Size:** target **~600–700 trajectory human-verified** (team tryhard, ~4 ngày verify tập trung 2 người). Mỗi tầng độ khó **≥150**. **Chất > lượng vẫn là luật** (R5): thà verify kỹ 600 còn hơn ẩu 700. Dedup ngữ nghĩa bắt buộc trước verify (LLM gen hay lặp ý).
- **VỨT** bộ 44 hard của W5 (invalid). **GIỮ** easy/medium cũ làm draft → re-validate qua pipeline mới + bổ sung `expected_observation`/`expected_final_state`.

---

## 6. Executable environment (NEW)

Nâng cấp quan trọng nhất của bản RIVF-best, và là long pole → **bắt đầu ngay**.

- **36 mock-tool = 36 hàm Python** trả output **canned/deterministic** (bảng tra / template theo input). Giữ ĐƠN GIẢN và deterministic.
- Cho phép: (a) Cổng 2 ở §5; (b) `verify_final_outcome` thành **thật** (hiện là stub trả True); (c) M2 retrieval thật; (d) verify multi-step → mở khoá hard tier executable.
- Double payoff: vừa làm benchmark mạnh hơn hẳn (executable + multi-step, ngang BFCL/τ-bench), vừa sửa cái lỗ verifier đã làm GRPO collapse → nếu chạy F5 thì reward signal hard-step không còn hỏng.

---

## 7. Pilot / oracle protocol

(Tận dụng `pilot/simulate_modes.py` — đã chạy 4 mode/step.)

1. Mỗi step, chạy CẢ 4 mode offline → ghi `success` (executable verifier, binary) + `cost` (**token thật + $ cho M3**, không chỉ multiplier {1,3,5,15}).
2. **Oracle policy:** mỗi step chọn `argmin{cost(m) : success(m)=True}`; không mode nào success → mode cao nhất (hoặc đánh fail).
3. `cost_oracle = Σ cost(oracle_mode)`; `cost_fixedM1 = 3 × num_steps`.
4. **Savings = (1 − cost_oracle / cost_fixedM1) × 100%** ở mức TSR ≥ always-think.
5. **Figure chính (Pareto):** x=cost/query, y=TSR; vẽ fixed M0/M1/M2/M3, oracle, random, ReDAct, ATLAS-style, (LLM-only). Oracle ở top-left. Gap fixed-M1 → oracle = cơ hội.
6. **Báo savings theo TỪNG tầng độ khó + tổng**, minh bạch phân bố độ khó (vì savings phụ thuộc nặng vào phân bố này).

**Output paper §3–§5:** Fig 1 heatmap mode×difficulty; Fig 2 Pareto; Table per-mode token/$/success.

---

## 8. Baselines + metrics

**Baselines (đã implement trong `baselines/run_baselines.py` — reuse, chỉ sửa model defer):** fixed M0, fixed M1, LLM-only, ReDAct (perplexity defer), ATLAS-style, random, oracle, (BudgetSwitch/GRPO chỉ nếu chạy F5).
- ⚠️ LLM-only + M3 đổi sang **GPT-4o-mini** (xem §4 contamination).

**Metrics:** Task Success Rate (final outcome, executable); cost/query (token + $); latency p50/p95; mode distribution; per-step success; **tất cả phân tầng theo độ khó**.

---

## 9. Code reuse map

**GIỮ NGUYÊN (reuse, không xóa):**
- `data/vitool/schema.py` — Trajectory/Step/Difficulty/Language (extend: dùng `expected_observation`, populate `expected_final_state`).
- `data/vitool/tool_catalog.json` — 36 tool (extend: thêm mock implementation).
- `verifiers/universal/schema_verifier.py` — `verify_step_outcome` (reuse), `verify_final_outcome` (extend → executable thật).
- `modes/m0_fast`, `m1_think`, `m2_think_search` — reuse (Qwen-3B).
- `modes/m3_defer` — reuse **nhưng đổi model → GPT-4o-mini**.
- `models/tool_use_lora/` — **agent của paper**, dùng lại bản W3 (loss 0.42, 90% easy). Không train lại trừ khi format/catalog đổi nhiều.
- `pilot/simulate_modes.py`, `pilot/analyze_pilot.py` — lõi oracle analysis.
- `baselines/run_baselines.py` — 8 baseline (sửa model defer).
- `training/reward.py`, `scripts/verify_no_oracle_leak.sh` — **giữ cho F5 optional**.
- Notebooks W1–W5 — giữ làm lịch sử/log.

**THÊM MỚI:**
- `env/mock_tools.py` — 36 mock-tool deterministic + final-state checker (executable env).
- `data/build_dataset.py` — pipeline gen→schema-gate→executable-gate→stratify (≥50/bin, hard=multi-step executable).
- `eval/oracle.py` — tính oracle policy + savings per-difficulty + overall.
- `eval/figures.py` — heatmap + Pareto generator.
- (F5 optional) `training/grpo_vitool.py` — train lại mode-routing trên ViTool + executable reward + **eval ĐÚNG** (chat-template + masked first-token argmax).

**CẮT khỏi lõi:** GRPO mode-routing (giữ file, chỉ dùng nếu làm F5; KHÔNG nằm headline).

---

## 10. Critical rules (updated)

```
# ViTool v10.0 — CRITICAL RULES
#
# R1 (CONTAMINATION): generator ≠ defer/baseline model.
#     Gold DRAFT = Llama-3.3-70B (Groq). M3 + LLM-only baseline = GPT-4o-mini.
#     Vi phạm → "quality ceiling" = model tự chấm chính nó ≈ giả → F3 vô nghĩa.
#
# R2 (EXECUTABLE VERIFICATION): gold hợp lệ CHỈ KHI chạy thật tới expected_final_state.
#     Không run được → loại. KHÔNG dùng static param-match cho multi-step.
#
# R3 (NO PRE-COMMIT NUMBERS): số X/Y/Z điền sau experiment, không pre-commit abstract.
#
# R4 (F5 only — NO ORACLE LEAK): nếu chạy GRPO, reward KHÔNG reference oracle_mode/expected_mode.
#     Verify: bash scripts/verify_no_oracle_leak.sh
#
# R5 (HUMAN GATE): chất lượng đến từ schema-gate + executable-gate + human-verify,
#     KHÔNG từ "gen kỹ hơn". Gen lại bao nhiêu lần mà thiếu 3 cổng vẫn dính lỗi cũ.
```

---

## 11. Paper structure (6 trang IEEE)

**Chật — đóng khung "benchmark + finding sắc", không tham.**

| § | Title | ~Trang | Nội dung |
|---|---|---|---|
| 1 | Introduction | 1.0 | Hook overthink/underthink cho tool agent; 3 contributions |
| 2 | Related Work | 0.75 | Test-time compute (BCAE), agentic RL (ATLAS), multilingual FC; ViTool ở giao điểm |
| 3 | ViTool Benchmark | 1.25 | Catalog 36 tool executable, 5 domain, stratification, pipeline 3 cổng, kappa |
| 4 | Mode-Success Analysis (F1) | 1.0 | Heatmap mode×difficulty; overthink-easy / underthink-hard |
| 5 | Oracle-Gap & When-to-Defer (F2, F3) | 1.0 | Pareto, savings per-difficulty, M3 chỉ đáng ở lát mỏng |
| 6 | (optional) Diagnostic: Learning the Policy (F5) | 0.5 | GRPO collapse + lý do; HOẶC cắt nếu hết giờ |
| 7 | Conclusion + Limitations | 0.25 | Recap + future work + honest limitations |
| Refs | — | 0.25 | ~20 refs |

VN (F4) lồng vào §3 (data) + §4–5 (phân tích), không tách section riêng để tiết kiệm trang.

---

## 12. Risk & fallback

| Risk | Prob | Mitigation |
|---|---|---|
| Timeline 15/7 quá chật | **Cao** | Go/No-Go gate ~25/6 (xem ROADMAP); thiếu → lùi SOICT |
| Advisor không kịp | **Cao** | Mời 2–3 NGAY (ngày 1), không chờ result |
| Executable env tốn thời gian | Trung | Bắt đầu ngay, tool deterministic đơn giản, song song với data |
| Human-verify chậm | Trung | 2 người, ~400–600 traj, chia domain; chất > lượng |
| Reviewer "benchmark VN đơn thuần incremental" | Trung | Đóng khung = benchmark **+ oracle-gap + executable + diagnostic**, không phải "thêm 1 dataset" |
| Contamination (gen=defer model) | Đã biết | R1: M3 = GPT-4o-mini |

**Fallback:** RIVF reject hoặc không kịp 15/7 → SOICT 2026 (12 trang CCIS, có chỗ cho executable multi-step + cross-lingual + F5 đầy đủ). Substance viết venue-agnostic để retarget dễ.

---

## 13. Honest assessment

**Mạnh:** literature thật & chắc (BCAE/ATLAS verified); infra thật (8 baseline, pilot engine, tool-use SFT 90% easy, modes chạy); nâng cấp executable làm benchmark mạnh hơn hẳn + sửa lỗ verifier; framing robust (không phụ thuộc GRPO).

**Rủi ro:** timeline 6 tuần cho paper đầu tay + dataset mới là aggressive; advisor chưa có; human-verify là cọc tốn sức.

**Odds thực tế:** ~55–70% nếu execute tốt + advisor co-author + figure oracle-gap rõ. KHÔNG phải 87–92%. Peer review có nhiễu ~5–10% không khử được.

**Giá trị hồ sơ TQ (Thanh Hoa/Bắc Đại):** RIVF/SOICT **không trên bảng CCF** → là tín hiệu "có nghiên cứu thật, biết hoàn thành" + đạn săn thầy (套磁) + chất liệu thư advisor, KHÔNG phải "publication mạnh". Là **một trụ, không phải vé vàng** — ghép với Olympic AI + GPA + thầy.

**Stop iterating. Start building.** Việc đầu tiên: dựng executable env + data pipeline (§5–§6), mời advisor (ROADMAP W0).

---

*v10.0 — Benchmark-first re-strategy của project BudgetSwitch. Headline: ViTool benchmark + oracle-gap. GRPO → diagnostic optional.*
