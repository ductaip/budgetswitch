# Advisor Email Templates

Send to 2-3 candidate advisors **in parallel** on W1 Day 5 (not sequentially — saves 2-3 weeks if first declines).

---

## Email 1: Primary template (Vietnamese)

**Subject line options (pick one):**
- `[NCKH] Đề xuất nghiên cứu - BudgetSwitch: Adaptive Compute Allocation cho SLM Agents`
- `[NCKH AI] Mong thầy/cô cân nhắc hướng dẫn đề tài về SLM agentic`
- `Sinh viên năm 3 xin tham vấn về đề tài NCKH agentic AI`

**Email body:**

```
Kính thưa thầy/cô [Tên đầy đủ],

Em là [Họ tên], MSSV [xxx], sinh viên năm 3 ngành AI Engineering 
tại [Trường]. Cùng với bạn [Họ tên B] (MSSV [yyy]), em đang chuẩn 
bị một đề tài NCKH về tối ưu compute allocation cho small language 
model agents - một hướng nghiên cứu agentic AI đang rất nóng năm 
2026 (gần đây có NVIDIA position paper "SLMs are the Future of 
Agentic AI", các thinking modes của DeepSeek-V4, paper ReDAct và 
BCAE về budget-aware reasoning).

Đề tài cụ thể của nhóm em mang tên BudgetSwitch: train một SLM 
3B (Qwen2.5-3B) học cách tự chọn compute mode (fast / think / 
think+search / defer-to-LLM) cho từng step trong tool-using 
trajectory, thông qua GRPO với verifiable reward (không cần human 
annotation và không cần LLM teacher). Vietnamese tool-use được 
dùng làm stress test benchmark.

Sau khi nghiên cứu kỹ literature (66+ papers reviewed) và iterate 
qua nhiều phiên bản đề xuất, nhóm em đã hoàn thiện được:

  1. Methodology defensible với reviewer am hiểu RL (clean GRPO,
     không có oracle leakage trong reward function)
  2. Roadmap 24 tuần thực tế với resource Kaggle T4 (~50 GPU-hours
     và ~$100 API budget - team tự chi trả)
  3. Risk mitigation plan với 5 fallback strategies
  4. Pilot study được thiết kế trước GRPO training để validate
     core hypothesis bằng data thực
  5. Target submission: RIVF 2026 hoặc SOICT 2026 (Springer CCIS,
     Scopus-indexed)

Em xin phép đính kèm full proposal (12 trang PDF) cho thầy/cô 
xem xét.

Em rất mong được thầy/cô cân nhắc hỗ trợ supervisorship cho 
project này, với vai trò Corresponding Author trên bài báo. 
Phần đóng góp từ thầy/cô mà nhóm em mong muốn:
  - Methodology guidance và direction-setting (định kỳ)
  - Feedback trên paper draft (chủ yếu Phase 5, W17-20)
  - Endorsement chính thức như Corresponding Author

Phần execution toàn bộ - coding, experiments, data construction, 
initial paper drafting - sẽ do nhóm em chịu trách nhiệm hoàn 
toàn. Nhóm em commit weekly progress updates và sẵn sàng adjust 
scope theo feedback của thầy/cô.

Nếu thầy/cô quan tâm và có thời gian, em xin được hẹn 1 buổi 
ngắn (15-30 phút, online hoặc trực tiếp) để trình bày chi tiết 
và lắng nghe góp ý. Em có thể linh hoạt theo lịch của thầy/cô.

Em xin chân thành cảm ơn thầy/cô đã dành thời gian xem xét đề 
xuất này.

Trân trọng,

[Họ tên A] - MSSV [xxx]
[Email] - [SĐT]

[Họ tên B] - MSSV [yyy]
[Email] - [SĐT]

---
Đính kèm:
- BudgetSwitch_Proposal.pdf (12 trang)
```

---

## Email 2: Follow-up if no reply after 5-7 days

**Subject:** `Re: [NCKH] Đề xuất nghiên cứu BudgetSwitch - Mong được thầy/cô phản hồi`

```
Kính thưa thầy/cô [Tên],

Em xin phép gửi follow-up về email đề xuất NCKH em đã gửi ngày 
[date], về đề tài BudgetSwitch (Adaptive Compute Allocation cho 
SLM Agents).

Em hoàn toàn hiểu thầy/cô bận rộn và đề xuất của em có thể chưa 
phù hợp với hướng nghiên cứu hiện tại. Em chỉ muốn xác nhận rằng 
thầy/cô đã nhận được email (đề phòng trường hợp vào spam) và 
hỏi xem thầy/cô có thể cho em biết về khả năng supervisorship 
trong khoảng 1-2 tuần tới được không?

Nhóm em sẽ bắt đầu execution phase từ tuần sau, và việc xác 
định advisor sớm sẽ giúp nhóm em điều chỉnh hướng project phù 
hợp với research focus của lab thầy/cô từ đầu, thay vì phải 
adjust giữa chừng.

Em xin chân thành cảm ơn thầy/cô.

Trân trọng,
[Họ tên A]
```

---

## Email 3: English version (for advisors who may prefer)

**Subject:** `[Research proposal] Adaptive compute allocation for SLM agents - undergraduate team seeking supervision`

```
Dear Professor [Last Name],

I'm [Full Name], a third-year AI Engineering student at [University]. 
Together with my teammate [Full Name B], I am preparing an 
undergraduate research project on adaptive compute allocation 
for tool-using small language model agents - a topic at the 
intersection of three active 2025-2026 trends: SLM-first agentic 
systems (NVIDIA position paper, June 2025), budget-aware test-time 
compute (BCAE, March 2026; AnytimeReasoner, May 2025), and 
agentic RL with verifiable rewards (ATLAS, March 2026; 
Fission-GRPO, January 2026).

The project, which we call BudgetSwitch, trains a Qwen2.5-3B 
tool-using agent to route each step of its trajectory to one 
of four compute modes (fast / think / think+search / defer-to-LLM) 
via Group Relative Policy Optimization with rewards drawn 
exclusively from deterministic step success and cost signals. 
The reward function uses no oracle mode labels and no 
LLM-teacher annotations, which we believe is methodologically 
important: the model discovers its mode-routing policy through 
GRPO group exploration alone. Vietnamese tool-use serves as 
the primary stress test benchmark.

After reviewing 66+ concurrent papers and iterating through 
several versions of the proposal, we have arrived at:

  - A methodologically clean approach (no oracle leakage in 
    the reward function, addressing a common pitfall in 
    RL-claimed work)
  - A 24-week execution roadmap calibrated to Kaggle T4 
    resources (~50 GPU-hours total, ~$100 API budget, both 
    funded by the team)
  - A pilot study designed to run before main GRPO training, 
    providing empirical motivation rather than relying solely 
    on cited results
  - Five fallback strategies covering various failure modes
  - Target submission: RIVF 2026 or SOICT 2026 (Springer CCIS, 
    Scopus-indexed)

I have attached the full proposal (12 pages) for your review.

We would be honored if you would consider supervising this 
project as Corresponding Author. We are seeking from you:

  - Periodic research direction guidance
  - Editorial feedback on paper drafts (primarily Phase 5, W17-20)
  - Formal endorsement as Corresponding Author on the submission

The team will assume full execution responsibility (coding, 
experiments, data construction, initial drafting). We commit 
to weekly progress updates and are open to scope adjustments 
based on your guidance.

If you are interested and your schedule permits, we would be 
grateful for a brief 15-30 minute meeting (online or in person, 
at your convenience) to discuss the proposal in more depth 
and receive your feedback.

Thank you very much for considering this request.

Best regards,

[Full Name A] - Student ID [xxx]
[Email] - [Phone]

[Full Name B] - Student ID [yyy]  
[Email] - [Phone]

---
Attached:
- BudgetSwitch_Proposal.pdf (12 pages)
```

---

## Strategy notes

**Do:**
- Send to all 2-3 candidates **on the same day** (parallel, not sequential)
- Attach the full PDF, not just text
- Use formal Vietnamese pronouns ("em" → "thầy/cô") consistent with academic culture
- Include both team members' contact info from the start
- Make the "what we need from you" section very specific (3 bullet points max)

**Don't:**
- Don't apologize excessively or undersell the project
- Don't ask for funding (we explicitly say "team tự chi trả")
- Don't promise the advisor extensive time commitment (they appreciate scope respect)
- Don't send the same exact email to multiple advisors who might compare notes — vary the opening paragraph slightly

**Choosing advisors (priority ranking):**

1. **Best:** PhD in AI/ML with recent (2024-2026) publications in NLP, RL, or agent systems
2. **Good:** PhD in Computer Science with general ML expertise and willingness to supervise NCKH
3. **Backup:** PhD in any relevant field with prior history of advising undergraduate research

**Red flags (skip these advisors):**

- Has never advised an undergraduate research project
- Hasn't published anything in the last 3 years
- Known to be very busy with administrative duties
- Research focus completely outside ML/NLP

---

## What to do after sending

**Day 5 evening:** Confirm all 3 emails sent. Document in `docs/advisor_outreach.md`.

**Day 6-7:** Continue W1 work. Don't refresh inbox every 5 minutes.

**Week 2 Day 3 (if no reply):** Send follow-up to top-priority advisor.

**Week 2 Day 5 (if still no reply):** Pivot to other professors via current course instructors who might introduce.

**Week 3 (if no reply from anyone):** Continue project independently. The README's research is sufficient to execute. Add advisor in W4-W6 if/when one becomes available; otherwise consider an external mentor (postdoc/PhD student) who might co-author.
