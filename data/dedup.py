"""
data/dedup.py — Semantic dedup cho ViTool (v10.0).

VÌ SAO CẦN: LLM (Llama-70B) gen 500+ task HARD chắc chắn lặp Ý (vd nhiều biến
thể "đặt vé HCM→Hà Nội"). Dedup chuỗi y hệt KHÔNG bắt được. Trùng ngữ nghĩa làm:
(a) team phí công verify đồ lặp, (b) reviewer chê benchmark thiếu đa dạng.

CÁCH LÀM: embed query bằng multilingual-e5-base (đúng model repo đã dùng cho M2),
greedy loại cái có cosine-sim > threshold với cái đã giữ. Trong cùng (domain,
difficulty) so chặt hơn; khác nhóm nới lỏng.

FALLBACK: nếu chưa cài sentence-transformers, dùng char n-gram Jaccard (kém hơn
nhưng đủ để pipeline/test chạy offline). Production NÊN dùng e5.

Dùng:
    from data.dedup import dedup_trajectories
    kept = dedup_trajectories(trajs, threshold=0.93)
Self-test (offline): python data/dedup.py --selftest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.vitool.schema import load_trajectories, save_trajectories  # noqa: E402


# ============================================================
# Embedding backend
# ============================================================
_E5 = None
_E5_TRIED = False


def _get_e5():
    global _E5, _E5_TRIED
    if _E5_TRIED:
        return _E5
    _E5_TRIED = True
    try:
        from sentence_transformers import SentenceTransformer
        _E5 = SentenceTransformer("intfloat/multilingual-e5-base")
        print("  [dedup] dùng multilingual-e5-base")
    except Exception as e:
        print(f"  [dedup] ⚠️ không load được e5 ({type(e).__name__}); fallback Jaccard n-gram. "
              f"Production nên `pip install sentence-transformers`.")
        _E5 = None
    return _E5


def _embed(texts: list[str]):
    """Trả ma trận embedding (numpy) hoặc None nếu phải fallback."""
    model = _get_e5()
    if model is None:
        return None
    # e5 convention: prefix "query: "
    embs = model.encode([f"query: {t}" for t in texts], normalize_embeddings=True,
                        show_progress_bar=False)
    return embs


# ---- fallback: char n-gram Jaccard ----
def _ngrams(s: str, n: int = 3) -> set:
    s = " ".join(s.lower().split())
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))} or {s}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ============================================================
# Dedup core
# ============================================================
def _cosine_matrix(embs):
    import numpy as np
    return np.asarray(embs) @ np.asarray(embs).T  # đã normalize → dot = cosine


def dedup_trajectories(trajs: list, threshold: float = 0.93,
                       cross_group_threshold: float | None = None) -> list:
    """
    Greedy dedup. Giữ thứ tự gốc; loại cái sim > threshold với cái ĐÃ giữ.
    - cùng (domain, difficulty): dùng `threshold` (chặt).
    - khác nhóm: dùng `cross_group_threshold` (mặc định threshold + 0.04, nới hơn).
    """
    if cross_group_threshold is None:
        cross_group_threshold = min(0.99, threshold + 0.04)
    if not trajs:
        return []

    queries = [t.query for t in trajs]
    groups = [(t.domain, max((s.difficulty.value for s in t.steps), default="easy")) for t in trajs]
    embs = _embed(queries)

    kept_idx: list[int] = []
    if embs is not None:
        import numpy as np
        embs = np.asarray(embs)
        for i in range(len(trajs)):
            dup = False
            for j in kept_idx:
                sim = float(embs[i] @ embs[j])
                thr = threshold if groups[i] == groups[j] else cross_group_threshold
                if sim > thr:
                    dup = True
                    break
            if not dup:
                kept_idx.append(i)
    else:
        # fallback Jaccard
        grams = [_ngrams(q) for q in queries]
        for i in range(len(trajs)):
            dup = False
            for j in kept_idx:
                sim = _jaccard(grams[i], grams[j])
                thr = threshold if groups[i] == groups[j] else cross_group_threshold
                if sim > thr:
                    dup = True
                    break
            if not dup:
                kept_idx.append(i)

    kept = [trajs[i] for i in kept_idx]
    removed = len(trajs) - len(kept)
    print(f"  [dedup] giữ {len(kept)}/{len(trajs)} (loại {removed} near-duplicate, thr={threshold})")
    return kept


# ============================================================
# Self-test (offline, dùng Jaccard fallback)
# ============================================================
def _selftest():
    from data.vitool.schema import Trajectory, Step, Difficulty, Language

    def mk(tid, query, tool="search_flight", diff="hard", dom="travel"):
        return Trajectory(traj_id=tid, query=query, language=Language.VI,
                          steps=[Step(expected_tool=tool, expected_params={}, difficulty=Difficulty(diff))],
                          source="test", domain=dom)

    trajs = [
        mk("a", "Đặt vé máy bay từ HCM đi Hà Nội ngày 20/7"),
        mk("b", "Đặt vé máy bay từ HCM đi Hà Nội vào ngày 20 tháng 7"),   # ~trùng a
        mk("c", "Đặt vé máy bay TP HCM ra Hà Nội 20/07/2026"),            # ~trùng a
        mk("d", "Tìm khách sạn ở Đà Nẵng cho 2 người", tool="search_hotel"),  # khác hẳn
        mk("e", "Kiểm tra số dư tài khoản ngân hàng", tool="check_balance", dom="banking"),  # khác domain
    ]
    # fallback Jaccard: threshold thấp hơn cho phù hợp n-gram
    kept = dedup_trajectories(trajs, threshold=0.55)
    ids = {t.traj_id for t in kept}
    assert "a" in ids, "phải giữ a"
    assert "d" in ids and "e" in ids, "câu khác hẳn phải được giữ"
    # ít nhất 1 trong b/c bị loại vì trùng a
    assert not ({"b", "c"} <= ids), "ít nhất 1 near-dup của a phải bị loại"
    print(f"✅ Dedup self-test PASS — giữ {sorted(ids)} (loại near-dup của 'a')")


def main():
    ap = argparse.ArgumentParser(description="Semantic dedup ViTool")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--in", dest="in_path")
    ap.add_argument("--out", dest="out_path")
    ap.add_argument("--threshold", type=float, default=0.93)
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not (args.in_path and args.out_path):
        raise SystemExit("Cần --in và --out (hoặc --selftest).")
    trajs = load_trajectories(args.in_path)
    kept = dedup_trajectories(trajs, threshold=args.threshold)
    save_trajectories(kept, args.out_path)
    print(f"✅ Saved {len(kept)} → {args.out_path}")


if __name__ == "__main__":
    main()