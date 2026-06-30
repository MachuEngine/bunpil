#!/usr/bin/env python
"""Phase 4: 출제 모듈 평가 스크립트
검색(Recall@5, MRR) / 문항 품질(LLM Judge) / 세트 제약 / Judge 신뢰도.
검색 평가: 실제 standards/regulations/past_exams 컬렉션 기반 골든셋 사용.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:1.5b")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db")

from app.common.llm import PromptTemplate, get_llm_backend
from app.common.rag import BGEEmbedder, BGEReranker, RAGRetriever, RAGStore

_GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "retrieval_golden_final.json")

def _load_retrieval_golden() -> list[dict]:
    with open(_GOLDEN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [item for item in data if item.get("reviewed")]

# ── 합성 문항 골든셋 (품질 평가용) ────────────────────────────────
ITEM_GOLDEN = [
    {
        "question": "민주주의의 핵심 원리로 옳은 것은?",
        "options": ["①국민 주권", "②군주 주권", "③귀족 통치", "④왕정 복고"],
        "answer": "①",
        "item_type": "객관식",
        "human_score": 4,
    },
    {
        "question": "권력 분립의 목적은?",
        "options": ["①권력 집중", "②효율 증대", "③견제와 균형", "④신속한 결정"],
        "answer": "③",
        "item_type": "객관식",
        "human_score": 4,
    },
    {
        "question": "시장 실패의 원인이 아닌 것은?",
        "options": ["①외부효과", "②공공재", "③완전경쟁", "④독과점"],
        "answer": "③",
        "item_type": "객관식",
        "human_score": 5,
    },
    {
        "question": "가격의 역할은?",
        "options": ["①정부 명령", "②자원 배분 신호", "③생산 금지", "④소비 제한"],
        "answer": "②",
        "item_type": "객관식",
        "human_score": 3,
    },
    {
        "question": "사회 보험이 보장하는 위험이 아닌 것은?",
        "options": ["①질병", "②실업", "③노령", "④사치"],
        "answer": "④",
        "item_type": "객관식",
        "human_score": 4,
    },
    {
        "question": "복지 정책의 주요 목표는?",
        "options": ["①경제 성장", "②빈곤 감소", "③수출 증대", "④군사력 강화"],
        "answer": "②",
        "item_type": "객관식",
        "human_score": 3,
    },
    {
        "question": "세계화의 부정적 영향은?",
        "options": ["①무역 증가", "②기술 이전", "③소득 격차 확대", "④분업 촉진"],
        "answer": "③",
        "item_type": "객관식",
        "human_score": 4,
    },
    {
        "question": "헌법에서 규정하는 대한민국의 국체는?",
        "options": ["①왕국", "②제국", "③민주공화국", "④연방국"],
        "answer": "③",
        "item_type": "객관식",
        "human_score": 5,
    },
    {  # 품질 낮은 문항 예시
        "question": "다음 중 민주주의와 관련 있는 것은?",
        "options": ["①민주주의", "②민주주의", "③민주주의", "④왕정"],
        "answer": "①",
        "item_type": "객관식",
        "human_score": 1,
    },
    {  # 품질 낮은 문항 예시
        "question": "경제는?",
        "options": ["①좋다", "②나쁘다", "③보통이다", "④모르겠다"],
        "answer": "①",
        "item_type": "객관식",
        "human_score": 1,
    },
]

# ── 세트 제약 검증용 합성 세트 ─────────────────────────────────────
SPEC = {
    "num_items": 5,
    "type_dist": {"객관식": 4, "서술형": 1},
    "difficulty_dist": {"상": 1, "중": 2, "하": 2},
    "standards": ["민주주의 원리 이해", "시장 경제 원리 이해"],
}

SYNTHETIC_SET = [
    {"item_type": "객관식", "difficulty": "상", "standard": "민주주의 원리 이해", "is_duplicate": False, "status": "approved"},
    {"item_type": "객관식", "difficulty": "중", "standard": "시장 경제 원리 이해", "is_duplicate": False, "status": "approved"},
    {"item_type": "객관식", "difficulty": "중", "standard": "민주주의 원리 이해", "is_duplicate": False, "status": "approved"},
    {"item_type": "객관식", "difficulty": "하", "standard": "시장 경제 원리 이해", "is_duplicate": False, "status": "approved"},
    {"item_type": "서술형", "difficulty": "하", "standard": "민주주의 원리 이해", "is_duplicate": False, "status": "approved"},
]

# ── 유틸리티 ────────────────────────────────────────────────────────

def _run_async(coro):
    import asyncio, concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result(timeout=300)


def cohen_kappa(human: list, llm: list, threshold: int = 3) -> float:
    """이진 Cohen's kappa: score >= threshold → positive."""
    n = len(human)
    h = [1 if x >= threshold else 0 for x in human]
    l = [1 if x >= threshold else 0 for x in llm]
    po = sum(hi == li for hi, li in zip(h, l)) / n
    ph = sum(h) / n
    pl = sum(l) / n
    pe = ph * pl + (1 - ph) * (1 - pl)
    return (po - pe) / (1 - pe) if pe < 1.0 else 1.0


# ── 평가 함수 ────────────────────────────────────────────────────────

def eval_retrieval(retriever: RAGRetriever, golden: list) -> dict:
    """Recall@5, MRR 계산. chunk_preview substring 매칭으로 정답 판정."""
    hits_at_5 = 0
    rr_sum = 0.0

    for item in golden:
        col = item["source_collection"]
        results = retriever.retrieve(item["query"], col, top_k=5, n_candidates=20)
        preview = item["chunk_preview"].strip()

        found_rank = None
        for rank, r in enumerate(results, 1):
            if preview and preview[:80] in r["text"]:
                found_rank = rank
                break

        if found_rank is not None:
            hits_at_5 += 1
            rr_sum += 1.0 / found_rank

    n = len(golden)
    return {
        "recall_at_5": hits_at_5 / n,
        "mrr": rr_sum / n,
        "n": n,
    }


JUDGE_TPL = PromptTemplate(
    system=(
        "문항을 3가지 기준으로 평가하세요. 각 점수는 1-5 정수, JSON으로만 응답하세요.\n"
        "기준: 정답유일성(오직 하나의 정답), 오답매력도(오답 선지가 그럴듯함), 근거성(교육과정 기반)\n"
        '형식: {"정답유일성": 정수, "오답매력도": 정수, "근거성": 정수}'
    ),
    few_shots=[
        {
            "user": '{"question":"민주주의 핵심 원리는?","options":["①국민주권","②왕정","③독재","④귀족"],"answer":"①"}',
            "assistant": '{"정답유일성": 5, "오답매력도": 3, "근거성": 4}',
        },
        {
            "user": '{"question":"경제는?","options":["①좋다","②나쁘다","③보통","④모름"],"answer":"①"}',
            "assistant": '{"정답유일성": 2, "오답매력도": 1, "근거성": 1}',
        },
    ],
)


def judge_one(item: dict, llm) -> dict:
    item_str = json.dumps(
        {"question": item["question"], "options": item.get("options", []), "answer": item.get("answer", "")},
        ensure_ascii=False,
    )
    messages = JUDGE_TPL.build(item_str)
    raw = _run_async(llm.generate(messages))
    try:
        s, e = raw.find("{"), raw.rfind("}") + 1
        scores = json.loads(raw[s:e]) if s >= 0 and e > s else {}
    except Exception:
        scores = {}
    return {
        "정답유일성": int(scores.get("정답유일성", 3)),
        "오답매력도": int(scores.get("오답매력도", 3)),
        "근거성": int(scores.get("근거성", 3)),
        "overall": round(
            (int(scores.get("정답유일성", 3)) + int(scores.get("오답매력도", 3)) + int(scores.get("근거성", 3))) / 3,
            2,
        ),
    }


def eval_item_quality(items: list, llm, limit: int = 8) -> dict:
    """LLM Judge로 문항 품질 평가. limit: LLM 호출 수 제한."""
    subset = items[:limit]
    results = []
    for item in subset:
        scores = judge_one(item, llm)
        results.append(scores)

    def avg(key):
        return round(sum(r[key] for r in results) / len(results), 2)

    return {
        "n": len(results),
        "avg_정답유일성": avg("정답유일성"),
        "avg_오답매력도": avg("오답매력도"),
        "avg_근거성": avg("근거성"),
        "avg_overall": avg("overall"),
        "pass_rate": round(sum(1 for r in results if r["overall"] >= 4.0) / len(results), 2),
    }


def eval_set_constraints(items: list, spec: dict) -> dict:
    """유형·난이도 분포, 커버리지, 중복률 함수 검증."""
    approved = [it for it in items if it.get("status") == "approved"]

    type_counts: dict = {}
    for it in approved:
        t = it.get("item_type", "")
        type_counts[t] = type_counts.get(t, 0) + 1

    diff_counts: dict = {}
    for it in approved:
        d = it.get("difficulty", "")
        diff_counts[d] = diff_counts.get(d, 0) + 1

    coverage_map = {s: 0 for s in spec.get("standards", [])}
    for it in approved:
        s = it.get("standard", "")
        if s in coverage_map:
            coverage_map[s] += 1

    dup_count = sum(1 for it in approved if it.get("is_duplicate"))

    type_ok = all(type_counts.get(k, 0) >= v for k, v in spec["type_dist"].items())
    diff_ok = all(diff_counts.get(k, 0) >= v for k, v in spec["difficulty_dist"].items())
    coverage_ok = all(v > 0 for v in coverage_map.values()) if coverage_map else True

    return {
        "type_dist": type_counts,
        "type_ok": type_ok,
        "diff_dist": diff_counts,
        "diff_ok": diff_ok,
        "coverage_map": coverage_map,
        "coverage_ok": coverage_ok,
        "dup_count": dup_count,
        "dup_rate": round(dup_count / max(len(approved), 1), 2),
        "total_approved": len(approved),
        "all_pass": type_ok and diff_ok and coverage_ok,
    }


def eval_judge_reliability(items_with_human: list, llm, limit: int = 8) -> dict:
    """LLM Judge 점수와 사람 라벨 일치율·kappa 측정."""
    subset = items_with_human[:limit]
    human_scores = []
    llm_scores = []

    for item in subset:
        h = item["human_score"]
        scores = judge_one(item, llm)
        l = round(scores["overall"])
        human_scores.append(h)
        llm_scores.append(l)

    agree = sum(h == l for h, l in zip(human_scores, llm_scores)) / len(human_scores)
    agree_pm1 = sum(abs(h - l) <= 1 for h, l in zip(human_scores, llm_scores)) / len(human_scores)
    kappa = cohen_kappa(human_scores, llm_scores, threshold=3)

    return {
        "n": len(subset),
        "exact_agreement": round(agree, 3),
        "agreement_within_1": round(agree_pm1, 3),
        "cohen_kappa": round(kappa, 3),
        "human_avg": round(sum(human_scores) / len(human_scores), 2),
        "llm_avg": round(sum(llm_scores) / len(llm_scores), 2),
    }


# ── 리포트 출력 ─────────────────────────────────────────────────────

def check(ok: bool) -> str:
    return "✓" if ok else "✗"


def print_report(retrieval: dict, quality: dict, constraints: dict, reliability: dict):
    print("\n" + "=" * 55)
    print("  분필 출제 모듈 평가 리포트")
    print("=" * 55)

    print(f"\n[1] 검색 성능 (n={retrieval['n']})")
    r5 = retrieval["recall_at_5"]
    mrr = retrieval["mrr"]
    print(f"  Recall@5 : {r5:.3f}  {check(r5 >= 0.8)} (기준 ≥ 0.8)")
    print(f"  MRR      : {mrr:.3f}  {check(mrr >= 0.6)} (참고값)")

    print(f"\n[2] 문항 품질 LLM Judge (n={quality['n']}, 5점 척도)")
    print(f"  정답유일성  : {quality['avg_정답유일성']:.2f}")
    print(f"  오답매력도  : {quality['avg_오답매력도']:.2f}")
    print(f"  근거성      : {quality['avg_근거성']:.2f}")
    print(f"  종합평균    : {quality['avg_overall']:.2f}  {check(quality['avg_overall'] >= 4.0)} (기준 ≥ 4.0)")
    print(f"  합격률(≥4.0): {quality['pass_rate']*100:.0f}%")

    print(f"\n[3] 세트 제약 검증")
    print(f"  유형 분포   : {constraints['type_dist']}  {check(constraints['type_ok'])}")
    print(f"  난이도 분포 : {constraints['diff_dist']}  {check(constraints['diff_ok'])}")
    print(f"  커버리지    : {constraints['coverage_map']}  {check(constraints['coverage_ok'])}")
    print(f"  중복률      : {constraints['dup_rate']*100:.0f}%  {check(constraints['dup_rate'] == 0.0)}")
    print(f"  전체 통과   : {check(constraints['all_pass'])}")

    print(f"\n[4] Judge 신뢰도 (n={reliability['n']})")
    k = reliability["cohen_kappa"]
    agree = reliability["agreement_within_1"]
    print(f"  정확 일치율 : {reliability['exact_agreement']:.3f}")
    print(f"  ±1 일치율   : {agree:.3f}  {check(agree >= 0.7)} (기준 ≥ 0.7)")
    print(f"  Cohen κ     : {k:.3f}  {check(k >= 0.4)} (기준 ≥ 0.4)")
    print(f"  사람 평균   : {reliability['human_avg']:.2f}")
    print(f"  LLM 평균    : {reliability['llm_avg']:.2f}")

    print("\n" + "=" * 55)
    note = "※ 개발 모델(1.5b)은 품질·Judge 수치가 낮을 수 있음. 7B(RunPod)에서 재평가 권장."
    print(note)
    print("=" * 55)


# ── 메인 ────────────────────────────────────────────────────────────

def main():
    print("=== Phase 4: 출제 모듈 평가 시작 ===\n")

    store = RAGStore()
    embedder = BGEEmbedder()
    reranker = BGEReranker()
    retriever = RAGRetriever(store, embedder, reranker)

    # 1. 검색 평가
    golden = _load_retrieval_golden()
    print(f"1. 검색 평가 (Recall@5, MRR) — 골든셋 {len(golden)}개...")
    retrieval_result = eval_retrieval(retriever, golden)
    print(f"   Recall@5={retrieval_result['recall_at_5']:.3f}, MRR={retrieval_result['mrr']:.3f}")

    # 2. 문항 품질 평가
    print("\n2. 문항 품질 LLM Judge (8개 샘플)...")
    llm = get_llm_backend()
    quality_result = eval_item_quality(ITEM_GOLDEN, llm, limit=8)
    print(f"   종합평균={quality_result['avg_overall']:.2f}/5, 합격률={quality_result['pass_rate']*100:.0f}%")

    # 3. 세트 제약 검증
    print("\n3. 세트 제약 함수 검증...")
    constraints_result = eval_set_constraints(SYNTHETIC_SET, SPEC)
    print(f"   전체 통과: {constraints_result['all_pass']}")

    # 4. Judge 신뢰도
    print("\n4. Judge 신뢰도 검증 (8개 샘플, 합성 사람 라벨)...")
    reliability_result = eval_judge_reliability(ITEM_GOLDEN, llm, limit=8)
    print(f"   κ={reliability_result['cohen_kappa']:.3f}, ±1 일치율={reliability_result['agreement_within_1']:.3f}")

    # 리포트
    print_report(retrieval_result, quality_result, constraints_result, reliability_result)


if __name__ == "__main__":
    main()
