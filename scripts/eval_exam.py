#!/usr/bin/env python
"""Phase 4: 출제 모듈 평가 스크립트
검색(Recall@5, MRR) / 문항 품질(LLM Judge) / 세트 제약 / Judge 신뢰도.
검색 평가: 실제 standards/regulations/past_exams 컬렉션 기반 골든셋 사용.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:1.5b")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db")

try:
    from langsmith import traceable
except ImportError:
    def traceable(**kwargs):
        def decorator(fn): return fn
        return decorator

from app.common.llm import PromptTemplate, get_judge_backend, get_llm_backend
from app.common.rag import BGEEmbedder, BGEReranker, RAGRetriever, RAGStore

_TRACE_META = {
    "model": os.getenv("OLLAMA_MODEL", "unknown"),
    "backend": os.getenv("LLM_BACKEND", "local"),
}

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
    # ── 추가 20건 (human_score: 사람이 미리 매긴 품질 점수 — LLM Judge와 일치율 검증용) ──
    # 5점 × 5건 — 정답 유일, 오답 매력적, 교육과정 근거 명확
    {
        "question": "세계인권선언(1948)에서 선언한 내용으로 옳지 않은 것은?",
        "options": ["①모든 사람은 생명권을 가진다", "②모든 사람은 교육받을 권리를 가진다", "③모든 사람은 특정 종교를 의무적으로 따라야 한다", "④모든 사람은 법 앞에 평등하다"],
        "answer": "③",
        "item_type": "객관식",
        "human_score": 5,
    },
    {
        "question": "비례대표제의 특징으로 옳은 것은?",
        "options": ["①지역 대표성이 강하다", "②사표 발생이 적어 다양한 정당이 의석을 획득할 수 있다", "③선거구가 작아 후보자와 유권자의 접촉이 쉽다", "④소수 정당이 의석을 얻기 어렵다"],
        "answer": "②",
        "item_type": "객관식",
        "human_score": 5,
    },
    {
        "question": "국제법에서 조약의 효력에 대한 설명으로 옳은 것은?",
        "options": ["①모든 국가에 자동으로 적용된다", "②서명한 당사국에만 구속력이 있다", "③국내법보다 항상 우선 적용된다", "④의회 비준 없이도 발효된다"],
        "answer": "②",
        "item_type": "객관식",
        "human_score": 5,
    },
    {
        "question": "누진세에 대한 설명으로 옳은 것은?",
        "options": ["①소득이 높을수록 세율이 낮아진다", "②모든 납세자에게 동일한 세율이 적용된다", "③소득이 높을수록 세율이 높아진다", "④소비 활동에만 부과된다"],
        "answer": "③",
        "item_type": "객관식",
        "human_score": 5,
    },
    {
        "question": "문화 상대주의적 관점에 대한 설명으로 옳은 것은?",
        "options": ["①자국 문화를 기준으로 타 문화를 평가한다", "②특정 문화가 다른 문화보다 우월하다고 본다", "③각 문화를 그 사회적 맥락에서 이해하고 존중한다", "④문화 간 우열을 명확히 구분할 수 있다고 본다"],
        "answer": "③",
        "item_type": "객관식",
        "human_score": 5,
    },
    # 4점 × 5건 — 대체로 좋으나 오답 매력도 또는 근거성이 약간 아쉬움
    {
        "question": "국가 간 상호 의존의 사례로 가장 적절한 것은?",
        "options": ["①한 나라가 모든 상품을 자국에서만 생산하는 것", "②한 나라의 금융 위기가 다른 나라 경제에 영향을 미치는 것", "③각국이 완전히 독립적인 경제 정책만 추구하는 것", "④한 나라가 외국과 일체의 무역을 하지 않는 것"],
        "answer": "②",
        "item_type": "객관식",
        "human_score": 4,
    },
    {
        "question": "우리나라 헌법에서 보장하는 사회권(사회적 기본권)의 사례로 적절한 것은?",
        "options": ["①종교의 자유", "②언론·출판의 자유", "③교육받을 권리", "④집회·결사의 자유"],
        "answer": "③",
        "item_type": "객관식",
        "human_score": 4,
    },
    {
        "question": "소선거구제의 특징으로 옳은 것은?",
        "options": ["①사표가 거의 발생하지 않는다", "②소수 정당이 의석을 얻기 쉽다", "③선거구가 작아 유권자가 후보자를 파악하기 쉽다", "④다양한 정치 세력이 고르게 대표된다"],
        "answer": "③",
        "item_type": "객관식",
        "human_score": 4,
    },
    {
        "question": "외부 불경제를 해소하기 위한 정부 정책으로 가장 적절한 것은?",
        "options": ["①오염 유발 기업에 보조금 지급", "②생산량 강제 증가 명령", "③오염 유발 기업에 환경세 부과", "④시장 가격 인하 규제"],
        "answer": "③",
        "item_type": "객관식",
        "human_score": 4,
    },
    {
        "question": "사회 이동 유형 중 세대 내 이동의 사례로 옳은 것은?",
        "options": ["①부모가 농부였는데 자녀가 의사가 된 경우", "②귀족 자녀가 귀족 지위를 그대로 유지한 경우", "③평사원이 같은 직장에서 임원으로 승진한 경우", "④중산층 부모를 둔 자녀가 중산층이 된 경우"],
        "answer": "③",
        "item_type": "객관식",
        "human_score": 4,
    },
    # 3점 × 4건 — 정답은 맞지만 오답이 너무 쉽거나 문장이 단순
    {
        "question": "법의 지배 원리에 대한 설명으로 옳은 것은?",
        "options": ["①지배자는 법의 적용을 받지 않는다", "②모든 사람은 법 앞에 평등하게 적용받는다", "③법은 권력자가 임의로 제정할 수 있다", "④국민 동의 없이도 법은 유효하다"],
        "answer": "②",
        "item_type": "객관식",
        "human_score": 3,
    },
    {
        "question": "시장 경제 체제의 특징이 아닌 것은?",
        "options": ["①사유재산 보장", "②자유로운 경쟁", "③국가의 생산수단 소유", "④가격 메커니즘에 의한 자원 배분"],
        "answer": "③",
        "item_type": "객관식",
        "human_score": 3,
    },
    {
        "question": "복지 국가의 역할로 볼 수 없는 것은?",
        "options": ["①사회보험 운영", "②공공부조 제공", "③사회 서비스 확대", "④기업의 이윤 극대화 지원"],
        "answer": "④",
        "item_type": "객관식",
        "human_score": 3,
    },
    {
        "question": "민주 선거의 4대 원칙에 해당하지 않는 것은?",
        "options": ["①보통 선거", "②평등 선거", "③간접 선거", "④비밀 선거"],
        "answer": "③",
        "item_type": "객관식",
        "human_score": 3,
    },
    # 2점 × 3건 — 정답이 모호하거나 복수 정답 가능
    {  # 모든 선지가 헌법 기본권에 해당 → 정답유일성 낮음
        "question": "다음 중 대한민국 헌법이 보장하는 기본권에 해당하는 것은?",
        "options": ["①신체의 자유", "②교육받을 권리", "③직업 선택의 자유", "④환경권"],
        "answer": "①",
        "item_type": "객관식",
        "human_score": 2,
    },
    {  # '긍정적 영향으로 보기 어려운 것'이 주관적 — ③ 외에도 논란 가능
        "question": "세계화가 개발도상국에 미치는 긍정적 영향으로 보기 어려운 것은?",
        "options": ["①선진 기술 도입 기회 확대", "②외국인 직접투자 유입 증가", "③전통 산업의 자생적 경쟁력 강화", "④경제 성장 가속화 가능성"],
        "answer": "③",
        "item_type": "객관식",
        "human_score": 2,
    },
    {  # ④가 정답이나 산업혁명도 민주주의 발전 요인으로 볼 수 있어 해석 여지 있음
        "question": "민주주의 발전에 직접적으로 기여한 역사적 사건으로 보기 어려운 것은?",
        "options": ["①영국 마그나카르타(1215)", "②프랑스 대혁명(1789)", "③미국 독립선언(1776)", "④산업혁명에 의한 생산성 향상"],
        "answer": "④",
        "item_type": "객관식",
        "human_score": 2,
    },
    # 1점 × 3건 — 선지 중복·교과 무관·무의미 문항
    {  # 선지가 무의미하고 문항과 무관
        "question": "다음 중 인권의 특징으로 옳은 것은?",
        "options": ["①인권은 인권이다", "②인권은 인권이 아니다", "③인권은 중요하다", "④왕정"],
        "answer": "①",
        "item_type": "객관식",
        "human_score": 1,
    },
    {  # 사회 교과 무관 내용
        "question": "물의 끓는점은 몇 °C인가?",
        "options": ["①50°C", "②100°C", "③150°C", "④200°C"],
        "answer": "②",
        "item_type": "객관식",
        "human_score": 1,
    },
    {  # 오답이 너무 명확해 변별력 없음
        "question": "사회란 무엇인가?",
        "options": ["①사람들이 관계를 맺으며 함께 사는 집단", "②사람이 전혀 없는 공간", "③동물만 존재하는 환경", "④아무것도 없는 상태"],
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

@traceable(name="eval_retrieval", run_type="chain", metadata=_TRACE_META)
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


@traceable(name="judge_one", run_type="llm", metadata=_TRACE_META)
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


@traceable(name="eval_item_quality", run_type="chain", metadata=_TRACE_META)
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


@traceable(name="eval_judge_reliability", run_type="chain", metadata=_TRACE_META)
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

@traceable(name="eval_exam_run", run_type="chain", metadata=_TRACE_META)
def main():
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        print("LangSmith 트레이싱: 활성화됨")
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
    _dist = {}
    for it in ITEM_GOLDEN:
        s = it["human_score"]
        _dist[s] = _dist.get(s, 0) + 1
    print(f"\nhuman_score 분포: { {k: _dist[k] for k in sorted(_dist)} } (n={len(ITEM_GOLDEN)})")
    llm = get_llm_backend()
    judge_llm = get_judge_backend()
    _gen_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    _judge_model = os.getenv("OLLAMA_JUDGE_MODEL")
    _fallback = "(폴백)" if not _judge_model else ""
    print(f"[LLM] 생성: {_gen_model} | Judge: {_judge_model or _gen_model} {_fallback}".rstrip())
    print(f"\n2. 문항 품질 LLM Judge ({len(ITEM_GOLDEN)}개)...")
    quality_result = eval_item_quality(ITEM_GOLDEN, judge_llm, limit=len(ITEM_GOLDEN))
    print(f"   종합평균={quality_result['avg_overall']:.2f}/5, 합격률={quality_result['pass_rate']*100:.0f}%")

    # 3. 세트 제약 검증
    print("\n3. 세트 제약 함수 검증...")
    constraints_result = eval_set_constraints(SYNTHETIC_SET, SPEC)
    print(f"   전체 통과: {constraints_result['all_pass']}")

    # 4. Judge 신뢰도
    print(f"\n4. Judge 신뢰도 검증 ({len(ITEM_GOLDEN)}개, 합성 사람 라벨)...")
    reliability_result = eval_judge_reliability(ITEM_GOLDEN, judge_llm, limit=len(ITEM_GOLDEN))
    print(f"   κ={reliability_result['cohen_kappa']:.3f}, ±1 일치율={reliability_result['agreement_within_1']:.3f}")

    # 리포트
    print_report(retrieval_result, quality_result, constraints_result, reliability_result)


if __name__ == "__main__":
    main()
