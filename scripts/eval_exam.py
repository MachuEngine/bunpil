#!/usr/bin/env python
"""출제 모듈 평가 스크립트
검색(Recall@5, MRR) / 문항 품질(LLM Judge) / 구조 유사도 Judge 신뢰도 / Judge 신뢰도.
검색 평가: 실제 standards/regulations 컬렉션 기반 골든셋 사용.
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

_STRUCTURE_GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "structure_golden.json")


def _load_structure_golden() -> list[dict]:
    with open(_STRUCTURE_GOLDEN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("entries", [])


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
            "user": '{"question":"세계인권선언(1948)에서 선언한 내용으로 옳지 않은 것은?","options":["①모든 사람은 생명권을 가진다","②모든 사람은 교육받을 권리를 가진다","③모든 사람은 특정 종교를 의무적으로 따라야 한다","④모든 사람은 법 앞에 평등하다"],"answer":"③"}',
            "assistant": '{"정답유일성": 5, "오답매력도": 5, "근거성": 5}',
        },
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


STRUCTURE_JUDGE_TPL = PromptTemplate(
    system=(
        "예시 문제와 새로 생성된 문항 세트를 비교해 구조적 유사도를 평가하세요. "
        "문항 개수 일치 여부는 판단하지 마세요 — 개수는 예시 문제와 무관하게 별도로 지정되므로 "
        "이 평가와는 무관합니다.\n"
        "다음 3가지를 JSON으로만 응답하세요.\n"
        "기준: type_ratio_score(유형 구성 비율 유사도, 0.0~1.0), "
        "difficulty_match(난이도 구성 부합, true/false), overall_score(종합 유사도, 0~5 정수)\n"
        '형식: {"type_ratio_score": 실수, "difficulty_match": true/false, "overall_score": 정수}'
    ),
    few_shots=[
        {
            "user": '{"예시_문제": "1문항(객관식)", "생성된_세트": [{"item_type":"객관식","difficulty":"중"}]}',
            "assistant": '{"type_ratio_score": 1.0, "difficulty_match": true, "overall_score": 5}',
        },
        {
            "user": '{"예시_문제": "3문항(객관식2+서술형1)", "생성된_세트": [{"item_type":"객관식","difficulty":"하"}]}',
            "assistant": '{"type_ratio_score": 0.5, "difficulty_match": false, "overall_score": 1}',
        },
    ],
)


@traceable(name="judge_structure_one", run_type="llm", metadata=_TRACE_META)
def judge_structure_one(entry: dict, llm) -> dict:
    content = json.dumps(
        {"예시_문제": entry["passage_text"], "생성된_세트": entry["generated_items"]},
        ensure_ascii=False,
    )
    messages = STRUCTURE_JUDGE_TPL.build(content)
    raw = _run_async(llm.generate(messages))
    try:
        s, e = raw.find("{"), raw.rfind("}") + 1
        scores = json.loads(raw[s:e]) if s >= 0 and e > s else {}
    except Exception:
        scores = {}
    return {
        "type_ratio_score": float(scores.get("type_ratio_score", 0.0)),
        "difficulty_match": bool(scores.get("difficulty_match", False)),
        "overall_score": int(scores.get("overall_score", 0)),
    }


@traceable(name="eval_structure_judge", run_type="chain", metadata=_TRACE_META)
def eval_structure_judge(golden: list, llm, limit: int = 8) -> dict:
    """STRUCTURE_GOLDEN의 고정된 (passage_text, generated_items) 쌍에 대해 LLM에게
    구조 유사도 판단만 다시 시켜 사람 라벨(human_label)과 대조한다.
    골든셋이 비어 있으면(라벨링 전) 빈 결과를 반환한다.
    count_match(문항 개수 일치)는 2026-07-09부로 이 비교에서 제외됨 — 개수는 예시
    문제와 무관하게 spec["num_items"]로 별도 지정되고, len(draft_items)==num_items로
    코드가 직접 검증하므로 LLM Judge/사람 라벨 대조 대상이 아님(structure_golden.json
    _schema.count_match_deprecated 참고)."""
    subset = golden[:limit]
    if not subset:
        return {"n": 0, "note": "STRUCTURE_GOLDEN이 비어 있습니다 — 라벨링 후 재실행하세요."}

    difficulty_match_hits = []
    overall_diffs = []

    for entry in subset:
        judge = judge_structure_one(entry, llm)
        human = entry["human_label"]

        difficulty_match_hits.append(judge["difficulty_match"] == human["difficulty_match"])
        overall_diffs.append(abs(judge["overall_score"] - human["overall_score"]))

    n = len(subset)
    return {
        "n": n,
        "difficulty_match_agreement": round(sum(difficulty_match_hits) / n, 3),
        "overall_score_mae": round(sum(overall_diffs) / n, 3),
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


def print_report(retrieval: dict, quality: dict, structure: dict, reliability: dict):
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

    print(f"\n[3] 구조 유사도 Judge 신뢰도 (STRUCTURE_GOLDEN, n={structure['n']})")
    if structure["n"] == 0:
        print(f"  {structure.get('note', '')}")
    else:
        print(f"  difficulty_match 일치율 : {structure['difficulty_match_agreement']:.3f}")
        print(f"  overall_score MAE       : {structure['overall_score_mae']:.3f}")

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

    # 3. 구조 유사도 Judge 신뢰도
    structure_golden = _load_structure_golden()
    print(f"\n3. 구조 유사도 Judge 신뢰도 검증 (STRUCTURE_GOLDEN {len(structure_golden)}개)...")
    structure_result = eval_structure_judge(structure_golden, judge_llm, limit=len(structure_golden) or 1)
    print(f"   n={structure_result['n']}")

    # 4. Judge 신뢰도
    print(f"\n4. Judge 신뢰도 검증 ({len(ITEM_GOLDEN)}개, 합성 사람 라벨)...")
    reliability_result = eval_judge_reliability(ITEM_GOLDEN, judge_llm, limit=len(ITEM_GOLDEN))
    print(f"   κ={reliability_result['cohen_kappa']:.3f}, ±1 일치율={reliability_result['agreement_within_1']:.3f}")

    # 리포트
    print_report(retrieval_result, quality_result, structure_result, reliability_result)


if __name__ == "__main__":
    main()
