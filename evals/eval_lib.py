"""출제 모듈 평가 공용 라이브러리 — golden 로더, judge 템플릿/함수, 평가 함수.

`eval_exam.py`가 실행 스크립트 겸 이 로직들의 정의처였는데, `eval_record.py`/
`eval_example_retrieval.py`/`compare_models.py`/`compare_distractor_quality.py`/
`compare_judge_models.py` 5곳이 거기서 직접 import해 쓰면서 "실행 스크립트인데
사실상 공용 모듈" 상태가 됐다. 여기로 분리해 eval_exam.py는 그 결과를 조합해
리포트를 출력하는 진입점 역할만 하도록 축소한다(2026-07-18).

주의: load_dotenv()/env 기본값 세팅/init_langsmith_project()는 각 호출 스크립트가
이 모듈을 import하기 전에 이미 실행한다고 가정한다(기존 스크립트들의 관례 유지) —
이 모듈 자체에서는 다시 호출하지 않는다.
"""
import asyncio
import concurrent.futures
import json
import os

try:
    from langsmith import traceable
except ImportError:
    def traceable(**kwargs):
        def decorator(fn): return fn
        return decorator

from app.common.llm import PromptTemplate
from app.common.rag import RAGRetriever
from app.modules.exam.judge import STRUCTURE_JUDGE_TPL, judge_structure  # noqa: F401 (재노출)

_TRACE_META = {
    "model": os.getenv("OLLAMA_MODEL", "unknown"),
    "backend": os.getenv("LLM_BACKEND", "local"),
}

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "golden")
_GOLDEN_PATH = os.path.join(_GOLDEN_DIR, "retrieval_golden_final.json")
_ITEM_GOLDEN_PATH = os.path.join(_GOLDEN_DIR, "item_golden.json")
_STRUCTURE_GOLDEN_PATH = os.path.join(_GOLDEN_DIR, "structure_golden.json")


# ── golden 로더 ─────────────────────────────────────────────────────

def _load_retrieval_golden() -> list[dict]:
    with open(_GOLDEN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [item for item in data if item.get("reviewed")]


def _load_item_golden() -> list[dict]:
    with open(_ITEM_GOLDEN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("entries", [])


ITEM_GOLDEN = _load_item_golden()


def _load_structure_golden() -> list[dict]:
    """human_label이 채워진(라벨링 완료된) 엔트리만 반환한다.
    라벨링 전 엔트리(human_label: null)는 retrieval_golden의 reviewed:false와 같은 개념으로 스킵."""
    with open(_STRUCTURE_GOLDEN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [e for e in data.get("entries", []) if e.get("human_label")]


# ── 유틸리티 ────────────────────────────────────────────────────────

def _run_async(coro):
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


# ── 검색 평가 ────────────────────────────────────────────────────────

@traceable(name="eval_retrieval", run_type="chain", metadata=_TRACE_META)
def eval_retrieval(retriever: RAGRetriever, golden: list) -> dict:
    """Recall@5, MRR 계산. chunk_preview substring 매칭으로 정답 판정.

    n_candidates를 명시하지 않고 `retrieve()`의 기본값을 따른다(2026-08-03 변경) —
    이전엔 20을 하드코딩했는데, 기본값이 10으로 바뀌면서 eval이 프로덕션과 다른
    설정을 측정하게 되는 검증-배포 불일치가 생겼다. 같은 종류의 불일치를
    2026-07-23 judge에서 이미 한 번 겪었으므로(bunpil_roadmap.md) 반복하지 않는다.
    """
    hits_at_5 = 0
    rr_sum = 0.0

    for item in golden:
        col = item["source_collection"]
        results = retriever.retrieve(item["query"], col, top_k=5)
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


# ── 문항 품질 Judge ──────────────────────────────────────────────────

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


@traceable(name="score_items", run_type="chain", metadata=_TRACE_META)
def score_items(items: list, llm, limit: int | None = None) -> list[dict]:
    """items 각각을 judge_one()으로 정확히 1회만 채점.

    문항 품질 평균(eval_item_quality)과 사람 라벨 대비 신뢰도(eval_judge_reliability)가
    이전에는 같은 골든셋을 각자 judge_one()으로 다시 채점해 LLM 호출이 중복됐다
    (ITEM_GOLDEN 30개 기준 실질 60회). 이 함수로 한 번만 채점한 결과를 두 함수가
    공유한다.
    """
    subset = items[:limit] if limit is not None else items
    return [{"item": item, "scores": judge_one(item, llm)} for item in subset]


def eval_item_quality(scored: list[dict]) -> dict:
    """score_items() 결과로 문항 품질 평균/합격률 계산 (LLM 재호출 없음)."""
    results = [s["scores"] for s in scored]

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


def eval_judge_reliability(scored: list[dict]) -> dict:
    """score_items() 결과와 human_score를 비교해 일치율·kappa 계산 (LLM 재호출 없음)."""
    human_scores = [s["item"]["human_score"] for s in scored]
    llm_scores = [round(s["scores"]["overall"]) for s in scored]

    agree = sum(h == l for h, l in zip(human_scores, llm_scores)) / len(human_scores)
    agree_pm1 = sum(abs(h - l) <= 1 for h, l in zip(human_scores, llm_scores)) / len(human_scores)
    kappa = cohen_kappa(human_scores, llm_scores, threshold=3)

    return {
        "n": len(scored),
        "exact_agreement": round(agree, 3),
        "agreement_within_1": round(agree_pm1, 3),
        "cohen_kappa": round(kappa, 3),
        "human_avg": round(sum(human_scores) / len(human_scores), 2),
        "llm_avg": round(sum(llm_scores) / len(llm_scores), 2),
    }


# ── 구조 유사도 Judge ────────────────────────────────────────────────
# STRUCTURE_JUDGE_TPL·채점 로직은 app/modules/exam/judge.py로 이동(2026-07-23) —
# 런타임 judge_node와 오프라인 eval이 완전히 같은 함수를 공유하도록 통합(검증-배포
# 일치). 여기서는 @traceable 데코레이터만 씌운 얇은 wrapper로 재노출한다.

@traceable(name="judge_structure_one", run_type="llm", metadata=_TRACE_META)
def judge_structure_one(entry: dict, llm) -> dict:
    return judge_structure(entry["passage_text"], entry["generated_items"], llm)


@traceable(name="score_structure", run_type="chain", metadata=_TRACE_META)
def score_structure(golden: list, llm, limit: int | None = None) -> list[dict]:
    """golden 각 항목을 judge_structure_one()으로 정확히 1회만 채점.

    score_items()와 같은 이유로 분리했다(2026-08-04). 이전에는 리포트용
    eval_structure_judge()와 LangSmith Experiments의 evaluator가 같은 골든셋을
    각자 채점해 LLM 호출이 2배였다(STRUCTURE_GOLDEN 45개 기준 실질 90회).
    이 함수로 한 번만 채점한 결과를 두 소비자가 공유한다.
    """
    subset = golden[:limit] if limit is not None else golden
    return [{"entry": entry, "judge": judge_structure_one(entry, llm)} for entry in subset]


def eval_structure_judge(scored: list[dict]) -> dict:
    """score_structure() 결과와 human_label을 대조해 일치율·MAE 계산 (LLM 재호출 없음).

    STRUCTURE_GOLDEN의 고정된 (passage_text, generated_items) 쌍에 대한 LLM 판단을
    사람 라벨(human_label)과 비교한다. 골든셋이 비어 있으면(라벨링 전) 빈 결과를 반환한다.
    count_match(문항 개수 일치)는 2026-07-09부로 이 비교에서 제외됨 — 개수는 예시
    문제와 무관하게 spec["num_items"]로 별도 지정되고, len(draft_items)==num_items로
    코드가 직접 검증하므로 LLM Judge/사람 라벨 대조 대상이 아님(structure_golden.json
    _schema.count_match_deprecated 참고)."""
    if not scored:
        return {"n": 0, "note": "STRUCTURE_GOLDEN이 비어 있습니다 — 라벨링 후 재실행하세요."}

    difficulty_match_hits = []
    overall_diffs = []
    count_match_code_hits = []  # LLM/사람 대조 대상 아님 — 골든셋 엔트리 자체의 데이터 정합성 확인용

    for s in scored:
        entry, judge = s["entry"], s["judge"]
        human = entry["human_label"]

        difficulty_match_hits.append(judge["difficulty_match"] == human["difficulty_match"])
        overall_diffs.append(abs(judge["overall_score"] - human["overall_score"]))
        count_match_code_hits.append(len(entry.get("generated_items", [])) == entry.get("num_items"))

    n = len(scored)
    return {
        "n": n,
        "count_match_code_rate": round(sum(count_match_code_hits) / n, 3),
        "difficulty_match_agreement": round(sum(difficulty_match_hits) / n, 3),
        "overall_score_mae": round(sum(overall_diffs) / n, 3),
    }
