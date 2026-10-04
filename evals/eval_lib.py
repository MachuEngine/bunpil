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
from app.common.llm.backends.openrouter import OpenRouterBackend
from app.common.rag import RAGRetriever
from app.modules.exam.judge import STRUCTURE_JUDGE_TPL, judge_structure  # noqa: F401 (재노출)
from evals.stats import cluster_bootstrap_ci, mean_ci, weighted_kappa

_TRACE_META = {
    "model": os.getenv("OLLAMA_MODEL", "unknown"),
    "backend": os.getenv("LLM_BACKEND", "local"),
}

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "golden")
_GOLDEN_PATH = os.path.join(_GOLDEN_DIR, "retrieval_golden_final.json")
_ITEM_GOLDEN_PATH = os.path.join(_GOLDEN_DIR, "item_golden.json")
_STRUCTURE_GOLDEN_PATH = os.path.join(_GOLDEN_DIR, "structure_golden.json")
_ITEM_QUALITY_GOLDEN_PATH = os.path.join(_GOLDEN_DIR, "item_quality_golden.json")
_QUALITY_CRITERIA = ("정답유일성", "오답매력도", "근거성")


def get_offline_judge_backend() -> OpenRouterBackend:
    """오프라인 정기 평가(eval_exam.py) 전용 Judge 백엔드 — 런타임 게이트의
    `get_judge_backend()`(.env JUDGE_BACKEND)와 **완전히 분리**한다(2026-10 결정).

    이유: 오프라인 평가가 "런타임 Judge를 믿어도 되는가"를 검증하는 역할인데, 같은 모델로
    검증하면 자기 자신을 신뢰도 기준으로 삼는 셈이 된다. 생성 모델과 계열이 다른 별도
    Judge(기본 anthropic/claude-sonnet-5.5, OpenRouter)로 재채점했을 때 사람 라벨 대비 κ가
    더 높게 나왔다(evals/eval_item_quality_runs.py reliability 비교 결과,
    data/golden/_judge_selection_round2.json). OPENROUTER_API_KEY 미설정 시 이 생성자 자체는
    실패하지 않고, 첫 generate() 호출 시 ChatOpenRouterBackend 생성 시점에 실패한다
    (fail-fast, get_judge_backend()와 같은 철학 — 신뢰도 수치를 내기 전에 멈춘다)."""
    model = os.getenv("OFFLINE_JUDGE_MODEL", "anthropic/claude-sonnet-5.5")
    return OpenRouterBackend(model=model)


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


@traceable(name="eval_retrieval_per_query", run_type="chain", metadata=_TRACE_META)
def eval_retrieval_per_query(retriever: RAGRetriever, golden: list) -> list[dict]:
    """eval_retrieval()과 동일한 판정 로직으로, 쿼리 단위 hit(0/1)·rr을 그대로 노출한다.

    eval_retrieval()의 집계(recall_at_5·mrr·n)만으로는 질의 단위 부트스트랩 CI를 낼 수
    없어 추가했다 — eval_retrieval()의 기존 반환값은 바꾸지 않고, CI가 필요한 소비자
    (eval_exam.py)만 이 함수를 따로 호출한다. 쿼리 원문 대신 id만 남긴다(최소 노출)."""
    rows = []
    for item in golden:
        col = item["source_collection"]
        results = retriever.retrieve(item["query"], col, top_k=5)
        preview = item["chunk_preview"].strip()

        found_rank = None
        for rank, r in enumerate(results, 1):
            if preview and preview[:80] in r["text"]:
                found_rank = rank
                break

        rows.append({
            "id": item["id"],
            "hit": 1 if found_rank is not None else 0,
            "rr": (1.0 / found_rank) if found_rank is not None else 0.0,
        })
    return rows


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
    item_dict = {"question": item["question"], "options": item.get("options", []), "answer": item.get("answer", "")}
    # 2026-10-03: stimulus(<보기>·자료 제시문)가 있으면 포함한다 — 합답형·자료형 문항은
    # <보기>를 봐야 정답유일성을 판단할 수 있다. 없거나 빈 문자열이면 기존과 완전히 같은
    # JSON(ITEM_GOLDEN이 이 조건으로 이미 측정돼 있어 바꾸지 않음).
    stimulus = item.get("stimulus")
    if stimulus:
        item_dict["stimulus"] = stimulus
    item_str = json.dumps(item_dict, ensure_ascii=False)
    messages = JUDGE_TPL.build(item_str)
    raw = _run_async(llm.generate(messages))
    try:
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s >= 0 and e > s:
            scores = json.loads(raw[s:e])
            parse_failed = False
        else:
            scores = {}
            parse_failed = True
    except Exception:
        scores = {}
        parse_failed = True
    return {
        "정답유일성": int(scores.get("정답유일성", 3)),
        "오답매력도": int(scores.get("오답매력도", 3)),
        "근거성": int(scores.get("근거성", 3)),
        "overall": round(
            (int(scores.get("정답유일성", 3)) + int(scores.get("오답매력도", 3)) + int(scores.get("근거성", 3))) / 3,
            2,
        ),
        "parse_failed": parse_failed,
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
    """score_items() 결과로 문항 품질 평균/합격률/저품질 비율 계산 (LLM 재호출 없음).

    2026-08-08: 기준별 **저품질 비율**(≤3점)을 함께 낸다. 평균만 보면 실제로 무엇을
    고쳐야 하는지가 가려지기 때문이다 — 실측 n=62에서 오답매력도 분포가
    {2:4, 3:5, 4:51, 5:2}로 **82%가 4점**에 뭉쳐 있었고, 평균 3.823을 목표 4.0으로
    끌어올리는 경로는 "보통 문항을 더 좋게"가 아니라 **"하위 14.5%를 없애기"** 하나뿐이었다
    (그 9건을 4점으로 만들면 평균 4.032). Judge가 5점을 거의 안 주므로(62건 중 2건)
    평균의 실질 상한도 4.03이라 해상도가 낮다 — 비율 지표가 더 잘 움직이고 조치도 명확하다.
    정답유일성에서 "치명적 실패율"을 따로 보는 것과 같은 이유(EVAL.md 26절).
    """
    results = [s["scores"] for s in scored]
    n = len(results)

    def avg(key):
        return round(sum(r[key] for r in results) / n, 2)

    def low_rate(key):
        """해당 기준이 3점 이하인 비율 — 평균을 끌어내리는 실제 원인."""
        return round(sum(1 for r in results if r[key] <= 3) / n, 3)

    return {
        "n": n,
        "avg_정답유일성": avg("정답유일성"),
        "avg_오답매력도": avg("오답매력도"),
        "avg_근거성": avg("근거성"),
        "avg_overall": avg("overall"),
        "low_rate_정답유일성": low_rate("정답유일성"),
        "low_rate_오답매력도": low_rate("오답매력도"),
        "low_rate_근거성": low_rate("근거성"),
        "pass_rate": round(sum(1 for r in results if r["overall"] >= 4.0) / n, 2),
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


# ── 문항 품질 Judge 신뢰도 — item_quality_golden.json(사람 라벨 91건) ──────────
# 2026-10 결정: 정기 평가의 Judge 신뢰도 기준 라벨셋을 ITEM_GOLDEN(30건, Claude 합성
# human_score)에서 이쪽(사람이 직접 라벨링한 91건, 기준별 1~5점)으로 교체한다. ITEM_GOLDEN은
# 이력으로 보존하고(위 eval_judge_reliability·eval_item_quality는 그대로 둠), 정기 평가
# (eval_exam.py)만 이 신뢰도 쪽을 쓴다.

def _load_item_quality_golden() -> list[dict]:
    """item_quality_golden.json 95건 중 cannot_judge이거나 기준(정답유일성·오답매력도·근거성)
    중 하나라도 null인 4건을 제외한 91건만 반환한다 — 사람이 판단을 보류한 항목이라
    Judge 신뢰도 계산 대상이 아니다(evals/eval_item_quality_runs.py usable_human_labels와
    같은 기준)."""
    with open(_ITEM_QUALITY_GOLDEN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    usable = []
    for e in data.get("entries", []):
        hl = e.get("human_label") or {}
        if hl.get("cannot_judge") or any(hl.get(c) is None for c in _QUALITY_CRITERIA):
            continue
        usable.append(e)
    return usable


@traceable(name="score_item_quality_golden", run_type="chain", metadata=_TRACE_META)
def score_item_quality_golden(entries: list[dict], llm, limit: int | None = None) -> list[dict]:
    """item_quality_golden.json 엔트리 각각을 judge_one()으로 정확히 1회만 채점.

    score_items()를 그대로 쓰지 않는 이유: 이 골든셋의 엔트리는 {"item": {...}, "human_label":
    {기준별 점수...}, "passage_id": ...}로 ITEM_GOLDEN(엔트리 자체가 문항)과 모양이 다르다.
    judge_one()에는 entry["item"]만 넘기고, entry 전체(passage_id 포함)는 결과에 남겨야
    eval_item_quality_golden_reliability()가 지문 단위 클러스터 CI를 낼 수 있다."""
    subset = entries[:limit] if limit is not None else entries
    return [{"entry": e, "scores": judge_one(e["item"], llm)} for e in subset]


def eval_item_quality_golden_reliability(scored: list[dict]) -> dict:
    """score_item_quality_golden() 결과와 기준별 human_label을 대조해 가중 κ·MAE·편향을
    지문(passage_id) 단위 클러스터 부트스트랩 95% CI와 함께 계산한다(LLM 재호출 없음).

    eval_judge_reliability()(ITEM_GOLDEN, overall 단일값·이진 κ·CI 없음)와는 별개 지표다 —
    이 함수가 2026-10부터 정기 평가(eval_exam.py)의 공식 Judge 신뢰도 수치이고,
    eval_judge_reliability()는 이력 비교용으로 남긴다. eval_item_quality_runs.py의
    compute_reliability_report()(여러 Judge·judged 캐시 조인용)는 모양이 안 맞아(이 함수는
    judge_one()으로 매번 새로 채점하는 1패스 결과라 judged 캐시·model_map 조인이 필요 없음)
    재사용하지 않고, 그 안에서 쓰는 evals/stats.py 원시 함수(weighted_kappa·
    cluster_bootstrap_ci·mean_ci)만 직접 재사용한다."""
    out = {"n": len(scored), "criteria": {}}
    for c in _QUALITY_CRITERIA:
        human_vals = [s["entry"]["human_label"][c] for s in scored]
        judge_vals = [s["scores"][c] for s in scored]
        clusters = [s["entry"]["passage_id"] for s in scored]

        kappa_point, kappa_lo, kappa_hi = cluster_bootstrap_ci(
            list(zip(human_vals, judge_vals)), clusters,
            lambda sample: weighted_kappa(
                [a for a, b in sample], [b for a, b in sample], labels=(1, 2, 3, 4, 5),
            ),
        )
        mae_values = [abs(j - h) for h, j in zip(human_vals, judge_vals)]
        mae_point, mae_lo, mae_hi = mean_ci(mae_values, clusters)
        bias_values = [j - h for h, j in zip(human_vals, judge_vals)]
        bias_point, bias_lo, bias_hi = mean_ci(bias_values, clusters)

        def _r(x):
            return round(x, 3) if x is not None else None

        out["criteria"][c] = {
            "weighted_kappa": _r(kappa_point), "kappa_ci": [_r(kappa_lo), _r(kappa_hi)],
            "mae": _r(mae_point), "mae_ci": [_r(mae_lo), _r(mae_hi)],
            "bias": _r(bias_point), "bias_ci": [_r(bias_lo), _r(bias_hi)],
            "human_avg": round(sum(human_vals) / len(human_vals), 2),
            "llm_avg": round(sum(judge_vals) / len(judge_vals), 2),
        }
    return out


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


# ── 구조 유사도 Judge — 항목 단위 CI(experiments/compare_judge_models.py에서 이동) ─────
# 2026-10: compare_judge_models.py(Judge 후보 비교)와 eval_exam.py(정기 평가) 둘 다 같은
# 계산(항목별 judge-human 쌍 추출 → 항목 단위 클러스터 부트스트랩 CI)이 필요해 공용 위치로
# 옮겼다. compare_judge_models.py는 이 두 함수를 그대로 import해 쓰고(동작 불변), 반복
# 회차 평균(_average_repeats)처럼 이 모듈이 쓰지 않는 로직은 그쪽에 남겨뒀다.

def structure_item_rows(scored: list[dict]) -> list[dict]:
    """score_structure() 결과 1회분을 항목별 (id, judge/human overall·difficulty_match)
    dict 리스트로 변환 — eval_structure_judge()는 집계만 내므로, CI 계산에 필요한
    항목 단위 쌍은 score_structure()가 이미 들고 있는 entry/judge를 그대로 꺼내 쓴다.

    likely_parse_failed: judge_structure()는 judge_one()과 달리 parse_failed 플래그를
    노출하지 않는다(app/modules/exam/judge.py — 파싱 실패 시 조용히 전부 기본값으로
    채운다). type_ratio_score=0.0·overall_score=0·difficulty_match=False가 동시에
    나온 경우를 "파싱 실패로 추정"하는 근사치로만 쓴다 — 진짜로 0점을 준 경우와
    구분할 수 없으므로 집계 시 반드시 '추정'으로 표기한다."""
    rows = []
    for s in scored:
        entry, judge = s["entry"], s["judge"]
        human = entry["human_label"]
        likely_parse_failed = (
            judge["type_ratio_score"] == 0.0
            and judge["overall_score"] == 0
            and judge["difficulty_match"] is False
        )
        rows.append({
            "id": entry["id"],
            "judge_overall": judge["overall_score"],
            "judge_difficulty_match": judge["difficulty_match"],
            "human_overall": human["overall_score"],
            "human_difficulty_match": human["difficulty_match"],
            "likely_parse_failed": likely_parse_failed,
        })
    return rows


def structure_ci(averaged_rows: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """항목 단위로 평균된 rows({"id", "mae", "bias", "difficulty_hit"} 형태 — 반복이 있으면
    compare_judge_models.py의 `_average_repeats()`로 회차 평균을 먼저 낸 결과, 반복이 1회면
    그 값 자체)에 클러스터(=항목 id) 부트스트랩 CI를 적용한다.

    rows 하나가 항목 하나이므로 클러스터 부트스트랩은 사실상 항목 단위 일반 부트스트랩과
    같다 — structure_golden은 항목마다 지문이 달라 다른 클러스터링 기준이 없다."""
    ids = [r["id"] for r in averaged_rows]

    def _mean_of(key):
        def fn(rows: list[dict]) -> float | None:
            vals = [r[key] for r in rows]
            return sum(vals) / len(vals) if vals else None
        return fn

    out = {}
    for key, label in (
        ("mae", "overall_mae"),
        ("difficulty_hit", "difficulty_match_agreement"),
        ("bias", "bias_judge_minus_human"),
    ):
        point, lo, hi = cluster_bootstrap_ci(averaged_rows, ids, _mean_of(key), n_boot=n_boot, seed=seed)
        out[label] = {
            "point": round(point, 3) if point is not None else None,
            "ci_lo": round(lo, 3) if lo is not None else None,
            "ci_hi": round(hi, 3) if hi is not None else None,
        }
    return out
