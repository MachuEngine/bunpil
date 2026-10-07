#!/usr/bin/env python
"""출제 모듈 평가 스크립트
검색(Recall@5, MRR) / 문항 품질 Judge 신뢰도 / 구조 유사도 Judge 신뢰도.
검색 평가: 실제 standards/regulations 컬렉션 기반 골든셋 사용.

judge 템플릿·골든셋 로더·평가 함수 본체는 eval_lib.py로 분리됨(2026-07-18) —
compare_models.py/compare_judge_models.py/compare_distractor_quality.py/
eval_record.py/eval_example_retrieval.py가 이 스크립트를 직접 import해 쓰던
것을 eval_lib.py 하나로 정리. 이 파일은 그 결과를 조합해 리포트를 출력하는
진입점 역할만 한다.

2026-10 Judge 분리(get_offline_judge_backend()): 이 스크립트가 신뢰도를 재는 Judge는
런타임 게이트 Judge(get_judge_backend(), .env JUDGE_BACKEND)가 아니라 별도의 오프라인
평가 전용 Judge(기본 anthropic/claude-sonnet-5.5, OpenRouter)다 — 같은 모델로 자기 자신의
신뢰도를 재면 검증 의미가 없기 때문이다(evals/eval_lib.py get_offline_judge_backend()
docstring 참고). 런타임 judge_node·.env의 JUDGE_BACKEND는 건드리지 않는다.

2026-10 골든셋 교체: 문항 품질 Judge 신뢰도의 기준 라벨셋을 ITEM_GOLDEN(item_golden.json,
30건, Claude 합성 human_score)에서 item_quality_golden.json(사람이 직접 라벨링한 91건,
기준별 1~5점, cannot_judge 4건 제외)으로 바꾼다. ITEM_GOLDEN은 이력으로만 보존하고
(evals/eval_lib.py의 eval_judge_reliability()·eval_item_quality() 자체는 지우지 않음),
이 스크립트의 정기 평가에서는 더 쓰지 않는다.
"""
import datetime
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# CHROMA_PERSIST_DIR: 로컬 .env는 배포 경로(/data/chroma_db)로 설정돼 있어 로컬에서
# 그대로 실행하면 RAGStore 초기화가 실패한다(evals/local_env.py 참고). 셸 명시 여부는
# load_dotenv() 호출 전에 캡처해야 한다.
_had_chroma_dir = "CHROMA_PERSIST_DIR" in os.environ
load_dotenv()

os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:1.5b")

from evals.local_env import use_local_chroma_dir
use_local_chroma_dir(_had_chroma_dir)

from app.common.llm.tracing import init_langsmith_project
init_langsmith_project()

try:
    from langsmith import traceable
except ImportError:
    def traceable(**kwargs):
        def decorator(fn): return fn
        return decorator

from app.common.rag import BGEEmbedder, BGEReranker, RAGRetriever, RAGStore

from eval_lib import (
    _GOLDEN_DIR,
    _TRACE_META,
    _load_item_quality_golden,
    _load_retrieval_golden,
    _load_structure_golden,
    eval_item_quality,
    eval_item_quality_golden_reliability,
    eval_retrieval_per_query,
    eval_structure_judge,
    get_offline_judge_backend,
    score_item_quality_golden,
    score_structure,
    structure_ci,
    structure_item_rows,
)
from evals.stats import mean_ci

_RESULT_PATH = os.path.join(_GOLDEN_DIR, "_eval_exam_latest.json")
_QUALITY_CRITERIA = ("정답유일성", "오답매력도", "근거성")


def _code_version() -> str:
    """golden_gen/gen_item_quality_golden.py `_code_version()`과 같은 값을 내는 로직을
    여기도 둔다(그 함수는 언더스코어 프라이빗이라 모듈 경계를 넘어 직접 가져오지 않음)."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(__file__),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


# ── 판정 기준(확정/미확정/판정 불가) ────────────────────────────────────────
# 2026-10 결정: CI가 0(또는 비교 대상)을 포함하면 "판정 불가", 목표 임계값 대비 CI 하한이
# 임계값 미만이면 "미확정", 그 외 "확정". 두 축(임계값 게이트용/0-기준 비교용)으로 나눈다.

def verdict_threshold(lo: float | None, hi: float | None, threshold: float) -> str:
    """'≥ 임계값' 게이트(κ·Recall 등)의 판정 — CI 하한이 임계값 미만이면 미확정."""
    if lo is None or hi is None:
        return "판정 불가(CI 산출 불가)"
    if lo < threshold:
        return "미확정(CI 하한이 임계값 미만)"
    return "확정"


def verdict_diff_from_zero(lo: float | None, hi: float | None) -> str:
    """편향(judge-human)처럼 0을 기준으로 삼는 값의 판정 — CI가 0을 포함하면 판정 불가."""
    if lo is None or hi is None:
        return "판정 불가(CI 산출 불가)"
    if lo <= 0 <= hi:
        return "판정 불가(CI가 0을 포함)"
    return "확정"


def _single_pass_structure_rows(rows: list[dict]) -> list[dict]:
    """structure_item_rows() 결과(반복 없음, 1패스)를 structure_ci()가 받는
    {"id", "mae", "bias", "difficulty_hit"} 모양으로 바꾼다 — experiments/compare_judge_models.py
    의 `_average_repeats([rows])`와 결과가 같다(반복이 1개뿐이면 "평균"이 그 값 자체).
    eval_exam.py는 judge_structure_one()을 항목당 1회만 호출해 반복이 없으므로
    _average_repeats()를 그대로 가져오지 않고 이 1패스 전용 변환만 둔다."""
    return [
        {
            "id": r["id"],
            "mae": abs(r["judge_overall"] - r["human_overall"]),
            "bias": r["judge_overall"] - r["human_overall"],
            "difficulty_hit": 1.0 if r["judge_difficulty_match"] == r["human_difficulty_match"] else 0.0,
        }
        for r in rows
    ]


# ── 리포트 출력 ─────────────────────────────────────────────────────

def check(ok: bool) -> str:
    return "✓" if ok else "✗"


def print_report(judge_model: str, retrieval: dict, quality: dict, item_reliability: dict,
                  structure: dict, structure_ci_result: dict) -> None:
    print("\n" + "=" * 60)
    print("  분필 출제 모듈 평가 리포트")
    print(f"  오프라인 평가 Judge: {judge_model} (런타임 게이트 Judge와 별도, 2026-10 분리)")
    print("=" * 60)

    print(f"\n[1] 검색 성능 (retrieval_golden, n={retrieval['n']})")
    r5, r5_ci = retrieval["recall_at_5"], retrieval["recall_at_5_ci"]
    mrr, mrr_ci = retrieval["mrr"], retrieval["mrr_ci"]
    r5_verdict = verdict_threshold(r5_ci[0], r5_ci[1], 0.8)
    print(f"  Recall@5 : {r5:.3f} CI={r5_ci}  {check(r5 >= 0.8)} (기준 ≥ 0.8) → {r5_verdict}")
    print(f"  MRR      : {mrr:.3f} CI={mrr_ci}  (참고값, 게이트 없음)")

    print(f"\n[2] 문항 품질 Judge 신뢰도 (item_quality_golden 사람 라벨 {item_reliability['n']}건 — "
          f"생성 품질 아님, 실측 생성 품질은 eval_ragas.py 참고, 5점 척도)")
    for c in _QUALITY_CRITERIA:
        cr = item_reliability["criteria"][c]
        k_verdict = verdict_threshold(cr["kappa_ci"][0], cr["kappa_ci"][1], 0.4)
        b_verdict = verdict_diff_from_zero(cr["bias_ci"][0], cr["bias_ci"][1])
        print(f"  [{c}]")
        print(f"    가중 κ   : {cr['weighted_kappa']} CI={cr['kappa_ci']}  {check(cr['weighted_kappa'] is not None and cr['weighted_kappa'] >= 0.4)} (기준 ≥ 0.4) → {k_verdict}")
        print(f"    MAE      : {cr['mae']} CI={cr['mae_ci']}")
        print(f"    편향(judge-human): {cr['bias']} CI={cr['bias_ci']} → {b_verdict}")
        print(f"    사람 평균/LLM 평균: {cr['human_avg']} / {cr['llm_avg']}")

    print(f"\n  [참고] item_quality_golden을 Judge가 매긴 종합 점수 자체(ITEM_GOLDEN의 avg_overall과 같은 성격)")
    print(f"  종합평균    : {quality['avg_overall']:.2f}  {check(quality['avg_overall'] >= 4.0)} (기준 ≥ 4.0)")
    print(f"  합격률(≥4.0): {quality['pass_rate']*100:.0f}%")

    print(f"\n[3] 구조 유사도 Judge 신뢰도 (structure_golden, n={structure['n']})")
    if structure["n"] == 0:
        print(f"  {structure.get('note', '')}")
    else:
        dma_ci = structure_ci_result["difficulty_match_agreement"]
        mae_ci = structure_ci_result["overall_mae"]
        bias_ci = structure_ci_result["bias_judge_minus_human"]
        bias_verdict = verdict_diff_from_zero(bias_ci["ci_lo"], bias_ci["ci_hi"])
        print(f"  difficulty_match 일치율 : {dma_ci['point']} CI=[{dma_ci['ci_lo']}, {dma_ci['ci_hi']}] (게이트 없음, 참고값)")
        print(f"  overall_score MAE       : {mae_ci['point']} CI=[{mae_ci['ci_lo']}, {mae_ci['ci_hi']}] (게이트 없음, 참고값)")
        print(f"  편향(judge-human)       : {bias_ci['point']} CI=[{bias_ci['ci_lo']}, {bias_ci['ci_hi']}] → {bias_verdict}")
        print(f"  (참고) count_match_code : {structure['count_match_code_rate']:.3f} — 골든셋 생성 시점에 num_items를 실제로 맞춘 비율(사람 대조 아님)")

    print("\n" + "=" * 60)


# ── LangSmith Experiments 연동 ────────────────────────────────────────

def run_langsmith_experiments(scored_quality: list[dict], scored_structure: list[dict]) -> None:
    """문항 품질·구조 유사도 Judge 신뢰도를 LangSmith Experiments에 기록한다.
    LANGCHAIN_TRACING_V2가 꺼져 있으면 조용히 건너뜀(선택 기능).

    2026-10: 문항 품질 쪽 데이터셋 소스를 ITEM_GOLDEN에서 item_quality_golden.json(사람
    라벨 91건)으로 교체 — 골든셋 자체가 바뀌었으므로(위 모듈 docstring) 여기도 맞춰 바꾼다.
    question 텍스트가 아니라 골든셋 id로 조회 키를 쓴다(원문 노출 최소화).

    2026-10: 골든셋·평가 기준(overall 단일값 → 기준별 점수)이 바뀌었는데 데이터셋 이름과
    experiment_prefix를 그대로 두면 같은 이름 아래 서로 다른 기준의 실험이 섞인다 —
    `-v2`를 붙여 분리한다. 구조 Judge(structure_golden) 쪽은 내용이 바뀌지 않았으므로
    그대로 둔다.

    2026-08-04부터 유지된 원칙: **이미 채점된 결과를 받아 조회만 한다**(재채점 없음) —
    main()이 리포트용으로 이미 채점한 것과 같은 골든셋·같은 judge를 또 돌리면 호출이
    두 배가 된다(91 + 45 = 136회 추가).
    """
    from langsmith_experiments import experiments_enabled, identity_target, sync_dataset
    if not experiments_enabled():
        return

    from langsmith import Client, evaluate

    client = Client()
    print("\n[LangSmith Experiments 연동]")

    quality_examples = [
        {
            "inputs": {"id": s["entry"]["id"]},
            "outputs": {"human_label": s["entry"]["human_label"]},
        }
        for s in scored_quality
    ]
    sync_dataset(
        client, "bunpil-item-quality-judge-v2", quality_examples,
        description="문항 품질(정답유일성·오답매력도·근거성) LLM Judge 신뢰도 — "
                     "item_quality_golden.json(사람 라벨 91건, 2026-10부터 ITEM_GOLDEN 대체)과 동기화됨",
    )

    scores_by_id = {s["entry"]["id"]: s["scores"] for s in scored_quality}

    def item_quality_evaluator(inputs: dict, reference_outputs: dict) -> list[dict]:
        scores = scores_by_id[inputs["id"]]
        human = reference_outputs.get("human_label", {})
        results = [{"key": "judge_overall", "score": scores["overall"]}]
        for c in _QUALITY_CRITERIA:
            results.append({"key": f"abs_diff_{c}", "score": abs(scores[c] - human.get(c, 0))})
        return results

    evaluate(
        identity_target, data="bunpil-item-quality-judge-v2",
        evaluators=[item_quality_evaluator],
        experiment_prefix="item-quality-judge-v2", metadata=_TRACE_META,
    )
    print("  - item-quality-judge-v2 실험 기록 완료")

    if scored_structure:
        structure_examples = [
            {
                "inputs": {
                    "passage_text": s["entry"]["passage_text"],
                    "generated_items": s["entry"]["generated_items"],
                },
                "outputs": {"human_label": s["entry"]["human_label"]},
            }
            for s in scored_structure
        ]
        sync_dataset(
            client, "bunpil-structure-judge", structure_examples,
            description="구조 유사도 LLM Judge 신뢰도 — structure_golden.json(human_label 채워진 항목만)과 동기화됨",
        )

        # passage_text는 STRUCTURE_GOLDEN 안에서 유일(45/45 확인) — 조회 키로 사용.
        judge_by_passage = {s["entry"]["passage_text"]: s["judge"] for s in scored_structure}

        def structure_judge_evaluator(inputs: dict, reference_outputs: dict) -> list[dict]:
            judge = judge_by_passage[inputs["passage_text"]]
            human = reference_outputs.get("human_label", {})
            return [
                {"key": "overall_score_diff", "score": abs(judge["overall_score"] - human.get("overall_score", 0))},
                {"key": "difficulty_match_agree", "score": 1.0 if judge["difficulty_match"] == human.get("difficulty_match") else 0.0},
                {"key": "type_ratio_score", "score": judge["type_ratio_score"]},
            ]

        evaluate(
            identity_target, data="bunpil-structure-judge",
            evaluators=[structure_judge_evaluator],
            experiment_prefix="structure-judge", metadata=_TRACE_META,
        )
        print("  - structure-judge 실험 기록 완료")
    else:
        print("  - structure-judge: 라벨링된 항목이 없어 건너뜀")


# ── 결과 저장 ───────────────────────────────────────────────────────

def build_result_record(
    judge_model: str, retrieval: dict, quality: dict, item_reliability: dict,
    structure: dict, structure_ci_result: dict,
) -> dict:
    """data/golden/_eval_exam_latest.json에 저장할 레코드 — 수치·CI·판정·메타만
    담는다(하드룰 4: 문항 원문·지문 원문 없음)."""
    item_criteria = {}
    for c in _QUALITY_CRITERIA:
        cr = item_reliability["criteria"][c]
        item_criteria[c] = {
            **cr,
            "kappa_verdict": verdict_threshold(cr["kappa_ci"][0], cr["kappa_ci"][1], 0.4),
            "bias_verdict": verdict_diff_from_zero(cr["bias_ci"][0], cr["bias_ci"][1]),
        }

    structure_record = None
    if structure["n"]:
        bias_ci = structure_ci_result["bias_judge_minus_human"]
        structure_record = {
            "n": structure["n"],
            "difficulty_match_agreement": structure_ci_result["difficulty_match_agreement"],
            "overall_score_mae": structure_ci_result["overall_mae"],
            "bias_judge_minus_human": bias_ci,
            "bias_verdict": verdict_diff_from_zero(bias_ci["ci_lo"], bias_ci["ci_hi"]),
            "count_match_code_rate": structure["count_match_code_rate"],
        }

    return {
        "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "code_version": _code_version(),
        "offline_judge_model": judge_model,
        "retrieval": {
            "n": retrieval["n"],
            "recall_at_5": retrieval["recall_at_5"], "recall_at_5_ci": retrieval["recall_at_5_ci"],
            "recall_at_5_verdict": verdict_threshold(
                retrieval["recall_at_5_ci"][0], retrieval["recall_at_5_ci"][1], 0.8,
            ),
            "mrr": retrieval["mrr"], "mrr_ci": retrieval["mrr_ci"],
        },
        "item_quality_golden_reliability": {"n": item_reliability["n"], "criteria": item_criteria},
        "item_quality_golden_judge_score": {
            "avg_overall": quality["avg_overall"], "pass_rate": quality["pass_rate"],
        },
        "structure_golden_reliability": structure_record,
    }


# ── 메인 ────────────────────────────────────────────────────────────

@traceable(name="eval_exam_run", run_type="chain", metadata=_TRACE_META)
def main():
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        print("LangSmith 트레이싱: 활성화됨")
    print("=== 출제 모듈 평가 시작 ===\n")

    store = RAGStore()
    embedder = BGEEmbedder()
    reranker = BGEReranker()
    retriever = RAGRetriever(store, embedder, reranker)

    judge_llm = get_offline_judge_backend()
    _judge_model = getattr(judge_llm, "model", None) or os.getenv("OFFLINE_JUDGE_MODEL", "anthropic/claude-sonnet-5.5")
    print(f"[Judge] 오프라인 평가 전용: {_judge_model} (런타임 게이트 Judge와 분리)")

    # 1. 검색 평가 — 질의 단위 부트스트랩 CI(evals/stats.py)
    # 2026-10: eval_retrieval()을 따로 또 호출하지 않는다 — eval_retrieval_per_query()가
    # 같은 판정 로직으로 질의 단위 hit/rr을 이미 내놓으므로, Recall@5·MRR 집계(eval_retrieval()과
    # 같은 계산식: hit 평균, rr 평균)를 그 결과에서 바로 뽑으면 질의마다 retrieve()를
    # 두 번 부르지 않아도 된다(eval_retrieval() 자체와 다른 호출자는 그대로 둠).
    golden = _load_retrieval_golden()
    print(f"\n1. 검색 평가 (Recall@5, MRR) — 골든셋 {len(golden)}개...")
    per_query = eval_retrieval_per_query(retriever, golden)
    ids = [r["id"] for r in per_query]
    n = len(per_query)
    retrieval_agg = {
        "recall_at_5": sum(r["hit"] for r in per_query) / n,
        "mrr": sum(r["rr"] for r in per_query) / n,
        "n": n,
    }
    recall_point, recall_lo, recall_hi = mean_ci([r["hit"] for r in per_query], ids)
    mrr_point, mrr_lo, mrr_hi = mean_ci([r["rr"] for r in per_query], ids)
    retrieval_result = {
        **retrieval_agg,
        "recall_at_5_ci": [round(recall_lo, 3) if recall_lo is not None else None,
                           round(recall_hi, 3) if recall_hi is not None else None],
        "mrr_ci": [round(mrr_lo, 3) if mrr_lo is not None else None,
                   round(mrr_hi, 3) if mrr_hi is not None else None],
    }
    print(f"   Recall@5={retrieval_result['recall_at_5']:.3f} CI={retrieval_result['recall_at_5_ci']}, "
          f"MRR={retrieval_result['mrr']:.3f} CI={retrieval_result['mrr_ci']}")

    # 2. 문항 품질 Judge 신뢰도 — item_quality_golden.json 사람 라벨 91건, judge_one 1회만 호출
    item_quality_golden = _load_item_quality_golden()
    print(f"\n2. 문항 품질 Judge 신뢰도 검증 (item_quality_golden {len(item_quality_golden)}건, "
          f"judge_one 1회만 호출)...")
    scored_quality = score_item_quality_golden(item_quality_golden, judge_llm)
    quality_result = eval_item_quality(scored_quality)
    item_reliability_result = eval_item_quality_golden_reliability(scored_quality)
    print(f"   종합평균={quality_result['avg_overall']:.2f}/5, 합격률={quality_result['pass_rate']*100:.0f}%")
    for c in _QUALITY_CRITERIA:
        cr = item_reliability_result["criteria"][c]
        print(f"   [{c}] κ={cr['weighted_kappa']} CI={cr['kappa_ci']}, MAE={cr['mae']}, 편향={cr['bias']}")

    # 3. 구조 유사도 Judge 신뢰도 — structure_golden 45건, judge_structure_one 1회만 호출.
    #    2026-07-23부터 이 Judge가 런타임 judge_node와 동일한 코드(app/modules/exam/judge.py)를
    #    공유하지만, 2026-10부터 "오프라인 평가 전용 Judge"로 호출 모델 자체는 런타임과 분리됐다
    #    (모듈 docstring 참고) — 이 수치는 그 분리된 Judge의 신뢰도다.
    structure_golden = _load_structure_golden()
    print(f"\n3. 구조 유사도 Judge 신뢰도 검증 (structure_golden {len(structure_golden)}개, "
          f"judge_structure_one 1회만 호출)...")
    scored_structure = score_structure(structure_golden, judge_llm)
    structure_result = eval_structure_judge(scored_structure)
    structure_ci_result = (
        structure_ci(_single_pass_structure_rows(structure_item_rows(scored_structure))) if scored_structure
        else {"overall_mae": {"point": None, "ci_lo": None, "ci_hi": None},
              "difficulty_match_agreement": {"point": None, "ci_lo": None, "ci_hi": None},
              "bias_judge_minus_human": {"point": None, "ci_lo": None, "ci_hi": None}}
    )
    print(f"   n={structure_result['n']}")

    # 리포트
    print_report(_judge_model, retrieval_result, quality_result, item_reliability_result,
                 structure_result, structure_ci_result)

    # 결과 저장 — 수치·CI·판정만(원문 없음, 하드룰 4).
    record = build_result_record(
        _judge_model, retrieval_result, quality_result, item_reliability_result,
        structure_result, structure_ci_result,
    )
    with open(_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {_RESULT_PATH}")

    # 4. LangSmith Experiments 기록 (선택 — LANGCHAIN_TRACING_V2=true일 때만)
    #    이미 채점한 결과를 넘긴다 — evaluator가 재채점하면 LLM 호출·trace가 2배가 된다.
    run_langsmith_experiments(scored_quality, scored_structure)


if __name__ == "__main__":
    main()
