#!/usr/bin/env python
"""출제 모듈 평가 스크립트
검색(Recall@5, MRR) / 문항 품질(LLM Judge) / 구조 유사도 Judge 신뢰도 / Judge 신뢰도.
검색 평가: 실제 standards/regulations 컬렉션 기반 골든셋 사용.

judge 템플릿·골든셋 로더·평가 함수 본체는 eval_lib.py로 분리됨(2026-07-18) —
compare_models.py/compare_judge_models.py/compare_distractor_quality.py/
eval_record.py/eval_example_retrieval.py가 이 스크립트를 직접 import해 쓰던
것을 eval_lib.py 하나로 정리. 이 파일은 그 결과를 조합해 리포트를 출력하는
진입점 역할만 한다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:1.5b")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db")

from app.common.llm.tracing import init_langsmith_project
init_langsmith_project()

try:
    from langsmith import traceable
except ImportError:
    def traceable(**kwargs):
        def decorator(fn): return fn
        return decorator

from app.common.llm import get_judge_backend, get_llm_backend
from app.common.rag import BGEEmbedder, BGEReranker, RAGRetriever, RAGStore

from eval_lib import (
    ITEM_GOLDEN,
    _TRACE_META,
    _load_retrieval_golden,
    _load_structure_golden,
    eval_item_quality,
    eval_judge_reliability,
    eval_retrieval,
    eval_structure_judge,
    judge_one,
    judge_structure_one,
    score_items,
)


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
        print(f"  (참고) count_match_code : {structure['count_match_code_rate']:.3f} — 골든셋 생성 시점에 num_items를 실제로 맞춘 비율(사람 대조 아님)")

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


# ── LangSmith Experiments 연동 ────────────────────────────────────────

def run_langsmith_experiments(judge_llm) -> None:
    """문항 품질·구조 유사도 Judge 신뢰도를 LangSmith Experiments에 기록한다.
    LANGCHAIN_TRACING_V2가 꺼져 있으면 조용히 건너뜀(선택 기능)."""
    from langsmith_experiments import experiments_enabled, identity_target, sync_dataset
    if not experiments_enabled():
        return

    from langsmith import Client, evaluate

    client = Client()
    print("\n[LangSmith Experiments 연동]")

    item_examples = [
        {
            "inputs": {
                "question": it["question"], "options": it["options"],
                "answer": it["answer"], "item_type": it.get("item_type", ""),
            },
            "outputs": {"human_score": it["human_score"]},
        }
        for it in ITEM_GOLDEN
    ]
    sync_dataset(
        client, "bunpil-item-quality-judge", item_examples,
        description="문항 품질(정답유일성·오답매력도·근거성) LLM Judge 신뢰도 — item_golden.json과 동기화됨",
    )

    def item_quality_evaluator(inputs: dict, reference_outputs: dict) -> list[dict]:
        scores = judge_one(inputs, judge_llm)
        human = reference_outputs.get("human_score", 0)
        llm_score = round(scores["overall"])
        return [
            {"key": "judge_overall", "score": scores["overall"]},
            {"key": "abs_diff_from_human", "score": abs(llm_score - human)},
            {"key": "exact_match", "score": 1.0 if llm_score == human else 0.0},
        ]

    evaluate(
        identity_target, data="bunpil-item-quality-judge",
        evaluators=[item_quality_evaluator],
        experiment_prefix="item-quality-judge", metadata=_TRACE_META,
    )
    print("  - item-quality-judge 실험 기록 완료")

    structure_golden = _load_structure_golden()
    if structure_golden:
        structure_examples = [
            {
                "inputs": {"passage_text": e["passage_text"], "generated_items": e["generated_items"]},
                "outputs": {"human_label": e["human_label"]},
            }
            for e in structure_golden
        ]
        sync_dataset(
            client, "bunpil-structure-judge", structure_examples,
            description="구조 유사도 LLM Judge 신뢰도 — structure_golden.json(human_label 채워진 항목만)과 동기화됨",
        )

        def structure_judge_evaluator(inputs: dict, reference_outputs: dict) -> list[dict]:
            judge = judge_structure_one(inputs, judge_llm)
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

    # 2. 문항 품질 평가 + Judge 신뢰도 — judge_one()을 골든셋당 1회만 호출해 공유
    #    (이전에는 eval_item_quality/eval_judge_reliability가 각자 다시 채점해 60회 중복 호출됨)
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
    print(f"\n2. 문항 품질 LLM Judge + Judge 신뢰도 검증 ({len(ITEM_GOLDEN)}개, judge_one 1회만 호출)...")
    scored_items = score_items(ITEM_GOLDEN, judge_llm)
    quality_result = eval_item_quality(scored_items)
    reliability_result = eval_judge_reliability(scored_items)
    print(f"   종합평균={quality_result['avg_overall']:.2f}/5, 합격률={quality_result['pass_rate']*100:.0f}%")
    print(f"   κ={reliability_result['cohen_kappa']:.3f}, ±1 일치율={reliability_result['agreement_within_1']:.3f}")

    # 3. 구조 유사도 Judge 신뢰도 — get_judge_backend()로 채점. 2026-07-23부터 이 Judge가
    #    런타임 judge_node와 동일한 코드(app/modules/exam/judge.py)를 공유하므로, 이 수치가
    #    곧 실제 배포된 judge의 신뢰도다(검증-배포 불일치 해소).
    structure_golden = _load_structure_golden()
    print(f"\n3. 구조 유사도 Judge 신뢰도 검증 (STRUCTURE_GOLDEN {len(structure_golden)}개)...")
    structure_result = eval_structure_judge(structure_golden, judge_llm, limit=len(structure_golden) or 1)
    print(f"   n={structure_result['n']}")

    # 리포트
    print_report(retrieval_result, quality_result, structure_result, reliability_result)

    # 4. LangSmith Experiments 기록 (선택 — LANGCHAIN_TRACING_V2=true일 때만)
    run_langsmith_experiments(judge_llm)


if __name__ == "__main__":
    main()
