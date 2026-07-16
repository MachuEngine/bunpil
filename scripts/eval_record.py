#!/usr/bin/env python
"""Phase 6: 생기부 모듈 평가 스크립트.
안전 지표 우선 — 마스킹 누락률(FN) / 사실 추가율 / 규정 위반 검출 Recall/F1 /
regulations RAG 검색 품질(Recall@5, MRR, 참고용).
데이터: 마스킹·사실추가·위반 판정은 합성 골든셋만 사용. regulations 검색 품질만
예외적으로 실제 인덱싱된 로컬 chroma_db(읽기 전용 검색)를 사용 — 빈 컬렉션으로는
검색 품질을 잴 수 없어 다른 eval 스크립트(eval_exam.py 등)와 동일하게 맞춤.
"""
import asyncio
import concurrent.futures
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

from app.common.llm import PromptTemplate, get_llm_backend
from app.common.rag import get_retriever
from app.modules.record.chain import RecordChain
from app.modules.record.masker import mask_pii

from eval_exam import _load_retrieval_golden, eval_retrieval

_TRACE_META = {
    "model": os.getenv("OLLAMA_MODEL", "unknown"),
    "backend": os.getenv("LLM_BACKEND", "local"),
}

# ── 골든셋 정의 ──────────────────────────────────────────────────────

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "golden")


def _load_golden(filename: str) -> list[dict]:
    with open(os.path.join(_GOLDEN_DIR, filename), encoding="utf-8") as f:
        data = json.load(f)
    return data.get("entries", [])


# [A] 마스킹 테스트 20건 (합성)
MASKING_GOLDEN = _load_golden("masking_golden.json")

# [B] 메모→윤문 사실 추가 테스트 20건 (합성)
HALLUCINATION_GOLDEN = _load_golden("hallucination_golden.json")

# [C] 규정 위반 탐지 골든셋 50건 (label 1=위반, 0=정상)
VIOLATION_GOLDEN = _load_golden("violation_golden.json")


def _run_async(coro):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result(timeout=300)


# ── 평가 함수 ────────────────────────────────────────────────────────

def eval_masking(golden: list) -> dict:
    """마스킹 누락률(FN), 오탐률(FP) 계산."""
    tp = fp = fn = tn = 0

    for item in golden:
        _, found = mask_pii(item["text"])
        expected = set(item["pii"])
        detected = set(found)

        if expected:
            # PII 있는 케이스: 기대 유형이 모두 감지되면 TP
            if expected <= detected:
                tp += 1
            else:
                fn += 1
        else:
            # PII 없는 케이스: 아무것도 감지 안 하면 TN
            if not detected:
                tn += 1
            else:
                fp += 1

    total_pii = tp + fn
    total_clean = tn + fp
    recall = tp / total_pii if total_pii else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    fn_rate = fn / total_pii if total_pii else 0.0

    return {
        "n": len(golden),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "fn_rate": round(fn_rate, 3),
    }


NLI_TPL = PromptTemplate(
    system=(
        "아래 [메모]에 없는 새로운 사실이 [윤문]에 추가되었으면 YES, 없으면 NO로만 응답하세요.\n"
        "단순 표현 변경·문체 다듬기는 추가로 보지 않습니다.\n"
        "**정도부사·평가어(예: 효과적으로, 성공적으로, 적극적으로)만 붙고 메모에 있던 행위 자체는 "
        "그대로면 NO입니다.** 반면 메모에 없던 구체적인 행위나 결과(예: 무엇을 관리했다, 무엇을 "
        "완수/달성/수상했다)가 새로 서술되면 YES입니다."
    ),
    few_shots=[
        {
            "user": "[메모] 수학 발표를 잘 함.\n[윤문] 수학 교과 발표에 적극 참여하였음.",
            "assistant": "NO",
        },
        {
            "user": "[메모] 조별 과제에서 열심히 함.\n[윤문] 조별 과제에서 1등을 수상하였음.",
            "assistant": "YES",
        },
        {
            "user": (
                "[메모] 친구들 의견 듣고 조율함.\n"
                "[윤문] 친구들의 의견을 청취하고 효과적으로 조율하여 협업 활동에 성공적으로 참여함."
            ),
            "assistant": "NO",
        },
        {
            "user": (
                "[메모] 조별 과제에서 리더 역할.\n"
                "[윤문] 조별 과제에서 리더로서 역할을 맡아 조업을 관리하고 팀원들과 협력하여 성공적으로 완수함."
            ),
            "assistant": "YES",
        },
    ],
    cot_prefix="",
)


@traceable(name="eval_hallucination", run_type="llm", metadata=_TRACE_META)
def eval_hallucination(golden: list, chain: RecordChain, llm) -> dict:
    """사실 추가율: 메모에 없는 내용 포함 여부 측정."""
    keyword_fn = 0   # 금지 키워드 기반 탐지
    nli_fn = 0       # LLM Judge 기반 탐지
    n = len(golden)

    for item in golden:
        out = _run_async(chain.run(item["memo"]))
        polished = out["polished"]

        # (1) 금지 키워드 검사
        if any(kw in polished for kw in item["forbidden"]):
            keyword_fn += 1

        # (2) NLI-style LLM Judge
        prompt = f"[메모] {item['memo']}\n[윤문] {polished}"
        messages = NLI_TPL.build(prompt)
        raw = _run_async(llm.generate(messages)).strip().upper()
        if raw.startswith("YES"):
            nli_fn += 1

    return {
        "n": n,
        "keyword_hallucination": keyword_fn,
        "keyword_hallucination_rate": round(keyword_fn / n, 3),
        "nli_hallucination": nli_fn,
        "nli_hallucination_rate": round(nli_fn / n, 3),
    }


@traceable(name="eval_regulation_retrieval", run_type="chain", metadata=_TRACE_META)
def eval_regulation_retrieval(retriever) -> dict:
    """regulations 컬렉션 RAG 검색 품질(Recall@5, MRR) — 참고용.

    eval_exam.py의 retrieval_golden_final.json(standards 12 + regulations 10,
    사람 검수 완료)은 지금까지 두 컬렉션을 합쳐 하나의 Recall@5로만 보고해왔다.
    record 모듈의 _step_validate가 실제로 의존하는 건 regulations 검색이므로
    그 10건만 분리해 별도로 측정한다. eval_exam.py의 기존 합산 수치는 그대로 둠
    (bunpil_roadmap.md에 그 수치 기준의 과거 실험 기록이 남아있어 비교 연속성 유지).
    n=10으로 표본이 작아 참고용 — 통과/실패 기준(all_ok)에는 포함하지 않는다.
    """
    golden = _load_retrieval_golden()
    regulations_golden = [item for item in golden if item["source_collection"] == "regulations"]
    return eval_retrieval(retriever, regulations_golden)


@traceable(name="eval_violation_detection", run_type="chain", metadata=_TRACE_META)
def eval_violation_detection(golden: list, chain: RecordChain) -> dict:
    """규정 위반 검출 Recall / F1 측정."""
    tp = fp = fn = tn = 0

    for item in golden:
        # validate 스텝만 직접 호출
        state = {
            "memo": item["text"],
            "masked": item["text"],
            "pii_found": [],
            "polished": item["text"],
            "violations": [],
            "attempt": 0,
        }
        result = _run_async(chain._step_validate(state))
        detected = len(result["violations"]) > 0
        expected = item["label"] == 1

        if expected and detected:
            tp += 1
        elif expected and not detected:
            fn += 1
        elif not expected and detected:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "n": len(golden),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


# ── 리포트 출력 ─────────────────────────────────────────────────────

def check(ok: bool) -> str:
    return "✓" if ok else "✗"


def print_report(mask: dict, halluc: dict, viol: dict, reg_retrieval: dict):
    print("\n" + "=" * 55)
    print("  분필 생기부 모듈 평가 리포트")
    print("=" * 55)

    fn_rate = mask["fn_rate"]
    print(f"\n[1] PII 마스킹 (n={mask['n']})")
    print(f"  TP={mask['tp']}  FN={mask['fn']}  FP={mask['fp']}  TN={mask['tn']}")
    print(f"  Recall    : {mask['recall']:.3f}  {check(mask['recall'] >= 1.0)} (기준 = 1.0)")
    print(f"  Precision : {mask['precision']:.3f}")
    print(f"  누락률(FN): {fn_rate:.3f}  {check(fn_rate == 0.0)} (목표 = 0)")

    print(f"\n[2] 사실 추가율 (n={halluc['n']})")
    k_rate = halluc["keyword_hallucination_rate"]
    n_rate = halluc["nli_hallucination_rate"]
    print(f"  키워드 기반 사실추가 : {halluc['keyword_hallucination']}건  {check(k_rate == 0.0)} (목표 = 0)")
    print(f"  NLI Judge 사실추가  : {halluc['nli_hallucination']}건  {check(n_rate == 0.0)} (목표 = 0)")
    print(f"  키워드 추가율       : {k_rate:.3f}")
    print(f"  NLI 추가율          : {n_rate:.3f}")

    print(f"\n[3] 규정 위반 검출 (n={viol['n']})")
    print(f"  TP={viol['tp']}  FP={viol['fp']}  FN={viol['fn']}  TN={viol['tn']}")
    print(f"  Recall    : {viol['recall']:.3f}  {check(viol['recall'] >= 0.95)} (기준 ≥ 0.95)")
    print(f"  Precision : {viol['precision']:.3f}")
    print(f"  F1        : {viol['f1']:.3f}")

    print(f"\n[4] regulations RAG 검색 품질 (n={reg_retrieval['n']}, 참고용 — 표본 작아 통과 기준 없음)")
    print(f"  Recall@5  : {reg_retrieval['recall_at_5']:.3f}")
    print(f"  MRR       : {reg_retrieval['mrr']:.3f}")

    all_ok = fn_rate == 0.0 and k_rate == 0.0 and viol["recall"] >= 0.95
    print(f"\n  전체 통과 : {check(all_ok)}")
    print("\n" + "=" * 55)
    print("※ 개발 모델(1.5b) 기준. 7B(RunPod)에서 재평가 권장.")
    print("=" * 55)


# ── 메인 ────────────────────────────────────────────────────────────

@traceable(name="eval_record_run", run_type="chain", metadata=_TRACE_META)
def main():
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        print("LangSmith 트레이싱: 활성화됨")
    print("=== Phase 6: 생기부 모듈 평가 시작 ===\n")

    chain = RecordChain()
    llm = get_llm_backend()

    print("1. PII 마스킹 평가 (20건)...")
    mask_result = eval_masking(MASKING_GOLDEN)
    print(f"   누락률(FN)={mask_result['fn_rate']:.3f}, Recall={mask_result['recall']:.3f}")

    print(f"\n2. 사실 추가율 평가 ({len(HALLUCINATION_GOLDEN)}건, NLI Judge)...")
    halluc_result = eval_hallucination(HALLUCINATION_GOLDEN, chain, llm)
    print(f"   키워드={halluc_result['keyword_hallucination']}건, NLI={halluc_result['nli_hallucination']}건")

    print(f"\n3. 규정 위반 검출 평가 ({len(VIOLATION_GOLDEN)}건)...")
    viol_result = eval_violation_detection(VIOLATION_GOLDEN, chain)
    print(f"   Recall={viol_result['recall']:.3f}, F1={viol_result['f1']:.3f}")

    print("\n4. regulations RAG 검색 품질 평가 (참고용, n=10)...")
    reg_retrieval_result = eval_regulation_retrieval(get_retriever())
    print(f"   Recall@5={reg_retrieval_result['recall_at_5']:.3f}, MRR={reg_retrieval_result['mrr']:.3f}")

    print_report(mask_result, halluc_result, viol_result, reg_retrieval_result)


if __name__ == "__main__":
    main()
