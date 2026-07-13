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

from app.common.llm.tracing import init_langsmith_project
init_langsmith_project()

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

_ITEM_GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "item_golden.json")


def _load_item_golden() -> list[dict]:
    with open(_ITEM_GOLDEN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("entries", [])


ITEM_GOLDEN = _load_item_golden()

_STRUCTURE_GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "structure_golden.json")


def _load_structure_golden() -> list[dict]:
    """human_label이 채워진(라벨링 완료된) 엔트리만 반환한다.
    라벨링 전 엔트리(human_label: null)는 retrieval_golden의 reviewed:false와 같은 개념으로 스킵."""
    with open(_STRUCTURE_GOLDEN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [e for e in data.get("entries", []) if e.get("human_label")]


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


# overall_score 채점 기준 — structure_golden.json의 _schema.overall_score_rubric과 반드시 동기 유지.
# 2026-07-11 첫 정식 측정에서 사람 라벨은 이 rubric(중복·복사·주제 이탈 감점)을 따르는데
# Judge 프롬프트는 유형·난이도 구조만 물어봐 overall κ가 -0.103까지 무너지는 미정렬을 확인,
# rubric을 프롬프트에 주입함(EVAL.md 5절 참고).
STRUCTURE_JUDGE_TPL = PromptTemplate(
    system=(
        "예시 문제와 새로 생성된 문항 세트를 비교해 평가하세요. "
        "문항 개수 일치 여부는 판단하지 마세요 — 개수는 별도로 코드가 검증합니다.\n"
        "다음 3가지를 JSON으로만 응답하세요.\n"
        "- type_ratio_score(유형 구성 비율 유사도, 0.0~1.0)\n"
        "- difficulty_match(난이도 구성 부합, true/false)\n"
        "- overall_score(0~5 정수): 단순한 유형·난이도 일치가 아니라, 예시의 주제·형식을 "
        "유지하면서 '새로운' 문항 세트로 변환했는지를 종합 평가합니다. "
        "예시 문제를 그대로 복사한 경우, 세트 안에 같은 문항이 반복되는 경우(문장이 완전히 "
        "동일하지 않아도 표현만 바꿔 사실상 같은 것을 묻는 패러프레이즈 반복도 포함— "
        "예: '다음 중 A인 것은?'과 '다음은 A를 의미하는가?'처럼 형식만 다를 뿐 같은 질문), "
        "주제가 이탈한 경우, 교육과정에 없는 개념을 지어낸 경우(환각), "
        "한국어가 아닌 텍스트가 섞인 경우는 반드시 감점하세요. "
        "**단, 같은 주제·개념 범주 안에서도 서로 다른 지점(정의, 사례 적용, 원인, 결과, "
        "비교 등)을 묻는 문항들은 표현이 비슷해 보여도 반복이 아니므로 감점하지 마세요. "
        "실질적으로 같은 것을 묻는 경우에만 반복으로 간주하세요.**\n"
        "  5: 유형·난이도·주제·형식이 매우 잘 맞고, 새 문항으로 충분히 변형되며 중복·심각한 오류 없음\n"
        "  4: 전반적으로 잘 맞으나 경미한 반복, 표현 오류, 일부 내용 결함\n"
        "  3: 핵심 구조는 유지하지만 뚜렷한 중복, 오류, 환각, 일부 유형·주제 손상\n"
        "  2: 일부 구조만 재현하며 유형 누락, 큰 주제 이탈, 심한 품질 저하\n"
        "  1: 원문 단순 복사 또는 완전 중복에 의존해 새 문항 생성으로 보기 어려움(형식 일치는 최소한 있음)\n"
        "  0: 유형 완전 반전, 언어 오염, 구조 붕괴 등으로 사실상 사용 불가\n"
        "판단이 애매하다고 해서 무조건 3점으로 두지 마세요. 3점도 다른 점수와 마찬가지로 "
        "명확한 근거가 있을 때만 주는 점수입니다 — 각 점수 정의를 다시 검토해 가장 부합하는 "
        "점수를 선택하세요.\n"
        '형식: {"type_ratio_score": 실수, "difficulty_match": true/false, "overall_score": 정수}'
    ),
    few_shots=[
        {
            "user": '{"예시_문제": "1. 시장 실패의 원인은?(객관식)", "생성된_세트": [{"question":"공공재의 특성으로 옳은 것은?","item_type":"객관식","difficulty":"중"}]}',
            "assistant": '{"type_ratio_score": 1.0, "difficulty_match": true, "overall_score": 5}',
        },
        {
            # 유형·난이도는 일치하지만 세트 내부가 완전 중복 → 구조 점수와 무관하게 낮은 overall
            "user": '{"예시_문제": "1. 기본권 중 자유권은?(객관식)", "생성된_세트": [{"question":"기본권 중 자유권에 해당하는 것은?","item_type":"객관식","difficulty":"중"},{"question":"기본권 중 자유권에 해당하는 것은?","item_type":"객관식","difficulty":"중"},{"question":"기본권 중 자유권에 해당하는 것은?","item_type":"객관식","difficulty":"중"}]}',
            "assistant": '{"type_ratio_score": 1.0, "difficulty_match": true, "overall_score": 1}',
        },
        {
            "user": '{"예시_문제": "1. 선거 원칙?(객관식2+서술형1)", "생성된_세트": [{"question":"보통 선거의 의미는?","item_type":"객관식","difficulty":"하"}]}',
            "assistant": '{"type_ratio_score": 0.5, "difficulty_match": false, "overall_score": 2}',
        },
        {
            # 문장이 다르지만 사실상 같은 질문(패러프레이즈 반복) — 텍스트 유사도로는 안 잡히지만
            # 감점 대상. 2026-07-12: Judge가 이 유형을 놓쳐 overall을 과대평가하는 것이 확인됨
            # (str_048류 사례, EVAL.md 5절 참고).
            "user": '{"예시_문제": "1. 소비자의 기본 권리로 옳은 것은?(객관식)", "생성된_세트": [{"question":"소비자의 기본 권리에 해당하는 것은?","item_type":"객관식","difficulty":"중"},{"question":"소비자가 갖는 권리로 옳은 것은?","item_type":"객관식","difficulty":"중"},{"question":"다음 중 소비자 권리에 해당하는 것은 무엇인가?","item_type":"객관식","difficulty":"중"}]}',
            "assistant": '{"type_ratio_score": 1.0, "difficulty_match": true, "overall_score": 1}',
        },
        {
            # 3점 앵커(2026-07-12 추가): 기존 few-shot 점수 분포가 {1,1,2,4,5}로 3점이
            # 비어있어, 애매한 사례(특히 유의어 치환 반복류, str_010/047 참고)를 만나면
            # Judge가 판단을 회피하듯 3점으로 수렴하는 경향이 확인됨(n=45 재측정, EVAL.md
            # 5절). 세트 절반은 유의어 치환 반복(문항1·2), 나머지 절반은 서로 다른 지점을
            # 묻는 정상 문항(문항3·4)인 "부분적 결함" 사례로 3점을 명확히 앵커링.
            "user": '{"예시_문제": "1. 지방분권이 필요한 이유로 가장 적절한 것은?(객관식)", "생성된_세트": [{"question":"지방분권이 필요한 배경으로 가장 적절한 것은?","item_type":"객관식","difficulty":"중"},{"question":"지방분권이 요구되는 이유 중 가장 적절한 것은?","item_type":"객관식","difficulty":"중"},{"question":"지방분권 실시 이후 나타날 수 있는 부작용으로 옳은 것은?","item_type":"객관식","difficulty":"중"},{"question":"지방분권과 중앙집권의 균형을 맞추기 위한 제도적 장치를 서술하시오.","item_type":"서술형","difficulty":"중"}]}',
            "assistant": '{"type_ratio_score": 0.5, "difficulty_match": true, "overall_score": 3}',
        },
        {
            # 균형 예시: 같은 주제(선거)라도 서로 다른 지점(원칙 구분, 제도 비교, 사례 적용)을
            # 물어 실질적으로 다른 문항 — 표현이 비슷해 보여도 반복으로 감점하면 안 됨.
            # 2026-07-12: 패러프레이즈 반복 few-shot만 넣었더니 Judge가 과도하게 엄격해져
            # (단일 문항조차 감점) 상관관계가 무너진 것을 확인, 균형 문구+예시로 보완.
            "user": '{"예시_문제": "1. 민주 선거의 기본 원칙은?(객관식)", "생성된_세트": [{"question":"보통 선거 원칙의 의미로 옳은 것은?","item_type":"객관식","difficulty":"중"},{"question":"평등 선거와 보통 선거 원칙의 차이를 서술하시오.","item_type":"서술형","difficulty":"중"}]}',
            "assistant": '{"type_ratio_score": 0.5, "difficulty_match": true, "overall_score": 4}',
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
    count_match_code_hits = []  # LLM/사람 대조 대상 아님 — 골든셋 엔트리 자체의 데이터 정합성 확인용

    for entry in subset:
        judge = judge_structure_one(entry, llm)
        human = entry["human_label"]

        difficulty_match_hits.append(judge["difficulty_match"] == human["difficulty_match"])
        overall_diffs.append(abs(judge["overall_score"] - human["overall_score"]))
        count_match_code_hits.append(len(entry.get("generated_items", [])) == entry.get("num_items"))

    n = len(subset)
    return {
        "n": n,
        "count_match_code_rate": round(sum(count_match_code_hits) / n, 3),
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

    # 5. LangSmith Experiments 기록 (선택 — LANGCHAIN_TRACING_V2=true일 때만)
    run_langsmith_experiments(judge_llm)


if __name__ == "__main__":
    main()
