#!/usr/bin/env python
"""RAG 품질 평가: Faithfulness, Answer Relevancy.

**Ragas 라이브러리를 쓰지 않고 알고리즘을 직접 구현했다.** 이유: `ragas`(0.3.9,
0.4.3 둘 다 확인)가 `langchain_community.chat_models.vertexai`에서 `ChatVertexAI`를
무조건 import하는데, 이 경로가 최신 `langchain-community`(0.4.2)에서 완전히
제거됐다 — ragas 쪽 알려진 상류 버그(GitHub vibrantlabsai/ragas #2741, #2745).
호환되는 구버전 `langchain-community`로 낮추면 이번엔 `langchain-core`가
0.3.x대로 끌려 내려가 이 프로젝트가 이미 쓰는 langgraph/langchain-openai/
langchain-ollama(전부 langchain-core 1.x 요구) 전체가 깨진다. 사용자 확인 후
Ragas 패키지 대신 Ragas의 실제 알고리즘만 우리 LLM Judge + 기존 BGEEmbedder로
재구현하기로 결정(EVAL.md 참고).

측정 대상: 출제 모듈의 RAG 검색→생성 흐름. passage_text를 "질문", 검색된
achievement-standard 청크를 "컨텍스트", 실제로 생성된 문항 세트를 "답변"으로
매핑한다(retrieval_golden_final.json은 검색 자체의 Recall/MRR 평가용이라 이
목적에 맞는 (question, context, answer) 3종 세트가 없어, gen_structure_golden.py의
PASSAGE_SAMPLES + 실제 그래프 실행으로 새로 구성한다).

- Faithfulness: 답변을 원자적 주장(atomic claim) 목록으로 분해한 뒤, 각 주장이
  컨텍스트로 뒷받침되는지 LLM에게 판정시켜 (뒷받침된 주장 수 / 전체 주장 수)로 계산.
  Ragas 논문·구현의 핵심 알고리즘을 그대로 따름.
- Answer Relevancy: 답변으로부터 역질문(이 답변이 나올 법한 질문) 3개를 생성시킨 뒤,
  원래 질문과의 임베딩 코사인 유사도 평균으로 계산. 역시 Ragas의 원 알고리즘을 따름.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "golden_gen"))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:7b")
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
from app.common.rag import get_embedder, get_retriever
from app.modules.exam.tools import _HANGUL_RE, _HAN_RE  # 언어 오염 검출 재사용

from eval_lib import eval_item_quality, score_items
from gen_structure_golden import PASSAGE_SAMPLES  # noqa: E402

_TRACE_META = {"model": os.getenv("OLLAMA_MODEL", "unknown"), "backend": os.getenv("LLM_BACKEND", "local")}

# 8개 샘플 — gen_structure_golden.py의 PASSAGE_SAMPLES 중 standards가 명확한 것으로 선정
_SAMPLE_IDS = ["str_001", "str_002", "str_006", "str_009", "str_013", "str_020", "str_023", "str_033"]


def _run_async(coro):
    import asyncio, concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result(timeout=300)


def _is_contaminated(text: str) -> bool:
    """qwen2.5:7b가 확률적으로 중국어를 섞어내는 문제(TROUBLESHOOTING.md)가 이
    스크립트의 claim 분해·역질문 생성 호출에서도 재현됨(시스템 프롬프트에 "한국어로만"
    지시를 넣어도 완전히 막히지 않음) — app/modules/exam/tools.py의 _check_korean()과
    동일한 판정 로직을 재사용해 오염된 claim/질문을 채점에서 제외한다."""
    hangul = len(_HANGUL_RE.findall(text))
    han = len(_HAN_RE.findall(text))
    if han and (hangul == 0 or han / (han + hangul) >= 0.05):
        return True
    return False


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


# ── 1. RAG 대상 데이터 구성 (question=passage_text, context=검색 결과, answer=실제 생성) ──

def _item_to_statement(it: dict) -> str:
    """문항을 검증 가능한 형태로 변환. 질문 문장만으로는 원자적 주장으로 쪼갤
    근거가 없어(의문문이라 참/거짓 판단 대상이 아님) 객관식은 정답 선지 내용까지
    포함시켜 "질문 (정답: ...)" 형태로 만든다 — 서술형은 정해진 정답이 없어 한계로 남음."""
    q = it.get("question", "")
    if it.get("item_type") == "객관식" and it.get("answer"):
        opts = it.get("options", [])
        match = next((o for o in opts if o.strip().startswith(it["answer"].strip())), None)
        if match:
            return f"{q} (정답: {match})"
    return q


def build_sample(sample: dict) -> dict:
    """실제 그래프를 호출해 문항을 생성하고, 같은 standards로 실제 검색을 수행해
    (question, context, answer)를 구성한다."""
    from app.modules.exam import ExamSpec, get_exam_graph
    from app.modules.exam.tools import get_draft_items, init_session

    spec: ExamSpec = {
        "passage_text": sample["passage_text"],
        "standards": sample.get("standards", []),
        "num_items": sample["num_items"],
    }
    init_session()
    graph = get_exam_graph()
    state = graph.invoke(
        {
            "spec": spec, "budget": 1, "draft_items": [], "agent_messages": [],
            "validation_passed": False, "similarity_judge_result": {},
        }
    )
    items = get_draft_items()
    answer = "\n".join(f"- {_item_to_statement(it)}" for it in items)

    query = ", ".join(sample.get("standards", [])) or sample["passage_text"][:50]
    context_chunks = get_retriever().retrieve(query, "standards", top_k=3)
    context = "\n".join(c["text"] for c in context_chunks)

    return {
        "id": sample["id"],
        "question": sample["passage_text"],
        "context": context,
        "answer": answer,
        "n_items": len(items),
        "items": items,  # 개별 문항 원본 — 실측 생성 품질(judge_one) 채점용, 재생성 없이 재사용
    }


# ── 2. Faithfulness: 답변 → 원자적 주장 분해 → 컨텍스트 대조 ──

CLAIM_DECOMPOSE_TPL = PromptTemplate(
    system=(
        "아래 [답변]을 더 이상 쪼갤 수 없는 짧은 사실 주장(atomic claim) 목록으로 분해하세요. "
        "각 주장은 독립적으로 참/거짓을 판단할 수 있는 서술문(평서문)이어야 합니다. "
        "**[답변]이 '(정답: ...)'가 붙은 객관식 문제 형태라면, 질문 자체를 그대로 옮기지 말고 "
        "'질문+정답'이 실제로 의미하는 사실 내용을 서술문으로 바꿔서 추출하세요** "
        "(예: '~은?(정답: ②A)' → 'A는 ~이다'). 정답 표시가 없는 서술형 질문은 그 안에 담긴 "
        "전제나 개념만 주장으로 뽑고, 답할 수 없는 질문 자체는 주장에 포함하지 마세요. "
        "**반드시 한국어로만 응답하세요(중국어·영어 등 다른 언어 섞지 말 것).**\n"
        '형식: {"claims": ["주장1", "주장2", ...]}'
    ),
    few_shots=[
        {
            "user": "[답변]\n- 지방분권이 필요한 배경으로 가장 적절한 것은 지역 실정에 맞는 행정 실현이다.",
            "assistant": '{"claims": ["지방분권이 필요한 배경 중 하나는 지역 실정에 맞는 행정 실현이다."]}',
        },
        {
            "user": "[답변]\n- 헌법이 보장하는 기본권 중 자유권에 해당하는 것은? (정답: ② 신체의 자유)",
            "assistant": '{"claims": ["신체의 자유는 헌법이 보장하는 기본권 중 자유권에 해당한다."]}',
        },
        {
            "user": "[답변]\n- 다음 중 정치 참여 방법이 아닌 것은?",
            "assistant": '{"claims": []}',
        },
    ],
    cot_prefix="",
)

CLAIM_VERIFY_TPL = PromptTemplate(
    system=(
        "아래 [컨텍스트]만 근거로 [주장]이 뒷받침되는지 판단하세요. 컨텍스트에 명시돼 있거나 "
        "합리적으로 추론 가능하면 SUPPORTED, 컨텍스트에 없거나 모순되면 UNSUPPORTED로만 응답하세요."
    ),
    few_shots=[
        {
            "user": "[컨텍스트] 지방분권은 지역 특성에 맞는 행정을 가능하게 한다.\n[주장] 지방분권은 지역 실정에 맞는 행정을 가능하게 한다.",
            "assistant": "SUPPORTED",
        },
        {
            "user": "[컨텍스트] 지방분권은 지역 특성에 맞는 행정을 가능하게 한다.\n[주장] 지방분권은 국가 예산을 두 배로 늘린다.",
            "assistant": "UNSUPPORTED",
        },
    ],
    cot_prefix="",
)


@traceable(name="faithfulness_one", run_type="chain", metadata=_TRACE_META)
def faithfulness_one(item: dict, llm) -> dict:
    """claims가 0개인 경우(예: 정답 없는 서술형 질문만 있어 검증 가능한 주장이 없음)는
    "불성실하다(0.0)"가 아니라 "이 항목은 채점 대상이 아님"이므로 score를 None으로
    반환 — 호출부(eval_ragas)에서 평균 계산 시 제외한다."""
    if not item["answer"].strip():
        return {"claims": [], "supported": 0, "total": 0, "score": None}

    raw = _run_async(llm.generate(CLAIM_DECOMPOSE_TPL.build(f"[답변]\n{item['answer']}")))
    try:
        s, e = raw.find("{"), raw.rfind("}") + 1
        claims = json.loads(raw[s:e]).get("claims", []) if s >= 0 and e > s else []
    except Exception:
        claims = []
    claims = [c for c in claims if not _is_contaminated(c)]
    if not claims:
        return {"claims": [], "supported": 0, "total": 0, "score": None}

    supported = 0
    for claim in claims:
        verdict = _run_async(
            llm.generate(CLAIM_VERIFY_TPL.build(f"[컨텍스트] {item['context']}\n[주장] {claim}"))
        ).strip().upper()
        if verdict.startswith("SUPPORTED"):
            supported += 1

    return {
        "claims": claims,
        "supported": supported,
        "total": len(claims),
        "score": round(supported / len(claims), 3),
    }


# ── 3. Answer Relevancy: 답변 → 역질문 생성 → 임베딩 코사인 유사도 ──

REVERSE_QUESTION_TPL = PromptTemplate(
    system=(
        "아래 [답변]을 보고, 이 답변이 나올 법한 질문을 3개 생성하세요. "
        "실제 원래 질문을 모르는 상태에서 답변만 보고 역으로 추측하세요. "
        "**반드시 한국어로만 응답하세요(중국어·영어 등 다른 언어 섞지 말 것).**\n"
        '형식: {"questions": ["질문1", "질문2", "질문3"]}'
    ),
    few_shots=[
        {
            "user": "[답변]\n- 지방분권이 필요한 배경으로 가장 적절한 것은 지역 실정에 맞는 행정 실현이다.",
            "assistant": '{"questions": ["지방분권은 왜 필요한가?", "지방분권의 목적은 무엇인가?", "지방자치가 필요한 이유는 무엇인가?"]}',
        },
    ],
    cot_prefix="",
)


@traceable(name="answer_relevancy_one", run_type="chain", metadata=_TRACE_META)
def answer_relevancy_one(item: dict, llm, embedder) -> dict:
    """questions가 0개(파싱 실패·언어 오염으로 전부 걸러짐)인 경우도 faithfulness와
    동일하게 "관련 없음(0.0)"이 아니라 "채점 불가"이므로 score를 None으로 반환."""
    if not item["answer"].strip():
        return {"reverse_questions": [], "score": None}

    raw = _run_async(llm.generate(REVERSE_QUESTION_TPL.build(f"[답변]\n{item['answer']}")))
    try:
        s, e = raw.find("{"), raw.rfind("}") + 1
        questions = json.loads(raw[s:e]).get("questions", []) if s >= 0 and e > s else []
    except Exception:
        questions = []
    questions = [q for q in questions if not _is_contaminated(q)]
    if not questions:
        return {"reverse_questions": [], "score": None}

    vectors = embedder.embed([item["question"]] + questions)
    q_vec, reverse_vecs = vectors[0], vectors[1:]
    sims = [_cosine(q_vec, rv) for rv in reverse_vecs]

    return {
        "reverse_questions": questions,
        "similarities": [round(s, 3) for s in sims],
        "score": round(sum(sims) / len(sims), 3) if sims else 0.0,
    }


@traceable(name="eval_ragas", run_type="chain", metadata=_TRACE_META)
def eval_ragas(sample_ids: list[str]) -> dict:
    llm = get_llm_backend()
    judge_llm = get_judge_backend()  # 실측 생성 품질(judge_one)용 — ITEM_GOLDEN과 같은 Judge라야 비교 가능
    embedder = get_embedder()
    samples = [s for s in PASSAGE_SAMPLES if s["id"] in sample_ids]

    results = []
    generated_items: list[dict] = []
    for i, sample in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {sample['id']} 처리 중...")
        item = build_sample(sample)
        if item["n_items"] == 0:
            print(f"  -> 문항 0개 생성, 건너뜀")
            continue
        generated_items.extend(item["items"])
        faith = faithfulness_one(item, llm)
        rel = answer_relevancy_one(item, llm, embedder)
        print(f"  -> faithfulness={faith['score']}, answer_relevancy={rel['score']}")
        results.append({"id": item["id"], "n_items": item["n_items"], "faithfulness": faith, "answer_relevancy": rel})

    n = len(results)
    if n == 0:
        return {"n": 0, "note": "유효한 샘플이 없습니다."}

    faith_scores = [r["faithfulness"]["score"] for r in results if r["faithfulness"]["score"] is not None]
    rel_scores = [r["answer_relevancy"]["score"] for r in results if r["answer_relevancy"]["score"] is not None]
    avg_faith = round(sum(faith_scores) / len(faith_scores), 3) if faith_scores else None
    avg_rel = round(sum(rel_scores) / len(rel_scores), 3) if rel_scores else None

    # 실측 생성 품질(참고값, 게이트 없음) — ITEM_GOLDEN(고정 30문항)이 아니라 이번 실행에서
    # 실제로 생성된 문항 기준. 재생성 없이 위 루프에서 이미 만든 문항을 그대로 재사용한다.
    item_quality_scored = score_items(generated_items, judge_llm) if generated_items else []
    item_quality = eval_item_quality(item_quality_scored) if item_quality_scored else None

    return {
        "n": n,
        "n_faithfulness_applicable": len(faith_scores),
        "n_answer_relevancy_applicable": len(rel_scores),
        "avg_faithfulness": avg_faith,
        "avg_answer_relevancy": avg_rel,
        "item_quality": item_quality,
        "item_quality_scored": item_quality_scored,
        "results": results,
    }


# ── LangSmith Experiments 연동 ────────────────────────────────────────

def run_langsmith_experiments(results: list[dict]) -> None:
    """RAG Faithfulness/Answer Relevancy를 LangSmith Experiments에 기록한다.

    2026-08-04: **이미 eval_ragas()가 생성·채점한 결과를 받아 조회만 한다**(재생성
    안 함). 이전에는 target 함수가 build_sample()을 직접 호출해, 콘솔 리포트용으로
    eval_ragas()가 이미 생성한 8개 샘플을 **여기서 다시 생성**했다 — temperature 0.7의
    확률적 생성이라 두 번째 생성은 리포트에 찍힌 것과 다른 문항이 나올 수 있었고,
    ReAct 에이전트 전체 루프(가장 비싼 부분)가 실행당 2배로 돌았다. 문항 품질·구조
    Judge(eval_exam.py, 2026-08-04에 먼저 고쳤음)와 같은 종류의 중복 호출 문제였다.
    """
    from langsmith_experiments import experiments_enabled, identity_target, sync_dataset
    if not experiments_enabled():
        return

    from langsmith import Client, evaluate

    client = Client()
    print("\n[LangSmith Experiments 연동]")

    samples = [s for s in PASSAGE_SAMPLES if s["id"] in _SAMPLE_IDS]
    examples = [
        {"inputs": {"id": s["id"], "passage_text": s["passage_text"],
                    "standards": s.get("standards", []), "num_items": s["num_items"]}}
        for s in samples
    ]
    sync_dataset(
        client, "bunpil-rag-quality", examples,
        description="RAG Faithfulness/Answer Relevancy(Ragas 알고리즘 자체 구현) — gen_structure_golden.PASSAGE_SAMPLES 기준 실제 생성",
    )

    # id는 PASSAGE_SAMPLES 안에서 유일 — 재생성 없이 조회 키로 쓴다.
    results_by_id = {r["id"]: r for r in results}

    def faithfulness_evaluator(outputs: dict) -> dict:
        r = results_by_id.get(outputs.get("id"))
        if r is None:
            return {"key": "faithfulness", "score": None, "comment": "문항 0개 생성"}
        return {"key": "faithfulness", "score": r["faithfulness"]["score"]}

    def answer_relevancy_evaluator(outputs: dict) -> dict:
        r = results_by_id.get(outputs.get("id"))
        if r is None:
            return {"key": "answer_relevancy", "score": None, "comment": "문항 0개 생성"}
        return {"key": "answer_relevancy", "score": r["answer_relevancy"]["score"]}

    evaluate(
        identity_target, data="bunpil-rag-quality",
        evaluators=[faithfulness_evaluator, answer_relevancy_evaluator],
        experiment_prefix="rag-quality", metadata=_TRACE_META,
    )
    print("  - rag-quality 실험 기록 완료")


def print_report(result: dict):
    print("\n" + "=" * 55)
    print("  분필 RAG 품질 평가 (Ragas 알고리즘 자체 구현)")
    print("=" * 55)
    if result.get("note"):
        # eval_ragas()가 n==0(전 샘플 문항 생성 실패)일 때 반환하는 축약 dict —
        # avg_faithfulness 등 나머지 키가 아예 없으므로 여기서 끝낸다(KeyError 방지).
        print(f"\n{result['note']} (n={result.get('n', 0)})")
        print("=" * 55)
        return
    print(f"\nn = {result['n']}")
    faith_str = f"{result['avg_faithfulness']:.3f}" if result["avg_faithfulness"] is not None else "N/A"
    print(f"Faithfulness (평균, n={result.get('n_faithfulness_applicable', 0)}) : {faith_str}")
    rel_str = f"{result['avg_answer_relevancy']:.3f}" if result["avg_answer_relevancy"] is not None else "N/A"
    print(f"Answer Relevancy (평균, n={result.get('n_answer_relevancy_applicable', 0)}) : {rel_str}")
    print("=" * 55)

    quality = result.get("item_quality")
    print("\n[실측 생성 품질] (참고값, 게이트 없음)")
    print("  ITEM_GOLDEN(고정 30문항)이 아니라 이번 실행에서 실제로 생성된 문항 기준")
    print("  — ITEM_GOLDEN 기반 문항품질·kappa·±1 지표(eval_exam.py)는 Judge 신뢰도를")
    print("  재는 것이지 생성 품질이 아니다. 이 지표가 생성 품질 쪽이다.")
    if quality is None:
        print("  n=0 (생성된 문항 없음, 채점 불가)")
    else:
        print(f"  n={quality['n']}")
        for name in ("정답유일성", "오답매력도", "근거성"):
            print(
                f"    {name:<12}평균 {quality[f'avg_{name}']:.2f}  "
                f"저품질(≤3) {quality[f'low_rate_{name}']*100:.0f}%"
            )
        print(f"    종합평균    : {quality['avg_overall']:.2f} (참고, 게이트 없음)")
    print("=" * 55)


if __name__ == "__main__":
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        print("LangSmith 트레이싱: 활성화됨")
    print("=== RAG 품질 평가 시작 ===\n")

    result = eval_ragas(_SAMPLE_IDS)
    print_report(result)

    _out_path = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "_ragas_eval_results.json")
    with open(_out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {_out_path}")

    run_langsmith_experiments(result.get("results", []))
