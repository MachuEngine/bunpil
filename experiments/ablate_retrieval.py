#!/usr/bin/env python
"""검색(RAG) 제거 실험 — 생성 단계가 search_standards(성취기준 검색) 결과를 실제로
쓰는지 확인한다.

배경(관측, 2026-10): 세 생성 모델 모두 search_standards를 호출하지만, 저장된 문항의
standard 필드는 15~26%만 채워지고 그 값은 전부 모델이 지은 주제명(성취기준 코드 0건)이다.
검색 결과 자체도 교육과정 "내용 요소" 목록 조각이라 코드가 없다. 이 스크립트는 운영 생성
모델(gpt-6-luna)로 지문 33개 × 조건 2(normal / no_retrieval) × 반복 2회를 생성해, 검색
결과를 빼도 문항 품질이 달라지지 않는지(=생성이 검색 결과를 실제로 쓰지 않는다는 신호인지)
Judge(기본 anthropic/claude-sonnet-5.5)로 비교한다.

서브커맨드:
  generate  지문 1개당 1회 운영 경로(그대로)를 실행해
            data/golden/_retrieval_ablation/{condition}_r{repeat}.jsonl에 즉시 append
            (이어서 실행 가능). no_retrieval 조건은 app.modules.exam.tools의
            get_retriever를 빈 결과를 돌려주는 가짜로 monkeypatch한다 — 도구 자체·
            프롬프트는 그대로 두고(프롬프트가 이미 "관련 자료가 없다고 나오면 성취기준
            없이 진행"이라고 지시함), 검색 계층만 끈다.
  judge     4개 run 파일(조건 2 × 반복 2)의 객관식 문항을 judge_one()으로 1회씩 채점해
            data/golden/_retrieval_ablation/judged_<judge-slug>_<condition>_r<repeat>.jsonl
            캐시에 append(이어서 실행).
  compare   지문 단위 짝 비교(같은 지문에서 normal 평균 vs no_retrieval 평균, 두 반복을
            지문 안에서 평균) + 규칙 지표(게이트 통과율·목표 개수 달성률·평균 시도 수·
            지연·search_standards 호출 수·standard 필드 기재율·성취기준 코드 패턴
            비율) + 반복 간 차이(참고)를 계산해 data/golden/_retrieval_ablation/
            comparison.json에 저장.

하드룰 2(마스킹은 모델 호출 이전): generate는 golden_gen/gen_item_quality_golden.py의
`_run_passage`를 그대로 재사용한다 — 그 함수가 app.main._build_spec()으로 PII 마스킹 →
요청 분석을 모델 호출(그래프 실행) 이전에 끝내는 운영 경로를 그대로 탄다.
하드룰 4(로그·캐시에 원문 금지): 콘솔 출력·run 파일·judged 캐시 모두 지문 원문을 새로
찍지 않는다(run 파일의 items 보존은 gen_item_quality_golden.py와 같은 전례를 따름 —
합성 데이터, 사람 라벨링 재료는 아니고 Judge 비교 재료). 자가 점검은 검색 결과의 "종류"
(자료없음/결과없음/결과있음)만 출력하고 원문은 출력하지 않는다.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# CHROMA_PERSIST_DIR: 로컬 .env는 /data/chroma_db로 설정돼 있어(배포 환경 기준) 로컬에서
# 실행하면 실패한다. 셸에 이미 설정돼 있으면(배포 환경 등) 그 값을 존중하고, 없으면
# load_dotenv()가 .env 값으로 채우기 전에 미리 표시해둔 뒤 로컬 경로로 덮어쓴다
# (공용 헬퍼로 교체, 2026-10-07 — evals/eval_exam.py 등 다른 스크립트와 로직 통일).
_had_chroma_dir = "CHROMA_PERSIST_DIR" in os.environ
load_dotenv()

from evals.local_env import use_local_chroma_dir
use_local_chroma_dir(_had_chroma_dir)

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "golden")
_INPUTS_PATH = os.path.join(_GOLDEN_DIR, "item_quality_inputs.json")
_OUT_DIR = os.path.join(_GOLDEN_DIR, "_retrieval_ablation")

_CONDITIONS = ("normal", "no_retrieval")
_REPEATS = (1, 2)
_GEN_MODEL = "gpt-6-luna"  # 운영 생성 모델(2026-10 사용자 결정) — 이 실험은 1개 모델만 다룬다.
_CRITERIA = ("정답유일성", "오답매력도", "근거성")

_DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-5.5"

# standard 필드에 성취기준 코드가 있는지 판정하는 근사 패턴 — 2022 개정 교육과정 코드는
# "학년군+과목약어+영역-일련번호"(숫자+한글+숫자-숫자) 형태다(예: "9사(일사)01-01"). 완벽한
# 정규식은 아니고 "코드처럼 보이는 문자열"을 거르는 참고 지표다.
_STANDARD_CODE_RE = re.compile(r"\d+[가-힣]+\d+-\d+")

_METRIC_DESCRIPTIONS = {
    "정답유일성": "객관식 문항에 정답이 오직 하나인지(Judge 1~5)",
    "오답매력도": "오답 선지가 그럴듯한지(Judge 1~5)",
    "근거성": "교육과정 근거 충실도(Judge 1~5) — Judge 신뢰도(가중 κ) 미확정(0.56)이라 참고",
    "overall": "세 기준 평균(Judge)",
    "gate_pass_rate": "최종 게이트(구조 유사도 Judge) 통과율",
    "count_ok_rate": "목표 문항 개수 달성률",
    "avg_attempts": "세트당 평균 시도 수(적을수록 효율적)",
    "latency_sec": "세트당 생성 소요 시간(초, 적을수록 빠름)",
    "search_standards_calls": "세트당 search_standards 호출 수",
    "standard_filled_rate": "저장된 문항 중 standard 필드가 채워진 비율",
    "standard_code_rate": "standard 필드에 성취기준 코드 패턴이 있는 비율",
}


def _load_inputs() -> list[dict]:
    with open(_INPUTS_PATH, encoding="utf-8") as f:
        return json.load(f)["entries"]


def select_entries(all_entries: list[dict], ids: list[str] | None = None) -> list[dict]:
    """--ids가 있으면 그 id만, 없으면 전체(LLM 호출 없음, 테스트 가능)."""
    if ids:
        id_set = set(ids)
        return [e for e in all_entries if e["id"] in id_set]
    return list(all_entries)


def load_run_records(path: str) -> list[dict]:
    """run/judged jsonl 파일을 읽어 레코드 리스트로 반환한다. 파일이 없으면 빈 리스트."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _run_path(condition: str, repeat: int) -> str:
    return os.path.join(_OUT_DIR, f"{condition}_r{repeat}.jsonl")


def _judge_slug(model: str) -> str:
    return model.replace("/", "__")


def _judged_path(judge_model: str, condition: str, repeat: int) -> str:
    return os.path.join(_OUT_DIR, f"judged_{_judge_slug(judge_model)}_{condition}_r{repeat}.jsonl")


def judged_item_keys(records: list[dict]) -> set[tuple[str, str]]:
    """judged 캐시에서 이미 채점 완료된 (run_id, item_id) 집합 — 이어서 실행 시 스킵 판정."""
    return {(r["run_id"], r["item_id"]) for r in records}


# ── generate: no_retrieval 패치 + 집계 ───────────────────────────────────

def classify_search_standards_output(text: str) -> str:
    """search_standards()의 반환 문자열을 세 범주로 분류한다(원문은 보존하지 않음, 하드룰 4).

    자료없음: standards 컬렉션이 비어 있음(get_store().count==0) — 이 실험이 전제하는
              "검색은 되는데 생성이 안 쓴다"는 상태가 아니다.
    결과없음: 컬렉션은 있지만 이 질의로 찾은 성취기준이 없음 — no_retrieval 패치가
              걸렸을 때 기대하는 값.
    결과있음: 성취기준 내용이 돌아옴(정상 검색) — normal 조건에서 기대하는 값."""
    if text == "교육과정 성취기준 자료 없음":
        return "자료없음"
    if text == "관련 성취기준 없음":
        return "결과없음"
    return "결과있음"


class _EmptyRetriever:
    """no_retrieval 조건에서 get_retriever()를 대체하는 가짜 — retrieve()가 항상 빈
    리스트를 돌려줘 search_standards가 '관련 성취기준 없음'을 반환하게 한다. 도구 자체·
    프롬프트는 수정하지 않는다(프롬프트가 이미 그 경우 성취기준 없이 진행하라고 지시함)."""

    def retrieve(self, *args, **kwargs) -> list:
        return []


class CountingRetriever:
    """실제(또는 가짜) retriever를 감싸 retrieve() 호출 수와 빈 결과 횟수를 센다.
    reset()으로 지문마다 집계를 끊어 세트 단위 집계를 얻는다."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0
        self.empty = 0

    def retrieve(self, *args, **kwargs):
        self.calls += 1
        results = self._inner.retrieve(*args, **kwargs)
        if not results:
            self.empty += 1
        return results

    def reset(self) -> None:
        self.calls = 0
        self.empty = 0


def apply_retrieval_patch(tools_module, condition: str, real_get_retriever) -> CountingRetriever:
    """tools_module.get_retriever를 조건에 맞게 monkeypatch하고, 호출 집계용
    CountingRetriever를 반환한다.

    no_retrieval: 내부를 _EmptyRetriever로 — search_standards가 항상 빈 결과를 본다.
    normal: 내부를 real_get_retriever()(실제 싱글톤)로 — 호출 집계만 추가하고 동작은
    그대로다."""
    inner = _EmptyRetriever() if condition == "no_retrieval" else real_get_retriever()
    counting = CountingRetriever(inner)
    tools_module.get_retriever = lambda: counting
    return counting


def cmd_generate(args: argparse.Namespace) -> None:
    # 모델 env는 golden_gen/gen_item_quality_golden.py의 gpt-6-luna 설정을 그대로
    # 재사용한다 — app.* import 전에 설정해야 한다(그 모듈의 관례와 동일).
    from golden_gen.gen_item_quality_golden import _JUDGE_ENV, _MODEL_ENV, _code_version, _run_passage

    for k, v in _MODEL_ENV[_GEN_MODEL].items():
        os.environ[k] = v
    for k, v in _JUDGE_ENV.items():
        os.environ[k] = v

    from app.common.llm.tracing import init_langsmith_project
    init_langsmith_project()

    import app.modules.exam.tools as tools_module
    from app.common.rag import get_retriever as _real_get_retriever

    counting = apply_retrieval_patch(tools_module, args.condition, _real_get_retriever)

    print(f"=== 조건: {args.condition} / repeat: {args.repeat_index} / 생성 모델: {_GEN_MODEL} ===")

    # 자가 점검 — 패치가 실제로 적용됐는지 결과 "종류"만 확인한다(원문 미출력, 하드룰 4).
    check_result = tools_module.search_standards.invoke({"query": "자가진단"})
    category = classify_search_standards_output(check_result)
    print(f"자가 점검 — search_standards 결과 종류={category}")
    if args.condition == "no_retrieval" and category != "결과없음":
        print("중단 — no_retrieval 패치가 기대대로 적용되지 않았습니다(결과 종류가 '결과없음'이 아님).")
        sys.exit(1)
    if args.condition == "normal" and category == "자료없음":
        print("중단 — standards 컬렉션이 비어 있어 normal 조건의 비교 기준이 되지 못합니다.")
        sys.exit(1)
    counting.reset()  # 자가 점검 호출은 집계에서 제외

    all_entries = _load_inputs()
    entries = select_entries(all_entries, ids=args.ids)

    out_path = _run_path(args.condition, args.repeat_index)
    os.makedirs(_OUT_DIR, exist_ok=True)
    done_ids = {r["id"] for r in load_run_records(out_path) if not r.get("error")}

    print(f"대상 지문 {len(entries)}개 (이미 완료 {len(done_ids & {e['id'] for e in entries})}개 스킵)\n")

    with open(out_path, "a", encoding="utf-8") as f:
        for i, entry in enumerate(entries, 1):
            if entry["id"] in done_ids:
                print(f"[{i}/{len(entries)}] {entry['id']} 이미 완료 — 스킵")
                continue
            counting.reset()
            start = time.perf_counter()
            try:
                record = _run_passage(_GEN_MODEL, entry, budget=5)
            except Exception as e:
                elapsed = time.perf_counter() - start
                record = {
                    "id": entry["id"],
                    "model": _GEN_MODEL,
                    "num_items": None,
                    "format": None,
                    "items": [],
                    "validation_passed": False,
                    "attempts": 0,
                    "attempt_log": [],
                    "input_tokens_total": None,
                    "output_tokens_total": None,
                    "cost_usd_est": None,
                    "wall_clock_sec": elapsed,
                    "error": {"type": type(e).__name__, "message": str(e)[:500]},
                    "code_version": _code_version(),
                }
                print(f"[{i}/{len(entries)}] {entry['id']} 실패: {type(e).__name__} ({elapsed:.1f}s)")
            else:
                print(
                    f"[{i}/{len(entries)}] {entry['id']} {record['wall_clock_sec']:.1f}s, "
                    f"{len(record['items'])}문항, 통과={record['validation_passed']}, "
                    f"search_standards 호출={counting.calls}(빈 결과 {counting.empty})"
                )
            record["condition"] = args.condition
            record["repeat_index"] = args.repeat_index
            record["search_standards_calls"] = counting.calls
            record["search_standards_empty_calls"] = counting.empty
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\n완료 — {out_path}")


# ── judge ────────────────────────────────────────────────────────────────

def cmd_judge(args: argparse.Namespace) -> None:
    from app.common.llm.backends.openrouter import OpenRouterBackend
    from evals.eval_item_quality_runs import mc_items_for_run
    from evals.eval_lib import judge_one
    from golden_gen.gen_item_quality_golden import dedupe_run_records

    judge_llm = OpenRouterBackend(model=args.judge_model)
    os.makedirs(_OUT_DIR, exist_ok=True)

    print(f"=== Judge: {args.judge_model} ===")
    for condition in _CONDITIONS:
        for repeat in _REPEATS:
            records = dedupe_run_records(load_run_records(_run_path(condition, repeat)))
            out_path = _judged_path(args.judge_model, condition, repeat)
            done = judged_item_keys(load_run_records(out_path))

            n_judged = n_skipped = 0
            with open(out_path, "a", encoding="utf-8") as f:
                for run in records:
                    for item in mc_items_for_run(run):
                        key = (run["id"], item["item_id"])
                        if key in done:
                            n_skipped += 1
                            continue
                        scores = judge_one(item, judge_llm)
                        rec = {
                            "run_id": run["id"],
                            "condition": condition,
                            "repeat_index": repeat,
                            "item_id": item["item_id"],
                            "scores": scores,
                            "judge_model": args.judge_model,
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        n_judged += 1
            print(f"[{condition} r{repeat}] 신규 채점 {n_judged}건, 캐시 스킵 {n_skipped}건 → {out_path}")


# ── compare: 지문 단위 짝 비교 + 규칙 지표 + 반복 간 잡음 ──────────────────

def standard_filled_rate_for_items(items: list[dict]) -> float | None:
    """저장된 문항 중 standard 필드가 채워진(공백 제외) 비율. items가 비어 있으면 None."""
    if not items:
        return None
    return sum(1 for it in items if str(it.get("standard") or "").strip()) / len(items)


def standard_code_rate_for_items(items: list[dict]) -> float | None:
    """standard 필드에 성취기준 코드 패턴(_STANDARD_CODE_RE)이 있는 비율. items가 비어
    있으면 None."""
    if not items:
        return None
    return sum(1 for it in items if _STANDARD_CODE_RE.search(str(it.get("standard") or ""))) / len(items)


def passage_level_gate_pass(records: list[dict]) -> dict[str, int]:
    return {r["id"]: int(bool(r.get("validation_passed"))) for r in records}


def passage_level_count_ok(records: list[dict]) -> dict[str, int]:
    return {
        r["id"]: int(r.get("num_items") is not None and len(r.get("items", [])) == r["num_items"])
        for r in records
    }


def passage_level_attempts(records: list[dict]) -> dict[str, int]:
    return {r["id"]: r.get("attempts", 0) for r in records if not r.get("error")}


def passage_level_latency(records: list[dict]) -> dict[str, float]:
    return {r["id"]: r.get("wall_clock_sec", 0.0) for r in records}


def passage_level_search_calls(records: list[dict]) -> dict[str, int]:
    return {r["id"]: r.get("search_standards_calls", 0) for r in records if not r.get("error")}


def passage_level_standard_filled(records: list[dict]) -> dict[str, float]:
    out = {}
    for r in records:
        rate = standard_filled_rate_for_items(r.get("items", []))
        if rate is not None:
            out[r["id"]] = rate
    return out


def passage_level_standard_code(records: list[dict]) -> dict[str, float]:
    out = {}
    for r in records:
        rate = standard_code_rate_for_items(r.get("items", []))
        if rate is not None:
            out[r["id"]] = rate
    return out


def passage_level_judge_scores(judged_records: list[dict], criterion: str) -> dict[str, float]:
    """judged 캐시((condition, repeat) 1개분)에서 지문(run_id)별 criterion 평균(문항이
    여러 개면 평균). criterion은 _CRITERIA 중 하나 또는 'overall'."""
    grouped: dict[str, list[float]] = {}
    for r in judged_records:
        grouped.setdefault(r["run_id"], []).append(r["scores"][criterion])
    return {pid: sum(vals) / len(vals) for pid, vals in grouped.items()}


def average_across_repeats(values_r1: dict[str, float], values_r2: dict[str, float]) -> dict[str, float]:
    """두 반복의 지문별 값을 지문 안에서 평균 — 두 반복 모두 값이 있는 지문만."""
    common = set(values_r1) & set(values_r2)
    return {pid: (values_r1[pid] + values_r2[pid]) / 2 for pid in common}


def rule_metrics_for_condition(records_r1: list[dict], records_r2: list[dict]) -> dict:
    """한 조건의 두 반복 run 레코드를 합쳐 낸 단순 비율/평균(참고용 요약, CI 없음) — 조건
    간 통계적 비교는 rule_pairwise_condition(지문 단위 CI)이 담당한다."""
    all_records = records_r1 + records_r2
    n = len(all_records)
    ok = [r for r in all_records if not r.get("error")]

    def _rate(fn):
        hits = sum(1 for r in all_records if fn(r))
        return round(hits / n, 3) if n else None

    gate_pass_rate = _rate(lambda r: bool(r.get("validation_passed")))
    count_ok_rate = _rate(
        lambda r: r.get("num_items") is not None and len(r.get("items", [])) == r["num_items"]
    )
    avg_attempts = round(sum(r.get("attempts", 0) for r in ok) / len(ok), 3) if ok else None
    avg_latency = round(sum(r.get("wall_clock_sec", 0.0) for r in all_records) / n, 3) if n else None
    avg_search_calls = (
        round(sum(r.get("search_standards_calls", 0) for r in ok) / len(ok), 3) if ok else None
    )
    all_items = [it for r in ok for it in r.get("items", [])]
    standard_filled_rate = standard_filled_rate_for_items(all_items)
    standard_code_rate = standard_code_rate_for_items(all_items)

    return {
        "n": n,
        "gate_pass_rate": gate_pass_rate,
        "count_ok_rate": count_ok_rate,
        "avg_attempts": avg_attempts,
        "avg_latency_sec": avg_latency,
        "avg_search_standards_calls": avg_search_calls,
        "standard_filled_rate": round(standard_filled_rate, 3) if standard_filled_rate is not None else None,
        "standard_code_rate": round(standard_code_rate, 3) if standard_code_rate is not None else None,
    }


# compare 서브커맨드의 짝 비교(지문 단위, higher_is_better 방향)에 쓰는 (지문별 값 추출
# 함수, higher_is_better) 매핑 — 적을수록 좋은 지표(시도 수·지연)는 False.
_RULE_PASSAGE_FNS = {
    "gate_pass_rate": (passage_level_gate_pass, True),
    "count_ok_rate": (passage_level_count_ok, True),
    "avg_attempts": (passage_level_attempts, False),
    "latency_sec": (passage_level_latency, False),
    "search_standards_calls": (passage_level_search_calls, True),
    "standard_filled_rate": (passage_level_standard_filled, True),
    "standard_code_rate": (passage_level_standard_code, True),
}


def compute_comparison(
    records_by_key: dict[tuple[str, int], list[dict]],
    judged_by_key: dict[tuple[str, int], list[dict]],
    judge_model: str,
) -> dict:
    """compare의 핵심 계산 전체(LLM 호출 없음) — cmd_compare와 테스트가 공유.

    records_by_key/judged_by_key 키는 (condition, repeat) — _CONDITIONS × _REPEATS
    전부(4개) 있어야 한다. records는 이미 dedupe_run_records() 적용된 값이어야 한다."""
    from evals.eval_item_quality_runs import paired_ci_verdict

    rule_metrics_by_condition = {
        c: rule_metrics_for_condition(records_by_key[(c, 1)], records_by_key[(c, 2)])
        for c in _CONDITIONS
    }

    def _quality_avg_across_repeats(condition: str, criterion: str) -> dict[str, float]:
        r1 = passage_level_judge_scores(judged_by_key[(condition, 1)], criterion)
        r2 = passage_level_judge_scores(judged_by_key[(condition, 2)], criterion)
        return average_across_repeats(r1, r2)

    quality_pairwise = {}
    for crit in (*_CRITERIA, "overall"):
        a = _quality_avg_across_repeats("normal", crit)
        b = _quality_avg_across_repeats("no_retrieval", crit)
        quality_pairwise[crit] = paired_ci_verdict(a, b, higher_is_better=True, label_a="normal", label_b="no_retrieval")

    def _rule_avg_across_repeats(condition: str, passage_fn) -> dict[str, float]:
        r1 = passage_fn(records_by_key[(condition, 1)])
        r2 = passage_fn(records_by_key[(condition, 2)])
        return average_across_repeats(r1, r2)

    rule_pairwise = {}
    for name, (fn, higher_is_better) in _RULE_PASSAGE_FNS.items():
        a = _rule_avg_across_repeats("normal", fn)
        b = _rule_avg_across_repeats("no_retrieval", fn)
        rule_pairwise[name] = paired_ci_verdict(a, b, higher_is_better=higher_is_better, label_a="normal", label_b="no_retrieval")

    repeat_noise = {}
    for c in _CONDITIONS:
        repeat_noise[c] = {}
        for crit in (*_CRITERIA, "overall"):
            r1 = passage_level_judge_scores(judged_by_key[(c, 1)], crit)
            r2 = passage_level_judge_scores(judged_by_key[(c, 2)], crit)
            repeat_noise[c][crit] = paired_ci_verdict(r1, r2, higher_is_better=True, label_a="r1", label_b="r2")

    return {
        "generation_model": _GEN_MODEL,
        "judge_model": judge_model,
        "metric_descriptions": _METRIC_DESCRIPTIONS,
        "groundedness_note": "근거성은 Judge 신뢰도(가중 κ) 미확정(0.56) 상태라 참고 지표로만 봅니다.",
        "rule_metrics_by_condition": rule_metrics_by_condition,
        "quality_pairwise_condition": quality_pairwise,
        "rule_pairwise_condition": rule_pairwise,
        "repeat_noise": repeat_noise,
    }


def _print_comparison(report: dict) -> None:
    """콘솔 출력 — 문항 원문은 출력하지 않는다(하드룰 4). 지표 이름 옆에 짧은 설명을 붙인다."""
    print(f"생성 모델: {report['generation_model']} / Judge: {report['judge_model']}")
    print(f"[참고] {report['groundedness_note']}")

    print("\n[조건별 규칙 지표] (두 반복 합산, 단순 비율·평균 — CI 없음)")
    for c in _CONDITIONS:
        m = report["rule_metrics_by_condition"][c]
        print(
            f"  {c} (n={m['n']})\n"
            f"    게이트 통과율={m['gate_pass_rate']}  목표 개수 달성률={m['count_ok_rate']}\n"
            f"    평균 시도 수={m['avg_attempts']}  세트당 지연(초)={m['avg_latency_sec']}\n"
            f"    search_standards 호출 수(세트당 평균)={m['avg_search_standards_calls']}\n"
            f"    standard 필드 기재율={m['standard_filled_rate']}  "
            f"성취기준 코드 패턴 비율={m['standard_code_rate']}"
        )

    print(
        "\n[문항 품질 짝 비교] normal vs no_retrieval "
        "(지문 단위, 두 반복을 지문 안에서 평균, CI가 0을 포함하면 판정 불가)"
    )
    for crit in (*_CRITERIA, "overall"):
        pw = report["quality_pairwise_condition"][crit]
        desc = report["metric_descriptions"][crit]
        print(f"  {crit}({desc}): n={pw['n']} mean_diff={pw['mean_diff']} CI={pw['ci']} → {pw['verdict']}")

    print("\n[규칙 지표 짝 비교] normal vs no_retrieval (지문 단위, 두 반복을 지문 안에서 평균)")
    for name, pw in report["rule_pairwise_condition"].items():
        desc = report["metric_descriptions"].get(name, "")
        print(f"  {name}({desc}): n={pw['n']} mean_diff={pw['mean_diff']} CI={pw['ci']} → {pw['verdict']}")

    print("\n[반복 간 차이(같은 조건 r1 vs r2)] 참고용 — 조건 차이가 반복 잡음보다 큰지 확인")
    for c in _CONDITIONS:
        print(f"  {c}")
        for crit in (*_CRITERIA, "overall"):
            pw = report["repeat_noise"][c][crit]
            print(f"    {crit}: n={pw['n']} mean_diff={pw['mean_diff']} CI={pw['ci']} → {pw['verdict']}")


def cmd_compare(args: argparse.Namespace) -> None:
    from golden_gen.gen_item_quality_golden import dedupe_run_records

    records_by_key = {
        (c, r): dedupe_run_records(load_run_records(_run_path(c, r)))
        for c in _CONDITIONS for r in _REPEATS
    }
    judged_by_key = {
        (c, r): load_run_records(_judged_path(args.judge_model, c, r))
        for c in _CONDITIONS for r in _REPEATS
    }

    report = compute_comparison(records_by_key, judged_by_key, args.judge_model)

    os.makedirs(_OUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUT_DIR, "comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _print_comparison(report)
    print(f"\n상세는 {out_path}에 저장했습니다.")


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="조건별로 운영 경로를 그대로 실행해 run jsonl에 append")
    p_gen.add_argument("--condition", required=True, choices=_CONDITIONS)
    p_gen.add_argument("--repeat-index", required=True, type=int, choices=_REPEATS)
    p_gen.add_argument("--ids", nargs="+", default=None, help="지정한 id만 실행")
    p_gen.set_defaults(func=cmd_generate)

    p_judge = sub.add_parser("judge", help="4개 run 파일의 객관식 문항을 Judge로 채점해 캐시에 append")
    p_judge.add_argument("--judge-model", default="anthropic/claude-sonnet-5.5")
    p_judge.set_defaults(func=cmd_judge)

    p_cmp = sub.add_parser("compare", help="normal vs no_retrieval 짝 비교 + 규칙 지표 + 반복 간 잡음")
    p_cmp.add_argument("--judge-model", default=_DEFAULT_JUDGE_MODEL)
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
