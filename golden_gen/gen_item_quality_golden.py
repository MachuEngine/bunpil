#!/usr/bin/env python
"""문항 품질(정답유일성·오답매력도·근거성) 라벨링용 골든셋 생성 + 모델 비교 지표 스크립트.

생성 모델 3종(qwen2.5:14b 로컬 Ollama / gpt-6-luna OpenRouter / gemini-3.8-flash OpenRouter)을
같은 입력 지문(`data/golden/item_quality_inputs.json`, 33개)으로 실제 운영 경로
(app.main._build_spec → get_exam_graph().stream(...))를 그대로 태워 생성한 문항을
(a) 모델을 블라인드한 사람 라벨링용 골든셋, (b) 모델별 비교용 규칙 지표로 가공한다.

서브커맨드:
  generate       지문 1개당 1회 그래프 실행 → data/golden/_item_quality_runs/{model}.jsonl에 즉시 append
  summarize      모델별 run 파일을 집계해 표로 출력(원문 미출력)
  build-labelset 3개 모델 run 파일에서 (지문, 모델)마다 객관식 1개씩 뽑아 블라인드 골든셋 생성
  export-sheet   JSON 직접 편집용 라벨링 시트(item_quality_labeling_sheet.json) 생성
  import-sheet   편집한 라벨링 시트의 라벨을 item_quality_golden.json에 반영

하드룰 2(마스킹은 모델 호출 이전): generate는 app.main._build_spec()을 그대로 호출해
PII 마스킹 → 요청 분석 순서를 운영과 동일하게 유지한다.
하드룰 4(로그·캐시에 원문 금지): 콘솔 출력은 id·모델·소요시간·문항 수·통과 여부만.
run 파일(_item_quality_runs/*.jsonl)에는 생성된 문항(items)을 그대로 적는다 — 이는
사람 라벨링(build-labelset)의 원재료로 쓰기 위한 것으로, structure_golden.json 등
기존 골든셋과 같은 전례(모델 출력 보존, 실제 학생 데이터 아님, 하드룰 1)를 따른다.
입력 지문 원문(passage_text)은 run 파일에 중복 저장하지 않는다 — build-labelset이
`data/golden/item_quality_inputs.json`에서 id로 다시 찾아 쓴다.
"""
import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# CHROMA_PERSIST_DIR: 로컬 .env는 배포 경로(/data/chroma_db)로 설정돼 있어 로컬에서
# 그대로 실행하면 RAGStore 초기화가 실패한다(evals/local_env.py 참고). 셸 명시 여부는
# load_dotenv() 호출 전에 캡처해야 한다.
_had_chroma_dir = "CHROMA_PERSIST_DIR" in os.environ
load_dotenv()

from evals.local_env import use_local_chroma_dir
use_local_chroma_dir(_had_chroma_dir)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "golden")
_INPUTS_PATH = os.path.join(_GOLDEN_DIR, "item_quality_inputs.json")
_RUNS_DIR = os.path.join(_GOLDEN_DIR, "_item_quality_runs")
_GOLDEN_PATH = os.path.join(_GOLDEN_DIR, "item_quality_golden.json")
_MODEL_MAP_PATH = os.path.join(_GOLDEN_DIR, "_item_quality_model_map.json")
_SHEET_PATH = os.path.join(_GOLDEN_DIR, "item_quality_labeling_sheet.json")

# 모델별 env — 반드시 app.* import 전에 설정한다(gen_structure_golden.py와 같은 관례).
# 평가의 API 호출은 OpenRouter로 통일한다(2026-10 사용자 결정) — qwen2.5-14b(로컬
# Ollama)만 예외, 나머지는 OpenRouter 경유로 호출해 같은 과금·요청 경로로 비교한다.
# Judge(JUDGE_BACKEND/OPENROUTER_JUDGE_MODEL)는 런타임과 동일하게 항상 gpt-5.6-luna로
# 명시 고정한다 — .env 값에 의존하지 않고, 생성 모델이 그대로 Judge로 새는 것도
# OPENROUTER_JUDGE_MODEL 미설정 시 즉시 실패(fail-fast)하는 factory.py
# get_judge_backend()가 보장한다(생성 모델 OPENROUTER_MODEL로 조용히 폴백하지 않음).
_MODEL_ENV = {
    "qwen2.5-14b": {"LLM_BACKEND": "local", "OLLAMA_MODEL": "qwen2.5:14b"},
    "gpt-6-luna": {"LLM_BACKEND": "openrouter", "OPENROUTER_MODEL": "openai/gpt-6-luna"},
    "gemini-3.8-flash": {"LLM_BACKEND": "openrouter", "OPENROUTER_MODEL": "google/gemini-3.8-flash"},
}
_JUDGE_ENV = {"JUDGE_BACKEND": "openrouter", "OPENROUTER_JUDGE_MODEL": "openai/gpt-5.6-luna"}
_MODELS = tuple(_MODEL_ENV)

# 단가표(1M 토큰당 USD, 입력/출력). 생성 모델 토큰만 — Judge(gpt-5.6-luna) 호출 비용은
# 세 모델 비교에 공통으로 끼는 고정비라 비교 목적상 제외한다.
# OpenRouter 경유로 바뀌어도 단가는 그대로(OpenRouter /api/v1/models 표기 단가가
# 각 제공사 직접 단가와 같음, 2026-10 확인).
_PRICE_PER_1M = {
    "qwen2.5-14b": (0.0, 0.0),
    "gpt-6-luna": (0.10, 0.50),
    "gemini-3.8-flash": (0.75, 3.75),
}

_SMOKE_FORMAT_ORDER = ("mc4", "mc5", "bogi_combo", "data")

_BROKEN_TOOL_CALL_NOTE = "도구 호출 형식이 손상"
_INCOMPLETE_NOTE = "아직 목표 문항 저장과 제출이 끝나지 않았습니다"
# graph.py agent_node가 도구 실행 예외·invalid_tool_calls에 돌려주는 ToolMessage의
# 공통 접두사(2026-10-07) — 도구별 오류 횟수 집계(tool_errors)에 쓴다.
_TOOL_CALL_ERROR_PREFIX = "도구 호출 오류"


def _run_path(model: str, smoke: bool) -> str:
    suffix = ".smoke.jsonl" if smoke else ".jsonl"
    return os.path.join(_RUNS_DIR, f"{model}{suffix}")


def _code_version() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(__file__),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _load_inputs() -> list[dict]:
    with open(_INPUTS_PATH, encoding="utf-8") as f:
        return json.load(f)["entries"]


def load_run_records(path: str) -> list[dict]:
    """run jsonl 파일을 읽어 레코드 리스트로 반환한다. 파일이 없으면 빈 리스트."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dedupe_run_records(records: list[dict]) -> list[dict]:
    """같은 id가 여러 번 append된 run(실패 후 재실행 등, 예: gpt-6-luna pilot-006)을
    id당 레코드 하나로 정리한다 — 그 id의 레코드 중 성공(error 없음)이 하나라도 있으면
    그중 마지막 것을, 전부 실패면 마지막 레코드를 쓴다. 반환 순서는 각 id가 입력에서
    처음 등장한 순서를 유지한다. cmd_summarize·evals/eval_item_quality_runs.py
    compare-generators가 공유한다."""
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for r in records:
        rid = r["id"]
        if rid not in groups:
            order.append(rid)
        groups.setdefault(rid, []).append(r)
    result = []
    for rid in order:
        group = groups[rid]
        successes = [r for r in group if not r.get("error")]
        result.append(successes[-1] if successes else group[-1])
    return result


# ── generate ─────────────────────────────────────────────────────────────

def select_entries(
    all_entries: list[dict],
    smoke: bool = False,
    ids: list[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """--smoke/--ids/--limit에 따라 처리할 입력 지문을 고른다(LLM 호출 없음, 테스트 가능).

    --smoke: 형식(mc4, mc5, bogi_combo, data)별 case_type=normal인 첫 지문 1개씩, 총 4개.
    --ids: 지정한 id만.
    둘 다 없으면 전체. --limit은 마지막에 앞에서부터 자른다."""
    if smoke:
        selected = []
        for fmt in _SMOKE_FORMAT_ORDER:
            for e in all_entries:
                if e["format"] == fmt and e["case_type"] == "normal":
                    selected.append(e)
                    break
    elif ids:
        id_set = set(ids)
        selected = [e for e in all_entries if e["id"] in id_set]
    else:
        selected = list(all_entries)
    if limit is not None:
        selected = selected[:limit]
    return selected


def _summarize_agent_messages(messages: list) -> dict:
    """한 번의 agent_node 실행(시도 1회)이 만든 메시지 리스트에서 집계값을 뽑는다.

    반환: {"tool_calls": {도구명: 호출 수}, "tool_errors": {도구명: 오류 ToolMessage 수},
    "malformed_retries": int, "incomplete_retries": int, "input_tokens": int|None,
    "output_tokens": int|None}. 토큰은 AIMessage.usage_metadata가 하나도 없으면
    None(없으면 null), 하나라도 있으면 있는 것만 합산한다.

    2026-10-07: tool_errors는 graph.py agent_node가 도구 실행 예외·깨진 인자에
    "도구 호출 오류"로 시작하는 ToolMessage를 돌려준 횟수를 도구별로 센다 — 검색
    백엔드 고장처럼 조용히 삼켜지던 환경 문제를 집계로 드러내기 위함(로그 경고와
    별개로, 생성 하네스 쪽 가시성)."""
    tool_calls: dict[str, int] = {}
    tool_errors: dict[str, int] = {}
    call_id_to_tool: dict[str, str] = {}
    malformed_retries = 0
    incomplete_retries = 0
    input_tokens = None
    output_tokens = None
    has_tokens = False
    first_human_seen = False
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", []) or []:
                tool_calls[tc["name"]] = tool_calls.get(tc["name"], 0) + 1
                if tc.get("id"):
                    call_id_to_tool[tc["id"]] = tc["name"]
            meta = getattr(m, "usage_metadata", None)
            if meta:
                has_tokens = True
                input_tokens = (input_tokens or 0) + (meta.get("input_tokens") or 0)
                output_tokens = (output_tokens or 0) + (meta.get("output_tokens") or 0)
        elif isinstance(m, ToolMessage):
            content = str(m.content or "")
            if content.startswith(_TOOL_CALL_ERROR_PREFIX):
                tool_name = call_id_to_tool.get(m.tool_call_id, "unknown")
                tool_errors[tool_name] = tool_errors.get(tool_name, 0) + 1
        elif isinstance(m, HumanMessage):
            # 첫 HumanMessage는 agent_node가 매 시도 시작 시 보내는 고정 지시문("위 지침에
            # 따라 문항을 작성하세요") — 재요청이 아니므로 집계에서 제외한다.
            if not first_human_seen:
                first_human_seen = True
                continue
            content = str(m.content or "")
            if _BROKEN_TOOL_CALL_NOTE in content:
                malformed_retries += 1
            elif _INCOMPLETE_NOTE in content:
                incomplete_retries += 1
    if not has_tokens:
        input_tokens = output_tokens = None
    return {
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "malformed_retries": malformed_retries,
        "incomplete_retries": incomplete_retries,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _run_passage(model: str, entry: dict, budget: int = 5) -> dict:
    """지문 하나를 운영 경로(그대로)로 실행해 run 레코드 1건을 만든다.

    하드룰 2: _build_spec()이 PII 마스킹 → 요청 분석을 모델 호출(그래프 실행) 이전에
    끝낸다.

    init_session()은 graph.stream() 호출 전에 같은 스레드에서 먼저 불러야 한다
    (app/main.py _run_exam·_run_exam_events와 같은 패턴) — LangGraph가 각 노드를
    context.run()으로 격리 실행해서, plan_node 안에서만 set()하면 그 dict가 agent_node로
    전파되지 않는다(app/modules/exam/tools.py init_session() 주석 참고). 미리 부르지
    않으면 이 스크립트 프로세스의 첫 지문에서 _request_ctx가 비어 있어 plan_node의
    set()이 격리되고, get_draft_items()가 LookupError를 내거나 이전 호출의 dict를
    그대로 보는 문제가 있었다."""
    from app.main import _build_spec
    from app.modules.exam import get_exam_graph
    from app.modules.exam.tools import get_draft_items, init_session

    start = time.perf_counter()
    spec, _truncated, _pii_found = asyncio.run(_build_spec(entry["passage_text"]))
    init_session()
    graph = get_exam_graph()

    attempt_log: list[dict] = []
    for step in graph.stream({"spec": spec, "budget": budget}, stream_mode="updates"):
        for node_name, node_output in step.items():
            if node_name == "agent":
                summary = _summarize_agent_messages(node_output.get("agent_messages", []))
                summary["attempt"] = len(attempt_log) + 1
                attempt_log.append(summary)
            elif node_name == "judge" and attempt_log:
                attempt_log[-1]["judge_result"] = node_output.get("similarity_judge_result", {})
            elif node_name == "validate" and attempt_log:
                attempt_log[-1]["validation_feedback"] = node_output.get("validation_feedback", "")
                attempt_log[-1]["validation_passed"] = node_output.get("validation_passed", False)

    items = get_draft_items()
    elapsed = time.perf_counter() - start

    total_input = total_output = None
    has_tokens = False
    for a in attempt_log:
        if a.get("input_tokens") is not None:
            has_tokens = True
            total_input = (total_input or 0) + a["input_tokens"]
        if a.get("output_tokens") is not None:
            total_output = (total_output or 0) + a["output_tokens"]
    if not has_tokens:
        total_input = total_output = None

    cost = None
    if total_input is not None and total_output is not None:
        price_in, price_out = _PRICE_PER_1M[model]
        cost = (total_input / 1_000_000) * price_in + (total_output / 1_000_000) * price_out

    return {
        "id": entry["id"],
        "model": model,
        "num_items": spec.get("num_items"),
        "format": spec.get("format"),
        "items": items,
        "validation_passed": attempt_log[-1].get("validation_passed", False) if attempt_log else False,
        "attempts": len(attempt_log),
        "attempt_log": attempt_log,
        "input_tokens_total": total_input,
        "output_tokens_total": total_output,
        "cost_usd_est": cost,
        "wall_clock_sec": elapsed,
        "error": None,
        "code_version": _code_version(),
    }


def cmd_generate(args: argparse.Namespace) -> None:
    for k, v in _MODEL_ENV[args.model].items():
        os.environ[k] = v
    for k, v in _JUDGE_ENV.items():
        os.environ[k] = v

    from app.common.llm.tracing import init_langsmith_project
    init_langsmith_project()

    print(f"=== 생성 모델: {args.model} ({os.environ['LLM_BACKEND']}) / Judge: gpt-5.6-luna ({_JUDGE_ENV['JUDGE_BACKEND']}) ===")

    all_entries = _load_inputs()
    entries = select_entries(all_entries, smoke=args.smoke, ids=args.ids, limit=args.limit)

    out_path = _run_path(args.model, args.smoke)
    os.makedirs(_RUNS_DIR, exist_ok=True)
    done_ids = {r["id"] for r in load_run_records(out_path) if not r.get("error")}

    print(f"대상 지문 {len(entries)}개 (이미 완료 {len(done_ids & {e['id'] for e in entries})}개 스킵)\n")

    with open(out_path, "a", encoding="utf-8") as f:
        for i, entry in enumerate(entries, 1):
            if entry["id"] in done_ids:
                print(f"[{i}/{len(entries)}] {entry['id']} ({args.model}) 이미 완료 — 스킵")
                continue
            start = time.perf_counter()
            try:
                record = _run_passage(args.model, entry, budget=5)
            except Exception as e:
                elapsed = time.perf_counter() - start
                record = {
                    "id": entry["id"],
                    "model": args.model,
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
                print(f"[{i}/{len(entries)}] {entry['id']} ({args.model}) 실패: {type(e).__name__} ({elapsed:.1f}s)")
            else:
                n_items = len(record["items"])
                passed = record["validation_passed"]
                print(
                    f"[{i}/{len(entries)}] {entry['id']} ({args.model}) "
                    f"{record['wall_clock_sec']:.1f}s, {n_items}문항, 통과={passed}"
                )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\n완료 — {out_path}")


# ── summarize ────────────────────────────────────────────────────────────

def summarize_model(records: list[dict]) -> dict:
    """run 레코드 리스트를 집계한다(LLM 호출 없음, 테스트 가능). 원문 미사용.

    같은 id가 중복으로 append된 경우(재실행) dedupe_run_records()로 먼저 정리한다 —
    안 하면 지문 33개인데 n이 34건으로 집계되는 문제가 있었다."""
    records = dedupe_run_records(records)
    n = len(records)
    ok = [r for r in records if not r.get("error")]
    n_errors = n - len(ok)
    count_match = sum(
        1 for r in ok if r.get("num_items") is not None and len(r.get("items", [])) == r["num_items"]
    )
    validated = sum(1 for r in ok if r.get("validation_passed"))
    avg_attempts = (sum(r.get("attempts", 0) for r in ok) / len(ok)) if ok else 0.0
    malformed_total = sum(
        sum(a.get("malformed_retries", 0) for a in r.get("attempt_log", [])) for r in ok
    )
    # 2026-10-07: 도구 오류(ToolMessage "도구 호출 오류"로 시작) 합계 — 0이 아니면
    # 검색 백엔드 고장 같은 환경 문제를 바로 알 수 있다(graph.py agent_node 경고 로그와
    # 별개로, 생성 하네스 쪽 가시성).
    tool_errors_total = sum(
        sum((a.get("tool_errors") or {}).values())
        for r in ok for a in r.get("attempt_log", [])
    )
    avg_wall_clock = (sum(r.get("wall_clock_sec", 0.0) for r in records) / n) if n else 0.0
    costs = [r["cost_usd_est"] for r in ok if r.get("cost_usd_est") is not None]
    avg_cost = (sum(costs) / len(costs)) if costs else None
    return {
        "n_runs": n,
        "n_errors": n_errors,
        "count_match_rate": (count_match / len(ok)) if ok else 0.0,
        "validate_rate": (validated / len(ok)) if ok else 0.0,
        "avg_attempts": avg_attempts,
        "malformed_total": malformed_total,
        "tool_errors_total": tool_errors_total,
        "avg_wall_clock_sec": avg_wall_clock,
        "avg_cost_usd": avg_cost,
    }


def cmd_summarize(args: argparse.Namespace) -> None:
    header = (
        f"{'model':<18}{'n':>4}{'err':>5}{'count_ok%':>11}{'valid%':>9}"
        f"{'attempts':>10}{'malform':>9}{'tool_err':>10}{'sec':>8}{'cost$':>9}"
    )
    print(header)
    print("-" * len(header))
    for model in _MODELS:
        records = load_run_records(_run_path(model, args.smoke))
        s = summarize_model(records)
        cost_str = f"{s['avg_cost_usd']:.4f}" if s["avg_cost_usd"] is not None else "n/a"
        print(
            f"{model:<18}{s['n_runs']:>4}{s['n_errors']:>5}"
            f"{s['count_match_rate']*100:>10.1f}%{s['validate_rate']*100:>8.1f}%"
            f"{s['avg_attempts']:>10.2f}{s['malformed_total']:>9}{s['tool_errors_total']:>10}"
            f"{s['avg_wall_clock_sec']:>8.1f}{cost_str:>9}"
        )


# ── build-labelset ───────────────────────────────────────────────────────

def _collect_mc_choices(
    inputs: list[dict], runs_by_model: dict[str, list[dict]], seed: int = 0
) -> tuple[list[dict], list[dict]]:
    """(지문, 모델)마다 객관식 문항 1개를 시드 고정 무작위로 뽑는다.

    반환: (선택된 항목 리스트[결정론적 순서], missing 리스트). 선택된 항목은
    {"passage_id", "model", "passage_text", "item", "run_item_id"}."""
    rng = random.Random(seed)
    passages = {e["id"]: e for e in inputs}
    chosen = []
    missing = []
    for passage_id in sorted(passages):
        for model in sorted(runs_by_model):
            records_by_id = {r["id"]: r for r in runs_by_model[model]}
            rec = records_by_id.get(passage_id)
            if not rec or rec.get("error"):
                missing.append({"passage_id": passage_id, "model": model, "reason": "run_missing_or_error"})
                continue
            mc_items = [it for it in rec.get("items", []) if it.get("item_type") == "객관식"]
            if not mc_items:
                missing.append({"passage_id": passage_id, "model": model, "reason": "no_mc_item"})
                continue
            item = rng.choice(mc_items)
            chosen.append({
                "passage_id": passage_id,
                "model": model,
                "passage_text": passages[passage_id]["passage_text"],
                "item": {k: item.get(k, "") for k in ("question", "stimulus", "options", "answer")},
                "run_item_id": item.get("item_id", ""),
            })
    return chosen, missing


def build_labelset(
    inputs: list[dict], runs_by_model: dict[str, list[dict]], seed: int = 0
) -> tuple[list[dict], dict, list[dict]]:
    """블라인드(모델 정보 미포함) 라벨링용 entries·model_map·missing 리스트를 만든다.

    entries 순서는 시드 고정 셔플(같은 지문·모델이 연달아 나오지 않을 정도면 충분)."""
    chosen, missing = _collect_mc_choices(inputs, runs_by_model, seed)
    random.Random(seed).shuffle(chosen)
    entries = []
    model_map = {}
    for i, c in enumerate(chosen, 1):
        blind_id = f"iq_{i:03d}"
        entries.append({
            "id": blind_id,
            "passage_id": c["passage_id"],
            "passage_text": c["passage_text"],
            "item": c["item"],
            "human_label": {
                "정답유일성": None,
                "오답매력도": None,
                "근거성": None,
                "학생난이도": None,
                "cannot_judge": False,
                "reason": "",
            },
        })
        model_map[blind_id] = {
            "model": c["model"],
            "passage_id": c["passage_id"],
            "run_item_id": c["run_item_id"],
        }
    return entries, model_map, missing


def _has_existing_labels(path: str) -> bool:
    """이미 human_label이 하나라도 채워진 기존 파일이 있으면 True(라벨 유실 방지용 가드)."""
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for e in data.get("entries", []):
        hl = e.get("human_label", {}) or {}
        if hl.get("cannot_judge") or (hl.get("reason") or "").strip():
            return True
        if any(hl.get(k) is not None for k in ("정답유일성", "오답매력도", "근거성", "학생난이도")):
            return True
    return False


_GOLDEN_SCHEMA = {
    "description": (
        "문항 품질(정답유일성·오답매력도·근거성) 사람 라벨링용 골든셋. 생성 모델 3종"
        "(qwen2.5-14b/gpt-6-luna/gemini-3.8-flash)의 출력에서 (지문, 모델)마다 객관식"
        " 1개씩 뽑았고, 모델 정보는 라벨링 편향을 막기 위해 entries에서 블라인드 처리했다"
        " — 매핑은 data/golden/_item_quality_model_map.json에 별도 보관."
    ),
    "entry_fields": {
        "id": "블라인드 id (iq_001부터, 섞인 순서 — 모델/지문 추정 불가)",
        "passage_id": "원본 입력 지문 id (data/golden/item_quality_inputs.json 참고)",
        "passage_text": "교사 예시 문제 원문(합성, 하드룰 1)",
        "item": "채점 대상 문항 (question, stimulus, options, answer)",
        "human_label": (
            "정답유일성·오답매력도·근거성: 1~5 정수(evals/eval_lib.py JUDGE_TPL과 동일 기준 — "
            "정답유일성=오직 하나의 정답, 오답매력도=오답 선지가 그럴듯함, 근거성=교육과정 기반). "
            "학생난이도: 상/중/하 문자열(고등학생이 풀 때의 난이도, 교사 관점 직관적 판단 — "
            "평가셋 난이도 분포 확인용, 모델 비교 지표에는 쓰지 않음). "
            "판단 불가 시 cannot_judge=true. 세 기준(정답유일성·오답매력도·근거성) 중 2점 "
            "이하가 있으면 reason에 한 줄 근거."
        ),
    },
    "provenance": "golden_gen/gen_item_quality_golden.py generate + build-labelset.",
}

_MODEL_MAP_SCHEMA = {
    "description": "item_quality_golden.json의 블라인드 id → 실제 모델/원본 매핑(라벨링 중 참조 금지).",
    "map_fields": {"model": "생성 모델", "passage_id": "원본 입력 지문 id", "run_item_id": "run 파일 내 item_id"},
    "missing": "(지문, 모델) 쌍 중 run 실패 또는 객관식 문항이 없어 골든셋에 포함되지 못한 조합.",
}


def cmd_build_labelset(args: argparse.Namespace) -> None:
    if _has_existing_labels(_GOLDEN_PATH):
        print(f"중단 — {_GOLDEN_PATH}에 이미 채워진 human_label이 있습니다. 라벨 유실 방지를 위해 덮어쓰지 않습니다.")
        sys.exit(1)

    inputs = _load_inputs()
    runs_by_model = {m: load_run_records(_run_path(m, smoke=False)) for m in _MODELS}
    entries, model_map, missing = build_labelset(inputs, runs_by_model, seed=args.seed)

    os.makedirs(_GOLDEN_DIR, exist_ok=True)
    with open(_GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump({"_schema": _GOLDEN_SCHEMA, "entries": entries}, f, ensure_ascii=False, indent=2)
    with open(_MODEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({"_schema": _MODEL_MAP_SCHEMA, "map": model_map, "missing": missing}, f, ensure_ascii=False, indent=2)

    print(f"완료 — {_GOLDEN_PATH}에 {len(entries)}개, {_MODEL_MAP_PATH}에 매핑 저장")
    if missing:
        print(f"missing {len(missing)}건 (run 실패 또는 객관식 없음) — model_map.json의 'missing' 참고")


# ── export-sheet / import-sheet (JSON 직접 편집) ────────────────────────────
#
# tools/labeling.html(브라우저 도구) 대신 JSON 파일을 직접 편집해 라벨을 달 수 있게 하는
# 보조 경로. 라벨 기준 문구는 tools/labeling.html의 RUBRIC 상수를 그대로 가져왔다(두 도구가
# 같은 기준을 쓰도록 문구를 동기화 — RUBRIC을 고치면 이 상수도 같이 고칠 것).

_SHEET_GUIDE = [
    "이 파일은 JSON 편집기로 직접 라벨을 채우는 시트입니다. 각 문항의 \"라벨\" 블록만 채우세요"
    "(그 아래 예시문제_줄·발문·보기_줄·선지·표시된_정답은 참고용 원문이라 수정해도 반영되지 않습니다).",
    "정답유일성·오답매력도·근거성: 1~5 정수. 모델 정보는 제공하지 않습니다 — 추측하지 말고"
    " 문항 자체만 보고 판단하세요.",
    "학생난이도: \"상\"/\"중\"/\"하\" 문자열.",
    "정답유일성·오답매력도·근거성 중 하나라도 2점 이하이거나 판단불가를 true로 두면 근거를"
    " 반드시 채워야 합니다(비어 있으면 import-sheet가 반영을 거부합니다).",
    "",
    "[기준표]",
    "정답유일성",
    "  5: 표시된 정답이 맞고, 다른 선지는 명확히 오답이다",
    "  3: 표시된 정답이 맞지만, 해석에 따라 다른 선지도 정답이 될 여지가 있다",
    "  1: 정답이 둘 이상이거나, 표시된 정답이 틀렸다",
    "오답매력도",
    "  5: 모든 오답이 같은 개념 범주 안에서 그럴듯해, 내용을 알아야 고를 수 있다",
    "  3: 일부 오답이 너무 티가 나서 소거법으로 쉽게 지워진다",
    "  1: 오답 대부분이 문항과 무관하거나 말이 되지 않는다",
    "근거성",
    "  5: 고교 사회과 교육과정 내용과 정확히 맞고 사실 오류가 없다",
    "  3: 경미한 부정확함이나 교육과정 밖 내용이 섞여 있다",
    "  1: 명백한 사실 오류가 있다",
    "학생난이도: 상/중/하 — 고등학생이 이 문항을 풀 때의 난이도(교사 관점의 직관적 판단)",
    "2·4점은 각각 인접한 두 앵커 사이.",
]

_SCORE_KEYS = ("정답유일성", "오답매력도", "근거성")


def build_label_sheet_entry(entry: dict) -> dict:
    """golden entry 1개를 사람이 읽기 좋은 시트 항목으로 바꾼다(라벨 칸이 맨 위).

    passage_text·stimulus는 줄 단위 문자열 배열로 풀어 "\\n"이 그대로 보이는 문제를
    없앤다. stimulus가 비어 있으면 보기_줄 키 자체를 생략한다."""
    item = entry["item"]
    sheet_entry = {
        "id": entry["id"],
        "라벨": {
            "정답유일성": None, "오답매력도": None, "근거성": None, "학생난이도": None,
            "판단불가": False, "근거": "",
        },
        "예시문제_줄": entry["passage_text"].split("\n"),
        "발문": item["question"],
    }
    if item.get("stimulus"):
        sheet_entry["보기_줄"] = item["stimulus"].split("\n")
    sheet_entry["선지"] = item["options"]
    sheet_entry["표시된_정답"] = item["answer"]
    return sheet_entry


def build_label_sheet(entries: list[dict]) -> dict:
    """golden entries 전체를 _안내 + 문항 리스트로 묶은 시트 딕셔너리로 바꾼다."""
    return {
        "_안내": _SHEET_GUIDE,
        "문항": [build_label_sheet_entry(e) for e in entries],
    }


def _has_existing_sheet_labels(path: str) -> bool:
    """이미 라벨이 하나라도 채워진 시트 파일이 있으면 True(라벨 유실 방지용 가드)."""
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("문항", []):
        label = item.get("라벨", {}) or {}
        if label.get("판단불가") or (label.get("근거") or "").strip():
            return True
        if any(label.get(k) is not None for k in (*_SCORE_KEYS, "학생난이도")):
            return True
    return False


def validate_sheet_label(label: dict) -> list[str]:
    """시트 문항 1개의 '라벨' 블록을 검증한다. 오류 메시지 리스트(없으면 빈 리스트)."""
    errors = []
    for key in _SCORE_KEYS:
        v = label.get(key)
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or not (1 <= v <= 5)):
            errors.append(f"{key} 값이 1~5 정수나 null이 아님")
    difficulty = label.get("학생난이도")
    if difficulty is not None and difficulty not in ("상", "중", "하"):
        errors.append("학생난이도 값이 상/중/하나 null이 아님")
    needs_reason = bool(label.get("판단불가")) or any(
        isinstance(label.get(k), int) and not isinstance(label.get(k), bool) and label.get(k) <= 2
        for k in _SCORE_KEYS
    )
    if needs_reason and not str(label.get("근거") or "").strip():
        errors.append("2점 이하 또는 판단불가인데 근거가 비어 있음")
    return errors


def validate_sheet(sheet_items: list[dict], valid_ids: set[str]) -> list[tuple[str, list[str]]]:
    """시트 문항 전체를 검증한다. 반환: [(id, 오류 메시지 리스트), ...](유효한 문항은 제외)."""
    errors = []
    for item in sheet_items:
        item_id = item.get("id")
        item_errors = []
        if item_id not in valid_ids:
            item_errors.append("golden에 존재하지 않는 id")
        item_errors.extend(validate_sheet_label(item.get("라벨", {}) or {}))
        if item_errors:
            errors.append((item_id, item_errors))
    return errors


def apply_sheet_labels(golden_entries: list[dict], sheet_items: list[dict]) -> tuple[list[dict], int]:
    """검증을 통과한 sheet_items의 라벨을 golden_entries의 human_label에 반영한다.

    다른 필드는 그대로 두고 human_label만 갱신한다. 반환: (갱신된 entries,
    미완료 문항 수 — 정답유일성·오답매력도·근거성 중 하나라도 null인 경우)."""
    sheet_by_id = {item["id"]: (item.get("라벨") or {}) for item in sheet_items}
    updated = []
    incomplete = 0
    for entry in golden_entries:
        label = sheet_by_id.get(entry["id"])
        new_entry = dict(entry)
        if label is not None:
            new_entry["human_label"] = {
                "정답유일성": label.get("정답유일성"),
                "오답매력도": label.get("오답매력도"),
                "근거성": label.get("근거성"),
                "학생난이도": label.get("학생난이도"),
                "cannot_judge": bool(label.get("판단불가", False)),
                "reason": label.get("근거", "") or "",
            }
            if any(label.get(k) is None for k in _SCORE_KEYS):
                incomplete += 1
        updated.append(new_entry)
    return updated, incomplete


def cmd_export_sheet(args: argparse.Namespace) -> None:
    if _has_existing_sheet_labels(_SHEET_PATH):
        print(f"중단 — {_SHEET_PATH}에 이미 채워진 라벨이 있습니다. 라벨 유실 방지를 위해 덮어쓰지 않습니다.")
        sys.exit(1)

    with open(_GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)
    sheet = build_label_sheet(golden["entries"])

    os.makedirs(_GOLDEN_DIR, exist_ok=True)
    with open(_SHEET_PATH, "w", encoding="utf-8") as f:
        json.dump(sheet, f, ensure_ascii=False, indent=2)

    print(f"완료 — {_SHEET_PATH}에 {len(sheet['문항'])}개 문항 (라벨은 전부 빈 값)")


def cmd_import_sheet(args: argparse.Namespace) -> None:
    sheet_path = args.sheet or _SHEET_PATH
    with open(sheet_path, encoding="utf-8") as f:
        sheet = json.load(f)
    with open(_GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)

    valid_ids = {e["id"] for e in golden["entries"]}
    sheet_items = sheet.get("문항", [])
    errors = validate_sheet(sheet_items, valid_ids)
    if errors:
        print(f"검증 실패 — {len(errors)}개 문항, 반영하지 않습니다.")
        for item_id, msgs in errors:
            for msg in msgs:
                print(f"  {item_id}: {msg}")
        sys.exit(1)

    updated_entries, incomplete = apply_sheet_labels(golden["entries"], sheet_items)
    golden["entries"] = updated_entries
    with open(_GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2)

    print(f"완료 — {_GOLDEN_PATH}에 {len(sheet_items)}개 문항 반영 (미완료 {incomplete}개)")


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="지문별로 운영 경로를 그대로 실행해 run jsonl에 append")
    p_gen.add_argument("--model", required=True, choices=_MODELS)
    p_gen.add_argument("--smoke", action="store_true", help="형식별 1개(총 4개)만 실행")
    p_gen.add_argument("--ids", nargs="+", default=None, help="지정한 id만 실행")
    p_gen.add_argument("--limit", type=int, default=None, help="앞에서 N개만 실행")
    p_gen.set_defaults(func=cmd_generate)

    p_sum = sub.add_parser("summarize", help="모델별 run 파일을 집계해 표로 출력")
    p_sum.add_argument("--smoke", action="store_true", help="smoke run 파일을 대상으로 집계")
    p_sum.set_defaults(func=cmd_summarize)

    p_build = sub.add_parser("build-labelset", help="3개 모델 run 파일에서 블라인드 라벨링용 골든셋 생성")
    p_build.add_argument("--seed", type=int, default=0)
    p_build.set_defaults(func=cmd_build_labelset)

    p_export = sub.add_parser("export-sheet", help="JSON 직접 편집용 라벨링 시트 생성")
    p_export.set_defaults(func=cmd_export_sheet)

    p_import = sub.add_parser("import-sheet", help="편집한 라벨링 시트의 라벨을 골든셋에 반영")
    p_import.add_argument("--sheet", default=None, help="시트 파일 경로(기본: data/golden/item_quality_labeling_sheet.json)")
    p_import.set_defaults(func=cmd_import_sheet)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
