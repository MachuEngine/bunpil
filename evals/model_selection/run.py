#!/usr/bin/env python
"""모델 선정 평가 하네스 CLI.

    # 비용 없이 구조 확인
    .venv/bin/python -m evals.model_selection.run generate --split pilot --models gpt-6-luna --mock --limit 3
    # 예상 호출 수·최대 비용만 계산(호출 없음)
    .venv/bin/python -m evals.model_selection.run estimate --split pilot --models gpt-6-luna,gemini-3.8-flash --repeats 1
    # 실제 실행(OPENROUTER_API_KEY 필요, 비용 상한 필수)
    .venv/bin/python -m evals.model_selection.run generate --split pilot --models gpt-6-luna --max-cost 2
    .venv/bin/python -m evals.model_selection.run score --run-id <id> --judge claude-opus-5.5 --max-cost 3
    .venv/bin/python -m evals.model_selection.run summarize --run-id <id>

원칙: 인증 정보는 OPENROUTER_API_KEY 환경변수에서만 읽고 어디에도 쓰지 않는다. mock 결과로는 순위를 매기지 않는다.
요청 모델·endpoint와 실제 처리가 다른 호출이 섞인 작업은 집계에서 뺀다.
"""
import argparse
import asyncio
import contextlib
import csv
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))

from . import graders, judge_prompts as jp, metrics  # noqa: E402
from .adapters import fetch_generation, make_adapter  # noqa: E402
from .recorder import RESULTS_ROOT, Recorder, TaskStore, current_task, stable_hash  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
DATA_DIR = os.path.join(ROOT, "evals", "data")

# dry-run 견적용 호출당 토큰 가정 — 실측으로 갱신한다(docs/model-selection-plan.md 8절).
# 값은 **호출 1회당** 토큰이다. set: GPT-4o-mini budget=1 실측(15세트 113회 호출, 입력 329,791·출력 13,949 —
# 세트 합계 약 22K·0.9K, 호출당 약 2.9K·0.12K, EVAL.md:665-680)에 재시도(budget=5) 여유를 더해
# 세트당 12회 × 호출당 입력 3K·출력 0.17K(세트 합계 약 36K·2K)로 잡았다.
ESTIMATE = {
    "set": {"calls": 12, "input": 3000, "output": 170},
    "runtime_judge": {"calls": 2, "input": 3000, "output": 100},
    "explain": {"calls": 1, "input": 700, "output": 700},
    "revise": {"calls": 1.5, "input": 3500, "output": 900},
    "judge_item": {"calls": 1, "input": 2500, "output": 600},
    "judge_solve": {"calls": 1, "input": 600, "output": 60},
    "judge_pair": {"calls": 2, "input": 3500, "output": 200},
}
SAFETY_FACTOR = 1.5  # 최대 비용 = 추정 × 1.5 (재시도·긴 입력)


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_rows(split: str, limit: int | None) -> list[dict]:
    with open(os.path.join(DATA_DIR, f"{split}.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows[:limit] if limit else rows


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


@contextlib.contextmanager
def patched_models(gen, runtime_judge):
    """프로덕션 코드는 고치지 않고, 이 블록 안에서만 모델 선택 함수를 바꿔 끼운다(단일 스레드 전제)."""
    import app.common.llm as common_llm
    import app.modules.exam.explain as explain_mod
    import app.modules.exam.graph as graph_mod
    import app.modules.exam.revise as revise_mod

    saved = [(common_llm, "get_llm_backend"), (graph_mod, "get_langchain_model"), (graph_mod, "get_judge_backend"),
             (explain_mod, "get_llm_backend"), (revise_mod, "get_llm_backend")]
    originals = [getattr(m, n) for m, n in saved]
    common_llm.get_llm_backend = lambda: gen
    graph_mod.get_langchain_model = lambda temperature=0.7: gen.chat_model()
    graph_mod.get_judge_backend = lambda: runtime_judge
    explain_mod.get_llm_backend = lambda: gen
    revise_mod.get_llm_backend = lambda: gen
    try:
        yield
    finally:
        for (m, n), orig in zip(saved, originals):
            setattr(m, n, orig)


def _load_calls(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "calls.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def route_status(calls: list[dict]) -> str:
    """작업 하나에 속한 호출들의 경로 상태. "ok" | "mismatch" | "unverified".

    **확인된(route_ok=True) 결과만 집계에 넣는다**(조건 9). 응답을 못 받은 호출(error)은 경로 문제가 아니라
    API 실패라 판정에서 뺀다 — 그 작업은 실패로 집계된다. 에이전트 경로처럼 응답에 provider가 없으면 None으로
    남는데, 이를 통과시키면 fallback 결과가 원래 모델 결과로 섞일 수 있어 "unverified"로 제외한다."""
    produced = [c for c in calls if not c.get("error")]
    if any(c.get("route_ok") is False for c in produced):
        return "mismatch"
    if any(c.get("route_ok") is not True for c in produced):
        return "unverified"
    return "ok"


def _finalize_routes(run_dir: str, cfg: dict, mock: bool):
    """실행이 끝나면 경로·실비용 확인을 자동으로 돌린다(사람이 verify-routes를 잊어도 되게)."""
    if mock:
        return
    try:
        n = verify_routes(run_dir, cfg)
        print(f"경로·비용 확인: {n}건 갱신")
    except Exception as e:  # noqa: BLE001 — 확인 실패는 집계에서 unverified로 제외되므로 실행 자체는 멈추지 않는다
        print(f"⚠️ 경로 확인 실패({type(e).__name__}) — 미확인 결과는 집계에서 제외된다. 나중에 verify-routes를 다시 실행하세요.")


def _mask_strings(obj):
    """생성물 캐시에 PII가 남지 않게 모든 문자열을 마스킹한다(하드룰 4). 채점은 마스킹 전 원문으로 이미 끝났다."""
    from app.common.privacy import mask_pii

    if isinstance(obj, str):
        return mask_pii(obj)[0]
    if isinstance(obj, list):
        return [_mask_strings(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _mask_strings(v) for k, v in obj.items()}
    return obj


def _set_task(adapters, task: dict):
    for a in adapters:
        a.active_task = task
    return current_task.set(task)


# ── generate: G-T1(세트)·G-T4(개수 추출)·G-T2(해설)·G-T3(수정) ─────────────

def run_generate(args, cfg):
    from app.main import _build_spec, _mask_item, _parse_item
    from app.modules.exam import get_exam_graph
    from app.modules.exam.explain import explain_item
    from app.modules.exam.revise import revise_items
    from app.modules.exam.tools import get_draft_items, init_session

    if not args.trace:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    rows = load_rows(args.split, args.limit)
    model_keys = args.models.split(",")
    run_id = args.run_id or f"gen-{args.split}-{time.strftime('%Y%m%d-%H%M%S')}{'-mock' if args.mock else ''}"
    rec = Recorder(run_id, args.max_cost)
    store = TaskStore(rec.dir)
    judge_key = cfg["runtime_judge"]
    rec.write_meta({"command": "generate", "split": args.split, "models": model_keys, "repeats": args.repeats,
                    "runtime_judge": judge_key, "mock": args.mock, "commit": git_commit(),
                    "model_specs": {k: cfg["models"][k] for k in model_keys + [judge_key]},
                    "sensitive": args.sensitive, "n_inputs": len(rows)})
    gens = {k: make_adapter(k, cfg, rec, role="generation", mock=args.mock, sensitive=args.sensitive) for k in model_keys}
    runtime_judge = make_adapter(judge_key, cfg, rec, role="judge", mock=args.mock, sensitive=args.sensitive)
    rng = random.Random(args.seed)
    graph = get_exam_graph()

    for row in rows:
        order = model_keys[:]
        rng.shuffle(order)  # 입력마다 모델 순서를 섞어 번갈아 실행(배치 효과 방지)
        for repeat in range(args.repeats):
            for key in order:
                gen = gens[key]
                spec = cfg["models"][key]
                task_key = stable_hash({"cmd": "set", "model": key, "spec": spec, "input": row["id"],
                                        "passage": row["passage_text"], "repeat": repeat,
                                        "judge": cfg["models"][judge_key], "mock": args.mock})
                if store.get(task_key):
                    continue
                base = {"input_id": row["id"], "model": key, "repeat": repeat, "expected": row["expected"],
                        "format": row["format"], "case_type": row["case_type"]}
                token = _set_task([gen, runtime_judge], {**base, "task": "set"})
                result = {"kind": "set", **base, "mock": args.mock}
                try:
                    with patched_models(gen, runtime_judge):
                        t0 = time.perf_counter()
                        spec_obj, truncated, pii_found = asyncio.run(_build_spec(row["passage_text"]))
                        init_session()
                        state = graph.invoke({"spec": spec_obj, "budget": 5})
                        items = get_draft_items()
                        result.update({
                            "latency_s": time.perf_counter() - t0,
                            "items": items, "extracted_num_items": spec_obj["num_items"],
                            "validation_passed": state.get("validation_passed", False),
                            "pii_found": pii_found, "masked_passage": spec_obj["passage_text"],
                            "analyzed_format": spec_obj.get("format"), "format_notice": spec_obj.get("format_notice", ""),
                        })
                        result["grade"] = graders.grade_set(row, items, spec_obj["num_items"],
                                                            spec_obj.get("format"), spec_obj.get("format_notice", ""))

                        explains = []
                        for i, item in enumerate(items):
                            current_task.set({**base, "task": "explain", "item_index": i})
                            gen.active_task = {**base, "task": "explain", "item_index": i}
                            masked_item, _ = _mask_item(_parse_item(item))
                            t1 = time.perf_counter()
                            try:
                                text = asyncio.run(explain_item(masked_item))
                                explains.append({"text": text, "latency_s": time.perf_counter() - t1})
                            except Exception as e:  # noqa: BLE001
                                explains.append({"error": f"{type(e).__name__}: {str(e)[:200]}"})
                        result["explanations"] = explains

                        revisions = []
                        if items:
                            masked_items = [_mask_item(_parse_item(it))[0] for it in items]
                            for j, req in enumerate(row["revise_requests"]):
                                current_task.set({**base, "task": "revise", "request_index": j})
                                gen.active_task = {**base, "task": "revise", "request_index": j}
                                t2 = time.perf_counter()
                                try:
                                    resp = asyncio.run(revise_items(spec_obj["passage_text"], masked_items, [], req["instruction"]))
                                except Exception as e:  # noqa: BLE001
                                    resp = None
                                    revisions.append({"error": f"{type(e).__name__}: {str(e)[:200]}"})
                                grade = graders.grade_revise(masked_items, resp, req)
                                revisions.append({"request": req, "response": resp, "grade": grade,
                                                  "latency_s": time.perf_counter() - t2})
                        result["revisions"] = revisions
                except Exception as e:  # noqa: BLE001 — 예산 초과는 전체 중단
                    from .recorder import BudgetExceeded

                    result["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                    if isinstance(e, BudgetExceeded):
                        store.put(task_key, _mask_strings(result))
                        print(f"[중단] {e}")
                        _finalize_routes(rec.dir, cfg, args.mock)
                        return run_id
                finally:
                    current_task.reset(token)
                store.put(task_key, _mask_strings(result))
                print(f"  {row['id']} {key} r{repeat}: items={len(result.get('items', []))} "
                      f"err={bool(result.get('error'))} spent=${rec.spent:.4f}")
    print(f"run_id={run_id}  spent=${rec.spent:.4f}  dir={rec.dir}")
    _finalize_routes(rec.dir, cfg, args.mock)
    return run_id


# ── score: 검증된 Judge로 생성 결과 채점(Q1 정답 키·Q2 결함·Q4 해설) ────────────

def run_score(args, cfg):
    run_dir = os.path.join(RESULTS_ROOT, args.run_id)
    store = TaskStore(run_dir)
    rec = Recorder(args.run_id, args.max_cost)
    judge = make_adapter(args.judge, cfg, rec, role="judge", mock=args.mock)
    out = TaskStore(os.path.join(run_dir, f"score-{args.judge}"))
    for task in [t for t in store.data.values() if t.get("kind") == "set" and t.get("items")]:
        if task["model"] == args.judge:
            continue  # 자기 출력은 채점하지 않는다
        for i, item in enumerate(task["items"]):
            key = stable_hash({"score": args.judge, "task": task["task_key"], "i": i, "spec": cfg["models"][args.judge]})
            if out.get(key):
                continue
            exp = (task.get("explanations") or [{}])[i] if i < len(task.get("explanations") or []) else {}
            ctx = {"input_id": task["input_id"], "model": task["model"], "repeat": task["repeat"], "task": "score", "i": i}
            judge.active_task = ctx
            token = current_task.set(ctx)
            try:
                verdict = _judge_json(judge, jp.item_verdict_messages(task["masked_passage"], item, exp.get("text", "")),
                                      jp.item_verdict_schema())
                solved = _judge_json(judge, jp.solve_messages(item), jp.solve_schema()) if item.get("options") else None
            finally:
                current_task.reset(token)
            out.put(key, {"source_task": task["task_key"], "input_id": task["input_id"], "model": task["model"],
                          "repeat": task["repeat"], "item_index": i, "answer": item.get("answer"),
                          "verdict": verdict, "solve": solved})
    print(f"scored with {args.judge}  spent=${rec.spent:.4f}")
    _finalize_routes(run_dir, cfg, args.mock)


def _judge_json(judge, messages, schema) -> dict:
    """구조화 출력 요청 → 파싱 실패 시 1회 재요청 → 그래도 실패면 parse_error(기본값으로 채우지 않음)."""
    for attempt in range(2):
        try:
            raw = judge.complete(messages, max_tokens=1024, response_format=schema)
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        data = jp.parse_json(raw)
        if data is not None:
            return data
    return {"parse_error": True}


# ── judge-validate: Judge 후보 × 사람 gold label ────────────────────────────

def run_judge_validate(args, cfg):
    with open(args.gold, encoding="utf-8") as f:
        gold = [json.loads(line) for line in f]
    run_id = args.run_id or f"judge-{time.strftime('%Y%m%d-%H%M%S')}{'-mock' if args.mock else ''}"
    rec = Recorder(run_id, args.max_cost)
    store = TaskStore(rec.dir)
    rec.write_meta({"command": "judge-validate", "gold": os.path.relpath(args.gold, ROOT), "judges": args.judges.split(","),
                    "repeats": args.repeats, "mock": args.mock, "commit": git_commit(),
                    "model_specs": {k: cfg["models"][k] for k in args.judges.split(",")}})
    for key in args.judges.split(","):
        judge = make_adapter(key, cfg, rec, role="judge", mock=args.mock)
        for row in gold:
            for rep in range(args.repeats):
                tkey = stable_hash({"jv": key, "row": row.get("output_id") or row.get("pair_id"), "rep": rep,
                                    "spec": cfg["models"][key]})
                if store.get(tkey):
                    continue
                ctx = {"judge": key, "row": row.get("output_id") or row.get("pair_id"), "repeat": rep}
                judge.active_task, token = ctx, current_task.set(ctx)
                try:
                    if "pair_id" in row:
                        ab = _judge_json(judge, jp.pairwise_messages(row["passage_text"], row["a_text"], row["b_text"]), jp.pairwise_schema())
                        ba = _judge_json(judge, jp.pairwise_messages(row["passage_text"], row["b_text"], row["a_text"]), jp.pairwise_schema())
                        res = {"ab": ab.get("winner"), "ba_back": jp.swap_winner(ba.get("winner")),
                               "parse_error": bool(ab.get("parse_error") or ba.get("parse_error"))}
                    else:
                        v = _judge_json(judge, jp.item_verdict_messages(row["passage_text"], row["item"], row.get("explanation", "")),
                                        jp.item_verdict_schema())
                        s = _judge_json(judge, jp.solve_messages(row["item"]), jp.solve_schema()) if row["item"].get("options") else None
                        res = {"verdict": v, "solve": s, "parse_error": bool(v.get("parse_error"))}
                finally:
                    current_task.reset(token)
                store.put(tkey, {"judge": key, "repeat": rep, **ctx, **res})
    print(f"run_id={run_id}  spent=${rec.spent:.4f}")
    _finalize_routes(rec.dir, cfg, args.mock)
    summarize_judges(rec.dir, gold, args.mock)


def summarize_judges(run_dir: str, gold: list[dict], mock: bool):
    store = TaskStore(run_dir)
    calls_by = defaultdict(list)
    for c in _load_calls(run_dir):
        t = c.get("task") or {}
        calls_by[(t.get("judge"), t.get("row"), t.get("repeat"))].append(c)
    by_judge = defaultdict(list)
    excluded = defaultdict(lambda: defaultdict(int))
    for t in store.data.values():
        status = route_status(calls_by[(t["judge"], t["row"], t["repeat"])])
        if status != "ok":
            excluded[t["judge"]][status] += 1  # 다른 provider·미확인 판정은 Judge 성적에 넣지 않는다
            continue
        by_judge[t["judge"]].append(t)
    gold_by = {(g.get("output_id") or g.get("pair_id")): g for g in gold}
    rows = []
    for judge, results in by_judge.items():
        r0 = [t for t in results if t["repeat"] == 0]
        item_res = [(t, gold_by[t["row"]]) for t in r0 if "verdict" in t and gold_by[t["row"]].get("resolution") != "ambiguous"]
        pred_pass = [all((t["verdict"].get(k) or {}).get("verdict") == "yes" for k in jp.CRITICAL_KEYS) for t, _ in item_res]
        truth_pass = [g["gold"]["pass"] for _, g in item_res]
        truth_crit = [not g["gold"]["pass"] for _, g in item_res]
        judge_v = [(t["verdict"].get(k) or {}).get("verdict") for t, _ in item_res for k in jp.ITEM_KEYS]
        gold_v = [g["gold"].get(k) for _, g in item_res for k in jp.ITEM_KEYS]
        solve = [(t["solve"] or {}).get("answer") == g["item"]["answer"] for t, g in item_res if t.get("solve") and g.get("answer_confirmed")]
        pairs = [(t, gold_by[t["row"]]) for t in r0 if "ab" in t]
        repeat_pairs = defaultdict(list)
        for t in results:
            if "verdict" in t:
                repeat_pairs[t["row"]].append(json.dumps({k: (t["verdict"].get(k) or {}).get("verdict") for k in jp.ITEM_KEYS}))
        stable = [len(set(v)) == 1 for v in repeat_pairs.values() if len(v) > 1]
        rows.append({
            "judge": judge, "n_items": len(item_res), "n_pairs": len(pairs),
            "excluded_route_mismatch": excluded[judge]["mismatch"], "excluded_unverified": excluded[judge]["unverified"],
            "macro_f1_pass": metrics.macro_f1(truth_pass, pred_pass),
            "weighted_kappa": metrics.weighted_kappa(gold_v, judge_v),
            "critical_recall": metrics.recall(truth_crit, [not p for p in pred_pass]),
            "false_reject_rate": metrics.false_reject_rate(truth_pass, pred_pass),
            "solve_accuracy": sum(solve) / len(solve) if solve else None,
            "pairwise_accuracy": (sum(1 for t, g in pairs if t["ab"] == g["human_winner"]) / len(pairs)) if pairs else None,
            "position_consistency": metrics.position_consistency([t["ab"] for t, _ in pairs], [t["ba_back"] for t, _ in pairs]),
            "length_bias": metrics.length_bias([len(g["a_text"]) for _, g in pairs], [len(g["b_text"]) for _, g in pairs],
                                               [t["ab"] for t, _ in pairs], [g["human_winner"] for _, g in pairs]),
            "repeat_stability": sum(stable) / len(stable) if stable else None,
            "parse_error_rate": sum(1 for t in results if t.get("parse_error")) / len(results) if results else None,
            "mock": mock,
        })
    _write_table(run_dir, "judge_summary", rows)
    if mock:
        print("⚠️ MOCK 결과 — 배선 확인용이며 Judge 순위를 매기지 않는다.")
    for r in rows:
        print(r)
    return rows


# ── summarize: 생성 결과 집계 ───────────────────────────────────────────

def run_summarize(args, cfg):
    run_dir = os.path.join(RESULTS_ROOT, args.run_id)
    store = TaskStore(run_dir)
    calls = _load_calls(run_dir)
    meta = json.load(open(os.path.join(run_dir, "meta.json"), encoding="utf-8"))
    is_mock = meta.get("mock") or any(c.get("mock") for c in calls)

    calls_by_task = defaultdict(list)
    for c in calls:
        t = c.get("task") or {}
        if t.get("task") == "score":
            continue  # 선정용 Judge 채점 비용은 서비스 운영 비용(세트당 비용)이 아니다
        calls_by_task[(t.get("input_id"), t.get("model"), t.get("repeat"))].append(c)

    sets = [t for t in store.data.values() if t.get("kind") == "set"]
    by_model = defaultdict(list)
    excluded = defaultdict(lambda: defaultdict(int))
    for t in sets:
        tc = calls_by_task[(t["input_id"], t["model"], t["repeat"])]
        status = route_status(tc)
        if status != "ok":
            excluded[t["model"]][status] += 1  # fallback·다른 provider·미확인 경로 결과는 원래 모델 결과로 세지 않는다
            continue
        by_model[t["model"]].append((t, tc))

    summary, details = [], []
    for model, pairs in by_model.items():
        succ, clusters, lat, cost, lat_exp, lat_rev = [], [], [], [], [], []
        per_format, per_case = defaultdict(list), defaultdict(list)
        revise_parse, revise_scope, revise_intent, inj = [], [], [], []
        for t, tc in pairs:
            g = t.get("grade") or {}
            ok = bool(g) and not t.get("error") and g.get("R1_count_ok") and g.get("all_items_format_ok") \
                and g.get("all_items_korean_ok") and g.get("all_items_no_pii") and not g.get("injection_followed")
            succ.append(1.0 if ok else 0.0)
            clusters.append(t["input_id"])
            per_format[t["format"]].append(ok)
            per_case[t["case_type"]].append(ok)
            if t.get("latency_s") is not None:
                lat.append(t["latency_s"])
            cost.append(sum(c.get("cost_usd") or 0 for c in tc))
            lat_exp += [e["latency_s"] for e in t.get("explanations", []) if "latency_s" in e]
            for r in t.get("revisions", []):
                if "grade" in r:
                    revise_parse.append(r["grade"]["R6_parse_ok"])
                    if r["grade"]["R7_scope_ok"] is not None:
                        revise_scope.append(r["grade"]["R7_scope_ok"])
                    if r["grade"].get("R11_intent_ok") is not None:
                        revise_intent.append(r["grade"]["R11_intent_ok"])
                    lat_rev.append(r["latency_s"])
            if "injection_followed" in g:
                inj.append(g["injection_followed"])
            details.append({"model": model, "input_id": t["input_id"], "repeat": t["repeat"], "format": t["format"],
                            "case_type": t["case_type"], "success": ok, "n_items": g.get("n_items"),
                            "latency_s": t.get("latency_s"), "cost_usd": sum(c.get("cost_usd") or 0 for c in tc),
                            "error": t.get("error")})
        model_calls = [c for c in calls if c.get("model_key") == model]  # 이 모델이 직접 한 호출만
        point, lo, hi = metrics.cluster_bootstrap_ci(succ, clusters)
        n_succ = sum(succ)
        summary.append({
            "model": model, "n_sets": len(pairs), "excluded_route_mismatch": excluded[model]["mismatch"],
            "excluded_unverified": excluded[model]["unverified"],
            "set_success": point, "set_success_ci95": f"[{lo:.3f}, {hi:.3f}]" if lo is not None else None,
            "zero_item_rate": sum(1 for t, _ in pairs if (t.get("grade") or {}).get("zero_items")) / len(pairs),
            "num_items_extract_acc": _mean([(t.get("grade") or {}).get("R8_num_items_ok") for t, _ in pairs]),
            "format_classify_acc": _mean([(t.get("grade") or {}).get("R9_format_ok") for t, _ in pairs]),
            "unsupported_notice_acc": _mean([(t.get("grade") or {}).get("R10_notice_ok") for t, _ in pairs]),
            "revise_intent_acc": _mean(revise_intent),
            "injection_followed_rate": _mean(inj),
            "revise_parse_ok": _mean(revise_parse), "revise_scope_ok": _mean(revise_scope),
            "set_latency_p50": metrics.percentile(lat, 50), "set_latency_p95": metrics.percentile(lat, 95),
            "explain_latency_p95": metrics.percentile(lat_exp, 95), "revise_latency_p95": metrics.percentile(lat_rev, 95),
            "cost_per_set": _mean(cost), "cost_per_successful_set": (sum(cost) / n_succ) if n_succ else None,
            "api_failure_rate": _mean([bool(c.get("error")) for c in model_calls]),
            "retry_rate": _mean([(c.get("retries") or 0) > 0 for c in model_calls]),
            "cost_source": sorted({c.get("cost_source") for c in model_calls if c.get("cost_source")}),
            "by_format": {k: _mean(v) for k, v in per_format.items()},
            "by_case_type": {k: _mean(v) for k, v in per_case.items()},
            "mock": is_mock,
        })
    _merge_judge_scores(run_dir, summary)
    _write_table(run_dir, "summary", summary)
    _write_table(run_dir, "details", details)
    if is_mock:
        print("⚠️ MOCK 결과 — 배선 확인용이다. 순위·Pareto를 산출하지 않는다.")
    else:
        rows = [{"model": s["model"], "q": s["set_success"] or 0, "cost": s["cost_per_successful_set"] or 1e9,
                 "p95": s["set_latency_p95"] or 1e9} for s in summary]
        print("Pareto(성공률↑, 성공 1건당 비용↓, p95 지연↓):", metrics.pareto_front(rows, ("q",), ("cost", "p95")))
    for s in summary:
        print({k: v for k, v in s.items() if k not in ("by_format", "by_case_type")})


def _merge_judge_scores(run_dir: str, summary: list[dict]):
    """score-<judge>/ 결과를 모델별 지표로 합친다(Q1 정답 키 일치, X1~X4 치명적 오류율).

    Judge 판정 지표는 **검증을 통과한 Judge**로 채점했을 때만 의미가 있다(docs/model-selection-plan.md 2절)."""
    calls_by = defaultdict(list)
    for c in _load_calls(run_dir):
        t = c.get("task") or {}
        if t.get("task") == "score":
            calls_by[(c.get("model_key"), t.get("input_id"), t.get("model"), t.get("repeat"), t.get("i"))].append(c)
    for name in sorted(os.listdir(run_dir)):
        if not name.startswith("score-"):
            continue
        judge = name[len("score-"):]
        all_rows = list(TaskStore(os.path.join(run_dir, name)).data.values())
        rows = [r for r in all_rows
                if route_status(calls_by[(judge, r["input_id"], r["model"], r["repeat"], r["item_index"])]) == "ok"]
        for s in summary:
            mine = [r for r in rows if r["model"] == s["model"]]
            if not mine:
                continue
            def v(r, k):
                return ((r.get("verdict") or {}).get(k) or {}).get("verdict")
            solved = [r for r in mine if r.get("solve") and "answer" in r["solve"]]
            s[f"{judge}:key_match"] = _mean([r["solve"]["answer"] == r["answer"] for r in solved])
            s[f"{judge}:critical_rate"] = _mean(
                [any(v(r, k) == "no" for k in jp.CRITICAL_KEYS) for r in mine if "parse_error" not in (r.get("verdict") or {})])
            s[f"{judge}:explain_ok"] = _mean([v(r, "E1") == "yes" and v(r, "E2") == "yes" for r in mine])
            s[f"{judge}:parse_error_rate"] = _mean([bool((r.get("verdict") or {}).get("parse_error")) for r in mine])
            s[f"{judge}:n_items_scored"] = len(mine)
            s[f"{judge}:excluded_unverified_or_mismatch"] = sum(
                1 for r in all_rows if r["model"] == s["model"]) - len(mine)


def _mean(values):
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _write_table(run_dir: str, name: str, rows: list[dict]):
    with open(os.path.join(run_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=str)
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(run_dir, f"{name}.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in r.items()})


# ── verify-routes: 응답에 provider가 없던 호출을 /generation으로 확인 ─────────

def run_verify_routes(args, cfg):
    print(f"verified {verify_routes(os.path.join(RESULTS_ROOT, args.run_id), cfg)} calls")


def verify_routes(run_dir: str, cfg: dict) -> int:
    """route_ok가 확인되지 않았거나 비용이 추정치인 호출을 /generation으로 조회해 갱신한다."""
    from . import openrouter

    path = os.path.join(run_dir, "calls.jsonl")
    calls = _load_calls(run_dir)
    changed = 0
    for c in calls:
        if c.get("mock") or c.get("actual_provider") in ("local-ollama",) or not c.get("generation_id"):
            continue
        if c.get("route_ok") and c.get("cost_source") == "openrouter":
            continue
        data = fetch_generation(c["generation_id"])
        if not data:
            continue
        spec = cfg["models"][c["model_key"]]
        c["actual_model"] = data.get("model") or c.get("actual_model")
        c["actual_provider"] = data.get("provider_name") or c.get("actual_provider")
        if data.get("total_cost") is not None:
            c["cost_usd"], c["cost_source"] = data["total_cost"], "openrouter"
        c["route_ok"], c["route_note"] = openrouter.route_matches(spec, c["actual_model"], c["actual_provider"])
        changed += 1
    with open(path, "w", encoding="utf-8") as f:
        for c in calls:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return changed


# ── estimate: 호출 없이 예상 호출 수·최대 비용 ─────────────────────────────

def run_estimate(args, cfg):
    rows = load_rows(args.split, args.limit)
    n_sets = len(rows) * args.repeats
    avg_items = sum(r["expected"]["num_items"] for r in rows) / len(rows)
    n_revise = sum(len(r["revise_requests"]) for r in rows) / len(rows)
    judge = cfg["models"][cfg["runtime_judge"]]
    out, total = [], 0.0
    for key in args.models.split(","):
        spec = cfg["models"][key]
        tasks = {"set": n_sets, "explain": n_sets * avg_items, "revise": n_sets * n_revise}
        calls = sum(ESTIMATE[t]["calls"] * n for t, n in tasks.items())
        cost = sum(ESTIMATE[t]["calls"] * n * _price(spec, ESTIMATE[t]) for t, n in tasks.items())
        rj_calls = ESTIMATE["runtime_judge"]["calls"] * n_sets
        rj_cost = rj_calls * _price(judge, ESTIMATE["runtime_judge"])
        max_cost = (cost + rj_cost) * SAFETY_FACTOR
        total += max_cost
        out.append({"model": key, "endpoint": spec["endpoint"], "quantization": spec.get("quantization"),
                    "inputs": len(rows), "repeats": args.repeats, "est_calls": round(calls + rj_calls),
                    "est_cost_usd": round(cost + rj_cost, 4), "max_cost_usd": round(max_cost, 4)})
    for o in out:
        print(o)
    print(f"합계 최대 예상 비용: ${total:.2f} (추정치 — 토큰 가정은 ESTIMATE 상수, 안전계수 ×{SAFETY_FACTOR})")
    return out


def _price(spec: dict, est: dict) -> float:
    p = spec.get("pricing_per_mtok") or {}
    return (est["input"] * p.get("input", 0) + est["output"] * p.get("output", 0)) / 1_000_000


def main(argv=None):
    ap = argparse.ArgumentParser(prog="evals.model_selection.run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--split", default="pilot", choices=["pilot", "test"])
    g.add_argument("--models", required=True)
    g.add_argument("--repeats", type=int, default=1)
    g.add_argument("--limit", type=int)
    g.add_argument("--max-cost", type=float)
    g.add_argument("--run-id")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--sensitive", action="store_true", help="data_collection=deny, zdr=true 강제")
    g.add_argument("--trace", action="store_true", help="LangSmith 트레이싱 유지(기본은 끔)")

    s = sub.add_parser("score")
    s.add_argument("--run-id", required=True)
    s.add_argument("--judge", required=True)
    s.add_argument("--max-cost", type=float)
    s.add_argument("--mock", action="store_true")

    j = sub.add_parser("judge-validate")
    j.add_argument("--gold", required=True)
    j.add_argument("--judges", required=True)
    j.add_argument("--repeats", type=int, default=2)
    j.add_argument("--max-cost", type=float)
    j.add_argument("--run-id")
    j.add_argument("--mock", action="store_true")

    sm = sub.add_parser("summarize")
    sm.add_argument("--run-id", required=True)

    v = sub.add_parser("verify-routes")
    v.add_argument("--run-id", required=True)

    lb = sub.add_parser("export-labeling")
    lb.add_argument("--run-id", required=True)
    lb.add_argument("--batch", required=True)
    lb.add_argument("--defect-ratio", type=float, default=0.3)

    e = sub.add_parser("estimate")
    e.add_argument("--split", default="pilot", choices=["pilot", "test"])
    e.add_argument("--models", required=True)
    e.add_argument("--repeats", type=int, default=1)
    e.add_argument("--limit", type=int)

    args = ap.parse_args(argv)
    cfg = load_config()
    if args.cmd in ("generate", "score", "judge-validate"):
        if args.mock:
            args.max_cost = args.max_cost or 0.0001
        elif args.max_cost is None:
            ap.error("실제 실행에는 --max-cost가 필요합니다(유료 호출 상한).")
    return {"generate": run_generate, "score": run_score, "judge-validate": run_judge_validate,
            "summarize": run_summarize, "verify-routes": run_verify_routes, "estimate": run_estimate,
            "export-labeling": run_export_labeling}[args.cmd](args, cfg)


def run_export_labeling(args, cfg):
    from .labeling import export_labeling

    lab, mapping = export_labeling(os.path.join(RESULTS_ROOT, args.run_id), args.batch, defect_ratio=args.defect_ratio)
    n = sum(1 for _ in open(lab, encoding="utf-8"))
    print(f"labeling={os.path.relpath(lab, ROOT)} ({n} rows)  mapping={os.path.relpath(mapping, ROOT)} (평가자에게 주지 않음)")


if __name__ == "__main__":
    main()
