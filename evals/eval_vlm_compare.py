#!/usr/bin/env python
"""VLM 후보 비교 하네스 (모델 선정 계획 3단계, MODEL_SELECTION.md 7절·docs/plan 참고).

후보 5종(gpt-4o-mini 기준 / gpt-6-luna·gemini-3.8-flash OpenRouter / qwen3-vl-8b·gemma3-12b
로컬 Ollama)을 같은 골든셋(data/golden/vlm_extraction_golden.json)으로 비교한다.
evals/eval_vlm.py의 채점 함수(cer, wer, strip_figure_block, score_figure_entry,
score_adversarial)를 그대로 재사용하고, 새로 만드는 것은 결과를 파일로 쌓아 신뢰구간을
낼 수 있게 하는 하네스 자체다(eval_vlm.py는 결과를 저장하지 않아 CI를 못 낸다는 문제가
있었음).

서브커맨드:
  extract        골든셋 이미지를 실제 VLM으로 추출해 data/golden/_vlm_runs/<model>.jsonl에
                 이미지마다 즉시 append(이어서 실행 — 성공 기록이 있는 id는 건너뜀)
  judge          figure 항목의 서술을 Judge로 채점해
                 data/golden/_vlm_judged/<judge-slug>/<model>.jsonl 캐시에 append(이어서 실행)
  export-sheet   그림 전체(현재 골든셋의 figure 카테고리 건수 — 숫자에 의존하지 않음)에
                 후보 하나씩 시드 고정 무작위 균등 배정 → 블라인드 라벨링 시트 생성
  import-sheet   라벨링 시트를 검증 후 data/golden/vlm_figure_human_labels.json에 반영
                 (부분 진행 허용 — 검증 실패 문항만 제외하고 나머지는 반영)
  reliability    사람 라벨 대비 Judge 후보들의 가중 κ·MAE·편향(부트스트랩 CI), Judge 간
                 짝지은 비교, VLM 모델별 편향을 계산해 data/golden/_vlm_judge_selection.json에 저장
  compare        후보별 텍스트 CER/WER·그림 서술 품질·adversarial·지연을 종합하고, 기준
                 대비·후보 간 짝지은 비교(CI가 0 포함 시 판정 불가)를 계산해
                 data/golden/_vlm_comparison.json에 저장

하드룰 4(로그·캐시에 원문 금지): 콘솔 출력은 id·카테고리·수치만. run 파일(_vlm_runs/*.jsonl)에
raw_output을 저장하는 것은 예외다 — 전부 합성 이미지(하드룰 1)라 실제 학생 데이터나
교사 입력 원문이 아니다. judge 캐시·사람 라벨 파일에는 점수만 적고 서술 원문은 적지 않는다.

하드룰 2/3: 이미지 추출(/exam/extract) 경로는 마스킹 대상인 passage_text를 다루지 않는다
(런타임 경로와 달리 이 스크립트는 합성 이미지 → 텍스트 추출만 하고 PII 마스킹 단계가
원래 없는 경로 — app/main.py의 /exam/extract 자체도 마스킹 이전 단계임).
"""
import argparse
import asyncio
import itertools
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.common.llm.backends.openrouter import OpenRouterBackend
from evals.eval_vlm import _call_with_retry, cer, score_adversarial, score_figure_entry, strip_figure_block, wer
from evals.stats import cluster_bootstrap_ci, mean_ci, weighted_kappa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(ROOT, "data", "golden")
GOLDEN_PATH = os.path.join(GOLDEN_DIR, "vlm_extraction_golden.json")
RUNS_DIR = os.path.join(GOLDEN_DIR, "_vlm_runs")
JUDGED_DIR = os.path.join(GOLDEN_DIR, "_vlm_judged")
LABEL_SHEET_PATH = os.path.join(GOLDEN_DIR, "vlm_labeling_sheet.json")
LABEL_MAP_PATH = os.path.join(GOLDEN_DIR, "_vlm_label_map.json")
HUMAN_LABELS_PATH = os.path.join(GOLDEN_DIR, "vlm_figure_human_labels.json")
JUDGE_SELECTION_PATH = os.path.join(GOLDEN_DIR, "_vlm_judge_selection.json")
COMPARISON_PATH = os.path.join(GOLDEN_DIR, "_vlm_comparison.json")

# 후보별 env — app.common.llm import 전에 설정해야 한다(golden_gen/gen_item_quality_golden.py
# cmd_generate와 같은 관례). 키·env 값은 승인된 계획(encapsulated-herding-frost.md)의 매핑 그대로.
_CANDIDATE_ENV = {
    "gpt-4o-mini": {"VLM_BACKEND": "openai", "OPENAI_VLM_MODEL": "gpt-4o-mini"},
    "gpt-6-luna": {"VLM_BACKEND": "openrouter", "OPENROUTER_VLM_MODEL": "openai/gpt-6-luna"},
    "gemini-3.8-flash": {"VLM_BACKEND": "openrouter", "OPENROUTER_VLM_MODEL": "google/gemini-3.8-flash"},
    "qwen3-vl-8b": {"VLM_BACKEND": "local", "OLLAMA_VLM_MODEL": "qwen3-vl:8b"},
    "gemma3-12b": {"VLM_BACKEND": "local", "OLLAMA_VLM_MODEL": "gemma3:12b"},
}
_CANDIDATES = tuple(_CANDIDATE_ENV)
_LOCAL_CANDIDATES = {"qwen3-vl-8b", "gemma3-12b"}
_LOCAL_LATENCY_NOTE = "맥 M5 로컬 기준 — 운영 GPU와 다름"
_COST_NOTE = "미측정(백엔드가 토큰 수를 반환하지 않아 기록하지 않음)"
_JUDGE_SCORE_LABELS = (1, 2, 3, 4, 5)


# ── 공용 경로/로더 헬퍼 ──────────────────────────────────────────────────

def _judge_slug(judge_model: str) -> str:
    return judge_model.replace("/", "__")


def _run_path(model: str) -> str:
    return os.path.join(RUNS_DIR, f"{model}.jsonl")


def _judged_path(judge_model: str, model: str) -> str:
    return os.path.join(JUDGED_DIR, _judge_slug(judge_model), f"{model}.jsonl")


def _code_version() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(__file__),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _round3(x: float | None) -> float | None:
    return round(x, 3) if x is not None else None


def load_golden() -> dict:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_run_records(path: str) -> list[dict]:
    """run/judged jsonl 파일을 읽어 레코드 리스트로 반환한다. 파일이 없으면 빈 리스트."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dedupe_runs(records: list[dict]) -> dict[str, dict]:
    """같은 id가 여러 번 append된 run(재실행 등)을 id당 레코드 하나로 정리한다 — 성공
    (error 없음) 기록이 하나라도 있으면 그중 마지막 것을, 전부 실패면 마지막 레코드를 쓴다."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["id"], []).append(r)
    out = {}
    for fid, group in groups.items():
        successes = [r for r in group if not r.get("error")]
        out[fid] = successes[-1] if successes else group[-1]
    return out


def load_human_labels() -> dict:
    if not os.path.exists(HUMAN_LABELS_PATH):
        return {}
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        return json.load(f).get("labels", {})


def load_label_map() -> dict:
    if not os.path.exists(LABEL_MAP_PATH):
        return {}
    with open(LABEL_MAP_PATH, encoding="utf-8") as f:
        return json.load(f).get("map", {})


# ── extract: 골든셋 이미지를 실제 VLM으로 추출 ───────────────────────────

def select_golden_entries(
    entries: list[dict], ids: list[str] | None = None, limit: int | None = None,
) -> list[dict]:
    """--ids/--limit에 따라 처리할 골든셋 항목을 고른다(LLM 호출 없음, 테스트 가능)."""
    selected = [e for e in entries if e["id"] in set(ids)] if ids else list(entries)
    if limit is not None:
        selected = selected[:limit]
    return selected


async def extract_one(vlm, entry: dict) -> dict:
    """이미지 1건을 추출해 run 레코드(모델·code_version 제외)를 만든다. 실패 시 raw는
    빈 문자열, error에 예외 정보를 담고 cer/wer은 계산하지 않는다(None)."""
    img_path = os.path.join(GOLDEN_DIR, entry["image"])
    with open(img_path, "rb") as f:
        image_bytes = f.read()

    start = time.perf_counter()
    try:
        raw = await _call_with_retry(lambda: vlm.extract_text(image_bytes, "image/png"))
        error = None
    except Exception as exc:  # noqa: BLE001 — 평가 스크립트: 개별 실패를 기록하고 계속 진행
        raw = ""
        error = {"type": type(exc).__name__, "message": str(exc)[:500]}
    elapsed = time.perf_counter() - start

    if error is None:
        if entry["category"] == "figure":
            text_part, fig_text = strip_figure_block(raw)
        else:
            text_part, fig_text = raw, None
        c = cer(entry["ground_truth_text"], text_part)
        w = wer(entry["ground_truth_text"], text_part)
    else:
        fig_text = None
        c = w = None

    return {
        "id": entry["id"],
        "category": entry["category"],
        "raw_output": raw,
        "cer": c,
        "wer": w,
        "figure_description": fig_text,
        "latency_sec": round(elapsed, 3),
        "error": error,
    }


async def _run_extract(args: argparse.Namespace) -> None:
    for k, v in _CANDIDATE_ENV[args.model].items():
        os.environ[k] = v

    from app.common.llm import get_vlm_backend
    vlm = get_vlm_backend()

    golden_entries = load_golden()["entries"]
    entries = select_golden_entries(golden_entries, ids=args.ids, limit=args.limit)

    out_path = _run_path(args.model)
    os.makedirs(RUNS_DIR, exist_ok=True)
    done_ids = {r["id"] for r in load_run_records(out_path) if not r.get("error")}

    print(f"=== VLM 후보: {args.model} ({_CANDIDATE_ENV[args.model]['VLM_BACKEND']}) ===")
    print(f"대상 {len(entries)}건 (이미 완료 {len(done_ids & {e['id'] for e in entries})}건 스킵)\n")

    code_version = _code_version()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, entry in enumerate(entries, 1):
            if entry["id"] in done_ids:
                print(f"[{i}/{len(entries)}] {entry['id']} 이미 완료 — 스킵")
                continue
            result = await extract_one(vlm, entry)
            record = {**result, "model": args.model, "code_version": code_version}
            if record["error"]:
                print(f"[{i}/{len(entries)}] {entry['id']} ({entry['category']}) 실패: {record['error']['type']}")
            else:
                print(
                    f"[{i}/{len(entries)}] {entry['id']} ({entry['category']}) "
                    f"CER={record['cer']:.3f} WER={record['wer']:.3f} {record['latency_sec']}s"
                )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\n완료 — {out_path}")


def cmd_extract(args: argparse.Namespace) -> None:
    asyncio.run(_run_extract(args))


# ── judge: figure 항목의 서술을 Judge로 채점 ─────────────────────────────

async def _run_judge(args: argparse.Namespace) -> None:
    judge = OpenRouterBackend(model=args.judge_model)

    golden_entries = load_golden()["entries"]
    figures_by_id = {e["id"]: e for e in golden_entries if e["category"] == "figure"}

    models = args.models or list(_CANDIDATES)
    print(f"=== Judge: {args.judge_model} / 대상 모델: {', '.join(models)} / repeat={args.repeat} ===")

    for model in models:
        records_by_id = {r["id"]: r for r in load_run_records(_run_path(model)) if not r.get("error")}
        out_path = _judged_path(args.judge_model, model)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        done = {(r["id"], r["repeat"]) for r in load_run_records(out_path)}

        n_judged = n_skipped = 0
        with open(out_path, "a", encoding="utf-8") as f:
            for fid, rec in records_by_id.items():
                if fid not in figures_by_id:
                    continue
                for rep in range(args.repeat):
                    key = (fid, rep)
                    if key in done:
                        n_skipped += 1
                        continue
                    result = await score_figure_entry(
                        figures_by_id[fid]["figure_summary"], rec.get("raw_output", ""), judge,
                        call_with_retry=_call_with_retry,
                    )
                    out_record = {
                        "id": fid,
                        "model": model,
                        "judge_model": args.judge_model,
                        "repeat": rep,
                        "judge_score": result["judge_score"],
                        "had_description": result["fig_text"] is not None,
                    }
                    f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                    f.flush()
                    n_judged += 1
        print(f"[{model}] 신규 채점 {n_judged}건, 캐시 스킵 {n_skipped}건 → {out_path}")


def cmd_judge(args: argparse.Namespace) -> None:
    asyncio.run(_run_judge(args))


# ── export-sheet / import-sheet (블라인드 사람 라벨링, JSON 직접 편집) ───

_SHEET_GUIDE = [
    "이 파일은 JSON 편집기로 직접 라벨을 채우는 시트입니다. 각 문항의 \"라벨\" 블록만 채우세요"
    "(핵심사실·VLM_서술_줄은 참고용 원문이라 수정해도 반영되지 않습니다).",
    "모델 정보는 제공하지 않습니다 — 추측하지 말고 핵심사실과 VLM 서술만 보고 판단하세요.",
    "점수: 1~5 정수. 2점 이하면 근거를 반드시 채워야 합니다"
    "(비어 있으면 import-sheet가 그 문항만 반영을 거부합니다).",
    "",
    "[기준]",
    "5: 핵심 수치와 경향을 거의 다 담음",
    "4: 대부분 담았으나 일부 아쉬움",
    "3: 일부만 담음",
    "2: 핵심을 상당 부분 놓침",
    "1: 핵심을 놓쳤거나 서술 자체가 없음 (VLM_서술_줄이 [\"서술 없음\"]이면 이 경우)",
]

_SCORE_KEY = "점수"
_REASON_KEY = "근거"


def assign_models_balanced(ids: list[str], candidates: Sequence[str], seed: int = 0) -> dict[str, str]:
    """ids(정렬된 순서)마다 candidates 중 하나를 시드 고정 무작위로, 배정 수가 후보별로
    가능한 한 균등하게(나머지는 candidates 앞부분에 1개씩 추가 배정) 배정한다."""
    n, k = len(ids), len(candidates)
    pool: list[str] = []
    for c in candidates:
        pool.extend([c] * (n // k))
    remainder = n - len(pool)
    if remainder:
        pool.extend(candidates[:remainder])
    rng = random.Random(seed)
    rng.shuffle(pool)
    return dict(zip(ids, pool))


def build_label_sheet_entry(entry: dict, fig_text: str | None) -> dict:
    """golden figure entry 1개 + 추출된 서술로 시트 항목을 만든다(라벨 칸이 맨 위).
    서술이 없으면 ["서술 없음"]을 넣어 빈 서술임을 명시한다."""
    lines = fig_text.split("\n") if fig_text else ["서술 없음"]
    return {
        "라벨": {_SCORE_KEY: None, _REASON_KEY: ""},
        "id": entry["id"],
        "핵심사실": entry["figure_summary"],
        "VLM_서술_줄": lines,
    }


def has_existing_sheet_labels(path: str) -> bool:
    """이미 라벨이 하나라도 채워진 시트 파일이 있으면 True(라벨 유실 방지용 가드)."""
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("문항", []):
        label = item.get("라벨", {}) or {}
        if label.get(_SCORE_KEY) is not None or str(label.get(_REASON_KEY) or "").strip():
            return True
    return False


def cmd_export_sheet(args: argparse.Namespace) -> None:
    if has_existing_sheet_labels(LABEL_SHEET_PATH):
        print(f"중단 — {LABEL_SHEET_PATH}에 이미 채워진 라벨이 있습니다. 라벨 유실 방지를 위해 덮어쓰지 않습니다.")
        sys.exit(1)

    golden_entries = load_golden()["entries"]
    figures = sorted((e for e in golden_entries if e["category"] == "figure"), key=lambda e: e["id"])
    ids = [e["id"] for e in figures]
    assignment = assign_models_balanced(ids, _CANDIDATES, seed=args.seed)

    needed_models = sorted(set(assignment.values()))
    records_by_model = {m: {r["id"]: r for r in load_run_records(_run_path(m))} for m in needed_models}

    sheet_items = []
    for e in figures:
        model = assignment[e["id"]]
        rec = records_by_model.get(model, {}).get(e["id"])
        fig_text = rec.get("figure_description") if rec else None
        sheet_items.append(build_label_sheet_entry(e, fig_text))

    sheet = {"_안내": _SHEET_GUIDE, "문항": sheet_items}
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    with open(LABEL_SHEET_PATH, "w", encoding="utf-8") as f:
        json.dump(sheet, f, ensure_ascii=False, indent=2)

    label_map_doc = {
        "_schema": {
            "description": "vlm_labeling_sheet.json의 블라인드 id → 실제 후보 모델 매핑(라벨링 중 참조 금지).",
        },
        "map": {fid: {"model": assignment[fid]} for fid in ids},
    }
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map_doc, f, ensure_ascii=False, indent=2)

    counts = dict(Counter(assignment.values()))
    print(f"완료 — {LABEL_SHEET_PATH}에 {len(sheet_items)}개 문항 (후보별 배정 수: {counts})")


def validate_sheet_label(label: dict) -> list[str]:
    """시트 문항 1개의 '라벨' 블록을 검증한다. 오류 메시지 리스트(없으면 빈 리스트)."""
    errors = []
    score = label.get(_SCORE_KEY)
    if score is not None and (isinstance(score, bool) or not isinstance(score, int) or not (1 <= score <= 5)):
        errors.append(f"{_SCORE_KEY} 값이 1~5 정수나 null이 아님")
    reason = str(label.get(_REASON_KEY) or "").strip()
    if isinstance(score, int) and not isinstance(score, bool) and score <= 2 and not reason:
        errors.append(f"{score}점(2점 이하)인데 {_REASON_KEY}가 비어 있음")
    return errors


def validate_sheet(sheet_items: list[dict], valid_ids: set[str]) -> list[tuple[str, list[str]]]:
    """시트 문항 전체를 검증한다. 반환: [(id, 오류 메시지 리스트), ...](유효한 문항은 제외)."""
    errors = []
    for item in sheet_items:
        item_id = item.get("id")
        item_errors = []
        if item_id not in valid_ids:
            item_errors.append("골든셋(label_map)에 존재하지 않는 id")
        item_errors.extend(validate_sheet_label(item.get("라벨", {}) or {}))
        if item_errors:
            errors.append((item_id, item_errors))
    return errors


def apply_sheet_labels(
    existing_labels: dict, sheet_items: list[dict], valid_ids: set[str],
) -> tuple[dict, list[tuple[str, list[str]]], int]:
    """검증을 통과한 sheet_items의 점수를 existing_labels에 반영한다(부분 진행 허용 —
    검증에 실패한 문항만 건너뛰고 나머지는 반영, 아직 점수가 없는 문항은 저장할 것이
    없으므로 건너뜀). 반환: (갱신된 labels dict, 오류 리스트, 이번에 반영한 건수)."""
    errors = validate_sheet(sheet_items, valid_ids)
    error_ids = {item_id for item_id, _ in errors}
    updated = dict(existing_labels)
    n_saved = 0
    for item in sheet_items:
        item_id = item.get("id")
        if item_id in error_ids:
            continue
        label = item.get("라벨", {}) or {}
        score = label.get(_SCORE_KEY)
        if score is None:
            continue  # 아직 라벨링 전 — 저장할 것 없음
        updated[item_id] = {_SCORE_KEY: score, _REASON_KEY: str(label.get(_REASON_KEY) or "")}
        n_saved += 1
    return updated, errors, n_saved


def cmd_import_sheet(args: argparse.Namespace) -> None:
    sheet_path = args.sheet or LABEL_SHEET_PATH
    with open(sheet_path, encoding="utf-8") as f:
        sheet = json.load(f)

    label_map = load_label_map()
    valid_ids = set(label_map)
    existing = load_human_labels()
    sheet_items = sheet.get("문항", [])

    updated, errors, n_saved = apply_sheet_labels(existing, sheet_items, valid_ids)
    if errors:
        print(f"검증 실패 — {len(errors)}개 문항(해당 문항만 반영하지 않음)")
        for item_id, msgs in errors:
            for msg in msgs:
                print(f"  {item_id}: {msg}")

    doc = {
        "_schema": {
            "description": (
                "그림 서술 블라인드 사람 라벨(점수 1~5, 근거) — id는 vlm_labeling_sheet.json/"
                "_vlm_label_map.json과 공유."
            ),
        },
        "labels": updated,
    }
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    with open(HUMAN_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(f"완료 — 이번 {n_saved}건 반영, 누적 {len(updated)}건 → {HUMAN_LABELS_PATH}")


# ── reliability: 사람 라벨 vs Judge 후보 신뢰도(κ·MAE·편향) ──────────────

def judge_avg(scores: list[int]) -> float:
    return sum(scores) / len(scores)


def judge_avg_rounded(scores: list[int]) -> int:
    return round(judge_avg(scores))


def build_reliability_rows(
    human_labels: dict, label_map: dict, judged_by_judge: dict[str, dict[str, list[dict]]],
) -> list[dict]:
    """human_labels({fid: {"점수":int,...}}) + label_map({fid: {"model":str}})을 조인하고,
    judged_by_judge({judge_model: {model: judged 레코드 리스트}})에서 그 fid·model에 해당하는
    반복들을 모아 행을 만든다. judge_score가 None인 반복(채점 실패, 블록 없음 1점 처리는
    여기 포함됨)만 제외하는 게 아니라 None만 걸러낸다 — 반환:
    [{"id", "model", "human": int, "judges": {judge_model: [score, ...]}}]."""
    rows = []
    for fid, hl in human_labels.items():
        score = hl.get(_SCORE_KEY)
        if score is None:
            continue
        mapping = label_map.get(fid)
        if mapping is None:
            continue
        model = mapping["model"]
        judges: dict[str, list[int]] = {}
        for jm, by_model in judged_by_judge.items():
            scores = [
                r["judge_score"] for r in by_model.get(model, [])
                if r["id"] == fid and r["judge_score"] is not None
            ]
            if scores:
                judges[jm] = scores
        rows.append({"id": fid, "model": model, "human": score, "judges": judges})
    return rows


def per_repeat_kappas(rows_with_judge: list[dict], judge_model: str) -> list[float | None]:
    """반복 평균으로 뭉개지 않고, 반복 인덱스별로 따로 가중 κ를 계산한다(판정 안정성 확인용)."""
    max_repeats = max((len(r["judges"][judge_model]) for r in rows_with_judge), default=0)
    kappas: list[float | None] = []
    for k in range(max_repeats):
        pairs = [
            (r["human"], r["judges"][judge_model][k])
            for r in rows_with_judge if len(r["judges"][judge_model]) > k
        ]
        if len(pairs) < 2:
            kappas.append(None)
            continue
        human_vals = [p[0] for p in pairs]
        judge_vals = [p[1] for p in pairs]
        kappas.append(weighted_kappa(human_vals, judge_vals, labels=_JUDGE_SCORE_LABELS))
    return kappas


def paired_abs_error_diff(rows: list[dict], judge_a: str, judge_b: str) -> list[tuple[str, float]]:
    """두 Judge 모두 캐시가 있는 이미지에서 |judge-human| 오차 차이(A−B)를 id와 함께
    반환한다. 양수면 B가 그 이미지에서 더 정확(오차가 더 작음)."""
    pairs = []
    for r in rows:
        if judge_a in r["judges"] and judge_b in r["judges"]:
            err_a = abs(judge_avg(r["judges"][judge_a]) - r["human"])
            err_b = abs(judge_avg(r["judges"][judge_b]) - r["human"])
            pairs.append((r["id"], err_a - err_b))
    return pairs


def bias_by_vlm_candidate(rows: list[dict], judge_model: str) -> dict[str, dict]:
    """VLM 후보 모델별 (judge-human) 평균 편향 + 이미지 단위 클러스터 부트스트랩 CI."""
    result = {}
    for model in sorted({r["model"] for r in rows}):
        subset = [r for r in rows if r["model"] == model and judge_model in r["judges"]]
        if not subset:
            result[model] = {"n": 0, "bias": None, "ci_low": None, "ci_high": None}
            continue
        diffs = [judge_avg(r["judges"][judge_model]) - r["human"] for r in subset]
        clusters = [r["id"] for r in subset]
        point, lo, hi = mean_ci(diffs, clusters)
        result[model] = {"n": len(subset), "bias": _round3(point), "ci_low": _round3(lo), "ci_high": _round3(hi)}
    return result


def compute_judge_reliability(rows: list[dict], judge_models: list[str]) -> dict:
    """reliability의 핵심 계산 전체(LLM 호출 없음) — cmd_reliability와 테스트가 공유."""
    per_judge = {}
    for jm in judge_models:
        rows_with_judge = [r for r in rows if jm in r["judges"]]
        n = len(rows_with_judge)
        if n < 2:
            per_judge[jm] = {"n": n, "note": "표본 부족(2건 미만)"}
            continue

        human_vals = [r["human"] for r in rows_with_judge]
        judge_vals_rounded = [judge_avg_rounded(r["judges"][jm]) for r in rows_with_judge]
        clusters = [r["id"] for r in rows_with_judge]

        kappa_point, kappa_lo, kappa_hi = cluster_bootstrap_ci(
            list(zip(human_vals, judge_vals_rounded)), clusters,
            lambda sample: weighted_kappa(
                [a for a, b in sample], [b for a, b in sample], labels=_JUDGE_SCORE_LABELS,
            ),
        )
        mae_values = [abs(judge_avg(r["judges"][jm]) - r["human"]) for r in rows_with_judge]
        mae_point, mae_lo, mae_hi = mean_ci(mae_values, clusters)
        bias_values = [judge_avg(r["judges"][jm]) - r["human"] for r in rows_with_judge]
        bias_point, bias_lo, bias_hi = mean_ci(bias_values, clusters)
        repeat_kappas = [k for k in per_repeat_kappas(rows_with_judge, jm) if k is not None]

        per_judge[jm] = {
            "n": n,
            "weighted_kappa": _round3(kappa_point), "kappa_ci": [_round3(kappa_lo), _round3(kappa_hi)],
            "mae": _round3(mae_point), "mae_ci": [_round3(mae_lo), _round3(mae_hi)],
            "bias": _round3(bias_point), "bias_ci": [_round3(bias_lo), _round3(bias_hi)],
            "repeat_kappa_mean": _round3(sum(repeat_kappas) / len(repeat_kappas)) if repeat_kappas else None,
            "repeat_kappa_range": (
                [_round3(min(repeat_kappas)), _round3(max(repeat_kappas))] if repeat_kappas else None
            ),
        }

    pairwise = {}
    for i in range(len(judge_models)):
        for j in range(i + 1, len(judge_models)):
            judge_a, judge_b = judge_models[i], judge_models[j]
            diff_pairs = paired_abs_error_diff(rows, judge_a, judge_b)
            label = f"{judge_a} vs {judge_b}"
            if len(diff_pairs) < 2:
                pairwise[label] = {"n": len(diff_pairs), "note": "표본 부족(2건 미만)"}
                continue
            diffs = [d for _, d in diff_pairs]
            clusters = [fid for fid, _ in diff_pairs]
            point, lo, hi = mean_ci(diffs, clusters)
            if lo is not None and hi is not None and lo <= 0 <= hi:
                verdict = "판정 불가(CI가 0을 포함)"
            elif point is not None and point > 0:
                verdict = f"{judge_b}가 더 정확"
            else:
                verdict = f"{judge_a}가 더 정확"
            pairwise[label] = {
                "n": len(diff_pairs), "mean_abs_error_diff": _round3(point),
                "ci": [_round3(lo), _round3(hi)], "verdict": verdict,
            }

    bias_by_model = {jm: bias_by_vlm_candidate(rows, jm) for jm in judge_models}

    return {"n_usable": len(rows), "per_judge": per_judge, "pairwise": pairwise, "bias_by_vlm_model": bias_by_model}


def _print_reliability_report(report: dict, judge_models: list[str]) -> None:
    """콘솔 출력 — 서술 원문은 절대 출력하지 않는다(하드룰 4). 지표 이름 옆에 짧은 설명을 붙인다."""
    print(f"사람 라벨(점수 있는 이미지) {report['n_usable']}건 사용")

    for jm in judge_models:
        pj = report["per_judge"].get(jm, {})
        if pj.get("note"):
            print(f"\n[{jm}] {pj['note']} (n={pj.get('n', 0)})")
            continue
        print(f"\n[{jm}] (n={pj['n']})")
        print(f"  weighted κ(가중 카파, 사람과의 순서형 일치도, 1에 가까울수록 좋음)={pj['weighted_kappa']} CI={pj['kappa_ci']}")
        print(f"  repeat별 κ 평균/범위(반복마다 따로 계산한 κ의 안정성)={pj['repeat_kappa_mean']} / {pj['repeat_kappa_range']}")
        print(f"  MAE(평균 절대 오차, 낮을수록 좋음)={pj['mae']} CI={pj['mae_ci']}")
        print(f"  편향(judge-human 평균, 양수면 Judge가 사람보다 후함)={pj['bias']} CI={pj['bias_ci']}")

    if report["pairwise"]:
        print("\n[짝지은 비교: |judge-human| 오차 차이(A-B), CI가 0을 포함하면 판정 불가]")
        for label, pw in report["pairwise"].items():
            if pw.get("note"):
                print(f"  {label}: {pw['note']} (n={pw['n']})")
            else:
                print(f"  {label}: mean_diff={pw['mean_abs_error_diff']} CI={pw['ci']} → {pw['verdict']}")

    print("\n[VLM 후보별 편향(judge-human 평균) — 같은 계열을 후하게 주는지 확인용]")
    for jm in judge_models:
        print(f"  {jm}:")
        for model, b in report["bias_by_vlm_model"][jm].items():
            if b["n"] == 0:
                print(f"    {model}: 표본 없음")
            else:
                print(f"    {model}: n={b['n']} bias={b['bias']} CI=[{b['ci_low']}, {b['ci_high']}]")


def cmd_reliability(args: argparse.Namespace) -> None:
    human_labels = load_human_labels()
    label_map = load_label_map()
    if not human_labels or not label_map:
        print(f"중단 — {HUMAN_LABELS_PATH} 또는 {LABEL_MAP_PATH}이 없습니다. export-sheet/import-sheet를 먼저 실행하세요.")
        sys.exit(1)

    judged_by_judge = {
        jm: {model: load_run_records(_judged_path(jm, model)) for model in _CANDIDATES}
        for jm in args.judges
    }
    rows = build_reliability_rows(human_labels, label_map, judged_by_judge)
    report = compute_judge_reliability(rows, args.judges)
    report["rows"] = rows  # 문항별 상세 — 재분석이 API 재호출 없이 가능하도록(점수만, 원문 없음)

    with open(JUDGE_SELECTION_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _print_reliability_report(report, args.judges)
    print(f"\n문항별 상세는 {JUDGE_SELECTION_PATH}에 저장했습니다.")


# ── compare: 후보 종합 비교 ───────────────────────────────────────────────

def text_only_cer_wer(records_by_id: dict[str, dict], golden_by_id: dict[str, dict]) -> dict:
    ids = [
        fid for fid, g in golden_by_id.items()
        if g["category"] == "text_only" and fid in records_by_id and not records_by_id[fid].get("error")
    ]
    cers = [records_by_id[fid]["cer"] for fid in ids]
    wers = [records_by_id[fid]["wer"] for fid in ids]
    cer_point, cer_lo, cer_hi = mean_ci(cers, ids)
    wer_point, wer_lo, wer_hi = mean_ci(wers, ids)
    return {
        "n": len(ids),
        "cer_mean": _round3(cer_point), "cer_ci": [_round3(cer_lo), _round3(cer_hi)],
        "wer_mean": _round3(wer_point), "wer_ci": [_round3(wer_lo), _round3(wer_hi)],
    }


def figure_text_cer(records_by_id: dict[str, dict], golden_by_id: dict[str, dict]) -> dict:
    ids = [
        fid for fid, g in golden_by_id.items()
        if g["category"] == "figure" and fid in records_by_id and not records_by_id[fid].get("error")
    ]
    cers = [records_by_id[fid]["cer"] for fid in ids]
    cer_point, cer_lo, cer_hi = mean_ci(cers, ids)
    return {"n": len(ids), "cer_mean": _round3(cer_point), "cer_ci": [_round3(cer_lo), _round3(cer_hi)]}


def figure_judge_metrics(judged_records: list[dict]) -> dict:
    scored = [(r["id"], r["judge_score"]) for r in judged_records if r["judge_score"] is not None]
    ids = [fid for fid, _ in scored]
    scores = [s for _, s in scored]
    mean_point, mean_lo, mean_hi = mean_ci(scores, ids)
    low_flags = [1 if s <= 2 else 0 for s in scores]
    low_point, low_lo, low_hi = mean_ci(low_flags, ids)
    return {
        "n": len(scored),
        "mean": _round3(mean_point), "mean_ci": [_round3(mean_lo), _round3(mean_hi)],
        "low_quality_rate": _round3(low_point), "low_quality_ci": [_round3(low_lo), _round3(low_hi)],
    }


def missing_description_count(records_by_id: dict[str, dict], golden_by_id: dict[str, dict]) -> int:
    return sum(
        1 for fid, g in golden_by_id.items()
        if g["category"] == "figure" and fid in records_by_id
        and not records_by_id[fid].get("error") and records_by_id[fid].get("figure_description") is None
    )


def human_reference_mean(human_labels: dict, label_map: dict, model: str) -> dict:
    """사람 라벨 참고 평균 — 그 모델에 배정된 이미지만, 표본이 작아 참고용이다."""
    scores = [
        hl[_SCORE_KEY] for fid, hl in human_labels.items()
        if hl.get(_SCORE_KEY) is not None and (label_map.get(fid) or {}).get("model") == model
    ]
    return {
        "n": len(scores),
        "mean": round(sum(scores) / len(scores), 3) if scores else None,
        "note": "참고용(배정 건수가 적음)",
    }


def adversarial_results_for_model(records_by_id: dict[str, dict], golden_by_id: dict[str, dict]) -> dict:
    """score_adversarial(eval_vlm.py)을 그대로 재사용해 adversarial 항목을 채점한다."""
    entries = []
    for fid, g in golden_by_id.items():
        if g["category"] != "adversarial":
            continue
        rec = records_by_id.get(fid)
        vlm_output = rec.get("raw_output", "") if rec else ""
        entries.append({**g, "vlm_output": vlm_output})
    score_adversarial(entries)
    n_leak_fail = sum(1 for e in entries if e["adversarial_kind"] == "no_leak" and e["adversarial_pass"] is False)
    n_pii_fail = sum(1 for e in entries if e["adversarial_kind"] == "pii_variant" and e["adversarial_pass"] is False)
    return {
        "n": len(entries), "leak_fail": n_leak_fail, "pii_fail": n_pii_fail,
        "detail": [{"id": e["id"], "kind": e["adversarial_kind"], "pass": e["adversarial_pass"]} for e in entries],
    }


def latency_stats(records_by_id: dict[str, dict]) -> dict:
    vals = [r["latency_sec"] for r in records_by_id.values() if not r.get("error")]
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 3) if vals else None,
        "median": round(statistics.median(vals), 3) if vals else None,
    }


def paired_diff_ci(
    values_a: dict[str, float], values_b: dict[str, float],
    higher_is_better: bool, label_a: str, label_b: str,
) -> dict:
    """id -> 값 dict 두 개를 받아, 두 후보 모두 값이 있는 이미지만으로 차이(b-a)의 평균과
    이미지 단위 클러스터 부트스트랩 95% CI, 판정을 계산한다."""
    common = sorted(set(values_a) & set(values_b))
    if not common:
        return {"n": 0, "mean_diff": None, "ci": [None, None], "verdict": "표본 없음(공통 이미지 없음)"}
    diffs = [values_b[k] - values_a[k] for k in common]
    point, lo, hi = mean_ci(diffs, common)
    if lo is None or hi is None:
        verdict = "판정 불가(표본 부족)"
    elif lo <= 0 <= hi:
        verdict = "판정 불가(CI가 0을 포함)"
    else:
        better = label_b if (point > 0) == higher_is_better else label_a
        verdict = f"{better}가 더 우수"
    return {"n": len(common), "mean_diff": _round3(point), "ci": [_round3(lo), _round3(hi)], "verdict": verdict}


def cer_by_id(records_by_id: dict[str, dict], golden_by_id: dict[str, dict]) -> dict[str, float]:
    """text_only + figure(텍스트 부분)의 이미지별 CER — 짝지은 비교 재료."""
    return {
        fid: rec["cer"] for fid, rec in records_by_id.items()
        if not rec.get("error") and golden_by_id.get(fid, {}).get("category") in ("text_only", "figure")
        and rec.get("cer") is not None
    }


def judge_score_by_id(judged_records: list[dict]) -> dict[str, float]:
    """이미지별 Judge 점수(반복 평균, None 제외) — 짝지은 비교 재료."""
    grouped: dict[str, list[int]] = {}
    for r in judged_records:
        if r["judge_score"] is not None:
            grouped.setdefault(r["id"], []).append(r["judge_score"])
    return {fid: sum(v) / len(v) for fid, v in grouped.items()}


def compute_comparison(
    models: list[str],
    baseline: str,
    judge_model: str,
    records_by_model: dict[str, dict[str, dict]],
    judged_by_model: dict[str, list[dict]],
    golden_entries: list[dict],
    human_labels: dict,
    label_map: dict,
    code_version: str,
) -> dict:
    """compare의 핵심 계산 전체(LLM 호출 없음) — cmd_compare와 테스트가 공유.
    records_by_model은 이미 dedupe_runs() 적용된 {model: {id: record}} 형태여야 한다."""
    golden_by_id = {e["id"]: e for e in golden_entries}

    per_model = {}
    cer_by_model = {}
    judge_score_by_model = {}
    for m in models:
        records_by_id = records_by_model[m]
        judged_records = judged_by_model.get(m, [])
        per_model[m] = {
            "text_only": text_only_cer_wer(records_by_id, golden_by_id),
            "figure_text": figure_text_cer(records_by_id, golden_by_id),
            "figure_judge": figure_judge_metrics(judged_records),
            "figure_missing_description": missing_description_count(records_by_id, golden_by_id),
            "figure_human_reference": human_reference_mean(human_labels, label_map, m),
            "adversarial": adversarial_results_for_model(records_by_id, golden_by_id),
            "latency": latency_stats(records_by_id),
            "latency_note": _LOCAL_LATENCY_NOTE if m in _LOCAL_CANDIDATES else None,
            "cost_note": _COST_NOTE,
        }
        cer_by_model[m] = cer_by_id(records_by_id, golden_by_id)
        judge_score_by_model[m] = judge_score_by_id(judged_records)

    # 짝지은 비교: 모든 후보 조합(기준 대비는 그 안에 자동 포함) — "상위 후보"는 결과를 본
    # 뒤에야 알 수 있으므로, 사전에 임의로 고르지 않고 전체 조합을 계산해 판단 재료로 둔다.
    pairwise = {}
    for a, b in itertools.combinations(models, 2):
        pairwise[f"{a}_vs_{b}"] = {
            "cer_diff": paired_diff_ci(cer_by_model[a], cer_by_model[b], higher_is_better=False, label_a=a, label_b=b),
            "figure_judge_diff": paired_diff_ci(
                judge_score_by_model[a], judge_score_by_model[b], higher_is_better=True, label_a=a, label_b=b,
            ),
        }

    return {
        "code_version": code_version,
        "judge_model": judge_model,
        "baseline": baseline,
        "models": models,
        "per_model": per_model,
        "pairwise": pairwise,
    }


def _print_comparison(report: dict) -> None:
    """콘솔 출력 — 서술/발문 원문은 출력하지 않는다(하드룰 4)."""
    print(f"Judge: {report['judge_model']} / 기준 후보(baseline): {report['baseline']}")

    for m in report["models"]:
        pm = report["per_model"][m]
        print(f"\n[{m}]")
        t = pm["text_only"]
        print(f"  text_only(n={t['n']}) CER 평균={t['cer_mean']} CI={t['cer_ci']}  WER 평균={t['wer_mean']} CI={t['wer_ci']}")
        ft = pm["figure_text"]
        print(f"  figure 텍스트 부분(n={ft['n']}) CER 평균={ft['cer_mean']} CI={ft['cer_ci']}")
        fj = pm["figure_judge"]
        print(
            f"  figure 서술 Judge 점수(n={fj['n']}) 평균={fj['mean']} CI={fj['mean_ci']}  "
            f"저품질(≤2점) 비율={fj['low_quality_rate']} CI={fj['low_quality_ci']}"
        )
        print(f"  figure 서술 누락 수={pm['figure_missing_description']}")
        hr = pm["figure_human_reference"]
        print(f"  figure 사람 점수 평균(n={hr['n']}, {hr['note']})={hr['mean']}")
        adv = pm["adversarial"]
        print(f"  adversarial(n={adv['n']}) no_leak 위반={adv['leak_fail']}  pii_variant 필수 라벨 누락={adv['pii_fail']}")
        lat = pm["latency"]
        note = f" [{pm['latency_note']}]" if pm["latency_note"] else ""
        print(f"  지연(초) 평균/중앙값={lat['mean']}/{lat['median']}{note}  비용: {pm['cost_note']}")

    print("\n[짝지은 비교] (같은 이미지에 두 후보 모두 값이 있는 경우만, CI가 0을 포함하면 판정 불가)")
    for label, pw in report["pairwise"].items():
        print(f"  {label}")
        cd = pw["cer_diff"]
        print(f"    CER 차이: n={cd['n']} mean_diff={cd['mean_diff']} CI={cd['ci']} → {cd['verdict']}")
        jd = pw["figure_judge_diff"]
        print(f"    figure 서술 Judge 점수 차이: n={jd['n']} mean_diff={jd['mean_diff']} CI={jd['ci']} → {jd['verdict']}")


def cmd_compare(args: argparse.Namespace) -> None:
    models = args.models or [m for m in _CANDIDATES if os.path.exists(_run_path(m))]
    if not models:
        print("중단 — run 파일이 하나도 없습니다. extract를 먼저 실행하세요.")
        sys.exit(1)
    if args.baseline in models:
        models = [args.baseline] + [m for m in models if m != args.baseline]

    records_by_model = {m: dedupe_runs(load_run_records(_run_path(m))) for m in models}
    judged_by_model = {m: load_run_records(_judged_path(args.judge_model, m)) for m in models}
    golden_entries = load_golden()["entries"]
    human_labels = load_human_labels()
    label_map = load_label_map()

    report = compute_comparison(
        models, args.baseline, args.judge_model, records_by_model, judged_by_model,
        golden_entries, human_labels, label_map, _code_version(),
    )

    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _print_comparison(report)
    print(f"\n상세는 {COMPARISON_PATH}에 저장했습니다.")


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="골든셋 이미지를 실제 VLM으로 추출해 run jsonl에 append(이어서 실행)")
    p_extract.add_argument("--model", required=True, choices=_CANDIDATES)
    p_extract.add_argument("--ids", nargs="+", default=None, help="지정한 id만 추출")
    p_extract.add_argument("--limit", type=int, default=None, help="앞에서 N개만 추출")
    p_extract.set_defaults(func=cmd_extract)

    p_judge = sub.add_parser("judge", help="figure 항목의 서술을 Judge로 채점해 캐시에 append(이어서 실행)")
    p_judge.add_argument("--judge-model", required=True, help="예: anthropic/claude-sonnet-5.5, openai/gpt-5.6-luna")
    p_judge.add_argument("--models", nargs="+", choices=_CANDIDATES, default=None)
    p_judge.add_argument("--repeat", type=int, default=1)
    p_judge.set_defaults(func=cmd_judge)

    p_export = sub.add_parser("export-sheet", help="그림 항목 전체를 후보에 블라인드 균등 배정해 라벨링 시트 생성")
    p_export.add_argument("--seed", type=int, default=0)
    p_export.set_defaults(func=cmd_export_sheet)

    p_import = sub.add_parser("import-sheet", help="라벨링 시트를 검증 후 사람 라벨 파일에 반영(부분 진행 허용)")
    p_import.add_argument("--sheet", default=None, help="시트 파일 경로(기본: data/golden/vlm_labeling_sheet.json)")
    p_import.set_defaults(func=cmd_import_sheet)

    p_rel = sub.add_parser("reliability", help="사람 라벨 vs Judge 후보들로 κ·MAE·편향 분석")
    p_rel.add_argument("--judges", nargs="+", required=True, help="비교할 Judge 후보 모델 id")
    p_rel.set_defaults(func=cmd_reliability)

    p_cmp = sub.add_parser("compare", help="VLM 후보들을 정확도·서술 품질·adversarial·지연으로 종합 비교")
    p_cmp.add_argument("--judge-model", required=True)
    p_cmp.add_argument("--baseline", default="gpt-4o-mini", choices=_CANDIDATES)
    p_cmp.add_argument("--models", nargs="+", choices=_CANDIDATES, default=None)
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
