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
  make-defects   사람 점수 5인 서술에 결함(핵심값 변경·소폭 변경·항목 삭제·단위 제거·수치
                 제거)을 주입해 중간 품질(2~3점) 구간 검증용 셋을 만들고 블라인드 라벨링
                 시트(vlm_defect_labeling_sheet.json)를 생성
  import-defect-sheet  결함 셋 라벨링 시트를 검증 후 data/golden/vlm_defect_human_labels.json에 반영
  judge-defects  결함 셋 서술을 기존 score_figure_entry로 채점해
                 data/golden/_vlm_judged_defects/<judge-slug>.jsonl에 append(이어서 실행)
  reliability-defects  결함 셋 단독/기존 90건과의 합산 κ·MAE, 사람 점수 구간별·결함 유형별
                 집계를 계산해 data/golden/_vlm_judge_selection_defects.json에 저장
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
import re
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
from evals.eval_vlm import (
    _call_with_retry,
    _FIGURE_BLOCK,
    _MD_TABLE_BLOCK,
    cer,
    score_adversarial,
    score_figure_entry,
    strip_figure_block,
    wer,
)
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

# 결함 주입 셋(make-defects 이하) — 사람 라벨 90건이 5점·1점·4점에만 몰려 있어 Judge가
# 중간 품질(2~3점) 서술을 낮게 채점하는지 검증할 수 없었던 문제를 메우는 보조 셋.
DEFECT_ITEMS_PATH = os.path.join(GOLDEN_DIR, "_vlm_defect_items.json")
DEFECT_SHEET_PATH = os.path.join(GOLDEN_DIR, "vlm_defect_labeling_sheet.json")
DEFECT_HUMAN_LABELS_PATH = os.path.join(GOLDEN_DIR, "vlm_defect_human_labels.json")
DEFECT_JUDGED_DIR = os.path.join(GOLDEN_DIR, "_vlm_judged_defects")
DEFECT_JUDGE_SELECTION_PATH = os.path.join(GOLDEN_DIR, "_vlm_judge_selection_defects.json")

# 후보별 env — app.common.llm import 전에 설정해야 한다(golden_gen/gen_item_quality_golden.py
# cmd_generate와 같은 관례). 키·env 값은 승인된 계획(encapsulated-herding-frost.md)의 매핑 그대로.
_CANDIDATE_ENV = {
    "gpt-4o-mini": {"VLM_BACKEND": "openai", "OPENAI_VLM_MODEL": "gpt-4o-mini"},
    "gpt-6-luna": {"VLM_BACKEND": "openrouter", "OPENROUTER_VLM_MODEL": "openai/gpt-6-luna"},
    "gemini-3.8-flash": {"VLM_BACKEND": "openrouter", "OPENROUTER_VLM_MODEL": "google/gemini-3.8-flash"},
    # 2026-10-05: 기본 태그 qwen3-vl:8b는 생각 과정 버전이라 출력 한도(2048)를 생각에 다 써서
    # 그림 문항 61장 중 25장이 빈 응답이었다(think=false도 무시됨). 추출에는 생각이 필요 없어
    # instruct 버전으로 바꿨다. 한도를 늘리면 이미지당 5분 이상이라 운영 후보가 될 수 없다.
    "qwen3-vl-8b": {"VLM_BACKEND": "local", "OLLAMA_VLM_MODEL": "qwen3-vl:8b-instruct"},
    "gemma3-12b": {"VLM_BACKEND": "local", "OLLAMA_VLM_MODEL": "gemma3:12b"},
    # 2026-10-06: 운영 전환 검증용 — 선정된 gpt-6-luna를 운영과 같은 OpenAI 직접 경로로 재측정한다
    # (OpenRouter 경로는 max_tokens·temperature 차이를 흡수해 직접 경로 문제를 가렸다).
    "gpt-6-luna-openai": {"VLM_BACKEND": "openai", "OPENAI_VLM_MODEL": "gpt-6-luna"},
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


# ── make-defects: 중간 품질(2~3점) 검증용 결함 주입 ────────────────────────
#
# 사람 라벨 90건이 5점 76·4점 6·1점 8로 양 끝에만 몰려, Judge가 미묘한 결함(수치 하나
# 오기, 항목 누락 등)을 낮게 채점하는지 검증되지 않았다. 사람이 5점을 준 서술에 아래
# 5종 결함을 하나씩 코드로 주입해 2~3점 구간을 채운다. 전부 LLM 미사용·결정론적 문자열
# 처리(시드는 어느 소스를 쓸지 고르는 순서에만 쓰고, 변환 자체는 무작위가 아니다).

_DEFECT_TYPES = (
    "key_value_changed", "minor_value_changed", "item_dropped", "unit_removed", "numbers_removed",
)

# "NN단위로 가장 많/적/높/낮/크/작~" 패턴으로 핵심사실(figure_summary)의 최댓값·최솟값
# 항목을 찾는다. 숫자와 "(으)로" 사이에 단위·공백이 섞여도(예: "410만 원으로") 잡도록
# 폭을 넉넉히 둔다.
_SUMMARY_EXTREME = re.compile(
    r"(\d+(?:\.\d+)?)[^\d]{0,12}?(?:으로|로)\s*(?:압도적으로\s*)?가장\s*(많|적|높|낮|크|작)"
)
_MAX_WORDS = ("많", "높", "크")
_MIN_WORDS = ("적", "낮", "작")

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_YEAR_PREFIX = re.compile(r"(?:19|20)\d{2}")

_SCALE_PAREN = re.compile(r"\(단위\s*[:：]\s*[^)]*\)\s*")
# 긴 표현을 먼저 둔다 — 짧은 "원"이 "만 원"보다 앞서면 "만 원"의 "원"만 지워지고 "만"이
# 남는 식으로 어긋난다(find -regex 알테네이션과 같은 이유). 숫자 바로 뒤(공백 하나까지는
# 허용)에 붙은 경우만 단위로 보고 지운다 — 뒤에 digit lookbehind가 없으면 "불명"의
# "명"처럼 일반 낱말 끝 글자가 단위로 오인되어 지워진다.
_UNIT_WORDS = (
    "억\\s?달러", "만\\s?명", "만\\s?원", "억\\s?원", "만\\s?톤", "만점",
    "kg", "㎢", "%", "점", "원", "톤", "건", "석", "대", "명",
)
_UNIT_PATTERN = re.compile(r"(?<=\d)\s?(?:" + "|".join(_UNIT_WORDS) + ")")


def find_extreme_value(figure_summary: str) -> tuple[str, str] | None:
    """figure_summary에서 최댓값·최솟값으로 언급된 항목의 수치를 찾는다. 두 속성을 모두
    언급하는 표(예: "~은 19석으로 가장 적지만 ~은 42%로 가장 높고")처럼 여러 개 있으면
    먼저 나오는 것만 쓴다. 반환: (수치 문자열, "max"|"min") 또는 패턴이 없으면 None."""
    m = _SUMMARY_EXTREME.search(figure_summary)
    if not m:
        return None
    value_str, direction = m.group(1), m.group(2)
    return value_str, ("max" if direction in _MAX_WORDS else "min")


def _format_like(old_str: str, new_value: float) -> str:
    """new_value를 old_str과 같은 소수 자릿수로 표기한다(정수면 정수로)."""
    if "." in old_str:
        decimals = len(old_str.split(".")[1])
        return f"{new_value:.{decimals}f}"
    return str(int(round(new_value)))


def _find_value_span(text: str, value_str: str) -> tuple[int, int] | None:
    """text에서 value_str과 정확히 일치하는 숫자 구간을 찾는다(앞뒤로 숫자나 소수점
    이어짐이 없는 위치만 — "14"가 "14.8"의 일부로 잘못 매칭되지 않도록)."""
    pattern = re.compile(r"(?<!\d)" + re.escape(value_str) + r"(?!\d)(?!\.\d)")
    m = pattern.search(text)
    return (m.start(), m.end()) if m else None


def apply_key_value_changed(fig_text: str, figure_summary: str) -> str | None:
    """핵심사실에서 최댓값/최솟값으로 언급된 항목의 수치를 바꿔 "어느 항목이 가장
    큰가/작은가"라는 정답 판단 자체가 바뀌게 한다. 최댓값은 0.6배로 줄이고 최솟값은
    1.6배로 늘린다(같은 방향으로 더 극단으로 보내면 순위가 그대로라 정답이 안 바뀐다).
    핵심사실에 극값 표현이 없거나, 그 수치를 서술에서 찾지 못하거나, 변환 전후가
    같으면(반올림으로 값이 안 바뀌는 경우) None — 호출 쪽에서 다른 서술로 대체한다."""
    found = find_extreme_value(figure_summary)
    if found is None:
        return None
    value_str, kind = found
    span = _find_value_span(fig_text, value_str)
    if span is None:
        return None
    new_value = float(value_str) * (0.6 if kind == "max" else 1.6)
    new_str = _format_like(value_str, new_value)
    if new_str == value_str:
        return None
    start, end = span
    return fig_text[:start] + new_str + fig_text[end:]


def apply_minor_value_changed(fig_text: str, figure_summary: str) -> str | None:
    """최댓값·최솟값이 아닌 수치 하나를 15% 늘려 소폭만 바꾼다(순위를 바꾸는 게 목적이
    아니므로 key_value_changed가 쓰는 극값 수치는 건드리지 않는다). "YYYY년"의 연도
    숫자는 항목 값이 아니라 레이블이라 후보에서 제외한다. 바꿀 수치가 없거나 변환
    전후가 같으면 None."""
    found = find_extreme_value(figure_summary)
    extreme_value_str = found[0] if found else None
    for m in _NUMBER.finditer(fig_text):
        candidate = m.group(0)
        is_year = _YEAR_PREFIX.fullmatch(candidate) and fig_text[m.end():m.end() + 1] == "년"
        if candidate == extreme_value_str or is_year:
            continue
        new_str = _format_like(candidate, float(candidate) * 1.15)
        if new_str == candidate:
            continue
        return fig_text[:m.start()] + new_str + fig_text[m.end():]
    return None


def apply_item_dropped(fig_text: str, _figure_summary: str) -> str | None:
    """항목(라벨+수치 한 쌍) 하나를 통째로 삭제한다. 표 형태(줄마다 항목 하나, 숫자가
    있는 줄이 2개 이상)면 마지막 데이터 줄을, 한 줄에 쉼표로 나열된 형태면 마지막으로
    숫자가 있는 쉼표 구간을 지운다 — 앞쪽이 아니라 뒤쪽을 지우는 이유는 첫 항목 앞에
    자료 설명 서두("[자료: (단위: ...) ~그래프.")가 같이 붙어 있는 경우가 많아서다.
    "]"로 끝나면 떼어두고 처리한 뒤 다시 붙인다. 항목이 하나뿐이면(지우면 자료가 통째로
    사라짐) None."""
    body, suffix = fig_text, ""
    stripped = fig_text.rstrip()
    if stripped.endswith("]"):
        suffix = fig_text[len(stripped) - 1:]
        body = stripped[:-1]

    lines = body.split("\n")
    digit_line_idxs = [i for i, line in enumerate(lines) if re.search(r"\d", line)]
    if len(lines) > 1 and len(digit_line_idxs) >= 2:
        drop_idx = digit_line_idxs[-1]
        new_body = "\n".join(lines[:drop_idx] + lines[drop_idx + 1:])
        result = new_body + suffix
        return result if result != fig_text else None

    segments = body.split(",")
    digit_segment_idxs = [i for i, seg in enumerate(segments) if re.search(r"\d", seg)]
    if len(digit_segment_idxs) < 2:
        return None
    drop_idx = digit_segment_idxs[-1]
    new_body = ",".join(segments[:drop_idx] + segments[drop_idx + 1:])
    result = new_body + suffix
    return result if result != fig_text else None


def _strip_units(text: str) -> str:
    """단위·척도 표기를 지운다 — "(단위: 만 원)" 같은 괄호 표기와, 숫자 뒤에 바로 붙는
    %·원·톤·건·석·대·명 등의 단위 글자."""
    text = _SCALE_PAREN.sub("", text)
    text = _UNIT_PATTERN.sub("", text)
    return text


def apply_unit_removed(fig_text: str, _figure_summary: str) -> str | None:
    """단위·척도 표기를 모두 제거한다(수치는 남긴다). 제거할 게 없으면(원래 단위
    표기가 없던 서술) None — 단위를 하나도 못 지웠는데 공백 정리만으로 원문과 달라지는
    것(예: 표의 정렬용 여분 공백)을 결함으로 치지 않기 위해, 단위 제거 여부를 공백
    정리 전에 먼저 판단한다."""
    stripped = _strip_units(fig_text)
    if stripped == fig_text:
        return None
    return re.sub(r"[ \t]{2,}", " ", stripped).strip()


# ── numbers_removed: 숫자 제거가 아니라 경향 문장 생성 ─────────────────────
#
# 처음에는 숫자만 문자 그대로 지우는 방식이었으나, "년대"→"년대"(연도 라벨까지
# 훼손), 표 칸이 통째로 빈 채 남는 등 사람이 보기에도 결함 티가 나는 문장이 나와
# (2026-10 메인 세션 전수 검토) "경향만 말하는 서술"로 바꿨다. (라벨, 값) 쌍을 파싱해
# 라벨(연도·연대·월·범주명)은 그대로 두고 값만 지운 뒤, 시간축이면 증가/감소/정점
# 추이를, 범주형이면 최댓값/최솟값만 말하는 문장을 만든다. 전부 LLM 미사용·결정론적
# 문자열/정규식 처리다.

_SIGNED_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")  # 음수(기온 등) 값도 있어 부호 포함
# 라벨 자체가 숫자를 포함하는 경우(연도 "2018년", 연대 "1990년대", 월 "1월", 연령대
# "10대") — 값 파싱 시 이 패턴이 먼저 발견되면 그 뒤를 값으로 본다.
_LABEL_NUMBER_SUFFIX = re.compile(r"\d+(?:년대|년|월|대)")
# 모든 라벨이 이 패턴(연도/연대/월)으로 끝나야 "시간축"으로 본다 — "10대"의 "대"는
# 연령대라 시간축 판별에는 포함하지 않는다(파싱 시 라벨 보존 목적으로만 쓴다).
_TIME_AXIS_LABEL = re.compile(r"(?:년대|년|월)$")
_PARTICLE_SPLIT = re.compile(r"(?:은|는|이|가|의)\s+")
_TRAILING_QUOTE_PARTICLE = re.compile(r"^(.*['\"])(?:은|는|이|가)$")
_TABLE_ROW_SEP = re.compile(r"^[\s|:\-]+$")  # "|---|---|" 같은 표 구분선
_SENTENCE_DOT = re.compile(r"(?<!\d)\.(?!\d)")  # 소수점(3.8 등)은 문장 경계가 아니다
# "~별"(대륙별·분야별 등)은 축 전체를 가리키는 말이라 개별 항목 라벨로 쓰면 안 된다.
_BAD_LABEL_SUFFIXES = ("별",)
_BAD_LABEL_SUBSTRINGS = ("그래프", "표", "자료", "단위", "항목", "수치", "비율", "추이", "차트")


def _clean_label(label: str) -> str:
    """라벨 후보 문자열을 다듬는다 — "원그래프—운동"처럼 대시로 차트 유형이 붙은
    경우 대시 뒤만 남기고, "'법률안'이"처럼 인용부호 뒤에 조사가 바로 붙은 경우 그
    조사를 뗀다. 둘 다 아니면 마지막 조사(은/는/이/가/의) 뒤를 찾는다 — "갑국의 ...
    비중은 복지"처럼 조사 뒤에 진짜 라벨이 더 있으면 그 뒤만 쓰고, "갑 지역은"처럼
    조사가 (원래 끝 공백 기준으로) 구절의 맨 끝이면 조사 앞 구절의 첫 단어("갑")를
    라벨로 쓴다(조사와 그 사이에 낀 설명어 "지역은"은 버린다). 조사 뒤가 끝인지
    판단하려면 끝 공백이 신호이므로, 끝 공백을 미리 지우지 않고 맨 마지막에만
    정리한다 — 안 그러면 "국가"처럼 조사처럼 보이는 글자가 원래 단어의 일부인
    경우까지 "국"으로 잘라내는 오류가 생긴다(끝에 공백이 전혀 없던 입력이라
    조사+공백 패턴 자체가 매칭되지 않아야 비로소 안전해진다)."""
    label = label.lstrip(" :：·-")
    if not label.strip():
        return ""
    if "—" in label:
        label = label.rsplit("—", 1)[-1]

    quote_check = label.strip()
    m = _TRAILING_QUOTE_PARTICLE.match(quote_check)
    if m:
        label = m.group(1)
    else:
        matches = list(_PARTICLE_SPLIT.finditer(label))
        if matches:
            last = matches[-1]
            tail = label[last.end():].strip()
            if tail:
                label = tail
            else:
                head_words = label[:last.start()].strip().split()
                if head_words:
                    label = head_words[0]
    return label.strip("'\" :：·-")


def _labels_look_clean(labels: list[str]) -> bool:
    """라벨들이 실제 항목명처럼 보이는지 확인한다 — 파싱이 문장 중간을 잘못 잘라
    "구성 항목 및 비율은 개신교"처럼 서두 문구가 라벨에 섞여 들어간 경우를 거른다."""
    for label in labels:
        if not label or len(label) > 8:
            return False
        if label.endswith(_BAD_LABEL_SUFFIXES):
            return False
        if any(s in label for s in _BAD_LABEL_SUBSTRINGS):
            return False
    return True


def _label_value_from_segment(segment: str) -> tuple[str, str] | None:
    """쉼표로 나뉜 세그먼트 하나에서 (라벨, 값)을 뽑는다. 세그먼트 안에 "연도/연대/월/
    연령대" 패턴이 있으면(위치 무관 — "갑국의 실업률은 2018년 3.8%"처럼 앞에 다른
    말이 붙어도) 그걸 라벨로 쓰고 그 뒤에서 값을 찾는다. 없으면 첫 숫자 앞까지를
    라벨로 본다."""
    segment = segment.strip()
    if not segment:
        return None
    m = _LABEL_NUMBER_SUFFIX.search(segment)
    if m:
        label, rest = m.group(0), segment[m.end():]
    else:
        m2 = re.match(r"[^\d]+", segment)
        if not m2:
            return None
        label, rest = m2.group(0), segment[m2.end():]
    label = _clean_label(label)
    if not label:
        return None
    value_m = _SIGNED_NUMBER.search(rest)
    if not value_m:
        return None
    return label, value_m.group(0)


def _label_value_from_row(cells: list[str]) -> tuple[str, str] | None:
    """표 한 행의 셀 목록에서 (라벨, 첫 번째 수치 열의 값)을 뽑는다. 첫 셀이 라벨,
    그 뒤 셀 중 숫자가 있는 첫 번째 셀의 값을 쓰고 나머지 수치 열은 무시한다."""
    if not cells:
        return None
    label = _clean_label(cells[0])
    if not label:
        return None
    for c in cells[1:]:
        if re.search(r"\d", c):
            m = _SIGNED_NUMBER.search(c)
            if m:
                return label, m.group(0)
    return None


def _split_preamble(text: str) -> tuple[str, str]:
    """첫 문장이 숫자를 포함하지 않는 설명문이면 그 문장을 서두로 떼어내고, 아니면
    서두 없이 전체를 그대로 돌려준다."""
    dot = text.find(".")
    if dot != -1:
        first_sentence = text[:dot + 1]
        if not re.search(r"\d", first_sentence):
            return first_sentence.strip(), text[dot + 1:].strip()
    return "", text.strip()


def parse_label_value_pairs(fig_text: str) -> tuple[str, list[tuple[str, str]]] | None:
    """[자료: ...] 서술(또는 마크다운/슬래시 표)에서 (서두, [(라벨, 값), ...])을 뽑는다.
    "|"나 "/"가 있는 줄이 3개 이상(머리글+데이터 2개 이상)이면 표로 보고, 첫 delimiter
    줄을 머리글로 제외한 뒤 각 데이터 줄에서 라벨+첫 번째 수치 열을 뽑는다(서두 없음).
    아니면 첫 문장을 서두로 떼어내고 나머지를 쉼표로 나눠 각 조각에서 (라벨, 값)을
    뽑는다 — 다만 나머지에 문장 마침표(소수점 제외)가 2개를 넘으면 문장이 여러 개로
    쪼개진 구조(표 아닌 서술문 여러 개)로 보고 안전하게 포기한다(None). (라벨, 값)이
    2개 미만이면 None — 호출 쪽에서 다른 서술로 대체한다."""
    body = fig_text
    if body.rstrip().endswith("]"):
        body = body.rstrip()[:-1]
    if body.startswith("[자료"):
        colon = body.find(":")
        if colon == -1:
            colon = body.find("：")
        if colon != -1:
            body = body[colon + 1:]
    body = body.strip()

    lines = body.split("\n")
    delim_lines = [line for line in lines if "|" in line or "/" in line]
    if len(delim_lines) >= 3:
        data_lines = [
            line for line in delim_lines[1:] if not _TABLE_ROW_SEP.fullmatch(line.strip())
        ]
        pairs = []
        for line in data_lines:
            line = line.strip()
            if not re.search(r"\d", line):
                continue
            delim = "|" if "|" in line else "/"
            cells = [c.strip() for c in line.strip("|").split(delim) if c.strip()]
            pair = _label_value_from_row(cells)
            if pair:
                pairs.append(pair)
        return ("", pairs) if len(pairs) >= 2 else None

    preamble, rest = _split_preamble(body)
    if len(_SENTENCE_DOT.findall(rest)) > 1:
        return None
    pairs = []
    for segment in rest.split(","):
        pair = _label_value_from_segment(segment)
        if pair:
            pairs.append(pair)
    return (preamble, pairs) if len(pairs) >= 2 else None


def _has_final_consonant(syllable: str) -> bool:
    """한글 음절에 받침(종성)이 있는지 — 한글이 아니면(영문 등) 받침이 없다고 본다."""
    code = ord(syllable)
    if not (0xAC00 <= code <= 0xD7A3):
        return False
    return (code - 0xAC00) % 28 != 0


def _particle_i_ga(word: str) -> str:
    """단어의 마지막 글자 받침 유무로 "이/가" 조사를 고른다(받침 있으면 "이")."""
    if not word:
        return "가"
    return "이" if _has_final_consonant(word[-1]) else "가"


def _classify_trend(values: list[float]) -> str:
    """시간축 값들의 모양을 분류한다: 단조 증가("up")/감소("down"), 정점 후 감소
    ("peak")/저점 후 증가("trough"), 그 외 증감 반복("mixed")."""
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    if all(d >= 0 for d in diffs) and any(d > 0 for d in diffs):
        return "up"
    if all(d <= 0 for d in diffs) and any(d < 0 for d in diffs):
        return "down"
    peak_idx = max(range(len(values)), key=lambda i: values[i])
    if 0 < peak_idx < len(values) - 1:
        rising = all(values[i] <= values[i + 1] for i in range(peak_idx))
        falling = all(values[i] >= values[i + 1] for i in range(peak_idx, len(values) - 1))
        if rising and falling:
            return "peak"
    trough_idx = min(range(len(values)), key=lambda i: values[i])
    if 0 < trough_idx < len(values) - 1:
        falling = all(values[i] >= values[i + 1] for i in range(trough_idx))
        rising = all(values[i] <= values[i + 1] for i in range(trough_idx, len(values) - 1))
        if falling and rising:
            return "trough"
    return "mixed"


def _time_trend_sentence(labels: list[str], values: list[float]) -> str:
    trend = _classify_trend(values)
    first, last = labels[0], labels[-1]
    if trend == "up":
        return f"{first}부터 {last}까지 꾸준히 증가했다."
    if trend == "down":
        return f"{first}부터 {last}까지 꾸준히 감소했다."
    if trend == "peak":
        peak_label = labels[max(range(len(values)), key=lambda i: values[i])]
        return f"{first}부터 {last}까지 {peak_label}에 가장 높았다가 이후 감소했다."
    if trend == "trough":
        trough_label = labels[min(range(len(values)), key=lambda i: values[i])]
        return f"{first}부터 {last}까지 {trough_label}에 가장 낮았다가 이후 증가했다."
    return f"{first}부터 {last}까지 증감을 반복했다."


def _categorical_extreme_sentence(labels: list[str], values: list[float]) -> str:
    max_idx = max(range(len(values)), key=lambda i: values[i])
    min_idx = min(range(len(values)), key=lambda i: values[i])
    items = ", ".join(labels)
    max_label, min_label = labels[max_idx], labels[min_idx]
    return (
        f"{items} 중 {max_label}{_particle_i_ga(max_label)} 가장 크고 "
        f"{min_label}{_particle_i_ga(min_label)} 가장 작다."
    )


def apply_numbers_removed(fig_text: str, _figure_summary: str) -> str | None:
    """값을 모두 지우고 경향 문장으로 바꾼다 — 라벨(연도·연대·월·범주명)의 숫자는
    보존한다. 라벨이 전부 연도/연대/월이면 "~부터 ~까지 증가/감소/정점/증감 반복"
    문장을, 그 외(범주형)면 "~ 중 ~이 가장 크고 ~이 가장 작다" 문장을 만든다.
    (라벨, 값) 쌍을 2개 이상 못 찾거나 라벨이 지저분해 보이면(_labels_look_clean)
    None — 호출 쪽에서 다른 서술로 대체한다."""
    parsed = parse_label_value_pairs(fig_text)
    if parsed is None:
        return None
    preamble, pairs = parsed
    if len(pairs) < 2:
        return None
    labels = [label for label, _ in pairs]
    if not _labels_look_clean(labels):
        return None
    try:
        values = [float(v) for _, v in pairs]
    except ValueError:
        return None

    if all(_TIME_AXIS_LABEL.search(label) for label in labels):
        sentence = _time_trend_sentence(labels, values)
    else:
        sentence = _categorical_extreme_sentence(labels, values)

    head = f"{preamble} " if preamble else ""
    result = f"[자료: {head}{sentence}]"
    return result if result != fig_text else None


_DEFECT_TRANSFORMS = {
    "key_value_changed": apply_key_value_changed,
    "minor_value_changed": apply_minor_value_changed,
    "item_dropped": apply_item_dropped,
    "unit_removed": apply_unit_removed,
    "numbers_removed": apply_numbers_removed,
}


def replace_figure_block(raw_output: str, new_block: str) -> str:
    """raw_output 안의 [자료: ...] 블록(또는 마크다운 표 블록 — strip_figure_block과 같은
    두 패턴)을 new_block으로 통째로 교체한다. 블록 바깥 텍스트(발문·선택지 등)는 손대지
    않는다. 블록을 못 찾으면(호출 전에 figure_description이 있는 항목만 쓰므로 정상
    경로에서는 일어나지 않음) raw_output을 그대로 반환한다."""
    m = _FIGURE_BLOCK.search(raw_output)
    if m is None:
        m = _MD_TABLE_BLOCK.search(raw_output)
    if m is None:
        return raw_output
    return raw_output[:m.start()] + new_block + raw_output[m.end():]


def load_defect_sources() -> list[dict]:
    """사람 점수 5인 (그림, 모델) 서술만 결함 주입 원천으로 모은다 — 사람이 이미 "핵심
    수치와 경향을 거의 다 담았다"고 판단한 서술이어야 결함을 하나 주입했을 때 그 결함
    만의 효과를 볼 수 있다. id 순으로 반환해 셔플 전 순서를 결정론적으로 만든다."""
    human_labels = load_human_labels()
    label_map = load_label_map()
    golden_by_id = {e["id"]: e for e in load_golden()["entries"]}
    sources = []
    for fid in sorted(human_labels):
        hl = human_labels[fid]
        if hl.get(_SCORE_KEY) != 5:
            continue
        mapping = label_map.get(fid)
        if mapping is None:
            continue
        model = mapping["model"]
        rec = dedupe_runs(load_run_records(_run_path(model))).get(fid)
        if rec is None or rec.get("error") or rec.get("figure_description") is None:
            continue
        golden_entry = golden_by_id.get(fid)
        if golden_entry is None:
            continue
        sources.append({
            "figure_id": fid,
            "model": model,
            "figure_summary": golden_entry["figure_summary"],
            "raw_output": rec["raw_output"],
            "figure_description": rec["figure_description"],
        })
    return sources


def make_defect_items(
    sources: list[dict], seed: int = 0, per_type: int = 6, n_originals: int = 10,
) -> list[dict]:
    """결함 유형 5종 × per_type건 + 원본(결함 없음) n_originals건을 결정론적으로 뽑는다.
    서로 다른 그림을 우선 쓴다(원본도 결함용과 겹치지 않는 그림을 우선) — 소스가 부족할
    때만 그림이 겹친다. 변환이 불가능하거나(해당 유형을 적용할 수치가 없음 등) 변환
    전후가 같은 서술은 버리고 다음 소스로 넘어간다. 반환 항목에는 아직 defect_id가 없고
    (섞은 뒤에 매긴다), source_label_id·figure_id·model·defect_type·raw_output·
    figure_description·figure_summary가 있다."""
    shuffled = sorted(sources, key=lambda s: s["figure_id"])
    random.Random(seed).shuffle(shuffled)

    used_figure_ids: set[str] = set()

    def _pick(predicate, limit: int) -> list[tuple[dict, str]]:
        picked: list[tuple[dict, str]] = []
        for s in shuffled:  # 1차: 아직 쓰지 않은 그림만
            if len(picked) >= limit:
                break
            if s["figure_id"] in used_figure_ids:
                continue
            result = predicate(s)
            if result is None:
                continue
            picked.append((s, result))
            used_figure_ids.add(s["figure_id"])
        if len(picked) < limit:  # 2차: 부족하면 그림 중복을 허용
            picked_figure_ids = {s["figure_id"] for s, _ in picked}
            for s in shuffled:
                if len(picked) >= limit:
                    break
                if s["figure_id"] in picked_figure_ids:
                    continue
                result = predicate(s)
                if result is None:
                    continue
                picked.append((s, result))
        return picked

    items: list[dict] = []
    for defect_type in _DEFECT_TYPES:
        fn = _DEFECT_TRANSFORMS[defect_type]

        def _predicate(s: dict, fn=fn) -> str | None:
            new_desc = fn(s["figure_description"], s["figure_summary"])
            return None if new_desc is None or new_desc == s["figure_description"] else new_desc

        for s, new_desc in _pick(_predicate, per_type):
            items.append({
                "source_label_id": s["figure_id"],
                "figure_id": s["figure_id"],
                "model": s["model"],
                "defect_type": defect_type,
                "raw_output": replace_figure_block(s["raw_output"], new_desc),
                "figure_description": new_desc,
                "figure_summary": s["figure_summary"],
            })

    for s, desc in _pick(lambda s: s["figure_description"], n_originals):
        items.append({
            "source_label_id": s["figure_id"],
            "figure_id": s["figure_id"],
            "model": s["model"],
            "defect_type": "original",
            "raw_output": s["raw_output"],
            "figure_description": desc,
            "figure_summary": s["figure_summary"],
        })

    random.Random(seed + 1).shuffle(items)
    for i, item in enumerate(items, 1):
        item["defect_id"] = f"d{i:02d}"
    return items


def cmd_make_defects(args: argparse.Namespace) -> None:
    if has_existing_sheet_labels(DEFECT_SHEET_PATH):
        print(f"중단 — {DEFECT_SHEET_PATH}에 이미 채워진 라벨이 있습니다. 라벨 유실 방지를 위해 덮어쓰지 않습니다.")
        sys.exit(1)

    sources = load_defect_sources()
    items = make_defect_items(sources, seed=args.seed, per_type=args.per_type, n_originals=args.originals)

    os.makedirs(GOLDEN_DIR, exist_ok=True)
    with open(DEFECT_ITEMS_PATH, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=2)

    # 라벨링 시트 — export-sheet의 build_label_sheet_entry를 그대로 재사용한다(블라인드로
    # defect_type·model을 안 보여주는 것은 그 함수가 애초에 id·핵심사실·서술만 쓰는
    # 덕분에 자동으로 달성된다).
    sheet_items = [
        build_label_sheet_entry({"id": it["defect_id"], "figure_summary": it["figure_summary"]}, it["figure_description"])
        for it in items
    ]
    sheet = {"_안내": _SHEET_GUIDE, "문항": sheet_items}
    with open(DEFECT_SHEET_PATH, "w", encoding="utf-8") as f:
        json.dump(sheet, f, ensure_ascii=False, indent=2)

    counts = dict(Counter(it["defect_type"] for it in items))
    print(f"완료 — {len(items)}건 (유형별: {counts}) → {DEFECT_ITEMS_PATH}, {DEFECT_SHEET_PATH}")


# ── import-defect-sheet ──────────────────────────────────────────────────

def load_defect_items() -> list[dict]:
    if not os.path.exists(DEFECT_ITEMS_PATH):
        return []
    with open(DEFECT_ITEMS_PATH, encoding="utf-8") as f:
        return json.load(f)["items"]


def load_defect_human_labels() -> dict:
    if not os.path.exists(DEFECT_HUMAN_LABELS_PATH):
        return {}
    with open(DEFECT_HUMAN_LABELS_PATH, encoding="utf-8") as f:
        return json.load(f).get("labels", {})


def cmd_import_defect_sheet(args: argparse.Namespace) -> None:
    sheet_path = args.sheet or DEFECT_SHEET_PATH
    with open(sheet_path, encoding="utf-8") as f:
        sheet = json.load(f)

    defect_items = load_defect_items()
    if not defect_items:
        print(f"중단 — {DEFECT_ITEMS_PATH}가 없습니다. make-defects를 먼저 실행하세요.")
        sys.exit(1)
    valid_ids = {it["defect_id"] for it in defect_items}
    existing = load_defect_human_labels()
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
                "결함 주입 서술 블라인드 사람 라벨(점수 1~5, 근거) — id는 "
                "vlm_defect_labeling_sheet.json/_vlm_defect_items.json과 공유."
            ),
        },
        "labels": updated,
    }
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    with open(DEFECT_HUMAN_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(f"완료 — 이번 {n_saved}건 반영, 누적 {len(updated)}건 → {DEFECT_HUMAN_LABELS_PATH}")


# ── judge-defects ─────────────────────────────────────────────────────────

def _defect_judged_path(judge_model: str) -> str:
    return os.path.join(DEFECT_JUDGED_DIR, f"{_judge_slug(judge_model)}.jsonl")


async def _run_judge_defects(args: argparse.Namespace) -> None:
    judge = OpenRouterBackend(model=args.judge_model)
    items = load_defect_items()
    if not items:
        print(f"중단 — {DEFECT_ITEMS_PATH}가 없습니다. make-defects를 먼저 실행하세요.")
        sys.exit(1)

    out_path = _defect_judged_path(args.judge_model)
    os.makedirs(DEFECT_JUDGED_DIR, exist_ok=True)
    done = {(r["id"], r["repeat"]) for r in load_run_records(out_path)}

    n_judged = n_skipped = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for it in items:
            for rep in range(args.repeat):
                key = (it["defect_id"], rep)
                if key in done:
                    n_skipped += 1
                    continue
                result = await score_figure_entry(
                    it["figure_summary"], it["raw_output"], judge, call_with_retry=_call_with_retry,
                )
                out_record = {
                    "id": it["defect_id"],
                    "repeat": rep,
                    "judge_model": args.judge_model,
                    "judge_score": result["judge_score"],
                    "had_description": result["fig_text"] is not None,
                }
                f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f.flush()
                n_judged += 1
    print(f"신규 채점 {n_judged}건, 캐시 스킵 {n_skipped}건 → {out_path}")


def cmd_judge_defects(args: argparse.Namespace) -> None:
    asyncio.run(_run_judge_defects(args))


# ── reliability-defects ──────────────────────────────────────────────────

def build_defect_reliability_rows(
    human_labels: dict, defect_items: list[dict], judged_by_judge: dict[str, list[dict]],
) -> list[dict]:
    """결함 주입 사람 라벨 + judge-defects 캐시를 조인한다. judged_by_judge는
    {judge_model: judge-defects 캐시 레코드 리스트}(reliability의 judged_by_judge와
    달리 VLM 모델별로 나뉘지 않는다 — 결함 항목마다 소스가 하나뿐이라서다). 반환:
    [{"id": defect_id, "figure_id", "defect_type", "human": int, "judges": {judge_model: [score, ...]}}]."""
    items_by_id = {it["defect_id"]: it for it in defect_items}
    rows = []
    for did, hl in human_labels.items():
        score = hl.get(_SCORE_KEY)
        if score is None:
            continue
        item = items_by_id.get(did)
        if item is None:
            continue
        judges: dict[str, list[int]] = {}
        for jm, records in judged_by_judge.items():
            scores = [r["judge_score"] for r in records if r["id"] == did and r["judge_score"] is not None]
            if scores:
                judges[jm] = scores
        rows.append({
            "id": did, "figure_id": item["figure_id"], "defect_type": item["defect_type"],
            "human": score, "judges": judges,
        })
    return rows


def _weighted_kappa_mae_bias(rows: list[dict], judge_model: str, cluster_key: str) -> dict:
    """rows(각 {cluster_key, "human", "judges": {judge_model: [...]}})에서 judge_model의
    가중 κ·MAE·편향과 클러스터 단위 부트스트랩 CI를 계산한다. reliability의
    compute_judge_reliability와 같은 계산이지만, 클러스터 키를 바꿀 수 있어야 해서
    (결함+기존 합산은 "figure_id" 단위 클러스터가 필요) 별도 함수로 둔다."""
    rows_with_judge = [r for r in rows if judge_model in r["judges"]]
    n = len(rows_with_judge)
    if n < 2:
        return {"n": n, "note": "표본 부족(2건 미만)"}

    human_vals = [r["human"] for r in rows_with_judge]
    judge_vals_rounded = [judge_avg_rounded(r["judges"][judge_model]) for r in rows_with_judge]
    clusters = [r[cluster_key] for r in rows_with_judge]

    kappa_point, kappa_lo, kappa_hi = cluster_bootstrap_ci(
        list(zip(human_vals, judge_vals_rounded)), clusters,
        lambda sample: weighted_kappa([a for a, b in sample], [b for a, b in sample], labels=_JUDGE_SCORE_LABELS),
    )
    mae_values = [abs(judge_avg(r["judges"][judge_model]) - r["human"]) for r in rows_with_judge]
    mae_point, mae_lo, mae_hi = mean_ci(mae_values, clusters)
    bias_values = [judge_avg(r["judges"][judge_model]) - r["human"] for r in rows_with_judge]
    bias_point, bias_lo, bias_hi = mean_ci(bias_values, clusters)

    return {
        "n": n,
        "weighted_kappa": _round3(kappa_point), "kappa_ci": [_round3(kappa_lo), _round3(kappa_hi)],
        "mae": _round3(mae_point), "mae_ci": [_round3(mae_lo), _round3(mae_hi)],
        "bias": _round3(bias_point), "bias_ci": [_round3(bias_lo), _round3(bias_hi)],
    }


def judge_by_human_score_bucket(rows: list[dict], judge_model: str) -> dict[int, dict]:
    """사람 점수(1~5) 구간별로 그 judge의 평균 점수·MAE·건수를 계산한다 — 결함 주입으로
    2~3점 구간을 채운 효과(그 구간에서도 Judge가 사람과 맞게 낮춰 주는지)를 보는 용도."""
    result = {}
    for score in _JUDGE_SCORE_LABELS:
        subset = [r for r in rows if r["human"] == score and judge_model in r["judges"]]
        if not subset:
            result[score] = {"n": 0, "judge_mean": None, "mae": None}
            continue
        judge_means = [judge_avg(r["judges"][judge_model]) for r in subset]
        mae_values = [abs(jm - score) for jm in judge_means]
        result[score] = {
            "n": len(subset),
            "judge_mean": _round3(sum(judge_means) / len(judge_means)),
            "mae": _round3(sum(mae_values) / len(mae_values)),
        }
    return result


def defect_type_breakdown(rows: list[dict], judge_model: str) -> dict:
    """결함 유형별 사람 평균·judge 평균·검출률을 계산한다. 결함과 원본은 그림이 서로
    달라 1:1로 짝지을 원본이 없으므로, 그 judge가 "original" 행들에 준 평균 점수를
    기준선으로 두고 각 결함 유형 행의 judge 점수가 그보다 낮은 비율을 검출률로 쓴다."""
    original_scores = [
        judge_avg(r["judges"][judge_model]) for r in rows
        if r["defect_type"] == "original" and judge_model in r["judges"]
    ]
    baseline = sum(original_scores) / len(original_scores) if original_scores else None

    result = {}
    for dtype in ("original",) + _DEFECT_TYPES:
        subset = [r for r in rows if r["defect_type"] == dtype and judge_model in r["judges"]]
        if not subset:
            result[dtype] = {"n": 0, "human_mean": None, "judge_mean": None, "detect_rate": None}
            continue
        human_mean = sum(r["human"] for r in subset) / len(subset)
        judge_scores = [judge_avg(r["judges"][judge_model]) for r in subset]
        judge_mean = sum(judge_scores) / len(judge_scores)
        detect_rate = (
            None if dtype == "original" or baseline is None
            else sum(1 for js in judge_scores if js < baseline) / len(judge_scores)
        )
        result[dtype] = {
            "n": len(subset), "human_mean": _round3(human_mean), "judge_mean": _round3(judge_mean),
            "detect_rate": _round3(detect_rate),
        }
    return result


def compute_defect_reliability(
    old_rows: list[dict], defect_rows: list[dict], judge_models: list[str],
) -> dict:
    """reliability-defects의 핵심 계산 전체(LLM 호출 없음) — cmd_reliability_defects와
    테스트가 공유. old_rows는 reliability의 build_reliability_rows 결과(90건, "id"가
    곧 figure_id), defect_rows는 build_defect_reliability_rows 결과다.
    (a) 결함셋 단독, (b) 기존 90건+결함셋 합산(그림 단위 클러스터 — 같은 그림이 두 셋에
    모두 있으면 하나의 클러스터로 묶는다), (c) 사람 점수 구간별, (d) 결함 유형별,
    (e) Judge 간 짝지은 비교를 계산한다."""
    combined_rows = (
        [{"figure_id": r["id"], "human": r["human"], "judges": r["judges"]} for r in old_rows]
        + [{"figure_id": r["figure_id"], "human": r["human"], "judges": r["judges"]} for r in defect_rows]
    )

    defect_only = {jm: _weighted_kappa_mae_bias(defect_rows, jm, "figure_id") for jm in judge_models}
    combined = {jm: _weighted_kappa_mae_bias(combined_rows, jm, "figure_id") for jm in judge_models}
    by_human_score = {jm: judge_by_human_score_bucket(combined_rows, jm) for jm in judge_models}
    by_defect_type = {jm: defect_type_breakdown(defect_rows, jm) for jm in judge_models}

    pairwise = {}
    for i in range(len(judge_models)):
        for j in range(i + 1, len(judge_models)):
            judge_a, judge_b = judge_models[i], judge_models[j]
            diff_pairs = paired_abs_error_diff(defect_rows, judge_a, judge_b)
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

    return {
        "n_defect": len(defect_rows), "n_combined": len(combined_rows),
        "defect_only": defect_only, "combined": combined,
        "by_human_score": by_human_score, "by_defect_type": by_defect_type,
        "pairwise": pairwise,
    }


def _print_defect_reliability_report(report: dict, judge_models: list[str]) -> None:
    """콘솔 출력 — 서술 원문은 절대 출력하지 않는다(하드룰 4)."""
    print(f"결함셋 {report['n_defect']}건 / 기존 90건+결함셋 합산 {report['n_combined']}건")

    for jm in judge_models:
        print(f"\n[{jm}]")
        d = report["defect_only"].get(jm, {})
        if d.get("note"):
            print(f"  결함셋 단독: {d['note']} (n={d.get('n', 0)})")
        else:
            print(
                f"  결함셋 단독(n={d['n']}) weighted κ(가중 카파)={d['weighted_kappa']} CI={d['kappa_ci']}  "
                f"MAE(평균 절대 오차)={d['mae']} CI={d['mae_ci']}  편향(judge-human)={d['bias']} CI={d['bias_ci']}"
            )
        c = report["combined"].get(jm, {})
        if c.get("note"):
            print(f"  기존+결함 합산: {c['note']} (n={c.get('n', 0)})")
        else:
            print(
                f"  기존+결함 합산(n={c['n']}, 그림 단위 클러스터) weighted κ={c['weighted_kappa']} "
                f"CI={c['kappa_ci']}  MAE={c['mae']} CI={c['mae_ci']}"
            )
        print("  사람 점수 구간별 judge 평균/MAE(중간 품질 구간을 채운 효과 확인용):")
        for score, b in report["by_human_score"][jm].items():
            print(f"    {score}점(n={b['n']}): judge 평균={b['judge_mean']} MAE={b['mae']}")
        print("  결함 유형별 사람 평균/judge 평균/검출률(judge가 원본 평균보다 낮게 준 비율):")
        for dtype, b in report["by_defect_type"][jm].items():
            print(f"    {dtype}(n={b['n']}): 사람={b['human_mean']} judge={b['judge_mean']} 검출률={b['detect_rate']}")

    if report["pairwise"]:
        print("\n[Judge 간 짝지은 비교 — |judge-human| 오차 차이(A-B), 결함셋 기준, CI가 0을 포함하면 판정 불가]")
        for label, pw in report["pairwise"].items():
            if pw.get("note"):
                print(f"  {label}: {pw['note']} (n={pw['n']})")
            else:
                print(f"  {label}: mean_diff={pw['mean_abs_error_diff']} CI={pw['ci']} → {pw['verdict']}")


def cmd_reliability_defects(args: argparse.Namespace) -> None:
    human_labels = load_human_labels()
    label_map = load_label_map()
    if not human_labels or not label_map:
        print(f"중단 — {HUMAN_LABELS_PATH} 또는 {LABEL_MAP_PATH}이 없습니다. export-sheet/import-sheet를 먼저 실행하세요.")
        sys.exit(1)
    defect_human_labels = load_defect_human_labels()
    defect_items = load_defect_items()
    if not defect_human_labels or not defect_items:
        print(f"중단 — {DEFECT_HUMAN_LABELS_PATH} 또는 {DEFECT_ITEMS_PATH}가 없습니다. make-defects/import-defect-sheet를 먼저 실행하세요.")
        sys.exit(1)

    judged_by_judge_old = {
        jm: {model: load_run_records(_judged_path(jm, model)) for model in _CANDIDATES}
        for jm in args.judges
    }
    old_rows = build_reliability_rows(human_labels, label_map, judged_by_judge_old)

    judged_by_judge_defect = {jm: load_run_records(_defect_judged_path(jm)) for jm in args.judges}
    defect_rows = build_defect_reliability_rows(defect_human_labels, defect_items, judged_by_judge_defect)

    report = compute_defect_reliability(old_rows, defect_rows, args.judges)
    report["rows"] = defect_rows  # 문항별 상세 — 점수만, 원문 없음

    with open(DEFECT_JUDGE_SELECTION_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _print_defect_reliability_report(report, args.judges)
    print(f"\n문항별 상세는 {DEFECT_JUDGE_SELECTION_PATH}에 저장했습니다.")


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

    p_make_defects = sub.add_parser(
        "make-defects", help="사람 점수 5인 서술에 결함을 주입해 2~3점 구간 검증용 셋과 블라인드 라벨링 시트 생성",
    )
    p_make_defects.add_argument("--seed", type=int, default=0)
    p_make_defects.add_argument("--per-type", type=int, default=6)
    p_make_defects.add_argument("--originals", type=int, default=10)
    p_make_defects.set_defaults(func=cmd_make_defects)

    p_import_defects = sub.add_parser(
        "import-defect-sheet", help="결함 주입 라벨링 시트를 검증 후 사람 라벨 파일에 반영(부분 진행 허용)",
    )
    p_import_defects.add_argument("--sheet", default=None, help="시트 파일 경로(기본: data/golden/vlm_defect_labeling_sheet.json)")
    p_import_defects.set_defaults(func=cmd_import_defect_sheet)

    p_judge_defects = sub.add_parser("judge-defects", help="결함 주입 서술을 Judge로 채점해 캐시에 append(이어서 실행)")
    p_judge_defects.add_argument("--judge-model", required=True)
    p_judge_defects.add_argument("--repeat", type=int, default=3)
    p_judge_defects.set_defaults(func=cmd_judge_defects)

    p_rel_defects = sub.add_parser(
        "reliability-defects", help="결함셋 단독/기존 90건과의 합산 κ·MAE, 구간별·유형별 집계",
    )
    p_rel_defects.add_argument("--judges", nargs="+", required=True, help="비교할 Judge 후보 모델 id")
    p_rel_defects.set_defaults(func=cmd_reliability_defects)

    p_cmp = sub.add_parser("compare", help="VLM 후보들을 정확도·서술 품질·adversarial·지연으로 종합 비교")
    p_cmp.add_argument("--judge-model", required=True)
    p_cmp.add_argument("--baseline", default="gpt-4o-mini", choices=_CANDIDATES)
    p_cmp.add_argument("--models", nargs="+", choices=_CANDIDATES, default=None)
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
