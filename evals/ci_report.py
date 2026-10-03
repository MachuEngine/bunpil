#!/usr/bin/env python
"""저장된 평가 원자료(raw per-item 결과)에 신뢰구간(CI)만 다시 계산해 표로 출력한다.

API·LLM 호출이 전혀 없다 — 전부 `data/golden/_*.json`에 이미 쌓여 있는 과거 실행 결과를
읽어 evals/stats.py로 부트스트랩 CI만 낸다. eval_exam.py(정기 평가, 매번 새로 채점)와
역할이 다르다 — 이 스크립트는 "이미 낸 수치의 불확실성을 사후에 드러내는" 용도다.

다루는 원자료(EVAL_SUMMARY.md의 대표 수치 중 레포에 원자료가 남아 있는 것):
  - data/golden/_ragas_eval_results.json       — faithfulness·answer_relevancy·문항 품질
  - data/golden/_structure_judge_eval_results.json — 구조 Judge(qwen2.5:14b) MAE·편향·일치율
  - data/golden/_validate_gate_calibration.json   — 게이트 재보정(gpt-5.6-luna) MAE·편향·일치율
  - VLM(eval_vlm.py) — 이 스크립트는 리포트만 출력하고 결과를 파일에 저장하지 않는다
    (evals/eval_vlm.py 확인, main()에 저장 코드 없음) → "원자료 없음"으로 표기한다.

위 네 가지 외에, data/golden/*.json을 조사하다 발견한 다른 대표 수치 원자료
(_judge_comparison_results.json·_model_comparison_results.json 등)는 **집계값만
저장돼 있고 항목 단위 raw 점수가 없어** CI를 낼 수 없다 — `_NO_RAW_DATA_NOTES`에
그 이유와 함께 목록화해 둔다(EVAL_SUMMARY.md를 읽고 뽑은 대표 수치 중 원자료가
없는 나머지).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.stats import mean_ci

_GOLDEN_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "golden"))
_RAGAS_PATH = os.path.join(_GOLDEN_DIR, "_ragas_eval_results.json")
_STRUCTURE_JUDGE_EVAL_PATH = os.path.join(_GOLDEN_DIR, "_structure_judge_eval_results.json")
_VALIDATE_GATE_CALIBRATION_PATH = os.path.join(_GOLDEN_DIR, "_validate_gate_calibration.json")
_OUT_PATH = os.path.join(_GOLDEN_DIR, "_ci_report.json")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# EVAL_SUMMARY.md의 대표 수치 중, 조사 결과 원자료 파일에 항목 단위 raw 점수가 없어
# (집계된 평균/분포만 저장돼 있어) 이 스크립트로 CI를 낼 수 없는 것들. 각 수치가 실린
# 절과, 왜 안 되는지를 함께 적는다 — "찾아보지 않고 빠뜨린 게 아니라 확인 후 제외했다"는
# 것을 남기기 위함.
_NO_RAW_DATA_NOTES = [
    {
        "metric": "Judge 모델 비교 kappa/MAE(qwen2.5:7b vs gpt-5.6-luna, EVAL_SUMMARY 4.1절)",
        "file": "data/golden/_judge_comparison_results.json",
        "reason": "Judge별 집계된 reliability dict만 저장돼 있고 항목 단위 원점수가 없음",
    },
    {
        "metric": "생성 모델 비교 실패율·생성시간(Qwen2.5 7B/14B 등, EVAL_SUMMARY 4.2절)",
        "file": "data/golden/_model_comparison_results.json",
        "reason": "모델별 집계값만 저장돼 있고 지문 단위 원시 실행 로그가 없음",
    },
    {
        "metric": "VLM 추출 정확도(CER/WER·자료 서술 Judge 점수, EVAL_SUMMARY 3.3절(4)/MODEL_SELECTION.md §7)",
        "file": "evals/eval_vlm.py 실행 결과(저장 안 함)",
        "reason": "eval_vlm.py가 리포트만 콘솔에 출력하고 결과를 파일로 저장하지 않음(확인함, main() 참고)",
    },
]


def _round_ci(point: float | None, lo: float | None, hi: float | None, ndigits: int = 4) -> dict:
    return {
        "point": round(point, ndigits) if point is not None else None,
        "ci": [round(lo, ndigits) if lo is not None else None, round(hi, ndigits) if hi is not None else None],
    }


# ── RAGAS(_ragas_eval_results.json) ──────────────────────────────────────

def ragas_ci(data: dict) -> dict:
    """faithfulness·answer_relevancy는 results[].id 단위, 문항 품질은
    item_quality_scored[].item.item_id 단위로 CI를 낸다 — 둘 다 지문(원래 입력 질문)이
    아니라 문항/결과 레코드 하나가 클러스터라, 지문 하나에서 문항이 여러 개 나왔다면
    (문항 품질 쪽은 n=24인데 results는 n=7이라 그럴 가능성이 있음) 실제보다 느슨한
    CI일 수 있다 — 참고용 신호로만 쓸 것."""
    out = {}

    results = data.get("results", [])
    faith_vals = [r["faithfulness"]["score"] for r in results if r.get("faithfulness", {}).get("score") is not None]
    faith_ids = [r["id"] for r in results if r.get("faithfulness", {}).get("score") is not None]
    rel_vals = [r["answer_relevancy"]["score"] for r in results if r.get("answer_relevancy", {}).get("score") is not None]
    rel_ids = [r["id"] for r in results if r.get("answer_relevancy", {}).get("score") is not None]

    point, lo, hi = mean_ci(faith_vals, faith_ids)
    out["faithfulness"] = {"n": len(faith_vals), **_round_ci(point, lo, hi)}
    point, lo, hi = mean_ci(rel_vals, rel_ids)
    out["answer_relevancy"] = {"n": len(rel_vals), **_round_ci(point, lo, hi)}

    scored = data.get("item_quality_scored", [])
    for c in ("정답유일성", "오답매력도", "근거성", "overall"):
        vals = [s["scores"][c] for s in scored]
        ids = [s["item"]["item_id"] for s in scored]
        point, lo, hi = mean_ci(vals, ids)
        out[f"item_quality_{c}"] = {"n": len(vals), **_round_ci(point, lo, hi)}

    return out


# ── 구조 Judge(_structure_judge_eval_results.json) / 게이트 재보정(_validate_gate_calibration.json) ──
# 두 파일 다 per_item에 (사람 overall_score·difficulty_match, Judge overall_score·
# difficulty_match)이 들어 있다는 점은 같지만 필드 이름이 다르다(전자는 human_overall/
# judge_overall 평평한 구조, 후자는 human.overall_score/judge.overall_score 중첩 구조) —
# 그래서 호출부가 (overall_mae용, bias용, agreement용) 값 리스트를 직접 뽑아 넘기는
# 공용 계산 함수 하나로 통일한다.

def _mae_bias_agreement_ci(
    judge_overall: list[float], human_overall: list[float],
    judge_diff: list[bool], human_diff: list[bool], ids: list[str],
) -> dict:
    mae_vals = [abs(j - h) for j, h in zip(judge_overall, human_overall)]
    bias_vals = [j - h for j, h in zip(judge_overall, human_overall)]
    agree_vals = [1 if jd == hd else 0 for jd, hd in zip(judge_diff, human_diff)]

    out = {}
    point, lo, hi = mean_ci(mae_vals, ids)
    out["overall_score_mae"] = {"n": len(mae_vals), **_round_ci(point, lo, hi)}
    point, lo, hi = mean_ci(bias_vals, ids)
    out["bias_judge_minus_human"] = {"n": len(bias_vals), **_round_ci(point, lo, hi)}
    point, lo, hi = mean_ci(agree_vals, ids)
    out["difficulty_match_agreement"] = {"n": len(agree_vals), **_round_ci(point, lo, hi)}
    return out


def structure_judge_eval_ci(data: dict) -> dict:
    """data/golden/_structure_judge_eval_results.json — per_item[].{id, human_overall,
    judge_overall, human_diff, judge_diff}(평평한 구조)."""
    rows = data.get("per_item", [])
    return _mae_bias_agreement_ci(
        [r["judge_overall"] for r in rows], [r["human_overall"] for r in rows],
        [r["judge_diff"] for r in rows], [r["human_diff"] for r in rows],
        [r["id"] for r in rows],
    )


def validate_gate_calibration_ci(data: dict) -> dict:
    """data/golden/_validate_gate_calibration.json — per_item[].{id, judge: {overall_score,
    difficulty_match}, human: {overall_score, difficulty_match}}(중첩 구조)."""
    rows = data.get("per_item", [])
    return _mae_bias_agreement_ci(
        [r["judge"]["overall_score"] for r in rows], [r["human"]["overall_score"] for r in rows],
        [r["judge"]["difficulty_match"] for r in rows], [r["human"]["difficulty_match"] for r in rows],
        [r["id"] for r in rows],
    )


# ── 리포트 조립 ──────────────────────────────────────────────────────────

def _load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_report() -> dict:
    report = {"sources": {}, "no_raw_data": _NO_RAW_DATA_NOTES}

    ragas_data = _load_json(_RAGAS_PATH)
    report["sources"]["ragas"] = {
        "file": os.path.relpath(_RAGAS_PATH, _REPO_ROOT),
        "ci": ragas_ci(ragas_data) if ragas_data is not None else None,
        "note": None if ragas_data is not None else "원자료 없음(파일이 아직 생성되지 않음)",
    }

    struct_data = _load_json(_STRUCTURE_JUDGE_EVAL_PATH)
    report["sources"]["structure_judge_eval"] = {
        "file": os.path.relpath(_STRUCTURE_JUDGE_EVAL_PATH, _REPO_ROOT),
        "ci": structure_judge_eval_ci(struct_data) if struct_data is not None else None,
        "note": None if struct_data is not None else "원자료 없음(파일이 아직 생성되지 않음)",
    }

    gate_data = _load_json(_VALIDATE_GATE_CALIBRATION_PATH)
    report["sources"]["validate_gate_calibration"] = {
        "file": os.path.relpath(_VALIDATE_GATE_CALIBRATION_PATH, _REPO_ROOT),
        "ci": validate_gate_calibration_ci(gate_data) if gate_data is not None else None,
        "note": None if gate_data is not None else "원자료 없음(파일이 아직 생성되지 않음)",
    }

    report["sources"]["vlm"] = {
        "file": None,
        "ci": None,
        "note": "원자료 없음 — evals/eval_vlm.py는 리포트를 콘솔에만 출력하고 파일로 저장하지 않음",
    }

    return report


def _print_ci_row(label: str, entry: dict) -> None:
    print(f"    {label:<28}n={entry['n']:<4}point={entry['point']}  CI={entry['ci']}")


def print_report(report: dict) -> None:
    print("\n" + "=" * 60)
    print("  분필 평가 — 저장된 원자료 CI 재계산 리포트(API 호출 없음)")
    print("=" * 60)

    for key, label in (
        ("ragas", "RAGAS(eval_ragas.py)"),
        ("structure_judge_eval", "구조 Judge 신뢰도(qwen2.5:14b, 일회성 raw)"),
        ("validate_gate_calibration", "게이트 재보정(gpt-5.6-luna, 일회성 raw)"),
        ("vlm", "VLM 추출(eval_vlm.py)"),
    ):
        src = report["sources"][key]
        print(f"\n[{label}] 파일: {src['file']}")
        if src["ci"] is None:
            print(f"    {src['note']}")
            continue
        for metric, entry in src["ci"].items():
            _print_ci_row(metric, entry)

    print("\n[CI 산출 불가(원자료 없음) — EVAL_SUMMARY.md 대표 수치 중]")
    for item in report["no_raw_data"]:
        print(f"  - {item['metric']}")
        print(f"      파일: {item['file']}")
        print(f"      이유: {item['reason']}")

    print("\n" + "=" * 60)


def main() -> None:
    report = build_report()
    print_report(report)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {_OUT_PATH}")


if __name__ == "__main__":
    main()
