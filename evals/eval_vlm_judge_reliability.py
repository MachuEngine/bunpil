#!/usr/bin/env python
"""VLM figure Judge(get_judge_backend()로 "[자료: ...]" 서술 커버리지를 채점하는 축)의
신뢰도를 사람 라벨과 비교한다. LLM 재호출 없음 — data/golden/vlm_figure_judge_golden.json에
이미 고정된 (VLM 출력, Judge 점수) 쌍에 사람이 채운 human_label만 대조한다.

지표는 evals/stats.py의 weighted_kappa(1~5 척도 가중 Cohen's kappa)와
cluster_bootstrap_ci(클러스터 = 항목 id)로 계산한다 — 가중 κ·MAE·±1 일치율 각각의
95% 신뢰구간(CI)을 낸다. n=15로 작아서 점추정치 하나만 보면 과신하기 쉽기 때문에,
CI가 넓으면(예: 0을 포함) "판정 불가"로 보는 게 안전하다.

이진 κ(점수 ≥3 여부로 이분화)도 참고용으로 같이 낸다 — 다만 Judge 점수가 15건 중
13건이 3점 이상으로 쏠려 있어(즉 이분화하면 거의 다 "양성") 기대 일치(pe)가 1에
가까워지고 값이 불안정해지기 쉽다. 이 쏠림 정도를 보여주기 위해 최빈값 비율(=다수
클래스 비율)을 함께 출력한다.

human_label이 비어 있는(null) 항목은 아직 라벨링 전으로 보고 건너뛴다.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.stats import cluster_bootstrap_ci, weighted_kappa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_PATH = os.path.join(ROOT, "data", "golden", "vlm_figure_judge_golden.json")

_LABELS = [1, 2, 3, 4, 5]


def _binary_kappa_threshold3(human: list[int], llm: list[int]) -> float | None:
    """이진 Cohen's kappa(점수 >= 3 → 양성). 참고용 — docstring의 쏠림 설명 참고."""
    n = len(human)
    if n < 2:
        return None
    h = [1 if x >= 3 else 0 for x in human]
    l = [1 if x >= 3 else 0 for x in llm]
    po = sum(hi == li for hi, li in zip(h, l)) / n
    ph, pl = sum(h) / n, sum(l) / n
    pe = ph * pl + (1 - ph) * (1 - pl)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def _kappa_stat(sample: list[tuple[int, int]]) -> float | None:
    if len(sample) < 2:
        return None
    human = [h for h, _ in sample]
    llm = [l for _, l in sample]
    return weighted_kappa(human, llm, labels=_LABELS)


def _mae_stat(sample: list[tuple[int, int]]) -> float | None:
    if not sample:
        return None
    return sum(abs(h - l) for h, l in sample) / len(sample)


def _within1_stat(sample: list[tuple[int, int]]) -> float | None:
    if not sample:
        return None
    return sum(1 for h, l in sample if abs(h - l) <= 1) / len(sample)


def _fmt_ci(point: float | None, lo: float | None, hi: float | None) -> str:
    if point is None:
        return "계산 불가"
    if lo is None or hi is None:
        return f"{point:.3f} (CI 계산 불가)"
    return f"{point:.3f}  [95% CI {lo:.3f}, {hi:.3f}]"


def main():
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        doc = json.load(f)
    entries = doc["entries"]

    labeled = [e for e in entries if e.get("human_label") is not None and e.get("judge_score") is not None]
    unlabeled = [e["id"] for e in entries if e.get("human_label") is None]

    print(f"전체 {len(entries)}건 중 라벨링 완료 {len(labeled)}건, 미완료 {len(unlabeled)}건")
    if unlabeled:
        print(f"  미완료: {unlabeled}")

    if not labeled:
        print(
            "\n라벨링된 항목이 없습니다. data/golden/vlm_figure_judge_golden.json을 직접 "
            "열어 각 항목의 \"human_label\" 필드(현재 null)를 1~5 정수로 채우세요 — "
            "figure_summary(핵심 사실)가 vlm_figure_description(VLM이 서술한 자료 설명)에 "
            "얼마나 담겼는지를 5점=거의 다 담음, 3점=일부만 담음, 1점=핵심을 놓치거나 "
            "서술 자체가 없음 기준으로 판단합니다(judge_score를 매길 때 Judge에게 준 기준과 동일)."
        )
        return
    if len(labeled) < 2:
        print("\n라벨링된 항목이 너무 적어 지표를 계산할 수 없습니다 (최소 2건 필요).")
        return

    human = [e["human_label"] for e in labeled]
    llm = [e["judge_score"] for e in labeled]
    ids = [e["id"] for e in labeled]
    rows = list(zip(human, llm))

    kappa_point, kappa_lo, kappa_hi = cluster_bootstrap_ci(rows, ids, _kappa_stat)
    mae_point, mae_lo, mae_hi = cluster_bootstrap_ci(rows, ids, _mae_stat)
    within1_point, within1_lo, within1_hi = cluster_bootstrap_ci(rows, ids, _within1_stat)
    exact = sum(h == l for h, l in rows) / len(rows)

    print("\n" + "=" * 60)
    print("VLM figure Judge 신뢰도")
    print("=" * 60)
    print(f"  n = {len(labeled)}")
    print(f"  사람 평균 = {sum(human) / len(human):.2f}  |  Judge 평균 = {sum(llm) / len(llm):.2f}")
    print(f"  정확 일치율 = {exact:.3f}")
    print(f"  ±1 일치율 = {_fmt_ci(within1_point, within1_lo, within1_hi)}")
    print(f"  MAE = {_fmt_ci(mae_point, mae_lo, mae_hi)}")
    print(f"  가중 kappa(1~5 척도) = {_fmt_ci(kappa_point, kappa_lo, kappa_hi)}")

    judge_pos_ratio = sum(1 for x in llm if x >= 3) / len(llm)
    mode_ratio = max(judge_pos_ratio, 1 - judge_pos_ratio)
    binary_kappa = _binary_kappa_threshold3(human, llm)
    binary_str = "계산 불가" if binary_kappa is None else f"{binary_kappa:.3f}"
    print(
        f"  [참고용] 이진 kappa(threshold=3) = {binary_str}  "
        f"(Judge 점수의 다수 클래스 비율 = {mode_ratio:.2f} — 쏠려 있을수록 이 값이 불안정해짐)"
    )
    print("=" * 60)

    print("\n항목별:")
    for e in labeled:
        diff = e["judge_score"] - e["human_label"]
        flag = "" if diff == 0 else f"  (Judge {'+' if diff > 0 else ''}{diff})"
        print(f"  {e['id']}: 사람={e['human_label']} Judge={e['judge_score']}{flag}")

    if kappa_point is not None and kappa_point < 0.4:
        print("\n⚠️ 가중 kappa < 0.4 — 프로젝트가 Judge 채택 기준으로 삼았던 목표치(MODEL_SELECTION.md "
              "2절, item_golden 최초 목표)에 못 미칩니다. figure 축 Judge 점수를 그대로 신뢰하기 "
              "전에 프롬프트 보정이 필요할 수 있습니다.")
    if kappa_lo is not None and kappa_lo <= 0.0 <= (kappa_hi or 0.0):
        print("\n⚠️ 가중 kappa의 95% CI가 0을 포함합니다 — n이 작아 이 결과만으로 Judge가 사람과 "
              "일치한다/안 한다를 단정하기 어렵습니다.")


if __name__ == "__main__":
    main()
