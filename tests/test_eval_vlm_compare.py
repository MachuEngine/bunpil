"""evals/eval_vlm_compare.py 순수 로직 테스트. LLM/네트워크 호출 없음 — extract의 VLM
백엔드는 app.common.llm.get_vlm_backend를 가짜로 바꿔 네트워크를 전혀 타지 않는다.

시트 배정 균등성·블라인드·키 순서(라벨 보호 포함), import 검증, reliability 조인·κ
계산 경로, compare의 짝 비교 쌍 구성·판정 분기, extract의 캐시 스킵을 다룬다."""
import argparse
import json

import pytest

import evals.eval_vlm_compare as mod
from evals.eval_vlm_compare import (
    apply_sheet_labels,
    assign_models_balanced,
    build_label_sheet_entry,
    build_reliability_rows,
    cmd_export_sheet,
    cmd_extract,
    cmd_import_sheet,
    compute_comparison,
    compute_judge_reliability,
    has_existing_sheet_labels,
    paired_diff_ci,
    validate_sheet,
    validate_sheet_label,
)


# ── assign_models_balanced ────────────────────────────────────────────────

def test_assign_models_balanced_even_counts():
    ids = [f"f{i:02d}" for i in range(1, 91)]  # 90건(2026-10 범위 변경 반영)
    candidates = ("a", "b", "c", "d", "e")
    assignment = assign_models_balanced(ids, candidates, seed=0)
    assert set(assignment) == set(ids)
    counts = {}
    for v in assignment.values():
        counts[v] = counts.get(v, 0) + 1
    assert counts == {c: 18 for c in candidates}  # 90/5


def test_assign_models_balanced_handles_remainder():
    ids = [f"x{i}" for i in range(7)]
    candidates = ("a", "b", "c")
    assignment = assign_models_balanced(ids, candidates, seed=1)
    counts = {}
    for v in assignment.values():
        counts[v] = counts.get(v, 0) + 1
    assert sum(counts.values()) == 7
    assert max(counts.values()) - min(counts.values()) <= 1


def test_assign_models_balanced_deterministic_with_seed():
    ids = [f"f{i:02d}" for i in range(10)]
    candidates = ("a", "b")
    a1 = assign_models_balanced(ids, candidates, seed=5)
    a2 = assign_models_balanced(ids, candidates, seed=5)
    assert a1 == a2


# ── build_label_sheet_entry — 키 순서(라벨 맨 위)·블라인드·서술 없음 표시 ──

def test_build_label_sheet_entry_key_order_and_blind():
    entry = {"id": "f01", "figure_summary": "A지역 820, B지역 450"}
    sheet_entry = build_label_sheet_entry(entry, "[자료: A 820, B 450]")
    assert list(sheet_entry.keys()) == ["라벨", "id", "핵심사실", "VLM_서술_줄"]
    assert sheet_entry["라벨"] == {"점수": None, "근거": ""}
    assert sheet_entry["id"] == "f01"
    assert "model" not in sheet_entry  # 블라인드 — 모델 정보 미포함
    assert sheet_entry["VLM_서술_줄"] == ["[자료: A 820, B 450]"]


def test_build_label_sheet_entry_missing_description():
    entry = {"id": "f02", "figure_summary": "핵심 사실"}
    sheet_entry = build_label_sheet_entry(entry, None)
    assert sheet_entry["VLM_서술_줄"] == ["서술 없음"]


def test_build_label_sheet_entry_splits_multiline_description():
    entry = {"id": "f03", "figure_summary": "핵심 사실"}
    sheet_entry = build_label_sheet_entry(entry, "줄1\n줄2")
    assert sheet_entry["VLM_서술_줄"] == ["줄1", "줄2"]


# ── has_existing_sheet_labels — 라벨 보호 가드 ─────────────────────────────

def test_has_existing_sheet_labels_false_for_missing_file(tmp_path):
    assert has_existing_sheet_labels(str(tmp_path / "nope.json")) is False


def test_has_existing_sheet_labels_false_when_all_empty(tmp_path):
    path = tmp_path / "sheet.json"
    data = {"문항": [{"id": "f01", "라벨": {"점수": None, "근거": ""}}]}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert has_existing_sheet_labels(str(path)) is False


@pytest.mark.parametrize("label", [{"점수": 4, "근거": ""}, {"점수": None, "근거": "근거 있음"}])
def test_has_existing_sheet_labels_true_when_filled(tmp_path, label):
    path = tmp_path / "sheet.json"
    data = {"문항": [{"id": "f01", "라벨": label}]}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert has_existing_sheet_labels(str(path)) is True


def test_cmd_export_sheet_refuses_to_overwrite_filled_sheet(tmp_path, monkeypatch, capsys):
    sheet_path = tmp_path / "sheet.json"
    data = {"문항": [{"id": "f01", "라벨": {"점수": 5, "근거": ""}}]}
    sheet_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(mod, "LABEL_SHEET_PATH", str(sheet_path))

    with pytest.raises(SystemExit):
        cmd_export_sheet(argparse.Namespace(seed=0))
    assert "덮어쓰지 않습니다" in capsys.readouterr().out


# ── validate_sheet_label / validate_sheet ─────────────────────────────────

@pytest.mark.parametrize("label,expect_error", [
    ({"점수": None, "근거": ""}, False),
    ({"점수": 5, "근거": ""}, False),
    ({"점수": 1, "근거": "핵심을 놓침"}, False),
    ({"점수": 2, "근거": ""}, True),  # 2점 이하인데 근거 없음
    ({"점수": 6, "근거": ""}, True),  # 범위 밖
    ({"점수": True, "근거": ""}, True),  # bool은 int로 취급하지 않음
    ({"점수": "3", "근거": ""}, True),  # 문자열
])
def test_validate_sheet_label(label, expect_error):
    errors = validate_sheet_label(label)
    assert bool(errors) == expect_error


def test_validate_sheet_rejects_unknown_id():
    items = [{"id": "ghost", "라벨": {"점수": 5, "근거": ""}}]
    errors = validate_sheet(items, valid_ids={"f01"})
    assert len(errors) == 1
    assert errors[0][0] == "ghost"


def test_validate_sheet_passes_known_id_with_valid_label():
    items = [{"id": "f01", "라벨": {"점수": 5, "근거": ""}}]
    errors = validate_sheet(items, valid_ids={"f01"})
    assert errors == []


# ── apply_sheet_labels — 부분 진행 허용 ───────────────────────────────────

def test_apply_sheet_labels_partial_progress():
    existing = {"f00": {"점수": 3, "근거": "이전 라벨"}}
    sheet_items = [
        {"id": "f01", "라벨": {"점수": 4, "근거": ""}},       # 유효
        {"id": "f02", "라벨": {"점수": 2, "근거": ""}},       # 2점인데 근거 없음 — 무효
        {"id": "f03", "라벨": {"점수": None, "근거": ""}},    # 아직 라벨링 전 — 저장 안 함
    ]
    valid_ids = {"f01", "f02", "f03"}

    updated, errors, n_saved = apply_sheet_labels(existing, sheet_items, valid_ids)

    assert n_saved == 1
    assert updated["f01"] == {"점수": 4, "근거": ""}
    assert "f02" not in updated  # 검증 실패 — 반영 안 됨
    assert "f03" not in updated  # 미완료 — 반영 안 됨
    assert updated["f00"] == {"점수": 3, "근거": "이전 라벨"}  # 기존 라벨 보존
    assert len(errors) == 1 and errors[0][0] == "f02"


def test_cmd_import_sheet_reports_error_but_saves_valid_entries(tmp_path, monkeypatch, capsys):
    sheet_path = tmp_path / "sheet.json"
    sheet = {"문항": [
        {"id": "f01", "라벨": {"점수": 5, "근거": ""}},
        {"id": "f02", "라벨": {"점수": 1, "근거": ""}},  # 1점인데 근거 없음 — 무효
    ]}
    sheet_path.write_text(json.dumps(sheet, ensure_ascii=False), encoding="utf-8")

    label_map_path = tmp_path / "label_map.json"
    label_map_path.write_text(
        json.dumps({"map": {"f01": {"model": "gpt-4o-mini"}, "f02": {"model": "gemma3-12b"}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    human_labels_path = tmp_path / "human_labels.json"

    monkeypatch.setattr(mod, "LABEL_MAP_PATH", str(label_map_path))
    monkeypatch.setattr(mod, "HUMAN_LABELS_PATH", str(human_labels_path))
    monkeypatch.setattr(mod, "GOLDEN_DIR", str(tmp_path))

    cmd_import_sheet(argparse.Namespace(sheet=str(sheet_path)))

    out = capsys.readouterr().out
    assert "검증 실패" in out
    assert "f02" in out

    saved = json.loads(human_labels_path.read_text(encoding="utf-8"))
    assert saved["labels"]["f01"] == {"점수": 5, "근거": ""}
    assert "f02" not in saved["labels"]


# ── build_reliability_rows / compute_judge_reliability ────────────────────

def test_build_reliability_rows_joins_and_drops_none_scores():
    human_labels = {"f01": {"점수": 5, "근거": ""}, "f02": {"점수": 1, "근거": "근거"}}
    label_map = {"f01": {"model": "gpt-4o-mini"}, "f02": {"model": "qwen3-vl-8b"}}
    judged_by_judge = {
        "judgeA": {
            "gpt-4o-mini": [
                {"id": "f01", "judge_score": 5, "repeat": 0},
                {"id": "f01", "judge_score": 4, "repeat": 1},
            ],
            "qwen3-vl-8b": [{"id": "f02", "judge_score": None, "repeat": 0}],
        },
    }
    rows = build_reliability_rows(human_labels, label_map, judged_by_judge)
    assert len(rows) == 2

    row_f01 = next(r for r in rows if r["id"] == "f01")
    assert row_f01["model"] == "gpt-4o-mini"
    assert row_f01["judges"]["judgeA"] == [5, 4]

    row_f02 = next(r for r in rows if r["id"] == "f02")
    assert row_f02["judges"] == {}  # 전부 None이라 제외


def test_build_reliability_rows_skips_unlabeled_or_unmapped():
    human_labels = {"f01": {"점수": None, "근거": ""}, "f02": {"점수": 3, "근거": ""}}
    label_map = {"f02_other_model": {"model": "gpt-4o-mini"}}  # f02는 매핑에 없음
    rows = build_reliability_rows(human_labels, label_map, judged_by_judge={})
    assert rows == []


def test_compute_judge_reliability_perfect_agreement_kappa_is_one():
    rows = [
        {"id": "a", "model": "m1", "human": 5, "judges": {"j1": [5]}},
        {"id": "b", "model": "m1", "human": 1, "judges": {"j1": [1]}},
        {"id": "c", "model": "m2", "human": 3, "judges": {"j1": [3]}},
        {"id": "d", "model": "m2", "human": 4, "judges": {"j1": [4]}},
    ]
    report = compute_judge_reliability(rows, ["j1"])
    pj = report["per_judge"]["j1"]
    assert pj["n"] == 4
    assert pj["weighted_kappa"] == pytest.approx(1.0)
    assert pj["mae"] == pytest.approx(0.0)
    assert pj["bias"] == pytest.approx(0.0)
    # bias_by_vlm_model은 모델별로 존재(둘 다 완벽 일치라 0에 가까움)
    assert set(report["bias_by_vlm_model"]["j1"]) == {"m1", "m2"}


def test_compute_judge_reliability_insufficient_sample_notes():
    rows = [{"id": "a", "model": "m1", "human": 5, "judges": {"j1": [5]}}]
    report = compute_judge_reliability(rows, ["j1"])
    assert report["per_judge"]["j1"]["note"] == "표본 부족(2건 미만)"


def test_compute_judge_reliability_pairwise_present_for_two_judges():
    rows = [
        {"id": "a", "model": "m1", "human": 5, "judges": {"j1": [5], "j2": [5]}},
        {"id": "b", "model": "m1", "human": 1, "judges": {"j1": [1], "j2": [5]}},
        {"id": "c", "model": "m2", "human": 3, "judges": {"j1": [3], "j2": [3]}},
        {"id": "d", "model": "m2", "human": 4, "judges": {"j1": [4], "j2": [2]}},
    ]
    report = compute_judge_reliability(rows, ["j1", "j2"])
    assert "j1 vs j2" in report["pairwise"]
    pw = report["pairwise"]["j1 vs j2"]
    assert pw["n"] == 4
    assert pw["verdict"] in (
        "j1가 더 정확", "j2가 더 정확", "판정 불가(CI가 0을 포함)",
    )


# ── paired_diff_ci — compare의 짝 비교 판정 분기 ──────────────────────────

def test_paired_diff_ci_no_common_ids():
    result = paired_diff_ci({"a": 1.0}, {"b": 2.0}, higher_is_better=True, label_a="A", label_b="B")
    assert result == {"n": 0, "mean_diff": None, "ci": [None, None], "verdict": "표본 없음(공통 이미지 없음)"}


def test_paired_diff_ci_lower_is_better_clear_winner():
    # CER(낮을수록 좋음) — B가 A보다 CER이 뚜렷하게 높음(더 나쁨) → A가 더 우수
    values_a = {"i1": 0.01, "i2": 0.02, "i3": 0.015, "i4": 0.012}
    values_b = {"i1": 0.5, "i2": 0.52, "i3": 0.51, "i4": 0.49}
    result = paired_diff_ci(values_a, values_b, higher_is_better=False, label_a="A", label_b="B")
    assert result["n"] == 4
    assert result["verdict"] == "A가 더 우수"


def test_paired_diff_ci_higher_is_better_clear_winner():
    values_a = {"i1": 1.0, "i2": 1.0, "i3": 1.0}
    values_b = {"i1": 5.0, "i2": 5.0, "i3": 5.0}
    result = paired_diff_ci(values_a, values_b, higher_is_better=True, label_a="A", label_b="B")
    assert result["verdict"] == "B가 더 우수"


def test_paired_diff_ci_ci_includes_zero_is_inconclusive():
    values_a = {"i1": 1.0, "i2": 1.0, "i3": 1.0, "i4": 1.0}
    values_b = {"i1": 1.1, "i2": 0.9, "i3": 1.05, "i4": 0.95}  # 평균 차이 0 근처, 부호가 섞임
    result = paired_diff_ci(values_a, values_b, higher_is_better=True, label_a="A", label_b="B")
    assert result["verdict"] == "판정 불가(CI가 0을 포함)"


# ── compute_comparison — 짝 비교 쌍 구성 ──────────────────────────────────

def _text_only_record(cer_value: float) -> dict:
    return {"cer": cer_value, "wer": cer_value, "latency_sec": 1.0, "error": None, "category": "text_only"}


def test_compute_comparison_builds_all_pairwise_combinations():
    golden_entries = [{"id": f"t{i}", "category": "text_only", "ground_truth_text": "x"} for i in range(4)]
    records_by_model = {
        "m1": {f"t{i}": _text_only_record(0.0) for i in range(4)},
        "m2": {f"t{i}": _text_only_record(0.3) for i in range(4)},
        "m3": {f"t{i}": _text_only_record(0.15) for i in range(4)},
    }
    report = compute_comparison(
        ["m1", "m2", "m3"], "m1", "judgeX", records_by_model, judged_by_model={},
        golden_entries=golden_entries, human_labels={}, label_map={}, code_version="test",
    )
    assert set(report["pairwise"]) == {"m1_vs_m2", "m1_vs_m3", "m2_vs_m3"}

    cer_diff = report["pairwise"]["m1_vs_m2"]["cer_diff"]
    assert cer_diff["verdict"] == "m1가 더 우수"  # m1의 CER(0.0)이 m2(0.3)보다 낮음(더 좋음)

    # figure_judge 캐시가 없으므로 서술 Judge 차이는 공통 이미지가 없어 "표본 없음"
    assert report["pairwise"]["m1_vs_m2"]["figure_judge_diff"]["verdict"] == "표본 없음(공통 이미지 없음)"


# ── extract — 캐시 스킵 ────────────────────────────────────────────────────

class _FakeVLM:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    async def extract_text(self, image_bytes: bytes, mime_type: str) -> str:
        self.calls += 1
        return self.text


def _write_golden(tmp_path, fixed_text: str):
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    img_dir = golden_dir / "vlm_golden_images"
    img_dir.mkdir()
    entries = []
    for i in (1, 2):
        img_rel = f"vlm_golden_images/t{i:02d}.png"
        (golden_dir / img_rel).write_bytes(b"fake-png-bytes")
        entries.append({
            "id": f"t{i:02d}", "category": "text_only", "image": img_rel,
            "ground_truth_text": fixed_text,
        })
    golden_path = golden_dir / "vlm_extraction_golden.json"
    golden_path.write_text(json.dumps({"entries": entries}, ensure_ascii=False), encoding="utf-8")
    return golden_dir, golden_path


def test_cmd_extract_skips_already_completed_ids(tmp_path, monkeypatch, capsys):
    fixed_text = "고정된 추출 텍스트"
    golden_dir, golden_path = _write_golden(tmp_path, fixed_text)
    runs_dir = tmp_path / "_vlm_runs"

    monkeypatch.setattr(mod, "GOLDEN_DIR", str(golden_dir))
    monkeypatch.setattr(mod, "GOLDEN_PATH", str(golden_path))
    monkeypatch.setattr(mod, "RUNS_DIR", str(runs_dir))

    fake_vlm = _FakeVLM(fixed_text)
    monkeypatch.setattr("app.common.llm.get_vlm_backend", lambda: fake_vlm)

    args = argparse.Namespace(model="gpt-4o-mini", ids=None, limit=None)
    cmd_extract(args)

    run_path = runs_dir / "gpt-4o-mini.jsonl"
    lines = run_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert fake_vlm.calls == 2

    # 두 번째 실행 — 전부 이미 완료 상태라 추가 호출이 없어야 한다.
    cmd_extract(args)
    out = capsys.readouterr().out
    assert "이미 완료" in out
    assert fake_vlm.calls == 2  # 재호출 없음
    lines_after = run_path.read_text(encoding="utf-8").splitlines()
    assert len(lines_after) == 2  # 파일에 추가되지 않음
