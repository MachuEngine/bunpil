"""evals/eval_vlm_compare.py 순수 로직 테스트. LLM/네트워크 호출 없음 — extract의 VLM
백엔드는 app.common.llm.get_vlm_backend를 가짜로 바꿔 네트워크를 전혀 타지 않는다.

시트 배정 균등성·블라인드·키 순서(라벨 보호 포함), import 검증, reliability 조인·κ
계산 경로, compare의 짝 비교 쌍 구성·판정 분기, extract의 캐시 스킵, make-defects 이하
(결함 변환 5종·블록 교체·블라인드·reliability-defects 집계)를 다룬다."""
import argparse
import json

import pytest

import evals.eval_vlm_compare as mod
from evals.eval_vlm_compare import (
    apply_item_dropped,
    apply_key_value_changed,
    apply_minor_value_changed,
    apply_numbers_removed,
    apply_sheet_labels,
    apply_unit_removed,
    assign_models_balanced,
    build_defect_reliability_rows,
    build_label_sheet_entry,
    build_reliability_rows,
    cmd_export_sheet,
    cmd_extract,
    cmd_import_sheet,
    cmd_make_defects,
    compute_comparison,
    compute_defect_reliability,
    compute_judge_reliability,
    defect_type_breakdown,
    find_extreme_value,
    has_existing_sheet_labels,
    judge_by_human_score_bucket,
    make_defect_items,
    paired_diff_ci,
    parse_label_value_pairs,
    replace_figure_block,
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


# ── make-defects: 결함 변환 5종 ────────────────────────────────────────────

_POP_SUMMARY = (
    "A~D 4개 지역의 인구를 막대그래프로 비교. A지역이 820만 명으로 가장 많고, "
    "B(450)·C(610)·D(300) 순. 값이 가장 큰 막대는 A."
)


def test_find_extreme_value_max():
    assert find_extreme_value(_POP_SUMMARY) == ("820", "max")


def test_find_extreme_value_min():
    summary = "2018~2022년 실업률 꺾은선 그래프. 2022년이 3.0%로 가장 낮음."
    assert find_extreme_value(summary) == ("3.0", "min")


def test_find_extreme_value_none_without_pattern():
    assert find_extreme_value("그냥 설명만 있는 문장") is None


def test_apply_key_value_changed_flips_max_item():
    fig_text = "[자료: A 820, B 450, C 610, D 300]"
    result = apply_key_value_changed(fig_text, _POP_SUMMARY)
    assert result == "[자료: A 492, B 450, C 610, D 300]"  # 820*0.6=492 — 더 이상 최댓값이 아님


def test_apply_key_value_changed_scales_min_item_up():
    summary = "2018~2022년 실업률 꺾은선 그래프. 2022년이 3.0%로 가장 낮음."
    fig_text = "[자료: 2018년 3.8, 2019년 3.5, 2020년 4.9, 2021년 3.7, 2022년 3.0]"
    result = apply_key_value_changed(fig_text, summary)
    assert result == "[자료: 2018년 3.8, 2019년 3.5, 2020년 4.9, 2021년 3.7, 2022년 4.8]"  # 3.0*1.6=4.8


def test_apply_key_value_changed_none_without_extreme_pattern():
    assert apply_key_value_changed("[자료: A 820, B 450]", "그냥 설명") is None


def test_apply_key_value_changed_none_when_value_not_in_text():
    assert apply_key_value_changed("[자료: X 1, Y 2]", _POP_SUMMARY) is None


def test_apply_minor_value_changed_leaves_extreme_value_untouched():
    fig_text = "[자료: A 820, B 400, C 610, D 300]"
    result = apply_minor_value_changed(fig_text, _POP_SUMMARY)
    assert result == "[자료: A 820, B 460, C 610, D 300]"  # 400*1.15=460, 최댓값(820)은 그대로


def test_apply_minor_value_changed_skips_year_labels():
    summary = "2018~2022년 실업률 꺾은선 그래프. 2022년이 3.0%로 가장 낮음."
    fig_text = "[자료: 2018년 3.8, 2019년 3.5, 2020년 4.9, 2021년 3.7, 2022년 3.0]"
    result = apply_minor_value_changed(fig_text, summary)
    assert result is not None
    assert "2018" in result and "2019" in result  # 연도 숫자는 안 건드림
    assert "3.8" not in result  # 가장 먼저 나오는 비극값 수치(3.8)가 바뀜


def test_apply_minor_value_changed_none_when_no_candidate():
    assert apply_minor_value_changed("[자료: A 820]", _POP_SUMMARY) is None  # 820뿐이라 비극값 후보 없음


def test_apply_item_dropped_comma_list_drops_last_item():
    fig_text = "[자료: A당 38%, B당 29%, C당 19%, D당 14%]"
    assert apply_item_dropped(fig_text, "") == "[자료: A당 38%, B당 29%, C당 19%]"


def test_apply_item_dropped_table_rows_drops_last_data_row():
    table = (
        "| 지역 | 전체 의석(석) | 여성 의원 비율(%) |\n"
        "|------|----------------|--------------------|\n"
        "| 갑   | 42             | 24                 |\n"
        "| 을   | 28             | 36                 |\n"
    )
    result = apply_item_dropped(table, "")
    assert "을" not in result
    assert "갑" in result


def test_apply_item_dropped_none_when_only_one_item():
    assert apply_item_dropped("[자료: A 820]", "") is None


def test_apply_unit_removed_strips_percent_and_scale_note():
    fig_text = "[자료: 진보 32%, 중도 40%, 보수 24%, 응답 불명 4%]"
    assert apply_unit_removed(fig_text, "") == "[자료: 진보 32, 중도 40, 보수 24, 응답 불명 4]"


def test_apply_unit_removed_does_not_mangle_non_unit_words():
    # "응답 불명"의 "명"이 "4%"의 단위로 오인되어 지워지면 안 된다(숫자 바로 뒤가 아님).
    result = apply_unit_removed("[자료: 응답 불명 4%]", "")
    assert "불명" in result


def test_apply_unit_removed_none_when_nothing_to_strip():
    assert apply_unit_removed("[자료: A 820, B 450]", "") is None


def test_apply_unit_removed_none_for_pure_whitespace_diff():
    # 표의 정렬용 공백만 남고 지울 단위가 없으면(단위가 숫자에 안 붙어 있음) None이어야
    # 한다 — 공백 정리만으로 "변환됨"으로 치면 안 된다.
    table = "| 갑 | 42             | 24 |\n"
    assert apply_unit_removed(table, "") is None


# apply_numbers_removed — 숫자만 지우지 않고 경향 문장을 만든다(2026-10 메인 세션이
# "년대"→"년대"·빈 표 칸 등 티 나는 결함을 지적해 재설계). 값은 지우되 라벨(연도·연대·
# 월·범주명)의 숫자는 보존하고, 시간축이면 증가/감소/정점 추이를, 범주형이면 최댓값·
# 최솟값만 말하는 문장을 만든다.

def test_apply_numbers_removed_time_axis_monotonic_increase():
    fig_text = "[자료: 2000년 15.5%, 2005년 20.0%, 2010년 23.9%, 2015년 27.2%, 2020년 31.7%, 2023년 34.5%]"
    assert apply_numbers_removed(fig_text, "") == "[자료: 2000년부터 2023년까지 꾸준히 증가했다.]"


def test_apply_numbers_removed_time_axis_peak_then_decrease():
    fig_text = "[자료: 1970년 15%, 1990년 35%, 2010년 28%, 2020년 20%]"
    assert apply_numbers_removed(fig_text, "") == "[자료: 1970년부터 2020년까지 1990년에 가장 높았다가 이후 감소했다.]"


def test_apply_numbers_removed_categorical_max_min():
    fig_text = "[자료: 수도권 4350, 충청권 3920, 영남권 3680, 호남권 3100]"
    assert apply_numbers_removed(fig_text, "") == (
        "[자료: 수도권, 충청권, 영남권, 호남권 중 수도권이 가장 크고 호남권이 가장 작다.]"
    )


@pytest.mark.parametrize("fig_text,expected", [
    (
        "[자료: 갑국 520, 을국 310, 병국 270]",
        "[자료: 갑국, 을국, 병국 중 갑국이 가장 크고 병국이 가장 작다.]",
    ),
    (
        "[자료: 아시아 59%, 아프리카 18%, 유럽 9%, 아메리카 13%, 오세아니아 1%]",
        "[자료: 아시아, 아프리카, 유럽, 아메리카, 오세아니아 중 아시아가 가장 크고 오세아니아가 가장 작다.]",
    ),
])
def test_apply_numbers_removed_particle_follows_final_consonant(fig_text, expected):
    # 받침 있으면 "이"("병국이"), 받침 없으면 "가"("아시아가") — 유니코드 종성 계산.
    assert apply_numbers_removed(fig_text, "") == expected


def test_apply_numbers_removed_markdown_table_input():
    table = (
        "| 지역 | 전체 의석(석) | 여성 의원 비율(%) |\n"
        "|------|----------------|--------------------|\n"
        "| 갑   | 42             | 24                 |\n"
        "| 을   | 28             | 36                 |\n"
    )
    # 표는 "첫 번째 수치 열"(전체 의석)만 쓴다 — 두 번째 열(여성 비율)은 무시.
    assert apply_numbers_removed(table, "") == "[자료: 갑, 을 중 갑이 가장 크고 을이 가장 작다.]"


@pytest.mark.parametrize("fig_text,expected_pairs", [
    (
        "[자료: 1990년대 28%, 2000년대 20%, 2010년대 15%, 2020년대 11%]",
        [("1990년대", "28"), ("2000년대", "20"), ("2010년대", "15"), ("2020년대", "11")],
    ),
    (
        "[자료: 1월 5도, 4월 15도, 7월 28도, 10월 18도]",
        [("1월", "5"), ("4월", "15"), ("7월", "28"), ("10월", "18")],
    ),
    (
        "[자료: 10대 31%, 20대 52%, 30대 38%, 40대 21%]",
        [("10대", "31"), ("20대", "52"), ("30대", "38"), ("40대", "21")],
    ),
])
def test_parse_label_value_pairs_preserves_label_numbers(fig_text, expected_pairs):
    # 라벨 안의 숫자(연도·연대·월·연령대)는 값이 아니라 라벨이므로 그대로 남아야 한다.
    assert parse_label_value_pairs(fig_text) == ("", expected_pairs)


def test_apply_numbers_removed_none_for_multi_sentence_structure():
    # 문장이 여러 개로 나뉜 구조(지역마다 독립된 문장)는 안전하게 변환을 포기한다.
    fig_text = (
        "[자료: 갑 지역의 삶의 만족도는 6.5점, 평균 통근 시간은 38분. "
        "을 지역의 삶의 만족도는 7.2점, 평균 통근 시간은 22분.]"
    )
    assert apply_numbers_removed(fig_text, "") is None


def test_apply_numbers_removed_none_when_fewer_than_two_pairs():
    assert apply_numbers_removed("[자료: 서술 없음]", "") is None


def test_replace_figure_block_table_to_bracket_block_preserves_surrounding_text():
    # 원 서술이 마크다운 표였어도 numbers_removed 출력은 "[자료: ...]" 블록이다 —
    # replace_figure_block이 블록 종류가 바뀌어도 블록 밖 텍스트를 그대로 보존해야 한다.
    raw = "1. 표 문제입니다.\n\n| a | b |\n|---|---|\n| x | 10 |\n| y | 20 |\n\n① ② ③"
    table_block = "| a | b |\n|---|---|\n| x | 10 |\n| y | 20 |\n"
    new_block = apply_numbers_removed(table_block, "")
    assert new_block == "[자료: x, y 중 y가 가장 크고 x가 가장 작다.]"
    result = replace_figure_block(raw, new_block)
    assert result == f"1. 표 문제입니다.\n\n{new_block}\n① ② ③"


# ── replace_figure_block — 블록 바깥 텍스트 보존 ──────────────────────────

def test_replace_figure_block_preserves_surrounding_text():
    raw = "3. 문제입니다.\n\n[자료: A 820, B 450]\n\n① 1 ② 2"
    result = replace_figure_block(raw, "[자료: A 492, B 450]")
    assert result == "3. 문제입니다.\n\n[자료: A 492, B 450]\n\n① 1 ② 2"


def test_replace_figure_block_markdown_table():
    raw = "1. 표 문제\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n① ② ③"
    result = replace_figure_block(raw, "| a | b |\n|---|---|\n| 9 | 9 |\n")
    assert result.startswith("1. 표 문제\n\n| a | b |\n|---|---|\n| 9 | 9 |\n")
    assert result.endswith("① ② ③")


# ── make_defect_items — 유형별 개수·변환 불가 대체·블라인드 ───────────────

def _source(fid, model="gpt-4o-mini", summary=_POP_SUMMARY, fig_text="[자료: A 820명, B 450명, C 610명, D 300명]"):
    return {
        "figure_id": fid, "model": model, "figure_summary": summary,
        "raw_output": f"문제 {fid}\n\n{fig_text}\n\n선택지",
        "figure_description": fig_text,
    }


def test_make_defect_items_counts_per_type_and_originals():
    sources = [_source(f"f{i:02d}") for i in range(1, 21)]  # 20개 소스 — 5x3+5 필요량보다 충분
    items = make_defect_items(sources, seed=0, per_type=3, n_originals=5)
    from collections import Counter
    counts = Counter(it["defect_type"] for it in items)
    assert counts["original"] == 5
    for dtype in ("key_value_changed", "minor_value_changed", "item_dropped", "unit_removed", "numbers_removed"):
        assert counts[dtype] == 3
    assert len(items) == 5 * 3 + 5


def test_make_defect_items_skips_untransformable_source():
    # f01은 key_value_changed가 불가능한 소스(summary에 극값 패턴 없음) — 다른 소스로 대체돼야 한다.
    bad = _source("f01", summary="극값 표현이 없는 그냥 설명")
    good = [_source(f"f{i:02d}") for i in range(2, 6)]
    items = make_defect_items([bad] + good, seed=0, per_type=1, n_originals=0)
    kv_items = [it for it in items if it["defect_type"] == "key_value_changed"]
    assert len(kv_items) == 1
    assert kv_items[0]["figure_id"] != "f01"


def test_make_defect_items_different_figures_preferred():
    sources = [_source(f"f{i:02d}") for i in range(1, 41)]  # per_type*5 + originals = 40 — 정확히 소진
    items = make_defect_items(sources, seed=0, per_type=6, n_originals=10)
    figure_ids = [it["figure_id"] for it in items]
    assert len(set(figure_ids)) == len(figure_ids)  # 소스가 충분하면 전부 다른 그림


def test_make_defect_items_assigns_blind_sequential_defect_ids():
    sources = [_source(f"f{i:02d}") for i in range(1, 11)]
    items = make_defect_items(sources, seed=0, per_type=1, n_originals=2)
    ids = sorted(it["defect_id"] for it in items)
    assert ids == [f"d{i:02d}" for i in range(1, len(items) + 1)]


# ── cmd_make_defects — 라벨 보호·블라인드(시트에 defect_type·model 없음) ──

def test_cmd_make_defects_refuses_to_overwrite_filled_sheet(tmp_path, monkeypatch, capsys):
    sheet_path = tmp_path / "defect_sheet.json"
    data = {"문항": [{"id": "d01", "라벨": {"점수": 3, "근거": "근거"}}]}
    sheet_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(mod, "DEFECT_SHEET_PATH", str(sheet_path))

    with pytest.raises(SystemExit):
        cmd_make_defects(argparse.Namespace(seed=0, per_type=6, originals=10))
    assert "덮어쓰지 않습니다" in capsys.readouterr().out


def test_cmd_make_defects_sheet_is_blind_and_items_file_has_metadata(tmp_path, monkeypatch):
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    monkeypatch.setattr(mod, "GOLDEN_DIR", str(golden_dir))
    monkeypatch.setattr(mod, "DEFECT_SHEET_PATH", str(golden_dir / "defect_sheet.json"))
    monkeypatch.setattr(mod, "DEFECT_ITEMS_PATH", str(golden_dir / "defect_items.json"))
    monkeypatch.setattr(
        mod, "load_defect_sources", lambda: [_source(f"f{i:02d}") for i in range(1, 11)],
    )

    cmd_make_defects(argparse.Namespace(seed=0, per_type=1, originals=2))

    sheet = json.loads((golden_dir / "defect_sheet.json").read_text(encoding="utf-8"))
    for item in sheet["문항"]:
        assert set(item) == {"라벨", "id", "핵심사실", "VLM_서술_줄"}  # defect_type·model 없음(블라인드)
        assert item["라벨"] == {"점수": None, "근거": ""}

    items = json.loads((golden_dir / "defect_items.json").read_text(encoding="utf-8"))["items"]
    assert {it["defect_type"] for it in items} >= {"original", "key_value_changed"}
    sheet_ids = {item["id"] for item in sheet["문항"]}
    assert sheet_ids == {it["defect_id"] for it in items}  # 시트 id는 defect_id(블라인드)


# ── reliability-defects — 구간별·유형별 집계 ──────────────────────────────

def test_judge_by_human_score_bucket_counts_and_mae():
    rows = [
        {"human": 5, "judges": {"j1": [5]}},
        {"human": 5, "judges": {"j1": [4]}},
        {"human": 3, "judges": {"j1": [3]}},
        {"human": 2, "judges": {"j1": [4]}},
    ]
    bucket = judge_by_human_score_bucket(rows, "j1")
    assert bucket[5] == {"n": 2, "judge_mean": 4.5, "mae": 0.5}
    assert bucket[3] == {"n": 1, "judge_mean": 3.0, "mae": 0.0}
    assert bucket[2] == {"n": 1, "judge_mean": 4.0, "mae": 2.0}
    assert bucket[1] == {"n": 0, "judge_mean": None, "mae": None}
    assert bucket[4] == {"n": 0, "judge_mean": None, "mae": None}


def test_defect_type_breakdown_detect_rate_against_original_baseline():
    rows = [
        {"defect_type": "original", "human": 5, "judges": {"j1": [5]}},
        {"defect_type": "original", "human": 5, "judges": {"j1": [5]}},
        {"defect_type": "item_dropped", "human": 2, "judges": {"j1": [5]}},   # 검출 못함(원본 평균 5와 같음)
        {"defect_type": "item_dropped", "human": 2, "judges": {"j1": [2]}},  # 검출함
    ]
    breakdown = defect_type_breakdown(rows, "j1")
    assert breakdown["original"]["judge_mean"] == 5.0
    assert breakdown["original"]["detect_rate"] is None
    assert breakdown["item_dropped"]["n"] == 2
    assert breakdown["item_dropped"]["human_mean"] == 2.0
    assert breakdown["item_dropped"]["detect_rate"] == 0.5  # 2건 중 1건만 원본 평균보다 낮음
    assert breakdown["minor_value_changed"]["n"] == 0


def test_build_defect_reliability_rows_joins_and_drops_none_scores():
    human_labels = {"d01": {"점수": 2, "근거": "근거"}, "d02": {"점수": None, "근거": ""}}
    defect_items = [
        {"defect_id": "d01", "figure_id": "f03", "defect_type": "item_dropped"},
        {"defect_id": "d02", "figure_id": "f05", "defect_type": "original"},
    ]
    judged_by_judge = {"j1": [
        {"id": "d01", "judge_score": 2},
        {"id": "d01", "judge_score": 3},
        {"id": "d02", "judge_score": 5},
    ]}
    rows = build_defect_reliability_rows(human_labels, defect_items, judged_by_judge)
    assert len(rows) == 1  # d02는 사람 점수가 없어 제외
    row = rows[0]
    assert row == {"id": "d01", "figure_id": "f03", "defect_type": "item_dropped", "human": 2, "judges": {"j1": [2, 3]}}


def test_compute_defect_reliability_clusters_shared_figure_across_sets():
    # f03이 기존 90건 셋(old_rows)과 결함셋(defect_rows) 모두에 등장 — 그림 단위로 같은
    # 클러스터로 묶여야 한다(한쪽 집합의 중복 표본처럼 취급되면 안 됨).
    old_rows = [
        {"id": "f03", "model": "gemini-3.8-flash", "human": 5, "judges": {"j1": [5]}},
        {"id": "f09", "model": "gpt-4o-mini", "human": 5, "judges": {"j1": [5]}},
    ]
    defect_rows = [
        {"id": "d01", "figure_id": "f03", "defect_type": "item_dropped", "human": 2, "judges": {"j1": [2]}},
        {"id": "d02", "figure_id": "f15", "defect_type": "unit_removed", "human": 3, "judges": {"j1": [3]}},
    ]
    report = compute_defect_reliability(old_rows, defect_rows, ["j1"])
    assert report["n_defect"] == 2
    assert report["n_combined"] == 4
    assert report["defect_only"]["j1"]["n"] == 2
    assert report["combined"]["j1"]["n"] == 4
    assert report["by_human_score"]["j1"][2]["n"] == 1
    assert report["by_defect_type"]["j1"]["item_dropped"]["n"] == 1
