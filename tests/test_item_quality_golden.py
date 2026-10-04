"""golden_gen/gen_item_quality_golden.py — LLM 호출 없는 순수 로직 단위 테스트.

generate 서브커맨드가 실제로 모델을 호출하는 부분(agent_node의 LLM 응답 등)은 검증하기
어려워 제외하고, run 결과 파싱·집계·build-labelset 로직은 가짜 run jsonl(임시 디렉터리)로
검증한다. 다만 `_run_passage`의 `init_session()` → `graph.stream()` 호출 **순서**는
LLM 호출 없이 graph.stream만 가짜로 대체해 확인할 수 있어 포함한다(아래
`test_run_passage_calls_init_session_before_graph_stream`)."""
import json
import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from golden_gen.gen_item_quality_golden import (
    _has_existing_labels,
    _has_existing_sheet_labels,
    _run_passage,
    _summarize_agent_messages,
    apply_sheet_labels,
    build_label_sheet,
    build_label_sheet_entry,
    build_labelset,
    dedupe_run_records,
    load_run_records,
    select_entries,
    summarize_model,
    validate_sheet,
    validate_sheet_label,
)


# ── select_entries ───────────────────────────────────────────────────────

_ENTRIES = [
    {"id": "a1", "format": "mc4", "case_type": "normal"},
    {"id": "a2", "format": "mc4", "case_type": "hard"},
    {"id": "b1", "format": "mc5", "case_type": "hard"},
    {"id": "b2", "format": "mc5", "case_type": "normal"},
    {"id": "c1", "format": "bogi_combo", "case_type": "normal"},
    {"id": "d1", "format": "data", "case_type": "normal"},
]


def test_select_entries_smoke_picks_one_per_format_normal():
    selected = select_entries(_ENTRIES, smoke=True)
    assert [e["id"] for e in selected] == ["a1", "b2", "c1", "d1"]


def test_select_entries_ids_filter():
    selected = select_entries(_ENTRIES, ids=["a2", "d1"])
    assert {e["id"] for e in selected} == {"a2", "d1"}


def test_select_entries_limit():
    selected = select_entries(_ENTRIES, limit=2)
    assert [e["id"] for e in selected] == ["a1", "a2"]


def test_select_entries_default_is_all():
    assert select_entries(_ENTRIES) == _ENTRIES


# ── _summarize_agent_messages ────────────────────────────────────────────

def _ai(tool_calls=None, usage=None):
    kwargs = {"content": "", "tool_calls": tool_calls or []}
    if usage is not None:
        kwargs["usage_metadata"] = usage
    return AIMessage(**kwargs)


def test_summarize_agent_messages_counts_tool_calls_and_tokens():
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="위 지침에 따라 문항을 작성하세요."),
        _ai(
            tool_calls=[{"name": "save_item", "args": {}, "id": "1"}],
            usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        ),
        ToolMessage(content="ok", tool_call_id="1"),
        _ai(
            tool_calls=[
                {"name": "save_item", "args": {}, "id": "2"},
                {"name": "submit_for_review", "args": {}, "id": "3"},
            ],
            usage={"input_tokens": 150, "output_tokens": 30, "total_tokens": 180},
        ),
        ToolMessage(content="ok", tool_call_id="2"),
        ToolMessage(content="ok", tool_call_id="3"),
    ]
    result = _summarize_agent_messages(messages)
    assert result["tool_calls"] == {"save_item": 2, "submit_for_review": 1}
    assert result["malformed_retries"] == 0
    assert result["incomplete_retries"] == 0
    assert result["input_tokens"] == 250
    assert result["output_tokens"] == 50


def test_summarize_agent_messages_counts_malformed_and_incomplete_retries():
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="위 지침에 따라 문항을 작성하세요."),  # 최초 지시 — 집계 제외
        _ai(),
        HumanMessage(content="도구 호출 형식이 손상되었습니다. 설명 없이 다시 시도하세요."),
        _ai(),
        HumanMessage(content="아직 목표 문항 저장과 제출이 끝나지 않았습니다. 다시 시도하세요."),
        _ai(tool_calls=[{"name": "submit_for_review", "args": {}, "id": "9"}]),
    ]
    result = _summarize_agent_messages(messages)
    assert result["malformed_retries"] == 1
    assert result["incomplete_retries"] == 1
    assert result["input_tokens"] is None
    assert result["output_tokens"] is None


def test_summarize_agent_messages_missing_usage_metadata_is_null():
    messages = [HumanMessage(content="위 지침에 따라 문항을 작성하세요."), _ai()]
    result = _summarize_agent_messages(messages)
    assert result["input_tokens"] is None
    assert result["output_tokens"] is None


# ── _run_passage: init_session()이 graph.stream()보다 먼저 호출되는지 ──────
#
# app/modules/exam/tools.py init_session() 주석: LangGraph는 각 노드를 context.run()으로
# 격리 실행하므로, plan_node 안에서만 set()하면 그 dict가 agent_node로 전파되지 않는다.
# app/main.py _run_exam·_run_exam_events는 이 때문에 graph 실행 전에 같은 스레드에서
# init_session()을 먼저 부른다 — _run_passage도 같은 패턴을 따라야 한다. graph.stream()을
# 가짜로 바꿔 "stream이 시작되는 시점"에 get_draft_items()가 예외 없이 동작하는지로
# 호출 순서를 확인한다(실제 모델 호출 없음). tests/conftest.py의 autouse 픽스처가 매
# 테스트 전 `_request_ctx`를 빈 dict로 리셋해두므로, init_session()이 먼저 불리지
# 않으면 get_draft_items()의 ctx["items"] 접근이 KeyError로 바로 드러난다.
def test_run_passage_calls_init_session_before_graph_stream(monkeypatch):
    import app.main
    import app.modules.exam
    from app.modules.exam.tools import get_draft_items

    captured = {}

    async def fake_build_spec(passage_text):
        return {"passage_text": passage_text, "num_items": 1, "format": {}}, False, []

    class FakeGraph:
        def stream(self, state, stream_mode="updates"):
            # _run_passage가 init_session()을 먼저 불렀다면 여기서 get_draft_items()가
            # KeyError 없이 빈 리스트를 반환한다.
            captured["items_at_stream_start"] = get_draft_items()
            return iter([])  # 빈 스트림 — agent/judge/validate 노드 없이 바로 종료

    monkeypatch.setattr(app.main, "_build_spec", fake_build_spec)
    monkeypatch.setattr(app.modules.exam, "get_exam_graph", lambda: FakeGraph())

    record = _run_passage("qwen2.5-14b", {"id": "p1", "passage_text": "지문"})

    assert captured["items_at_stream_start"] == []
    assert record["items"] == []
    assert record["attempts"] == 0


# ── load_run_records / summarize_model ───────────────────────────────────

def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_load_run_records_missing_file_returns_empty(tmp_path):
    assert load_run_records(str(tmp_path / "nope.jsonl")) == []


def test_load_run_records_roundtrip(tmp_path):
    path = tmp_path / "qwen.jsonl"
    records = [{"id": "p1", "error": None}, {"id": "p2", "error": None}]
    _write_jsonl(path, records)
    assert load_run_records(str(path)) == records


def test_summarize_model_aggregates_rates_and_errors():
    records = [
        {
            "id": "p1", "error": None, "num_items": 2,
            "items": [{"item_type": "객관식"}, {"item_type": "객관식"}],
            "validation_passed": True, "attempts": 1,
            "attempt_log": [{"malformed_retries": 1}],
            "wall_clock_sec": 10.0, "cost_usd_est": 0.01,
        },
        {
            "id": "p2", "error": None, "num_items": 2,
            "items": [{"item_type": "객관식"}],  # 개수 불일치
            "validation_passed": False, "attempts": 3,
            "attempt_log": [{"malformed_retries": 0}, {"malformed_retries": 2}, {"malformed_retries": 0}],
            "wall_clock_sec": 30.0, "cost_usd_est": 0.03,
        },
        {"id": "p3", "error": {"type": "RuntimeError", "message": "boom"}, "wall_clock_sec": 1.0},
    ]
    s = summarize_model(records)
    assert s["n_runs"] == 3
    assert s["n_errors"] == 1
    assert s["count_match_rate"] == pytest.approx(0.5)
    assert s["validate_rate"] == pytest.approx(0.5)
    assert s["avg_attempts"] == pytest.approx(2.0)
    assert s["malformed_total"] == 3
    assert s["avg_wall_clock_sec"] == pytest.approx((10.0 + 30.0 + 1.0) / 3)
    assert s["avg_cost_usd"] == pytest.approx(0.02)


def test_summarize_model_empty_records_no_zero_division():
    s = summarize_model([])
    assert s["n_runs"] == 0
    assert s["count_match_rate"] == 0.0
    assert s["validate_rate"] == 0.0
    assert s["avg_cost_usd"] is None


# ── dedupe_run_records ────────────────────────────────────────────────────
# gpt-6-luna pilot-006처럼 같은 id가 실패 후 재실행으로 두 번 append된 run을 정리한다.

def test_dedupe_run_records_keeps_last_success_when_failure_then_success():
    records = [
        {"id": "p1", "error": {"type": "BadRequestError"}},
        {"id": "p1", "error": None, "items": [_mc_item()]},
    ]
    result = dedupe_run_records(records)
    assert len(result) == 1
    assert result[0]["error"] is None


def test_dedupe_run_records_keeps_last_success_even_when_success_then_failure():
    # 성공이 먼저, 그 뒤에 실패 재실행이 append된 순서도 "마지막 성공"을 쓴다(단순
    # last-wins가 아님).
    records = [
        {"id": "p1", "error": None, "items": [_mc_item("a")]},
        {"id": "p1", "error": {"type": "BadRequestError"}},
    ]
    result = dedupe_run_records(records)
    assert len(result) == 1
    assert result[0]["error"] is None
    assert result[0]["items"][0]["question"] == "a"


def test_dedupe_run_records_keeps_last_record_when_all_failed():
    records = [
        {"id": "p1", "error": {"type": "A"}},
        {"id": "p1", "error": {"type": "B"}},
    ]
    result = dedupe_run_records(records)
    assert len(result) == 1
    assert result[0]["error"]["type"] == "B"


def test_dedupe_run_records_preserves_first_seen_order_and_passes_through_unique_ids():
    records = [
        {"id": "p2", "error": None},
        {"id": "p1", "error": None},
        {"id": "p2", "error": {"type": "A"}},  # p2 재실행(실패) — 기존 성공을 유지
    ]
    result = dedupe_run_records(records)
    assert [r["id"] for r in result] == ["p2", "p1"]
    assert result[0]["error"] is None  # p2는 첫 성공 레코드를 유지


def test_summarize_model_dedupes_duplicate_ids_before_counting():
    records = [
        {
            "id": "p1", "error": {"type": "BadRequestError"}, "wall_clock_sec": 1.0,
        },
        {
            "id": "p1", "error": None, "num_items": 1,
            "items": [{"item_type": "객관식"}], "validation_passed": True, "attempts": 1,
            "attempt_log": [{"malformed_retries": 0}], "wall_clock_sec": 5.0, "cost_usd_est": 0.01,
        },
        {
            "id": "p2", "error": None, "num_items": 1,
            "items": [{"item_type": "객관식"}], "validation_passed": True, "attempts": 1,
            "attempt_log": [{"malformed_retries": 0}], "wall_clock_sec": 3.0, "cost_usd_est": 0.01,
        },
    ]
    s = summarize_model(records)
    assert s["n_runs"] == 2  # p1 중복 제거 후 2건(33개 지문 기준이면 34 -> 33과 같은 맥락)
    assert s["n_errors"] == 0  # p1은 성공 레코드로 정리됨


# ── build_labelset ────────────────────────────────────────────────────────

_INPUTS = [
    {"id": "p1", "passage_text": "지문 1"},
    {"id": "p2", "passage_text": "지문 2"},
]


def _mc_item(q="질문", item_id="it1"):
    return {
        "item_id": item_id, "question": q, "stimulus": "", "options": ["①", "②", "③", "④"],
        "answer": "①", "item_type": "객관식", "difficulty": "중", "standard": "",
    }


def _essay_item(item_id="it2"):
    return {
        "item_id": item_id, "question": "서술형 질문", "stimulus": "", "options": [],
        "answer": "", "item_type": "서술형", "difficulty": "중", "standard": "",
    }


def test_build_labelset_picks_one_mc_item_per_passage_model():
    runs_by_model = {
        "qwen2.5-14b": [
            {"id": "p1", "error": None, "items": [_mc_item("q1a"), _mc_item("q1b")]},
            {"id": "p2", "error": None, "items": [_mc_item("q2a")]},
        ],
        "gpt-6-luna": [
            {"id": "p1", "error": None, "items": [_mc_item("q1c")]},
            {"id": "p2", "error": None, "items": [_mc_item("q2b")]},
        ],
    }
    entries, model_map, missing = build_labelset(_INPUTS, runs_by_model, seed=0)
    assert len(entries) == 4  # 2 지문 x 2 모델
    assert missing == []
    assert len(model_map) == 4
    # 블라인드: entries에 모델 정보가 없어야 한다
    for e in entries:
        assert "model" not in e
        assert set(e.keys()) == {"id", "passage_id", "passage_text", "item", "human_label"}
        assert e["human_label"] == {
            "정답유일성": None, "오답매력도": None, "근거성": None, "학생난이도": None,
            "cannot_judge": False, "reason": "",
        }
    # model_map으로만 모델을 알 수 있다
    for blind_id, mapping in model_map.items():
        assert mapping["model"] in ("qwen2.5-14b", "gpt-6-luna")
        assert mapping["passage_id"] in ("p1", "p2")


def test_build_labelset_missing_when_no_mc_item_or_run_missing():
    runs_by_model = {
        "qwen2.5-14b": [
            {"id": "p1", "error": None, "items": [_essay_item()]},  # 객관식 없음
            # p2 run 자체가 없음
        ],
    }
    entries, model_map, missing = build_labelset(_INPUTS, runs_by_model, seed=0)
    assert entries == []
    assert model_map == {}
    reasons = {(m["passage_id"], m["model"]): m["reason"] for m in missing}
    assert reasons[("p1", "qwen2.5-14b")] == "no_mc_item"
    assert reasons[("p2", "qwen2.5-14b")] == "run_missing_or_error"


def test_build_labelset_run_with_error_is_missing():
    runs_by_model = {
        "qwen2.5-14b": [
            {"id": "p1", "error": {"type": "RuntimeError", "message": "x"}, "items": []},
            {"id": "p2", "error": None, "items": [_mc_item()]},
        ],
    }
    entries, model_map, missing = build_labelset(_INPUTS, runs_by_model, seed=0)
    assert len(entries) == 1
    assert entries[0]["passage_id"] == "p2"
    assert any(m["passage_id"] == "p1" and m["reason"] == "run_missing_or_error" for m in missing)


def test_build_labelset_reproducible_with_same_seed():
    runs_by_model = {
        "qwen2.5-14b": [
            {"id": "p1", "error": None, "items": [_mc_item("a"), _mc_item("b"), _mc_item("c")]},
        ],
    }
    e1, m1, _ = build_labelset(_INPUTS[:1], runs_by_model, seed=0)
    e2, m2, _ = build_labelset(_INPUTS[:1], runs_by_model, seed=0)
    assert e1 == e2
    assert m1 == m2


# ── _has_existing_labels ──────────────────────────────────────────────────

def test_has_existing_labels_false_for_missing_file(tmp_path):
    assert _has_existing_labels(str(tmp_path / "nope.json")) is False


def test_has_existing_labels_false_when_all_null(tmp_path):
    path = tmp_path / "golden.json"
    data = {
        "entries": [
            {
                "id": "iq_001",
                "human_label": {
                    "정답유일성": None, "오답매력도": None, "근거성": None, "학생난이도": None,
                    "cannot_judge": False, "reason": "",
                },
            }
        ]
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert _has_existing_labels(str(path)) is False


def test_has_existing_labels_true_when_any_score_filled(tmp_path):
    path = tmp_path / "golden.json"
    data = {
        "entries": [
            {
                "id": "iq_001",
                "human_label": {
                    "정답유일성": 4, "오답매력도": None, "근거성": None, "학생난이도": None,
                    "cannot_judge": False, "reason": "",
                },
            }
        ]
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert _has_existing_labels(str(path)) is True


def test_has_existing_labels_true_when_student_difficulty_filled(tmp_path):
    path = tmp_path / "golden.json"
    data = {
        "entries": [
            {
                "id": "iq_001",
                "human_label": {
                    "정답유일성": None, "오답매력도": None, "근거성": None, "학생난이도": "중",
                    "cannot_judge": False, "reason": "",
                },
            }
        ]
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert _has_existing_labels(str(path)) is True


def test_has_existing_labels_true_when_cannot_judge_set(tmp_path):
    path = tmp_path / "golden.json"
    data = {
        "entries": [
            {
                "id": "iq_001",
                "human_label": {"정답유일성": None, "오답매력도": None, "근거성": None, "cannot_judge": True, "reason": "자료 부족"},
            }
        ]
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert _has_existing_labels(str(path)) is True


# ── build_label_sheet_entry / build_label_sheet (JSON 직접 편집용 시트) ─────

def _golden_entry(id_="iq_001", stimulus="", passage_text="1행\n2행"):
    return {
        "id": id_,
        "passage_id": "p1",
        "passage_text": passage_text,
        "item": {
            "question": "질문입니다",
            "stimulus": stimulus,
            "options": ["① 가", "② 나"],
            "answer": "① 가",
        },
        "human_label": {
            "정답유일성": None, "오답매력도": None, "근거성": None, "학생난이도": None,
            "cannot_judge": False, "reason": "",
        },
    }


def test_build_label_sheet_entry_key_order_label_first():
    entry = build_label_sheet_entry(_golden_entry())
    # 라벨 칸이 맨 위 — id, 라벨, 예시문제_줄, 발문, (보기_줄,) 선지, 표시된_정답 순서
    assert list(entry.keys()) == ["id", "라벨", "예시문제_줄", "발문", "선지", "표시된_정답"]
    assert entry["라벨"] == {
        "정답유일성": None, "오답매력도": None, "근거성": None, "학생난이도": None,
        "판단불가": False, "근거": "",
    }


def test_build_label_sheet_entry_splits_passage_text_into_lines():
    entry = build_label_sheet_entry(_golden_entry(passage_text="첫줄\n둘째줄\n셋째줄"))
    assert entry["예시문제_줄"] == ["첫줄", "둘째줄", "셋째줄"]


def test_build_label_sheet_entry_omits_stimulus_key_when_empty():
    entry = build_label_sheet_entry(_golden_entry(stimulus=""))
    assert "보기_줄" not in entry


def test_build_label_sheet_entry_splits_stimulus_when_present():
    entry = build_label_sheet_entry(_golden_entry(stimulus="자료1\n자료2"))
    assert entry["보기_줄"] == ["자료1", "자료2"]
    # 보기_줄도 선지보다 앞, 라벨보다는 뒤
    keys = list(entry.keys())
    assert keys.index("라벨") < keys.index("보기_줄") < keys.index("선지")


def test_build_label_sheet_wraps_entries_with_guide():
    sheet = build_label_sheet([_golden_entry("iq_001"), _golden_entry("iq_002")])
    assert list(sheet.keys()) == ["_안내", "문항"]
    assert isinstance(sheet["_안내"], list) and len(sheet["_안내"]) > 0
    assert [e["id"] for e in sheet["문항"]] == ["iq_001", "iq_002"]


# ── _has_existing_sheet_labels ───────────────────────────────────────────

def _empty_sheet_label():
    return {"정답유일성": None, "오답매력도": None, "근거성": None, "학생난이도": None, "판단불가": False, "근거": ""}


def test_has_existing_sheet_labels_false_for_missing_file(tmp_path):
    assert _has_existing_sheet_labels(str(tmp_path / "nope.json")) is False


def test_has_existing_sheet_labels_false_when_all_empty(tmp_path):
    path = tmp_path / "sheet.json"
    data = {"_안내": [], "문항": [{"id": "iq_001", "라벨": _empty_sheet_label()}]}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert _has_existing_sheet_labels(str(path)) is False


def test_has_existing_sheet_labels_true_when_score_filled(tmp_path):
    path = tmp_path / "sheet.json"
    label = _empty_sheet_label()
    label["정답유일성"] = 4
    data = {"_안내": [], "문항": [{"id": "iq_001", "라벨": label}]}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert _has_existing_sheet_labels(str(path)) is True


# ── validate_sheet_label / validate_sheet ────────────────────────────────

def test_validate_sheet_label_accepts_all_null():
    assert validate_sheet_label(_empty_sheet_label()) == []


def test_validate_sheet_label_accepts_valid_scores_with_reason():
    label = _empty_sheet_label()
    label.update({"정답유일성": 5, "오답매력도": 4, "근거성": 3, "학생난이도": "상"})
    assert validate_sheet_label(label) == []


def test_validate_sheet_label_rejects_out_of_range_score():
    label = _empty_sheet_label()
    label["정답유일성"] = 6
    errors = validate_sheet_label(label)
    assert any("정답유일성" in e for e in errors)


def test_validate_sheet_label_rejects_non_int_score():
    label = _empty_sheet_label()
    label["오답매력도"] = "4"
    errors = validate_sheet_label(label)
    assert any("오답매력도" in e for e in errors)


def test_validate_sheet_label_rejects_invalid_difficulty():
    label = _empty_sheet_label()
    label["학생난이도"] = "최상"
    errors = validate_sheet_label(label)
    assert any("학생난이도" in e for e in errors)


def test_validate_sheet_label_requires_reason_when_score_low():
    label = _empty_sheet_label()
    label["근거성"] = 1
    errors = validate_sheet_label(label)
    assert any("근거" in e for e in errors)


def test_validate_sheet_label_requires_reason_when_cannot_judge():
    label = _empty_sheet_label()
    label["판단불가"] = True
    errors = validate_sheet_label(label)
    assert any("근거" in e for e in errors)


def test_validate_sheet_label_passes_when_low_score_has_reason():
    label = _empty_sheet_label()
    label.update({"근거성": 2, "근거": "교육과정 밖 내용 포함"})
    assert validate_sheet_label(label) == []


def test_validate_sheet_flags_unknown_id():
    items = [{"id": "iq_999", "라벨": _empty_sheet_label()}]
    errors = validate_sheet(items, valid_ids={"iq_001"})
    assert len(errors) == 1
    assert errors[0][0] == "iq_999"
    assert any("id" in msg for msg in errors[0][1])


def test_validate_sheet_no_errors_when_all_valid():
    items = [
        {"id": "iq_001", "라벨": _empty_sheet_label()},
        {"id": "iq_002", "라벨": _empty_sheet_label()},
    ]
    assert validate_sheet(items, valid_ids={"iq_001", "iq_002"}) == []


# ── apply_sheet_labels ────────────────────────────────────────────────────

def test_apply_sheet_labels_maps_keys_to_human_label():
    golden_entries = [_golden_entry("iq_001")]
    label = {
        "정답유일성": 5, "오답매력도": 4, "근거성": 3, "학생난이도": "중",
        "판단불가": False, "근거": "",
    }
    updated, incomplete = apply_sheet_labels(golden_entries, [{"id": "iq_001", "라벨": label}])
    assert updated[0]["human_label"] == {
        "정답유일성": 5, "오답매력도": 4, "근거성": 3, "학생난이도": "중",
        "cannot_judge": False, "reason": "",
    }
    assert incomplete == 0
    # 다른 필드는 그대로
    assert updated[0]["passage_id"] == "p1"


def test_apply_sheet_labels_maps_cannot_judge_and_reason():
    golden_entries = [_golden_entry("iq_001")]
    label = {
        "정답유일성": None, "오답매력도": None, "근거성": None, "학생난이도": None,
        "판단불가": True, "근거": "그림이 깨져서 판단 불가",
    }
    updated, incomplete = apply_sheet_labels(golden_entries, [{"id": "iq_001", "라벨": label}])
    assert updated[0]["human_label"]["cannot_judge"] is True
    assert updated[0]["human_label"]["reason"] == "그림이 깨져서 판단 불가"
    assert incomplete == 1  # 점수 3종이 모두 null


def test_apply_sheet_labels_leaves_entry_untouched_when_not_in_sheet():
    golden_entries = [_golden_entry("iq_001"), _golden_entry("iq_002")]
    label = {"정답유일성": 5, "오답매력도": 5, "근거성": 5, "학생난이도": "상", "판단불가": False, "근거": ""}
    updated, incomplete = apply_sheet_labels(golden_entries, [{"id": "iq_001", "라벨": label}])
    assert updated[1]["human_label"]["정답유일성"] is None  # iq_002는 시트에 없어 그대로
    assert incomplete == 0


def test_apply_sheet_labels_partial_progress_counts_incomplete():
    golden_entries = [_golden_entry("iq_001"), _golden_entry("iq_002")]
    sheet_items = [
        {"id": "iq_001", "라벨": {
            "정답유일성": 5, "오답매력도": 5, "근거성": 5, "학생난이도": "상",
            "판단불가": False, "근거": "",
        }},
        {"id": "iq_002", "라벨": {
            "정답유일성": 4, "오답매력도": None, "근거성": None, "학생난이도": None,
            "판단불가": False, "근거": "",
        }},
    ]
    updated, incomplete = apply_sheet_labels(golden_entries, sheet_items)
    assert incomplete == 1  # iq_002만 미완료
    assert updated[0]["human_label"]["정답유일성"] == 5
    assert updated[1]["human_label"]["오답매력도"] is None
