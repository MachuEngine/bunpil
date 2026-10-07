"""experiments/ablate_retrieval.py 순수 로직 + no_retrieval 패치 단위 테스트. LLM·네트워크 없음.

가짜 run 레코드(golden_gen/gen_item_quality_golden.py `_run_passage()`가 만드는 형태 +
이 실험이 덧붙이는 condition/repeat_index/search_standards_calls 필드)와 가짜 judged
레코드만 사용한다. no_retrieval 패치 테스트는 app.modules.exam.tools의 실제
search_standards 도구를 호출해 get_store/get_retriever를 monkeypatch로 가짜로
바꾼 뒤 반환값을 확인한다(도구 자체는 수정하지 않는다는 설계 확인용)."""
import sys

from experiments.ablate_retrieval import (
    CountingRetriever,
    _EmptyRetriever,
    apply_retrieval_patch,
    average_across_repeats,
    classify_search_standards_output,
    compute_comparison,
    judged_item_keys,
    passage_level_attempts,
    passage_level_count_ok,
    passage_level_gate_pass,
    passage_level_judge_scores,
    passage_level_latency,
    passage_level_search_calls,
    passage_level_standard_code,
    passage_level_standard_filled,
    rule_metrics_for_condition,
    select_entries,
    standard_code_rate_for_items,
    standard_filled_rate_for_items,
    _judged_path,
    _run_path,
)


# ── classify_search_standards_output ─────────────────────────────────────

def test_classify_empty_store():
    assert classify_search_standards_output("교육과정 성취기준 자료 없음") == "자료없음"


def test_classify_empty_results():
    assert classify_search_standards_output("관련 성취기준 없음") == "결과없음"


def test_classify_has_content():
    assert classify_search_standards_output("[1] 민주주의는...") == "결과있음"


# ── CountingRetriever / _EmptyRetriever ──────────────────────────────────

def test_empty_retriever_always_returns_empty_list():
    assert _EmptyRetriever().retrieve("질의") == []


def test_counting_retriever_counts_calls_and_empty_results():
    class _Fake:
        def __init__(self, results):
            self._results = results

        def retrieve(self, *a, **k):
            return self._results

    counting = CountingRetriever(_Fake([]))
    counting.retrieve("질의1")
    counting.retrieve("질의2")
    assert counting.calls == 2
    assert counting.empty == 2

    counting2 = CountingRetriever(_Fake([{"text": "x"}]))
    counting2.retrieve("질의")
    assert counting2.calls == 1
    assert counting2.empty == 0


def test_counting_retriever_reset_clears_counts():
    counting = CountingRetriever(_EmptyRetriever())
    counting.retrieve("질의")
    counting.reset()
    assert counting.calls == 0
    assert counting.empty == 0


# ── apply_retrieval_patch (가짜 tools_module) ────────────────────────────

class _FakeToolsModule:
    get_retriever = None


class _FakeRealRetriever:
    def __init__(self, results=None):
        self.retrieve_calls = 0
        self._results = results if results is not None else [{"text": "실제 검색 결과"}]

    def retrieve(self, *a, **k):
        self.retrieve_calls += 1
        return self._results


def test_apply_retrieval_patch_no_retrieval_forces_empty():
    tools_mod = _FakeToolsModule()
    real = _FakeRealRetriever()
    counting = apply_retrieval_patch(tools_mod, "no_retrieval", lambda: real)

    result = tools_mod.get_retriever().retrieve("질의")

    assert result == []
    assert counting.calls == 1
    assert counting.empty == 1
    assert real.retrieve_calls == 0  # no_retrieval은 real_get_retriever를 호출만 하지 않음(내부가 가짜)


def test_apply_retrieval_patch_normal_passes_through_and_counts():
    tools_mod = _FakeToolsModule()
    real = _FakeRealRetriever()
    counting = apply_retrieval_patch(tools_mod, "normal", lambda: real)

    result = tools_mod.get_retriever().retrieve("질의")

    assert result == [{"text": "실제 검색 결과"}]
    assert counting.calls == 1
    assert counting.empty == 0
    assert real.retrieve_calls == 1


# ── no_retrieval 패치가 실제 search_standards 도구에 적용되는지 ─────────────
# app.modules.exam.tools는 수정하지 않는다 — get_retriever/get_store만 바꿔서
# 도구의 분기(저장 데이터 없음 vs 검색 결과 없음)를 재현한다.

class _FakeStoreWithData:
    def count(self, name: str) -> int:
        return 42  # standards 컬렉션에 데이터가 있다고 가정


def test_no_retrieval_patch_makes_real_tool_return_no_result_message(monkeypatch):
    tools_mod = sys.modules.get("app.modules.exam.tools")
    if tools_mod is None:
        import app.modules.exam.tools as tools_mod  # noqa: F401 (최초 import)
        tools_mod = sys.modules["app.modules.exam.tools"]

    monkeypatch.setattr(tools_mod, "get_store", lambda: _FakeStoreWithData())
    original_get_retriever = tools_mod.get_retriever
    try:
        counting = apply_retrieval_patch(tools_mod, "no_retrieval", lambda: _FakeRealRetriever())
        result = tools_mod.search_standards.invoke({"query": "민주주의"})
    finally:
        tools_mod.get_retriever = original_get_retriever

    assert result == "관련 성취기준 없음"
    assert classify_search_standards_output(result) == "결과없음"
    assert counting.calls == 1
    assert counting.empty == 1


def test_normal_patch_makes_real_tool_return_content(monkeypatch):
    tools_mod = sys.modules["app.modules.exam.tools"]

    monkeypatch.setattr(tools_mod, "get_store", lambda: _FakeStoreWithData())
    original_get_retriever = tools_mod.get_retriever
    try:
        fake_hit = [{"text": "성취기준: 민주주의의 의미와 발전 과정을 이해한다."}]
        counting = apply_retrieval_patch(tools_mod, "normal", lambda: _FakeRealRetriever(fake_hit))
        result = tools_mod.search_standards.invoke({"query": "민주주의"})
    finally:
        tools_mod.get_retriever = original_get_retriever

    assert classify_search_standards_output(result) == "결과있음"
    assert counting.calls == 1
    assert counting.empty == 0


# ── select_entries ────────────────────────────────────────────────────────

def test_select_entries_returns_all_without_ids():
    entries = [{"id": "p1"}, {"id": "p2"}]
    assert select_entries(entries) == entries


def test_select_entries_filters_by_ids():
    entries = [{"id": "p1"}, {"id": "p2"}, {"id": "p3"}]
    result = select_entries(entries, ids=["p1", "p3"])
    assert [e["id"] for e in result] == ["p1", "p3"]


# ── judged_item_keys / 경로 헬퍼 ──────────────────────────────────────────

def test_judged_item_keys_builds_skip_set():
    records = [
        {"run_id": "p1", "item_id": "it1"},
        {"run_id": "p1", "item_id": "it2"},
        {"run_id": "p2", "item_id": "it1"},
    ]
    keys = judged_item_keys(records)
    assert keys == {("p1", "it1"), ("p1", "it2"), ("p2", "it1")}


def test_run_path_and_judged_path_naming():
    assert _run_path("normal", 1).endswith("normal_r1.jsonl")
    assert _run_path("no_retrieval", 2).endswith("no_retrieval_r2.jsonl")
    judged = _judged_path("anthropic/claude-sonnet-5.5", "normal", 1)
    assert judged.endswith("judged_anthropic__claude-sonnet-5.5_normal_r1.jsonl")


# ── standard 필드 지표 ────────────────────────────────────────────────────

def _item(item_id="it1", standard=""):
    return {
        "item_id": item_id, "item_type": "객관식", "question": "충분히 긴 질문입니다",
        "options": ["①", "②", "③", "④"], "answer": "①", "standard": standard,
    }


def test_standard_filled_rate_counts_non_empty_only():
    items = [_item("it1", standard="경제 주체의 합리적 선택"), _item("it2", standard=""), _item("it3", standard="  ")]
    assert standard_filled_rate_for_items(items) == 1 / 3


def test_standard_filled_rate_none_when_no_items():
    assert standard_filled_rate_for_items([]) is None


def test_standard_code_rate_detects_code_like_pattern():
    items = [
        _item("it1", standard="9사01-01"),  # 코드처럼 보임
        _item("it2", standard="민주주의와 법"),  # 모델이 지은 주제명
        _item("it3", standard=""),
    ]
    assert standard_code_rate_for_items(items) == 1 / 3


def test_standard_code_rate_none_when_no_items():
    assert standard_code_rate_for_items([]) is None


# ── passage_level_* ──────────────────────────────────────────────────────

def _run(id_, items=None, num_items=None, attempts=1, validation_passed=True,
          wall_clock=1.0, search_calls=2, error=None):
    items = items if items is not None else []
    return {
        "id": id_, "model": "gpt-6-luna",
        "num_items": num_items if num_items is not None else len(items),
        "format": "mc4", "items": items, "validation_passed": validation_passed,
        "attempts": attempts, "attempt_log": [{"attempt": 1}],
        "search_standards_calls": search_calls, "search_standards_empty_calls": 0,
        "wall_clock_sec": wall_clock, "error": error,
    }


def test_passage_level_gate_pass_and_count_ok():
    records = [
        _run("p1", items=[_item("a"), _item("b")], num_items=2, validation_passed=True),
        _run("p2", items=[_item("a")], num_items=2, validation_passed=False),
    ]
    assert passage_level_gate_pass(records) == {"p1": 1, "p2": 0}
    assert passage_level_count_ok(records) == {"p1": 1, "p2": 0}


def test_passage_level_attempts_excludes_error_runs():
    records = [_run("p1", attempts=2), _run("p2", attempts=3, error={"type": "RuntimeError"})]
    assert passage_level_attempts(records) == {"p1": 2}


def test_passage_level_latency_and_search_calls():
    records = [_run("p1", wall_clock=12.5, search_calls=3)]
    assert passage_level_latency(records) == {"p1": 12.5}
    assert passage_level_search_calls(records) == {"p1": 3}


def test_passage_level_standard_filled_and_code():
    records = [
        _run("p1", items=[_item("a", standard="9사01-01"), _item("b", standard="")], num_items=2),
        _run("p2", items=[], num_items=0),  # items 없음 -> 두 함수 모두 제외
    ]
    assert passage_level_standard_filled(records) == {"p1": 0.5}
    assert passage_level_standard_code(records) == {"p1": 0.5}


def test_passage_level_judge_scores_averages_multiple_items_per_passage():
    judged_records = [
        _judged("p1", "it1", 정답유일성=5),
        _judged("p1", "it2", 정답유일성=3),
        _judged("p2", "it1", 정답유일성=4),
    ]
    scores = passage_level_judge_scores(judged_records, "정답유일성")
    assert scores == {"p1": 4.0, "p2": 4.0}


# ── average_across_repeats ───────────────────────────────────────────────

def test_average_across_repeats_only_common_passages():
    r1 = {"p1": 4.0, "p2": 2.0}
    r2 = {"p1": 6.0, "p3": 9.0}
    assert average_across_repeats(r1, r2) == {"p1": 5.0}


# ── rule_metrics_for_condition ───────────────────────────────────────────

def test_rule_metrics_for_condition_combines_two_repeats():
    r1 = [_run("p1", validation_passed=True, attempts=1, wall_clock=2.0, search_calls=1)]
    r2 = [_run("p1", validation_passed=False, attempts=3, wall_clock=4.0, search_calls=3)]
    m = rule_metrics_for_condition(r1, r2)
    assert m["n"] == 2
    assert m["gate_pass_rate"] == 0.5
    assert m["avg_attempts"] == 2.0
    assert m["avg_latency_sec"] == 3.0
    assert m["avg_search_standards_calls"] == 2.0


# ── compute_comparison (핵심 통합 테스트) ────────────────────────────────

def _judged(run_id, item_id, 정답유일성=5, 오답매력도=5, 근거성=5):
    overall = round((정답유일성 + 오답매력도 + 근거성) / 3, 2)
    return {
        "run_id": run_id, "item_id": item_id,
        "scores": {
            "정답유일성": 정답유일성, "오답매력도": 오답매력도, "근거성": 근거성,
            "overall": overall, "parse_failed": False,
        },
    }


_PASSAGE_IDS = ["p1", "p2", "p3", "p4"]


def _make_records(condition: str, repeat: int, validation_passed: bool) -> list[dict]:
    return [
        _run(pid, items=[_item("it1")], num_items=1, validation_passed=validation_passed)
        for pid in _PASSAGE_IDS
    ]


def test_compute_comparison_detects_consistent_quality_gap():
    # no_retrieval이 모든 지문·반복에서 정답유일성을 normal보다 정확히 2점 낮게 받는
    # 가짜 데이터 — 지문 단위 분산이 0이라 CI가 좁아 "판정 불가"가 아니어야 한다.
    records_by_key = {
        ("normal", 1): _make_records("normal", 1, True),
        ("normal", 2): _make_records("normal", 2, True),
        ("no_retrieval", 1): _make_records("no_retrieval", 1, True),
        ("no_retrieval", 2): _make_records("no_retrieval", 2, True),
    }
    judged_by_key = {
        ("normal", 1): [_judged(pid, "it1", 정답유일성=5) for pid in _PASSAGE_IDS],
        ("normal", 2): [_judged(pid, "it1", 정답유일성=5) for pid in _PASSAGE_IDS],
        ("no_retrieval", 1): [_judged(pid, "it1", 정답유일성=3) for pid in _PASSAGE_IDS],
        ("no_retrieval", 2): [_judged(pid, "it1", 정답유일성=3) for pid in _PASSAGE_IDS],
    }

    report = compute_comparison(records_by_key, judged_by_key, "anthropic/claude-sonnet-5.5")

    pw = report["quality_pairwise_condition"]["정답유일성"]
    assert pw["n"] == 4
    assert pw["mean_diff"] == -2.0
    assert pw["ci"][0] is not None and pw["ci"][1] is not None
    assert pw["ci"][1] < 0  # CI가 0을 포함하지 않음 -> 판정 가능
    assert "normal" in pw["verdict"]


def test_compute_comparison_repeat_noise_is_inconclusive_when_identical():
    records_by_key = {
        (c, r): _make_records(c, r, True) for c in ("normal", "no_retrieval") for r in (1, 2)
    }
    # 반복 1·2가 완전히 동일한 점수 -> 반복 간 차이는 0, CI도 0을 포함해야 함.
    judged_by_key = {
        (c, r): [_judged(pid, "it1", 정답유일성=4) for pid in _PASSAGE_IDS]
        for c in ("normal", "no_retrieval") for r in (1, 2)
    }

    report = compute_comparison(records_by_key, judged_by_key, "anthropic/claude-sonnet-5.5")

    noise = report["repeat_noise"]["normal"]["정답유일성"]
    assert noise["mean_diff"] == 0.0
    assert "판정 불가" in noise["verdict"]


def test_compute_comparison_includes_groundedness_reference_note():
    records_by_key = {
        (c, r): _make_records(c, r, True) for c in ("normal", "no_retrieval") for r in (1, 2)
    }
    judged_by_key = {
        (c, r): [_judged(pid, "it1") for pid in _PASSAGE_IDS]
        for c in ("normal", "no_retrieval") for r in (1, 2)
    }
    report = compute_comparison(records_by_key, judged_by_key, "anthropic/claude-sonnet-5.5")
    assert "κ" in report["groundedness_note"]
    assert "κ" in report["metric_descriptions"]["근거성"]


# ── judge 캐시 스킵(이어서 실행) ─────────────────────────────────────────

def test_judge_cache_skip_logic_via_judged_item_keys():
    cached = [{"run_id": "p1", "item_id": "it1"}]
    done = judged_item_keys(cached)
    todo = [("p1", "it1"), ("p1", "it2"), ("p2", "it1")]
    remaining = [k for k in todo if k not in done]
    assert remaining == [("p1", "it2"), ("p2", "it1")]
