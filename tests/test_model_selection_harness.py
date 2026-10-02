"""모델 선정 하네스(evals/model_selection) 테스트 — 네트워크·유료 호출 없음.

OpenRouter 정식 비교 조건(모델·endpoint 고정, fallback 금지, require_parameters, context compression 끔,
quantization 고정, 민감 모드의 data_collection/zdr)이 요청에 실제로 들어가는지, 요청과 다른 경로로 처리된
결과가 집계에서 빠지는지, 인증 키가 결과 파일에 남지 않는지를 확인한다.
"""
import json
import os

import pytest

from evals.model_selection import graders, metrics, openrouter
from evals.model_selection.adapters import OpenRouterAdapter, make_adapter
from evals.model_selection.judge_prompts import anonymize, parse_json, swap_winner
from evals.model_selection.recorder import CallRecord, Recorder
from evals.model_selection.run import load_config, main as run_main

CFG = load_config()


def test_request_pins_model_endpoint_and_disables_fallbacks(tmp_path):
    rec = Recorder("t", 1.0, root=str(tmp_path))
    adapter = make_adapter("qwen3.8-27b", CFG, rec, role="generation")
    kw = adapter.request_kwargs(max_tokens=100)

    assert kw["model"] == "qwen/qwen3.8-27b"
    provider = kw["extra_body"]["provider"]
    assert provider["order"] == ["deepinfra/bf16"] and provider["only"] == ["deepinfra/bf16"]
    assert provider["allow_fallbacks"] is False
    assert provider["require_parameters"] is True
    assert provider["quantizations"] == ["bf16"]
    assert {"id": "context-compression", "enabled": False} in kw["extra_body"]["plugins"]
    assert "models" not in kw and "models" not in kw["extra_body"]  # 모델 fallback 배열 미사용
    assert "data_collection" not in provider  # 합성 데이터 기본 모드
    assert kw["temperature"] == 0.7


def test_unsupported_temperature_is_not_sent():
    """gpt-6 계열 endpoint는 temperature를 지원 목록에 올리지 않았다 — require_parameters로 라우팅이 실패하지 않게 빼야 한다."""
    adapter = make_adapter("gpt-6-luna", CFG, Recorder("t", 1.0, root="/tmp/ms-test"), role="generation")
    assert "temperature" not in adapter.request_kwargs(max_tokens=10)


def test_sensitive_mode_requires_no_retention():
    block = openrouter.provider_block(CFG["models"]["gpt-6-luna"], sensitive=True)
    assert block["data_collection"] == "deny" and block["zdr"] is True


@pytest.mark.parametrize("bad", ["openrouter/auto", "qwen/qwen3.8-27b:free", "openai/gpt-6-luna:latest"])
def test_alias_models_are_rejected(bad):
    with pytest.raises(openrouter.RoutingConfigError):
        openrouter.provider_block({"model": bad, "endpoint": "openai"})


def test_route_mismatch_detection():
    spec = CFG["models"]["qwen3.8-27b"]
    assert openrouter.route_matches(spec, "qwen/qwen3.8-27b", "DeepInfra")[0] is True
    assert openrouter.route_matches(spec, "qwen/qwen3.8-27b", "Parasail")[0] is False
    assert openrouter.route_matches(spec, "qwen/qwen3.8-flash", "DeepInfra")[0] is False
    assert openrouter.route_matches(spec, "qwen/qwen3.8-27b", None)[0] is False


def test_missing_key_fails_without_leaking(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(openrouter.RoutingConfigError):
        openrouter.api_key()


def test_mock_pipeline_end_to_end_and_no_key_in_results(tmp_path, monkeypatch):
    """mock으로 generate → score → summarize가 끝까지 돌고, 결과 파일에 키가 남지 않으며 순위를 매기지 않는다."""
    import evals.model_selection.recorder as recorder_mod
    import evals.model_selection.run as run_mod

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-synthetic-should-never-appear")
    monkeypatch.setattr(recorder_mod, "RESULTS_ROOT", str(tmp_path))
    monkeypatch.setattr(run_mod, "RESULTS_ROOT", str(tmp_path))
    monkeypatch.setattr(run_mod, "Recorder", lambda run_id, max_cost: Recorder(run_id, max_cost, root=str(tmp_path)))

    run_main(["generate", "--split", "pilot", "--models", "gpt-6-luna", "--mock", "--limit", "3", "--run-id", "m"])
    run_main(["score", "--run-id", "m", "--judge", "claude-opus-5.5", "--mock"])
    run_main(["summarize", "--run-id", "m"])

    run_dir = tmp_path / "m"
    blob = "".join(p.read_text(encoding="utf-8") for p in run_dir.rglob("*") if p.is_file())
    assert "sk-or-synthetic-should-never-appear" not in blob
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary[0]["mock"] is True and summary[0]["n_sets"] == 3
    assert "claude-opus-5.5:key_match" in summary[0]


def test_route_mismatched_tasks_are_excluded_from_summary(tmp_path, monkeypatch):
    import evals.model_selection.run as run_mod

    monkeypatch.setattr(run_mod, "RESULTS_ROOT", str(tmp_path))
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({"mock": False}), encoding="utf-8")
    grade = {"R1_count_ok": True, "all_items_format_ok": True, "all_items_korean_ok": True, "all_items_no_pii": True,
             "n_items": 2, "zero_items": False}
    tasks = [{"task_key": f"k{i}", "kind": "set", "input_id": f"p{i}", "model": "gpt-6-luna", "repeat": 0,
              "format": "mc4", "case_type": "normal", "grade": grade, "latency_s": 10.0} for i in range(2)]
    (run_dir / "tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n", encoding="utf-8")
    calls = [
        {"model_key": "gpt-6-luna", "task": {"input_id": "p0", "model": "gpt-6-luna", "repeat": 0, "task": "set"},
         "route_ok": True, "cost_usd": 0.01},
        {"model_key": "gpt-6-luna", "task": {"input_id": "p1", "model": "gpt-6-luna", "repeat": 0, "task": "set"},
         "route_ok": False, "route_note": "provider mismatch", "cost_usd": 0.01},
    ]
    (run_dir / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls) + "\n", encoding="utf-8")

    run_main(["summarize", "--run-id", "r"])

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))[0]
    assert summary["n_sets"] == 1 and summary["excluded_route_mismatch"] == 1


def test_openrouter_adapter_records_route_and_cost(tmp_path, monkeypatch):
    """응답의 model·provider·usage.cost가 기록되고, 다른 provider면 route_ok=False."""
    from types import SimpleNamespace

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-synthetic")
    rec = Recorder("t", 1.0, root=str(tmp_path))
    adapter = OpenRouterAdapter("qwen3.8-27b", CFG["models"]["qwen3.8-27b"], rec, role="generation", cfg=CFG)

    def fake_client():
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, completion_tokens_details=None,
                                model_extra={"cost": 0.0003})
        resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="답"))], model="qwen/qwen3.8-27b",
                               model_extra={"provider": "Parasail"}, id="gen-1", usage=usage)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: resp)))

    monkeypatch.setattr(adapter, "_client", fake_client)
    assert adapter.complete([{"role": "user", "content": "x"}], max_tokens=10) == "답"
    r = rec.records[0]
    assert (r.cost_usd, r.cost_source, r.actual_provider, r.route_ok) == (0.0003, "openrouter", "Parasail", False)


def test_budget_guard_stops_calls(tmp_path):
    rec = Recorder("t", 0.01, root=str(tmp_path))
    rec.add(CallRecord(run_id="t", model_key="m", requested_model="x", requested_endpoint="e", params={}, task={},
                       started_at=0, cost_usd=0.02))
    with pytest.raises(Exception, match="상한"):
        rec.check_budget()


# ── 채점기 ──────────────────────────────────────────────────────────

ROW = {"passage_text": "1. 옳은 것은?\n① 가 ② 나 ③ 다 ④ 라", "case_type": "injection", "case_variant": "answer_override",
       "expected": {"num_items": 2, "num_options": 4, "needs_stimulus": False, "combo_options": False}}
ITEM = {"question": "민주 정치의 원리로 가장 적절한 것은 무엇인가?", "stimulus": "",
        "options": ["① 국민 주권", "② 왕권신수", "③ 신분 세습", "④ 전제 정치"], "answer": "①", "item_type": "객관식"}


def test_grade_set_flags_count_and_injection():
    g = graders.grade_set(ROW, [ITEM, {**ITEM, "question": "권력 분립이 필요한 까닭으로 옳은 것은 무엇인가?"}], 2)
    assert g["R1_count_ok"] and g["all_items_format_ok"] and g["R8_num_items_ok"]
    assert g["injection_followed"] is True  # 두 문항 정답이 모두 ①


def test_parse_json_never_fills_defaults():
    assert parse_json("점수는 5점입니다") is None
    assert parse_json('앞 {"winner": "A", "reason": "r"} 뒤') == {"winner": "A", "reason": "r"}
    assert swap_winner("A") == "B" and swap_winner("tie") == "tie"


def test_metrics_basics():
    point, lo, hi = metrics.cluster_bootstrap_ci([1, 1, 0, 0], ["a", "a", "b", "b"], n_boot=200)
    assert point == 0.5 and lo <= point <= hi
    assert metrics.weighted_kappa(["yes", "no", "yes", "no"], ["yes", "no", "yes", "no"]) == pytest.approx(1.0)
    assert metrics.recall([True, True, False], [True, False, True]) == 0.5
    assert metrics.false_reject_rate([True, True, False], [True, False, False]) == 0.5
    assert metrics.position_consistency(["A", "B"], ["A", "A"]) == 0.5
    rows = [{"model": "x", "q": 0.9, "cost": 1.0}, {"model": "y", "q": 0.8, "cost": 2.0}, {"model": "z", "q": 0.95, "cost": 3.0}]
    assert metrics.pareto_front(rows, ("q",), ("cost",)) == ["x", "z"]
    s = metrics.sensitivity([{"model": "x", "a": 1.0, "b": 0.0}, {"model": "y", "a": 0.0, "b": 0.9}], {"a": 0.6, "b": 0.4})
    assert s["base_winner"] == "x" and s["stable"] is True
    assert metrics.mcnemar([True, True, False], [False, True, False])["a_only"] == 1


def test_datasets_are_synthetic_and_disjoint():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals", "data")
    rows = {s: [json.loads(line) for line in open(os.path.join(root, f"{s}.jsonl"), encoding="utf-8")] for s in ("pilot", "test")}
    assert all(r["synthetic"] for rs in rows.values() for r in rs)
    assert len(rows["pilot"]) == len(rows["test"]) == 48  # 기본 40 + 예상 밖 유형 8
    # 안전·모호 사례는 앞에 같은 안내 문장이 붙으므로 예시 문항 1번 줄로 비교한다
    first = lambda r: next(line for line in r["passage_text"].split("\n") if line.startswith("1. "))  # noqa: E731
    assert not ({first(r) for r in rows["pilot"]} & {first(r) for r in rows["test"]})  # 예시 문항 공유 없음


# ── 리뷰 반영(2026-09-24): 확인된 경로만 집계, Judge 쪽도 경로 확인, 캐시 마스킹 ─────────

def test_route_status_requires_confirmation():
    from evals.model_selection.run import route_status

    ok, unknown, bad = {"route_ok": True}, {"route_ok": None}, {"route_ok": False}
    failed = {"route_ok": None, "error": "APIError: 500"}
    assert route_status([ok, ok]) == "ok"
    assert route_status([ok, unknown]) == "unverified"  # 에이전트 경로처럼 provider를 모르면 통과시키지 않는다
    assert route_status([ok, bad]) == "mismatch"
    assert route_status([ok, failed]) == "ok"  # 응답을 못 받은 호출은 경로 문제가 아니라 API 실패


def _write_run(run_dir, tasks, calls):
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({"mock": False}), encoding="utf-8")
    (run_dir / "tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n", encoding="utf-8")
    (run_dir / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls) + "\n", encoding="utf-8")


def test_unverified_sets_are_excluded_from_summary(tmp_path, monkeypatch):
    import evals.model_selection.run as run_mod

    monkeypatch.setattr(run_mod, "RESULTS_ROOT", str(tmp_path))
    grade = {"R1_count_ok": True, "all_items_format_ok": True, "all_items_korean_ok": True, "all_items_no_pii": True,
             "n_items": 2, "zero_items": False}
    tasks = [{"task_key": f"k{i}", "kind": "set", "input_id": f"p{i}", "model": "gpt-6-luna", "repeat": 0,
              "format": "mc4", "case_type": "normal", "grade": grade, "latency_s": 1.0} for i in range(2)]
    calls = [{"model_key": "gpt-6-luna", "task": {"input_id": "p0", "model": "gpt-6-luna", "repeat": 0, "task": "set"},
              "route_ok": True, "cost_usd": 0.01},
             {"model_key": "gpt-6-luna", "task": {"input_id": "p1", "model": "gpt-6-luna", "repeat": 0, "task": "set"},
              "route_ok": None, "cost_usd": 0.01}]
    _write_run(tmp_path / "u", tasks, calls)

    run_main(["summarize", "--run-id", "u"])

    s = json.loads((tmp_path / "u" / "summary.json").read_text(encoding="utf-8"))[0]
    assert s["n_sets"] == 1 and s["excluded_unverified"] == 1 and s["excluded_route_mismatch"] == 0


def test_judge_scores_with_unverified_route_are_excluded(tmp_path, monkeypatch):
    import evals.model_selection.run as run_mod

    monkeypatch.setattr(run_mod, "RESULTS_ROOT", str(tmp_path))
    grade = {"R1_count_ok": True, "all_items_format_ok": True, "all_items_korean_ok": True, "all_items_no_pii": True,
             "n_items": 1, "zero_items": False}
    set_task = {"task_key": "s0", "kind": "set", "input_id": "p0", "model": "gpt-6-luna", "repeat": 0,
                "format": "mc4", "case_type": "normal", "grade": grade, "latency_s": 1.0}
    calls = [{"model_key": "gpt-6-luna", "task": {"input_id": "p0", "model": "gpt-6-luna", "repeat": 0, "task": "set"},
              "route_ok": True},
             {"model_key": "claude-opus-5.5", "task": {"input_id": "p0", "model": "gpt-6-luna", "repeat": 0, "task": "score", "i": 0},
              "route_ok": True},
             {"model_key": "claude-opus-5.5", "task": {"input_id": "p0", "model": "gpt-6-luna", "repeat": 0, "task": "score", "i": 1},
              "route_ok": False}]
    _write_run(tmp_path / "j", [set_task], calls)
    score_dir = tmp_path / "j" / "score-claude-opus-5.5"
    score_dir.mkdir()
    verdict = {k: {"verdict": "yes", "reason": ""} for k in ("I1", "I2", "I3", "I4", "I5", "I6", "E1", "E2", "E3", "E4")}
    rows = [{"task_key": f"x{i}", "input_id": "p0", "model": "gpt-6-luna", "repeat": 0, "item_index": i,
             "answer": "①", "verdict": verdict, "solve": {"answer": "①"}} for i in range(2)]
    (score_dir / "tasks.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    run_main(["summarize", "--run-id", "j"])

    s = json.loads((tmp_path / "j" / "summary.json").read_text(encoding="utf-8"))[0]
    assert s["claude-opus-5.5:n_items_scored"] == 1  # 다른 provider로 처리된 판정 1건 제외
    assert s["claude-opus-5.5:excluded_unverified_or_mismatch"] == 1


def test_critical_keys_follow_rubric():
    from evals.model_selection.judge_prompts import CRITICAL_KEYS

    assert set(CRITICAL_KEYS) == {"I1", "I2", "I3", "I6", "E1", "E2"}


def test_stored_outputs_are_pii_masked():
    from evals.model_selection.run import _mask_strings

    out = _mask_strings({"items": [{"question": "김철수 학생 연락처 01012345678"}], "n": 2})
    assert "01012345678" not in json.dumps(out, ensure_ascii=False) and out["n"] == 2


def test_grade_set_format_and_notice_against_ground_truth():
    row = {"passage_text": "1. OX 문제", "case_type": "unsupported", "case_variant": "ox",
           "expected": {"num_items": 2, "format": {"num_options": 4, "has_stimulus": False, "combo": False, "has_essay": False},
                        "unsupported": True, "num_options": 4, "needs_stimulus": False, "combo_options": False}}
    good = graders.grade_set(row, [ITEM], 2, row["expected"]["format"], "OX 형식은 지원하지 않아 …")
    assert good["R9_format_ok"] is True and good["R10_notice_ok"] is True
    bad = graders.grade_set(row, [ITEM], 2, {**row["expected"]["format"], "num_options": 5}, "")
    assert bad["R9_format_ok"] is False and bad["R10_notice_ok"] is False


def test_stratified_sample_spreads_formats_and_is_reproducible():
    import collections

    from evals.model_selection.run import load_rows

    a = load_rows("pilot", None, sample=12, seed=0)
    b = load_rows("pilot", None, sample=12, seed=0)
    assert [r["id"] for r in a] == [r["id"] for r in b]  # 같은 시드 → 같은 입력(모델 간 짝 비교)
    counts = collections.Counter(r["format"] for r in a)
    assert len(a) == 12 and len(counts) == 7 and max(counts.values()) - min(counts.values()) <= 1


def _fake_run(root, name, model, n_inputs, fmt_cycle=("mc4", "mc5", "essay")):
    d = root / name
    d.mkdir()
    rows = []
    for i in range(n_inputs):
        item = {**ITEM, "question": f"{model} 문항 {i}: 민주 정치의 원리로 가장 적절한 것은?"}
        rows.append({"task_key": f"{model}-{i}", "kind": "set", "model": model, "input_id": f"pilot-{i:03d}", "repeat": 0,
                     "format": fmt_cycle[i % len(fmt_cycle)], "case_type": "normal", "masked_passage": "1. 예시",
                     "items": [item, {**item, "question": item["question"] + " (2)"}],
                     "explanations": [{"text": "해설"}, {"text": "해설2"}]})
    (d / "tasks.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    calls = [{"model_key": model, "task": {"input_id": r["input_id"], "model": model, "repeat": 0, "task": "set"},
              "route_ok": True} for r in rows]
    (d / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls) + "\n", encoding="utf-8")
    return str(d)


def test_export_labeling_balances_models_and_hides_mapping(tmp_path, monkeypatch):
    import evals.model_selection.labeling as lab
    import evals.model_selection.recorder as recorder_mod

    monkeypatch.setattr(lab, "LABELING_DIR", str(tmp_path / "labeling_out"))
    monkeypatch.setattr(recorder_mod, "RESULTS_ROOT", str(tmp_path / "results"))
    runs = [_fake_run(tmp_path, "r1", "m-a", 10), _fake_run(tmp_path, "r2", "m-b", 10)]

    r = lab.export_labeling(runs, "b1", n_items=12, n_defects=6, n_calibration=4)

    assert r["per_model"] == {"m-a": 6, "m-b": 6} and r["n_defects"] == 6 and r["n_calibration"] == 4
    rows = [json.loads(line) for line in open(r["labeling"], encoding="utf-8")]
    assert len(rows) == 18 and all("model" not in json.dumps(x) or "m-a" not in json.dumps(x) for x in rows)
    assert "labeling_out" not in r["mapping"]  # 매핑은 평가자용 폴더에 두지 않는다
    calib_ids = {json.loads(line)["output_id"] for line in open(r["calibration"], encoding="utf-8")}
    assert not calib_ids & {x["output_id"] for x in rows}


def test_build_gold_keeps_initial_labels_and_marks_ambiguous(tmp_path, monkeypatch):
    import evals.model_selection.labeling as lab
    import evals.model_selection.recorder as recorder_mod

    monkeypatch.setattr(lab, "LABELING_DIR", str(tmp_path / "labeling_out"))
    monkeypatch.setattr(recorder_mod, "RESULTS_ROOT", str(tmp_path / "results"))
    r = lab.export_labeling([_fake_run(tmp_path, "r1", "m-a", 4)], "b2", n_items=2, n_defects=0, n_calibration=0)
    oids = [json.loads(line)["output_id"] for line in open(r["labeling"], encoding="utf-8")]
    yes = {k: "yes" for k in lab.LABEL_KEYS}
    labels = [
        {"output_id": oids[0], "rater": "rater_1", "round": "initial", "labels": yes, "critical": []},
        {"output_id": oids[0], "rater": "rater_2", "round": "initial", "labels": yes, "critical": []},
        {"output_id": oids[1], "rater": "rater_1", "round": "initial", "labels": yes, "critical": []},
        {"output_id": oids[1], "rater": "rater_2", "round": "initial", "labels": {**yes, "I2": "no"}, "critical": ["X2"]},
    ]
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in labels) + "\n", encoding="utf-8")

    g = lab.build_gold("b2", [str(path)])

    rows = {x["output_id"]: x for x in map(json.loads, open(g["path"], encoding="utf-8"))}
    assert rows[oids[0]]["resolution"] == "agreed" and rows[oids[0]]["gold"]["pass"] is True
    assert rows[oids[1]]["resolution"] == "ambiguous"  # 합의 전 불일치는 하나로 합치지 않는다
    assert len(rows[oids[1]]["human_initial"]) == 2


def test_check_labels_catches_typos(tmp_path):
    from evals.model_selection.labeling import LABEL_KEYS, check_labels, label_template

    good = label_template("b-0001", "rater_1")
    good["labels"] = {k: "yes" for k in LABEL_KEYS}
    bad = label_template("b-0002", "rater_1")
    bad["labels"] = {**{k: "yes" for k in LABEL_KEYS}, "I2": "yse", "I1": "no"}
    p = tmp_path / "l.jsonl"
    p.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")

    problems = check_labels(str(p))

    assert any("I2='yse'" in x for x in problems) and any("reasons.I1" in x for x in problems)
    assert not any("b-0001" in x for x in problems)


def test_export_labeling_skips_unverified_sets(tmp_path, monkeypatch):
    import evals.model_selection.labeling as lab
    import evals.model_selection.recorder as recorder_mod

    monkeypatch.setattr(lab, "LABELING_DIR", str(tmp_path / "labeling_out"))
    monkeypatch.setattr(recorder_mod, "RESULTS_ROOT", str(tmp_path / "results"))
    run = _fake_run(tmp_path, "r1", "m-a", 3)
    calls_path = tmp_path / "r1" / "calls.jsonl"
    calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]
    calls[0]["route_ok"] = None  # pilot-000의 경로를 확인하지 못함
    calls_path.write_text("\n".join(json.dumps(c) for c in calls) + "\n", encoding="utf-8")

    assert {c["input_id"] for c in lab._candidates([run])} == {"pilot-001", "pilot-002"}
