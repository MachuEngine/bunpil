"""VLM figure Judge 신뢰도 관련 순수 로직 테스트. LLM 호출 없음(가짜 judge 백엔드만 사용).

evals/eval_vlm.py의 _parse_judge_score/score_figure_entry(golden_gen 스크립트와 공유),
golden_gen/gen_vlm_judge_reliability_golden.py의 _has_existing_labels,
evals/eval_vlm_judge_reliability.py의 통계 계산 함수를 대상으로 한다."""
import asyncio
import json

import pytest

from evals.eval_vlm import _parse_judge_score, score_figure_entry
from evals.eval_vlm_judge_reliability import (
    _binary_kappa_threshold3,
    _kappa_stat,
    _mae_stat,
    _within1_stat,
    main as reliability_main,
)
from golden_gen.gen_vlm_judge_reliability_golden import _has_existing_labels


# ── _parse_judge_score ───────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("5", 5),
    ("1", 1),
    ("점수: 4점입니다", 4),
    ("3~4", 3),  # 이전 버그: "".join(digits)로 "34"가 되어 5로 잘렸음
    ("", None),
    ("자료가 부족합니다", None),
    ("6", None),  # 1~5 범위 밖 숫자만 있으면 유효한 점수 없음
])
def test_parse_judge_score(raw, expected):
    assert _parse_judge_score(raw) == expected


# ── score_figure_entry ───────────────────────────────────────────────────

class _FakeJudge:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    async def generate(self, messages, **kwargs):
        self.calls += 1
        return self.response


class _RaisingJudge:
    async def generate(self, messages, **kwargs):
        raise AssertionError("자료 블록이 없을 때는 Judge를 호출하면 안 됨")


async def _no_retry(coro_fn):
    return await coro_fn()


def test_score_figure_entry_no_block_skips_judge_call():
    judge = _RaisingJudge()
    result = asyncio.run(
        score_figure_entry("핵심 사실", "자료 설명이 전혀 없는 출력", judge, call_with_retry=_no_retry)
    )
    assert result["judge_score"] == 1
    assert result["fig_text"] is None
    assert "블록" in result["judge_note"]


def test_score_figure_entry_calls_judge_with_extracted_block():
    judge = _FakeJudge("4")
    vlm_output = "문제 본문\n[자료: A 820, B 450]\n선택지"
    result = asyncio.run(score_figure_entry("핵심 사실", vlm_output, judge, call_with_retry=_no_retry))
    assert judge.calls == 1
    assert result["judge_score"] == 4
    assert result["fig_text"] == "[자료: A 820, B 450]"


def test_score_figure_entry_parses_range_response_without_truncation():
    judge = _FakeJudge("3~4")
    vlm_output = "[자료: X 1, Y 2]"
    result = asyncio.run(score_figure_entry("핵심 사실", vlm_output, judge, call_with_retry=_no_retry))
    assert result["judge_score"] == 3


def test_score_figure_entry_judge_failure_returns_none_with_note():
    class _FailingJudge:
        async def generate(self, messages, **kwargs):
            raise RuntimeError("API 오류")

    result = asyncio.run(
        score_figure_entry("핵심 사실", "[자료: X 1]", _FailingJudge(), call_with_retry=_no_retry)
    )
    assert result["judge_score"] is None
    assert "judge 호출 실패" in result["judge_note"]


# ── _has_existing_labels (golden_gen/gen_vlm_judge_reliability_golden.py) ─

def test_has_existing_labels_false_for_missing_file(tmp_path):
    assert _has_existing_labels(str(tmp_path / "nope.json")) is False


def test_has_existing_labels_false_when_all_null(tmp_path):
    path = tmp_path / "golden.json"
    data = {"entries": [{"id": "f01", "human_label": None}, {"id": "f02", "human_label": None}]}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert _has_existing_labels(str(path)) is False


def test_has_existing_labels_true_when_any_filled(tmp_path):
    path = tmp_path / "golden.json"
    data = {"entries": [{"id": "f01", "human_label": None}, {"id": "f02", "human_label": 4}]}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert _has_existing_labels(str(path)) is True


# ── eval_vlm_judge_reliability.py 통계 함수 ───────────────────────────────

def test_kappa_stat_perfect_agreement():
    sample = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    assert _kappa_stat(sample) == pytest.approx(1.0)


def test_kappa_stat_too_few_rows_returns_none():
    assert _kappa_stat([(3, 3)]) is None


def test_mae_stat():
    sample = [(5, 4), (1, 1), (3, 5)]
    assert _mae_stat(sample) == pytest.approx((1 + 0 + 2) / 3)


def test_within1_stat():
    sample = [(5, 4), (1, 1), (3, 5)]  # diff = 1, 0, 2
    assert _within1_stat(sample) == pytest.approx(2 / 3)


def test_binary_kappa_threshold3_perfect_agreement():
    human = [1, 2, 4, 5, 5]
    llm = [1, 2, 4, 5, 5]
    assert _binary_kappa_threshold3(human, llm) == pytest.approx(1.0)


def test_binary_kappa_threshold3_too_few_returns_none():
    assert _binary_kappa_threshold3([3], [3]) is None


# ── main() — human_label 비어있을 때 안내만 하고 종료 ─────────────────────

def test_main_prints_guidance_when_all_unlabeled(tmp_path, monkeypatch, capsys):
    import evals.eval_vlm_judge_reliability as mod

    path = tmp_path / "golden.json"
    data = {"entries": [{"id": "f01", "judge_score": 5, "human_label": None}]}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(mod, "GOLDEN_PATH", str(path))

    reliability_main()

    out = capsys.readouterr().out
    assert "human_label" in out
    assert "1~5" in out


def test_main_reports_stats_when_labeled(tmp_path, monkeypatch, capsys):
    import evals.eval_vlm_judge_reliability as mod

    path = tmp_path / "golden.json"
    entries = [
        {"id": f"f{i:02d}", "judge_score": score, "human_label": label}
        for i, (score, label) in enumerate(
            [(5, 5), (4, 4), (1, 2), (5, 4), (3, 3), (5, 5)], start=1
        )
    ]
    path.write_text(json.dumps({"entries": entries}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(mod, "GOLDEN_PATH", str(path))

    reliability_main()

    out = capsys.readouterr().out
    assert "가중 kappa" in out
    assert "참고용" in out
