#!/usr/bin/env python
"""출제 모듈 Agent Trajectory Eval — LangSmith 트레이스 집계.

기존 eval(eval_exam.py 등)은 최종 산출물(문항 품질)만 채점한다. 이 스크립트는
과정(도구 호출 신뢰도, 재시도 원인)을 이미 LangSmith에 쌓인 트레이스에서
집계한다 — 앱 코드(graph.py/tools.py)는 건드리지 않고, LangGraph가 자동으로
남기는 노드/도구 run만 읽는다.

⚠️ 이 집계는 프로덕션 관측이 아니다. RunPod 크레딧 소진(bunpil_roadmap.md
"진행 상태 요약")으로 실사용 트래픽이 없어, 조회 대상은 사실상
LANGCHAIN_TRACING_V2=true로 실행한 로컬 eval/테스트 실행분이다("eval 실행
트레이스 기반 실패 모드 분류"로 설명할 것 — production observability라고
말하지 않는다).

전제: LANGCHAIN_API_KEY가 있어야 과거 트레이스를 조회할 수 있다. 트레이스를
새로 만들려면:
    LANGCHAIN_TRACING_V2=true LANGCHAIN_API_KEY=<키> CHROMA_PERSIST_DIR=./chroma_db \\
        python scripts/test_exam.py

분류 로직은 노드 run의 outputs 키 시그니처로 판별한다(run.name이 아님) —
LangGraph 버전에 따라 노드 run 이름 표기가 달라질 수 있지만, graph.py의 각
노드 함수가 반환하는 dict 키 구성(agent_messages/budget, draft_items 등)은
코드가 실제로 보장하는 계약이라 더 안정적이다. 도구 run은 @tool 데코레이터가
함수 이름을 그대로 tool.name으로 쓰는 LangChain 관례를 따르므로 run.name으로
식별한다.

graph.py/tools.py의 문자열 리터럴이 바뀌면 아래 상수들과 어긋나 조용히
집계가 틀어질 수 있다 — 그래서 아무 데도 안 걸리는 항목은 "unclassified"로
따로 집계해 드러낸다.

보안(하드룰 4): 트레이스에는 passage_text·생성 문항 원문이 들어있을 수
있으나, 이 스크립트는 카운트·카테고리만 출력한다. 원문은 콘솔에도 JSON
출력에도 쓰지 않는다.
"""
import argparse
import itertools
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.common.llm.tracing import init_langsmith_project
init_langsmith_project()


# ── graph.py/tools.py 리터럴과 동기화 필요 ──────────────────────────

_REJECTION_PREFIXES = (
    "저장 거부 —",
    "점수 기록 거부 —",
    "문항 폐기 거부 —",
    "형식 오류 — 수정 필요:",
    "Unknown tool:",
)

_EMPTY_RESULT_STRINGS = (
    "교육과정 자료 없음",
    "관련 규정 없음",
    "교육과정 성취기준 자료 없음",
    "관련 성취기준 없음",
)

_MALFORMED_RETRY_MARKERS = (
    "도구 호출 형식이 손상되었습니다",
    "아직 목표 문항 저장과 제출이 끝나지 않았습니다",
)

# validate_node의 feedback 문구(graph.py:296-311) — 형식·절차 실패 vs Judge 판단 불일치
_FORMAT_FEEDBACK_MARKERS = (
    "문항 개수 불일치",
    "품질 점수 미달 또는 미채점 문항",
)
_JUDGMENT_FEEDBACK_MARKERS = (
    "구조 유사도 미채점",
    "유형 비율 유사도 미달",
    "난이도 구성 불일치",
    "종합 구조 유사도 점수 미달",
)


# ── 노드 run 식별 (outputs 키 시그니처 기반) ────────────────────────

def _node_output(run) -> dict:
    """일부 버전은 노드 outputs를 {"output": {...}}로 한 겹 더 감싼다 — 양쪽 다 시도."""
    outputs = run.outputs
    if not isinstance(outputs, dict):
        return {}
    if set(outputs.keys()) == {"output"} and isinstance(outputs["output"], dict):
        return outputs["output"]
    return outputs


def _is_plan_run(run) -> bool:
    out = _node_output(run)
    return (
        "validation_feedback" in out
        and "similarity_judge_result" in out
        and "draft_items" not in out
    )


def _is_agent_run(run) -> bool:
    return "agent_messages" in _node_output(run)


def _is_validate_run(run) -> bool:
    out = _node_output(run)
    return "draft_items" in out and "validation_feedback" in out


def _classify_feedback(feedback: str) -> str:
    if not feedback:
        return "unclassified"
    hits_format = any(m in feedback for m in _FORMAT_FEEDBACK_MARKERS)
    hits_judgment = any(m in feedback for m in _JUDGMENT_FEEDBACK_MARKERS)
    if hits_format and hits_judgment:
        return "both"
    if hits_format:
        return "format"
    if hits_judgment:
        return "judgment"
    return "unclassified"


def _tool_output_text(run) -> str:
    outputs = run.outputs
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, dict):
        for key in ("output", "result", "content"):
            val = outputs.get(key)
            if isinstance(val, str):
                return val
        return json.dumps(outputs, ensure_ascii=False)
    return str(outputs or "")


def _classify_tool_run(run) -> str:
    if run.error:
        return "error"
    text = _tool_output_text(run)
    if any(text.startswith(p) for p in _REJECTION_PREFIXES):
        return "rejected"
    if any(s in text for s in _EMPTY_RESULT_STRINGS):
        return "empty_result"
    return "ok"


# ── 조회 + 집계 ──────────────────────────────────────────────────

def fetch_runs(client, project: str, days: int, limit: int) -> list:
    start_time = datetime.now(timezone.utc) - timedelta(days=days)
    # run_type 필터는 langsmith SDK 버전마다 지원 형태가 달라 클라이언트 측에서
    # 직접 나누는 편이 안전하다(이 스크립트의 원칙 — "분류는 전부 클라이언트 측").
    #
    # limit을 list_runs()에 그대로 넘기면 안 된다 — 이 SDK(0.10.0)는 limit을
    # 커서 페이지네이션 크기가 아니라 /runs/query 요청의 limit 필드로 그대로
    # 보내는데, LangSmith API는 요청당 최대 100까지만 허용한다(그 이상이면
    # "Limit exceeds maximum allowed value of 100" 400 에러). 대신 limit 없이
    # 호출해 커서 페이지네이션이 알아서 페이지를 넘기게 하고, 원하는 총량만큼만
    # 클라이언트 쪽에서 islice로 끊는다.
    runs_iter = client.list_runs(project_name=project, start_time=start_time)
    return list(itertools.islice(runs_iter, limit))


def aggregate(runs: list) -> dict:
    tool_runs = [r for r in runs if r.run_type == "tool"]
    llm_runs = [r for r in runs if r.run_type == "llm"]
    plan_runs = [r for r in runs if _is_plan_run(r)]
    agent_runs = [r for r in runs if _is_agent_run(r)]
    validate_runs = [r for r in runs if _is_validate_run(r)]

    # A. 도구 호출 신뢰도
    tool_status = Counter()
    tool_status_by_name: dict = defaultdict(Counter)
    for r in tool_runs:
        status = _classify_tool_run(r)
        tool_status[status] += 1
        tool_status_by_name[r.name or "unknown"][status] += 1

    n_tool = len(tool_runs)
    tool_summary = {
        "n": n_tool,
        "error_rate": round(tool_status["error"] / n_tool, 3) if n_tool else None,
        "rejection_rate": round(tool_status["rejected"] / n_tool, 3) if n_tool else None,
        "empty_result_rate": round(tool_status["empty_result"] / n_tool, 3) if n_tool else None,
        "by_tool": {name: dict(counts) for name, counts in tool_status_by_name.items()},
    }

    # B. 재시도 원인 분류 — validate_node 출력
    feedback_categories = Counter()
    for r in validate_runs:
        out = _node_output(r)
        if out.get("validation_passed"):
            feedback_categories["none(passed)"] += 1
        else:
            feedback_categories[_classify_feedback(out.get("validation_feedback", ""))] += 1

    # malformed tool-call 재작성 요청이 다음 LLM 호출 입력에 등장한 횟수
    # (graph.py agent_node가 messages에 append한 뒤 다음 llm.invoke()에 그대로 전달됨)
    malformed_count = 0
    for r in llm_runs:
        blob = json.dumps(r.inputs or {}, ensure_ascii=False)
        if any(marker in blob for marker in _MALFORMED_RETRY_MARKERS):
            malformed_count += 1
    n_llm = len(llm_runs)

    retry_summary = {
        "n_validate_runs": len(validate_runs),
        "feedback_categories": dict(feedback_categories),
        "malformed_tool_call_llm_calls": malformed_count,
        "malformed_tool_call_rate": round(malformed_count / n_llm, 3) if n_llm else None,
    }

    # C. 궤적 형태
    n_sessions = len(plan_runs)
    submit_calls = sum(1 for r in tool_runs if r.name == "submit_for_review")
    passed_count = sum(1 for r in validate_runs if _node_output(r).get("validation_passed"))

    trajectory_summary = {
        "n_sessions(plan_runs)": n_sessions,
        "n_agent_node_runs": len(agent_runs),
        "avg_agent_reentries_per_session": (
            round(len(agent_runs) / n_sessions, 2) if n_sessions else None
        ),
        "submit_for_review_calls": submit_calls,
        "validation_passed_rate": (
            round(passed_count / len(validate_runs), 3) if validate_runs else None
        ),
    }

    return {
        "tool_reliability": tool_summary,
        "retry_causes": retry_summary,
        "trajectory": trajectory_summary,
    }


# ── 리포트 출력 ─────────────────────────────────────────────────────

def print_report(result: dict, project: str, days: int) -> None:
    print("=" * 55)
    print("  분필 출제 모듈 — Agent Trajectory Eval")
    print(f"  project={project}  기간=최근 {days}일")
    print("=" * 55)

    t = result["tool_reliability"]
    print(f"\n[1] 도구 호출 신뢰도 (n={t['n']})")
    if t["n"] == 0:
        print("  도구 호출 트레이스 없음 — LANGCHAIN_TRACING_V2=true로 실행한 이력이")
        print("  있는지, project명이 맞는지(bunpil-dev/bunpil-prod) 확인하세요.")
    else:
        print(f"  오류율(error)    : {t['error_rate']}  (예외 발생 — 인자 형식 오류 등)")
        print(f"  거부율(rejected) : {t['rejection_rate']}  (가드레일 정상 동작 — 실패 아님)")
        print(f"  빈결과율(empty)  : {t['empty_result_rate']}  (RAG 검색 결과 없음)")
        print("  도구별 분해:")
        for name, counts in sorted(t["by_tool"].items()):
            print(f"    - {name}: {dict(counts)}")

    r = result["retry_causes"]
    print(f"\n[2] 재시도 원인 분류 (validate 실행 n={r['n_validate_runs']})")
    if r["n_validate_runs"] == 0:
        print("  validate 노드 트레이스 없음")
    else:
        for k, v in sorted(r["feedback_categories"].items()):
            print(f"    - {k}: {v}")
        print(
            f"  malformed tool-call 재작성 요청이 등장한 LLM 호출 수: "
            f"{r['malformed_tool_call_llm_calls']} (rate={r['malformed_tool_call_rate']})"
        )

    tr = result["trajectory"]
    print("\n[3] 궤적 형태")
    for k, v in tr.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 55)
    print("※ production observability 아님 — LANGCHAIN_TRACING_V2=true로 실행한")
    print("  로컬/eval 트레이스 집계임(RunPod 크레딧 소진, 실사용 트래픽 없음).")
    print("=" * 55)


def main() -> None:
    parser = argparse.ArgumentParser(description="출제 모듈 Agent Trajectory Eval")
    parser.add_argument(
        "--project", default=None,
        help="LangSmith 프로젝트명 (기본: 현재 LANGCHAIN_PROJECT 값)",
    )
    parser.add_argument("--days", type=int, default=30, help="조회 기간(일)")
    parser.add_argument("--limit", type=int, default=2000, help="조회할 최대 run 수(안전장치)")
    parser.add_argument("--json", default=None, help="집계 결과를 JSON으로도 저장할 경로")
    args = parser.parse_args()

    from langsmith import Client
    client = Client()

    project = args.project or os.environ.get("LANGCHAIN_PROJECT", "bunpil")
    print(f"LangSmith 프로젝트 '{project}'에서 최근 {args.days}일 트레이스 조회 중...")

    runs = fetch_runs(client, project, args.days, args.limit)
    print(f"조회된 run 수: {len(runs)}\n")

    result = aggregate(runs)
    print_report(result, project, args.days)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 저장: {args.json}")


if __name__ == "__main__":
    main()
