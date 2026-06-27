from typing_extensions import TypedDict


class ExamSpec(TypedDict):
    unit: str
    num_items: int
    type_dist: dict          # {"객관식": 3, "서술형": 1}
    difficulty_dist: dict    # {"상": 1, "중": 2, "하": 1}
    target: str              # "내신" | "수능형"
    standards: list          # 성취기준 목록
    passage_text: str        # 업로드 PDF 원문 (에이전트 프롬프트에 직접 삽입)


class DraftItem(TypedDict):
    item_id: str
    question: str
    options: list            # 객관식 선지. 서술형은 []
    answer: str
    item_type: str           # "객관식" | "서술형"
    difficulty: str          # "상" | "중" | "하"
    standard: str
    judge_score: float       # 0–5
    is_duplicate: bool
    status: str              # "approved" | "rejected"


class ExamState(TypedDict):
    spec: ExamSpec
    source_collection: str   # 더 이상 사용 안 함 — 하위 호환용으로 유지
    coverage_map: dict       # {standard: 승인 문항 수}
    draft_items: list        # 누적 문항 (validate 노드가 교체)
    budget: int              # 남은 재시도 횟수
    agent_messages: list     # 에이전트 메시지 (agent 노드가 교체)
    validation_passed: bool
    error: str
