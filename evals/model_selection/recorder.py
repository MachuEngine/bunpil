"""호출 기록·비용 상한·작업 결과 저장(재실행 시 재사용).

- 호출 1건 = CallRecord 1줄(`calls.jsonl`). 요청 모델·실제 처리 모델·provider·파라미터·토큰·비용·지연·오류.
- 작업 1건(세트 생성·해설·수정·Judge 판정) = TaskStore 1줄(`tasks.jsonl`). 같은 task_key가 있으면 다시 실행하지
  않고 저장된 결과를 쓴다 → 같은 조건 재실행이 호출 없이 재현된다.
- 인증 정보는 어떤 필드에도 들어가지 않는다(레코드에 헤더·키를 담지 않는다).
"""
import contextvars
import dataclasses
import hashlib
import json
import os
import threading
import time

RESULTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "evals", "results", "model_selection"
)

# 현재 호출이 어떤 작업에 속하는지 — adapter가 CallRecord에 붙인다
current_task: contextvars.ContextVar[dict] = contextvars.ContextVar("current_task", default={})


class BudgetExceeded(RuntimeError):
    pass


@dataclasses.dataclass
class CallRecord:
    run_id: str
    model_key: str
    requested_model: str
    requested_endpoint: str
    params: dict
    task: dict
    started_at: float
    latency_s: float | None = None
    actual_model: str | None = None
    actual_provider: str | None = None
    generation_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None
    cost_source: str | None = None  # "openrouter" | "estimate" | "local"
    route_ok: bool | None = None
    route_note: str = ""
    retries: int = 0
    error: str | None = None
    mock: bool = False


def stable_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


class Recorder:
    """실행 하나(run_id)의 호출 기록과 누적 비용을 관리한다. 스레드 안전."""

    def __init__(self, run_id: str, max_cost: float, root: str = RESULTS_ROOT):
        self.run_id = run_id
        self.max_cost = max_cost
        self.dir = os.path.join(root, run_id)
        os.makedirs(self.dir, exist_ok=True)
        self._lock = threading.Lock()
        self.spent = 0.0
        self.records: list[CallRecord] = []

    def check_budget(self):
        if self.spent >= self.max_cost:
            raise BudgetExceeded(f"누적 비용 ${self.spent:.4f}가 상한 ${self.max_cost:.2f}에 도달해 중단합니다.")

    def add(self, rec: CallRecord):
        with self._lock:
            self.records.append(rec)
            self.spent += rec.cost_usd or 0.0
            with open(os.path.join(self.dir, "calls.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(dataclasses.asdict(rec), ensure_ascii=False) + "\n")
            if rec.error:
                with open(os.path.join(self.dir, "errors.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"model_key": rec.model_key, "task": rec.task, "error": rec.error}, ensure_ascii=False) + "\n")

    def write_meta(self, meta: dict):
        with open(os.path.join(self.dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({**meta, "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, ensure_ascii=False, indent=2)


class TaskStore:
    """작업 결과 캐시. task_key → 결과 dict."""

    def __init__(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, "tasks.jsonl")
        self._lock = threading.Lock()
        self.data: dict[str, dict] = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.data[row["task_key"]] = row

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, row: dict):
        with self._lock:
            row = {**row, "task_key": key}
            self.data[key] = row
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
