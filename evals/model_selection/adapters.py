"""모델 adapter — 제공사 차이를 여기서 흡수한다.

모든 adapter는 두 가지 얼굴을 가진다.
- `generate(messages, **kw)` (async): 프로덕션 코드의 `LLMBackend`와 같은 모양. `explain_item`·
  `_analyze_request`·`judge_node`에 그대로 끼운다.
- `chat_model()`: 에이전트(`graph.py agent_node`)가 `bind_tools`로 쓰는 LangChain 채팅 모델.

지금은 OpenRouter(기본)·Ollama(로컬 기준 모델)·Mock 세 가지다. 벤더 공식 API로 최종 후보를 재검증할 때는
같은 인터페이스로 adapter를 하나 더 만들면 된다.
"""
import asyncio
import json
import re
import time

import httpx

from . import openrouter
from .recorder import CallRecord, Recorder, current_task

_RETRIABLE = ("429", "500", "502", "503", "504", "timeout", "timed out", "connection")


def _estimate_cost(spec: dict, prompt_tokens, completion_tokens) -> float | None:
    if prompt_tokens is None or completion_tokens is None:
        return None
    p = spec.get("pricing_per_mtok") or {}
    return (prompt_tokens * p.get("input", 0) + completion_tokens * p.get("output", 0)) / 1_000_000


class BaseAdapter:
    mock = False

    def __init__(self, key: str, spec: dict, recorder: Recorder, *, role: str, cfg: dict, sensitive: bool = False):
        self.key, self.spec, self.recorder, self.role, self.cfg, self.sensitive = key, spec, recorder, role, cfg, sensitive
        defaults = cfg["defaults"]["judge" if role == "judge" else "generation"]
        self.temperature = defaults["temperature"]

    def _new_record(self, params: dict) -> CallRecord:
        return CallRecord(
            run_id=self.recorder.run_id, model_key=self.key,
            requested_model=self.spec["model"], requested_endpoint=self.spec["endpoint"],
            # judge_structure()처럼 별도 스레드에서 도는 호출은 contextvar가 전달되지 않아 active_task로 대신한다
            params=params, task=dict(current_task.get() or getattr(self, "active_task", {})),
            started_at=time.time(), mock=self.mock,
        )

    async def generate(self, messages: list[dict], **kw) -> str:
        return await asyncio.to_thread(self.complete, messages, **kw)

    def complete(self, messages, *, max_tokens=1024, response_format=None, **_):  # pragma: no cover - 인터페이스
        raise NotImplementedError

    def chat_model(self):  # pragma: no cover - 인터페이스
        raise NotImplementedError


# ── OpenRouter ──────────────────────────────────────────────────────────

class OpenRouterAdapter(BaseAdapter):
    def _client(self):
        from openai import OpenAI

        o = self.cfg["openrouter"]
        return OpenAI(base_url=o["base_url"], api_key=openrouter.api_key(), timeout=o["timeout_s"], max_retries=0)

    def request_kwargs(self, *, max_tokens: int, response_format=None) -> dict:
        """실제로 보낼 요청 인자(키 제외). 테스트·dry-run에서도 쓴다."""
        kwargs = {
            "model": self.spec["model"],
            "max_tokens": max_tokens,
            "extra_body": openrouter.extra_body(self.spec, sensitive=self.sensitive),
            **openrouter.sampling_params(self.spec, self.temperature),
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    def complete(self, messages, *, max_tokens=1024, response_format=None, **_):
        self.recorder.check_budget()
        kwargs = self.request_kwargs(max_tokens=max_tokens, response_format=response_format)
        rec = self._new_record({k: v for k, v in kwargs.items() if k != "model"})
        client = self._client()
        text, last_err = None, None
        for attempt in range(self.cfg["openrouter"]["max_retries"] + 1):
            t0 = time.perf_counter()
            try:
                resp = client.chat.completions.create(messages=messages, **kwargs)
                rec.latency_s = time.perf_counter() - t0
                text = resp.choices[0].message.content or ""
                rec.actual_model = resp.model
                rec.actual_provider = (resp.model_extra or {}).get("provider")
                rec.generation_id = resp.id
                if resp.usage:
                    rec.prompt_tokens, rec.completion_tokens = resp.usage.prompt_tokens, resp.usage.completion_tokens
                    details = getattr(resp.usage, "completion_tokens_details", None)
                    rec.reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
                    cost = (resp.usage.model_extra or {}).get("cost")
                    rec.cost_usd, rec.cost_source = (cost, "openrouter") if cost is not None else (
                        _estimate_cost(self.spec, rec.prompt_tokens, rec.completion_tokens), "estimate")
                break
            except Exception as e:  # noqa: BLE001 — 오류는 기록하고 재시도 여부만 판단
                last_err = f"{type(e).__name__}: {str(e)[:300]}"
                if attempt < self.cfg["openrouter"]["max_retries"] and any(s in last_err.lower() for s in _RETRIABLE):
                    rec.retries += 1
                    time.sleep(2 ** attempt)
                    continue
                break
        if text is None:
            rec.error = last_err
        else:
            rec.route_ok, rec.route_note = openrouter.route_matches(self.spec, rec.actual_model, rec.actual_provider)
        self.recorder.add(rec)
        if text is None:
            raise RuntimeError(f"{self.key} 호출 실패: {last_err}")
        return text

    def chat_model(self):
        """에이전트용 — 프로젝트 의존성인 langchain-openai의 ChatOpenAI를 OpenRouter로 향하게 한다."""
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        o = self.cfg["openrouter"]
        return ChatOpenAI(
            model=self.spec["model"],
            base_url=o["base_url"],
            api_key=SecretStr(openrouter.api_key()),
            max_tokens=self.cfg["defaults"]["generation"]["max_tokens"]["agent"],
            temperature=self.temperature if self.spec.get("supports_temperature") else None,
            extra_body=openrouter.extra_body(self.spec, sensitive=self.sensitive),
            timeout=o["timeout_s"],
            max_retries=o["max_retries"],
            callbacks=[RecordingCallback(self)],
        )


def fetch_generation(generation_id: str, *, attempts: int = 4) -> dict | None:
    """GET /generation — 실제 provider·비용 확인. 통계가 늦게 올라올 수 있어 몇 번 기다린다."""
    for i in range(attempts):
        try:
            r = httpx.get(
                "https://openrouter.ai/api/v1/generation",
                params={"id": generation_id},
                headers={"Authorization": f"Bearer {openrouter.api_key()}"},
                timeout=30,
            )
            if r.status_code == 200:
                return r.json().get("data")
        except httpx.HTTPError:
            pass
        time.sleep(1.5 * (i + 1))
    return None


# ── 에이전트 호출 기록(LangChain 콜백) ──────────────────────────────────

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # pragma: no cover
    BaseCallbackHandler = object


class RecordingCallback(BaseCallbackHandler):
    """에이전트의 LLM 호출마다 CallRecord를 남긴다. provider는 응답에 없을 수 있어 나중에 /generation으로 확인."""

    def __init__(self, adapter: BaseAdapter):
        self.adapter = adapter
        self._start: dict = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
        self.adapter.recorder.check_budget()
        self._start[run_id] = (time.time(), time.perf_counter())

    def on_llm_end(self, response, *, run_id, **kw):
        started, t0 = self._start.pop(run_id, (time.time(), time.perf_counter()))
        rec = self.adapter._new_record({"path": "agent"})
        rec.started_at, rec.latency_s = started, time.perf_counter() - t0
        msg = response.generations[0][0].message if response.generations and response.generations[0] else None
        meta = getattr(msg, "response_metadata", {}) or {}
        usage = getattr(msg, "usage_metadata", None) or {}
        rec.actual_model = meta.get("model_name") or meta.get("model")
        rec.actual_provider = meta.get("provider")
        rec.generation_id = meta.get("id")
        rec.prompt_tokens, rec.completion_tokens = usage.get("input_tokens"), usage.get("output_tokens")
        rec.cost_usd = _estimate_cost(self.adapter.spec, rec.prompt_tokens, rec.completion_tokens)
        rec.cost_source = "local" if self.adapter.spec["access"] == "ollama" else "estimate"
        if self.adapter.spec["access"] == "ollama":
            rec.route_ok, rec.actual_provider = True, "local-ollama"
        self.adapter.recorder.add(rec)

    def on_llm_error(self, error, *, run_id, **kw):
        started, t0 = self._start.pop(run_id, (time.time(), time.perf_counter()))
        rec = self.adapter._new_record({"path": "agent"})
        rec.started_at, rec.latency_s = started, time.perf_counter() - t0
        rec.error = f"{type(error).__name__}: {str(error)[:300]}"
        self.adapter.recorder.add(rec)


# ── Ollama(로컬 기준 모델) ──────────────────────────────────────────────

class OllamaAdapter(BaseAdapter):
    def complete(self, messages, *, max_tokens=1024, response_format=None, **_):
        import os

        rec = self._new_record({"max_tokens": max_tokens, "temperature": self.temperature})
        payload = {
            "model": self.spec["model"], "messages": messages, "stream": False,
            "options": {"num_predict": max_tokens, "num_ctx": 16384, "temperature": self.temperature},
        }
        if response_format is not None:
            payload["format"] = "json"
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/chat", json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            rec.latency_s, rec.error = time.perf_counter() - t0, f"{type(e).__name__}: {str(e)[:300]}"
            self.recorder.add(rec)
            raise RuntimeError(f"{self.key} 호출 실패: {rec.error}") from e
        rec.latency_s = time.perf_counter() - t0
        rec.actual_model, rec.actual_provider, rec.route_ok = data.get("model"), "local-ollama", True
        rec.prompt_tokens, rec.completion_tokens = data.get("prompt_eval_count"), data.get("eval_count")
        rec.cost_usd, rec.cost_source = 0.0, "local"
        self.recorder.add(rec)
        return data["message"]["content"]

    def chat_model(self):
        from langchain_ollama import ChatOllama
        import os

        return ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), model=self.spec["model"],
            num_predict=self.cfg["defaults"]["generation"]["max_tokens"]["agent"], num_ctx=16384,
            temperature=self.temperature, callbacks=[RecordingCallback(self)],
        )


# ── Mock(파이프라인 검증 전용 — 순위 산출 금지) ─────────────────────────

class MockAdapter(BaseAdapter):
    """실제 모델을 부르지 않는다. 결과에 mock=True가 붙고, summarize는 mock 결과로 순위를 매기지 않는다.

    생성: 현재 작업의 기대 형식(expected)대로 문항을 만드는 도구 호출을 흉내 낸다 — 진짜 그래프·게이트를 통과시켜
    배선을 검증하기 위함이다. 단발 호출: 프롬프트 종류를 보고 형식에 맞는 고정 응답을 돌려준다."""

    mock = True

    def complete(self, messages, *, max_tokens=1024, response_format=None, **_):
        rec = self._new_record({"max_tokens": max_tokens, "mock": True})
        system = messages[0]["content"] if messages else ""
        joined = "\n".join(m["content"] for m in messages)
        expected = current_task.get().get("expected", {})
        if "requested_num_items" in system:  # 요청 분석(main._ANALYZE_PROMPT) — 정답 형식을 그대로 돌려준다
            fmt = expected.get("format") or {}
            text = json.dumps({"requested_num_items": expected.get("num_items", 2), **fmt,
                               "unsupported": ["모의 미지원"] if expected.get("unsupported") else [], "nearest": "4지 선다"},
                              ensure_ascii=False)
        elif "해설을 한국어로" in system:
            text = "정답의 근거: 모의 해설입니다.\n나머지 선지는 개념 범주가 다릅니다."
        elif "type_ratio_score" in system:
            text = json.dumps({"type_ratio_score": 1.0, "difficulty_match": True, "overall_score": 4})
        elif "JUDGE_TASK" in joined:
            text = _mock_judge_response(joined)
        else:
            text = "모의 응답"
        rec.latency_s, rec.actual_model, rec.actual_provider = 0.001, self.spec["model"], "mock"
        rec.prompt_tokens, rec.completion_tokens = len(joined) // 2, len(text) // 2
        rec.cost_usd, rec.cost_source, rec.route_ok = 0.0, "mock", True
        self.recorder.add(rec)
        return text

    def chat_model(self):
        return MockToolChatModel(adapter=self)


def _mock_judge_response(prompt: str) -> str:
    kind = re.search(r"JUDGE_TASK=(\w+)", prompt).group(1)
    seed = sum(map(ord, prompt)) % 7  # 결정론적이되 항목마다 다르게
    if kind == "item":
        yes = "yes" if seed else "no"
        return json.dumps({k: {"verdict": yes, "reason": "모의"} for k in ("I1", "I2", "I3", "I4", "I5", "I6", "E1", "E2", "E3", "E4")})
    if kind == "solve":
        return json.dumps({"answer": "①②③④⑤"[seed % 4], "confidence": 2})
    return json.dumps({"winner": ["A", "B", "tie", "both_bad", "unsure"][seed % 5], "reason": "모의"})


_MOCK_STEMS = [
    "민주 정치의 원리 가운데 권력 분립이 필요한 까닭으로 가장 적절한 것은?",
    "시장 경제에서 가격이 자원 배분에 하는 역할을 옳게 설명한 것은?",
    "지역 간 인구 이동이 도시 구조에 미치는 영향으로 옳은 것은?",
    "윤리적 판단에서 결과를 중시하는 관점의 특징으로 알맞은 것은?",
    "문화 변동 과정에서 나타나는 전파 현상의 사례로 적절한 것은?",
]

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class MockToolChatModel(BaseChatModel):
        """에이전트용 가짜 채팅 모델: 필요한 개수만큼 save_item을 호출하고 submit_for_review로 끝낸다."""

        adapter: object = None
        calls: int = 0

        @property
        def _llm_type(self) -> str:
            return "mock-tool-chat"

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            from app.modules.exam.tools import _OPTION_MARKS, get_draft_items

            expected = (current_task.get() or getattr(self.adapter, "active_task", {})).get("expected", {})
            n_target = expected.get("num_items", 2)
            saved = len(get_draft_items())
            self.calls += 1
            if saved < n_target:
                n_opt = expected.get("num_options", 4)
                combo = expected.get("combo_options", False)
                marks = _OPTION_MARKS[:n_opt]
                letters = ["ㄱ", "ㄴ", "ㄷ", "ㄹ"]
                options = (
                    [f"{m} {letters[i % 3]}, {letters[(i + 1) % 3]}" for i, m in enumerate(marks)] if combo
                    else [f"{m} 모의 선지 {saved}-{i} 내용입니다" for i, m in enumerate(marks)]
                )
                args = {
                    # 실제 중복 게이트(Jaccard 0.8)를 통과하도록 문항마다 다른 문장을 쓴다
                    "question": _MOCK_STEMS[saved % len(_MOCK_STEMS)],
                    "options": options, "answer": marks[0], "item_type": "객관식", "difficulty": "중",
                    "stimulus": "<보기>\nㄱ. 모의 진술 하나입니다\nㄴ. 모의 진술 둘입니다\nㄷ. 모의 진술 셋입니다"
                    if expected.get("needs_stimulus") else "",
                }
                call = {"name": "save_item", "args": args, "id": f"mock-{self.calls}"}
            else:
                call = {"name": "submit_for_review", "args": {}, "id": f"mock-{self.calls}"}
            self.adapter.complete([{"role": "user", "content": "agent-turn"}], max_tokens=1)  # 호출 기록만 남긴다
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])
except ImportError:  # pragma: no cover
    pass


def make_adapter(key: str, cfg: dict, recorder: Recorder, *, role: str, mock: bool = False, sensitive: bool = False):
    spec = cfg["models"][key]
    if role not in spec["roles"]:
        raise ValueError(f"{key}는 {role} 역할 후보가 아닙니다(config roles={spec['roles']}).")
    if mock:
        return MockAdapter(key, spec, recorder, role=role, cfg=cfg, sensitive=sensitive)
    if spec["access"] == "openrouter":
        return OpenRouterAdapter(key, spec, recorder, role=role, cfg=cfg, sensitive=sensitive)
    if spec["access"] == "ollama":
        return OllamaAdapter(key, spec, recorder, role=role, cfg=cfg, sensitive=sensitive)
    raise ValueError(f"알 수 없는 access: {spec['access']}")
