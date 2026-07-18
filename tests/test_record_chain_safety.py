"""생기부 사실보존 및 fail-closed 동작 테스트."""
import asyncio

from app.modules.record.chain import RecordChain


class _FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return self.responses.pop(0)


class _FakeStore:
    def __init__(self, count=1):
        self._count = count

    def count(self, collection):
        return self._count


class _FakeRetriever:
    def __init__(self, results=None):
        self.results = results if results is not None else [{"text": "합성 규정"}]

    def retrieve(self, *args, **kwargs):
        return self.results


def _chain(responses, *, count=1, results=None):
    chain = RecordChain.__new__(RecordChain)
    chain._llm = _FakeLLM(responses)
    chain._store = _FakeStore(count)
    chain._retriever = _FakeRetriever(results)
    return chain


def _state(polished="토론에서 근거를 들어 주장함"):
    return {
        "memo": "토론에서 근거를 들어 주장함",
        "masked": "토론에서 근거를 들어 주장함",
        "pii_found": [],
        "polished": polished,
        "violations": [],
        "generated_pii": [],
        "validation_status": "pending",
        "attempt": 0,
    }


def test_fact_addition_is_rejected():
    chain = _chain(["YES"])
    result = asyncio.run(chain._step_validate(_state("토론 대회에서 1등을 수상함")))

    assert result["validation_status"] == "violations_found"
    assert any("새로운 사실" in v for v in result["violations"])


def test_missing_regulations_is_validation_unavailable():
    chain = _chain(["NO"], count=0)
    result = asyncio.run(chain._step_validate(_state()))

    assert result["validation_status"] == "unavailable"
    assert any("규정 자료 없음" in v for v in result["violations"])


def test_failed_result_is_suppressed():
    chain = _chain(["메모에 없는 수상 사실을 추가함", "YES"])
    result = asyncio.run(chain.run("토론에서 근거를 들어 주장함", max_retry=1))

    assert result["polished"] == ""
    assert result["validation_status"] == "violations_found"


def test_retry_receives_violations_and_can_recover():
    chain = _chain([
        "토론 대회에서 1등을 수상함",
        "YES",
        "토론에서 근거를 들어 주장함",
        "NO",
        "OK",
    ])
    result = asyncio.run(chain.run("토론에서 근거를 들어 주장함", max_retry=2))

    retry_messages = chain._llm.calls[2]
    assert "이전 검증 결과" in retry_messages[-1]["content"]
    assert result["polished"] == "토론에서 근거를 들어 주장함"
    assert result["validation_status"] == "passed"


def test_generated_pii_is_remasked_before_validation():
    chain = _chain(["연락처 01012345678로 자료를 제출함"])
    result = asyncio.run(chain.run("자료를 제출함", max_retry=1))

    assert result["polished"] == ""
    assert "전화번호" in result["pii_found"]
    assert result["validation_status"] == "violations_found"
