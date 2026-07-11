"""생기부 윤문 Chain.

흐름: mask_pii → polish → validate → (위반 시 재시도) → 출력 + 교사 고지
보안: 마스킹은 모델 호출 전 / 입력 비저장 / 로그 PII 금지
"""
import logging
import re
from typing import List, TypedDict

from app.common.llm import get_llm_backend
from app.common.rag import get_retriever, get_store

from .masker import mask_pii
from .prompts import POLISH_TPL, VALIDATE_TPL

logger = logging.getLogger(__name__)

REGULATION_COLLECTION = "regulations"

WARNING = (
    "\n\n[교사 확인 사항]\n"
    "이 문장은 AI 보조 도구로 생성된 초안입니다. "
    "최종 기재 여부와 내용의 정확성은 담당 교사가 반드시 확인·책임져야 합니다."
)

# ── 규칙 기반 위반 탐지 (LLM 보완 — 명백한 패턴 결정론적 처리) ──
_RULE_NEGATIVE = ["불성실", "부족", "낮은 편", "어려움이 있음", "개선이 필요", "주의가 필요", "발전이 필요", "보충이 필요"]
_RULE_COMPARE  = ["에 비해", "보다 낮", "보다 부족", "하위권", "상위권", "서열", "비교할 때"]
# "학생보다 ... 느린" 처럼 비교 표현과 열등 서술어가 떨어져 있는 경우까지 잡기 위한 근접 매칭
_RULE_COMPARE_RE = re.compile(r"보다\s?.{0,25}?(느리|느린|느려|낮|부족|못하|뒤처|열등)")
_RULE_GUESS    = ["것 같", "로 보임", "것으로 추측", "말에 따르면"]
# 가정환경·종교·정치성향 언급은 내용의 긍/부정과 무관하게 그 자체가 위반(사생활·중립성 규정)
_RULE_BACKGROUND = ["가정형편", "가정환경", "편부모", "한부모", "저소득", "결손가정", "다문화가정"]
_RULE_RELIGION_POLITICS = ["종교적", "종교 활동", "신앙", "정치적", "정치성향", "지지 정당"]
# 외모·신체 언급은 호의적 서술이어도 그 자체가 위반(생기부 기재요령 — 외모 평가 금지)
_RULE_APPEARANCE = ["외모", "키가 작", "키가 커", "체격이", "생김새", "인상이 좋", "잘생", "예쁘"]


def _rule_violations(text: str) -> List[str]:
    """결정론적 키워드 기반 1차 위반 탐지."""
    found: List[str] = []
    if any(kw in text for kw in _RULE_NEGATIVE):
        found.append("VIOLATION: 부정적·비하적 표현 포함")
    if any(kw in text for kw in _RULE_COMPARE) or _RULE_COMPARE_RE.search(text):
        found.append("VIOLATION: 비교·서열화 표현 포함")
    if any(kw in text for kw in _RULE_GUESS):
        found.append("VIOLATION: 추측·미확인 표현 포함")
    if any(kw in text for kw in _RULE_BACKGROUND):
        found.append("VIOLATION: 가정환경 언급 포함")
    if any(kw in text for kw in _RULE_RELIGION_POLITICS):
        found.append("VIOLATION: 종교·정치성향 언급 포함")
    if any(kw in text for kw in _RULE_APPEARANCE):
        found.append("VIOLATION: 외모·신체 언급 포함")
    _, pii = mask_pii(text)
    if pii:
        found.append(f"VIOLATION: 개인정보({', '.join(pii)}) 포함")
    return found



class RecordState(TypedDict):
    memo: str
    masked: str
    pii_found: List[str]
    polished: str
    violations: List[str]
    attempt: int


class RecordOutput(TypedDict):
    masked_memo: str
    pii_found: List[str]
    polished: str
    violations: List[str]
    warning: str


class RecordChain:
    def __init__(self):
        self._store = get_store()
        self._retriever = get_retriever()
        self._llm = get_llm_backend()
        if self._store.count(REGULATION_COLLECTION) == 0:
            logger.warning(
                "regulations 컬렉션이 비어있습니다. "
                "scripts/index_regulations.py를 실행한 뒤 다시 시도하세요."
            )

    # ── 처리 스텝 ────────────────────────────────────────────────────

    def _step_mask(self, state: RecordState) -> RecordState:
        """① PII 마스킹 — 모델 호출 전 반드시 실행."""
        masked, found = mask_pii(state["memo"])
        if found:
            logger.info("PII 감지 유형: %s (내용 비기록)", found)
        return {**state, "masked": masked, "pii_found": found}

    async def _step_polish(self, state: RecordState) -> RecordState:
        """② 마스킹된 메모로 윤문 생성."""
        messages = POLISH_TPL.build(state["masked"])
        raw = await self._llm.generate(messages)
        polished = raw.strip()
        return {**state, "polished": polished}

    async def _step_validate(self, state: RecordState) -> RecordState:
        """③ 규정 RAG 검증 — 하이브리드(규칙+LLM) 위반 플래그 추출."""
        # 1단계: 결정론적 규칙 기반 (빠르고 확실한 패턴)
        violations: List[str] = _rule_violations(state["polished"])

        # 2단계: LLM 기반 (뉘앙스·복합 위반)
        try:
            results = self._retriever.retrieve(
                state["polished"], REGULATION_COLLECTION, top_k=3, n_candidates=10
            )
            if results:
                reg_text = "\n".join(r["text"] for r in results[:3])
                prompt = f"[규정]\n{reg_text}\n\n[문장]\n{state['polished']}"
                messages = VALIDATE_TPL.build(prompt)
                raw = (await self._llm.generate(messages)).strip()
                if not raw.upper().startswith("OK"):
                    violations.append(raw)
            else:
                logger.warning("regulations 컬렉션이 비어있어 LLM 검증을 건너뜁니다.")
        except Exception:
            logger.warning("regulations 검색 실패 — LLM 검증을 건너뜁니다.")

        return {**state, "violations": violations}

    # ── 공개 API ────────────────────────────────────────────────────

    async def run(self, memo: str, max_retry: int = 2) -> RecordOutput:
        """메모를 입력받아 윤문 결과를 반환. 위반 시 최대 max_retry 재시도."""
        state: RecordState = {
            "memo": memo,
            "masked": "",
            "pii_found": [],
            "polished": "",
            "violations": [],
            "attempt": 0,
        }

        # mask 는 한 번만
        state = self._step_mask(state)

        for attempt in range(max_retry):
            state["attempt"] = attempt
            state = await self._step_polish(state)
            state = await self._step_validate(state)
            if not state["violations"]:
                break

        return RecordOutput(
            masked_memo=state["masked"],
            pii_found=state["pii_found"],
            polished=state["polished"],
            violations=state["violations"],
            warning=WARNING,
        )


_instance: RecordChain = None


# BGEEmbedder/BGEReranker 로딩 비용이 크므로 프로세스당 한 번만 생성
def get_record_chain() -> RecordChain:
    global _instance
    if _instance is None:
        _instance = RecordChain()
    return _instance
