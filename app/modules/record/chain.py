"""생기부 윤문 Chain.

흐름: mask_pii → polish → validate → (위반 시 재시도) → 출력 + 교사 고지
보안: 마스킹은 모델 호출 전 / 입력 비저장 / 로그 PII 금지
"""
import logging
from typing import List, TypedDict

from app.common.llm import get_llm_backend, run_async
from app.common.rag import BGEEmbedder, BGEReranker, RAGRetriever, RAGStore

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
_RULE_COMPARE  = ["에 비해", "보다 낮", "보다 부족", "하위권", "상위권", "서열"]
_RULE_GUESS    = ["것 같", "로 보임", "것으로 추측", "말에 따르면"]


def _rule_violations(text: str) -> List[str]:
    """결정론적 키워드 기반 1차 위반 탐지."""
    found: List[str] = []
    if any(kw in text for kw in _RULE_NEGATIVE):
        found.append("VIOLATION: 부정적·비하적 표현 포함")
    if any(kw in text for kw in _RULE_COMPARE):
        found.append("VIOLATION: 비교·서열화 표현 포함")
    if any(kw in text for kw in _RULE_GUESS):
        found.append("VIOLATION: 추측·미확인 표현 포함")
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
        self._store = RAGStore()
        self._embedder = BGEEmbedder()
        self._reranker = BGEReranker()
        self._retriever = RAGRetriever(self._store, self._embedder, self._reranker)
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

    def _step_polish(self, state: RecordState) -> RecordState:
        """② 마스킹된 메모로 윤문 생성."""
        messages = POLISH_TPL.build(state["masked"])
        raw = run_async(self._llm.generate(messages))
        polished = raw.strip()
        return {**state, "polished": polished}

    def _step_validate(self, state: RecordState) -> RecordState:
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
                raw = run_async(self._llm.generate(messages)).strip()
                if not raw.upper().startswith("OK"):
                    violations.append(raw)
            else:
                logger.warning("regulations 컬렉션이 비어있어 LLM 검증을 건너뜁니다.")
        except Exception:
            logger.warning("regulations 검색 실패 — LLM 검증을 건너뜁니다.")

        return {**state, "violations": violations}

    # ── 공개 API ────────────────────────────────────────────────────

    def run(self, memo: str, max_retry: int = 2) -> RecordOutput:
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
            state = self._step_polish(state)
            state = self._step_validate(state)
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
