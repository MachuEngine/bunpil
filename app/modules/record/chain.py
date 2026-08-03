"""생기부 윤문 Chain.

흐름: mask_pii → polish → validate → (위반 시 재시도) → 출력 + 교사 고지
보안: 마스킹은 모델 호출 전 / 입력 비저장 / 로그 PII 금지
"""
import logging
import re
from typing import List, Literal, TypedDict

from app.common.llm import get_llm_backend
from app.common.rag import get_retriever, get_store

from .masker import mask_pii
from .prompts import FACT_CHECK_TPL, POLISH_TPL, VALIDATE_TPL

logger = logging.getLogger(__name__)

REGULATION_COLLECTION = "regulations"

WARNING = (
    "\n\n[교사 확인 사항]\n"
    "이 문장은 AI 보조 도구로 생성된 초안입니다. "
    "최종 기재 여부와 내용의 정확성은 담당 교사가 반드시 확인·책임져야 합니다."
)

# ── 규칙 기반 위반 탐지 (LLM 보완 — 명백한 패턴 결정론적 처리) ──
#
# ⚠️ 출처 주의 (2026-08-03 확인): 아래 6개 규칙 중 인덱싱된 교육부 기재요령에
# 실제 근거 조항이 있는 것은 _RULE_BACKGROUND 하나뿐이다. `기재요령_고등학교_2024.pdf`
# 원문 262,678자를 직접 검색한 결과 '종교'·'신앙'·'외모'·'용모'·'추측'은 0회 등장한다
# ('정치'는 3회 나오지만 전부 "정치활동으로 결석 시 출결 처리" 맥락이라 여기서 막으려는
# 것과 다르다). EVAL.md 14절 참고.
#
# 그럼에도 규칙을 유지하는 이유: 규정에 조항이 없다고 해서 학생의 종교·정치성향·외모를
# 공식 기록에 적는 것이 적절해지지는 않는다. 근거가 없다는 이유로 규칙을 빼면 시스템이
# 덜 안전해질 뿐이므로, 안전한 기본값으로 유지하고 출처를 정직하게 표기하는 쪽을 택했다.
#
# 미해결: 이 규칙들이 실제 교육 현장 규범과 일치하는지는 확인되지 않았다. 이 프로젝트에서
# 근거로 삼은 것은 VIOLATION_GOLDEN(Claude 합성)의 라벨이지 규정이 아니며, 그 골든셋으로
# 채점한 Recall 0.927은 "합성 정답지를 규칙이 맞힌 것"에 가깝다(순환). 지인 교사 확인 후
# 재판단 필요.
_RULE_NEGATIVE = ["불성실", "부족", "낮은 편", "어려움이 있음", "개선이 필요", "주의가 필요", "발전이 필요", "보충이 필요"]
_RULE_COMPARE  = ["에 비해", "보다 낮", "보다 부족", "하위권", "상위권", "서열", "비교할 때"]
# "학생보다 ... 느린" 처럼 비교 표현과 열등 서술어가 떨어져 있는 경우까지 잡기 위한 근접 매칭
_RULE_COMPARE_RE = re.compile(r"보다\s?.{0,25}?(느리|느린|느려|낮|부족|못하|뒤처|열등)")
_RULE_GUESS    = ["것 같", "로 보임", "것으로 추측", "말에 따르면"]
# 가정환경 언급 — 유일하게 규정 근거가 확인된 규칙.
# 근거: 기재요령 p24 "차. 부모(친인척 포함)의 사회･경제적 지위(직종명, 직업명, 직장명,
# 직위명 등) 암시 내용"은 어떠한 항목에도 기재할 수 없음.
# (교사는 "가정형편"이라 쓰고 규정은 "부모의 사회·경제적 지위"라 써서 공유 토큰이 없다 —
#  RAG 검색이 이 유형을 못 잡는 이유이기도 하다. EVAL.md 14절)
_RULE_BACKGROUND = ["가정형편", "가정환경", "편부모", "한부모", "저소득", "결손가정", "다문화가정"]
# 종교·정치성향 언급 — 규정 근거 없음(위 출처 주의 참고). 민감정보라는 판단에 따른 방어적 규칙.
_RULE_RELIGION_POLITICS = ["종교적", "종교 활동", "신앙", "정치적", "정치성향", "지지 정당"]
# 외모·신체 언급 — 규정 근거 없음(위 출처 주의 참고). 호의적 서술이어도 막는다는 것은
# 기재요령 조항이 아니라 이 프로젝트의 방어적 선택이다.
_RULE_APPEARANCE = ["외모", "키가 작", "키가 커", "체격이", "생김새", "인상이 좋", "잘생", "예쁘"]


def _rule_warnings(text: str) -> List[str]:
    """결정론적 키워드 기반 주의 표현 탐지 — **차단이 아니라 경고**(2026-08-03 변경).

    이전에는 `_rule_violations()`로서 하나라도 걸리면 윤문 결과를 통째로 숨겼다.
    그런데 키워드는 **서술 대상을 구분하지 못한다**는 것이 실측으로 드러났다:

        "사회 수업에서 정치적 다원주의 개념을 조사해 발표함"      → 차단됨(오탐)
        "가정환경에 따른 교육 격차를 주제로 보고서를 작성함"      → 차단됨(오탐)
        "아버지가 대기업 임원이라 경제에 관심이 많음"            → 통과됨(미탐)

    분필은 **사회 교사용** 도구라 정치·종교·가정환경은 교과가 다루는 주제어이기도
    하다. "학생 본인의 속성"과 "학생이 탐구한 주제"는 같은 단어를 쓰므로 `in`
    연산으로는 나눌 수 없다 — 규정 근거가 있는 `_RULE_BACKGROUND`도 똑같이
    오작동했다(위 예시 2·3번). 따라서 이 판단은 코드가 최종 결정할 수 있는
    성질이 아니라고 보고, **탐지는 하되 결정은 교사에게 넘긴다**.

    차단을 계속 유지하는 것은 CLAUDE.md 하드룰에 직접 걸리는 항목뿐이다
    (PII 생성·잔존, 메모에 없는 사실 추가, 검증 시스템 실패) — `_step_validate` 참고.
    """
    found: List[str] = []
    if any(kw in text for kw in _RULE_NEGATIVE):
        found.append("WARNING: 부정적·비하적 표현일 수 있음")
    if any(kw in text for kw in _RULE_COMPARE) or _RULE_COMPARE_RE.search(text):
        found.append("WARNING: 비교·서열화 표현일 수 있음")
    if any(kw in text for kw in _RULE_GUESS):
        found.append("WARNING: 추측·미확인 표현일 수 있음")
    if any(kw in text for kw in _RULE_BACKGROUND):
        found.append("WARNING: 가정환경 언급일 수 있음(기재요령 p24 — 부모의 사회·경제적 지위 암시 금지)")
    if any(kw in text for kw in _RULE_RELIGION_POLITICS):
        found.append("WARNING: 종교·정치성향 언급일 수 있음")
    if any(kw in text for kw in _RULE_APPEARANCE):
        found.append("WARNING: 외모·신체 언급일 수 있음")
    return found


def _pii_violations(text: str) -> List[str]:
    """윤문 결과에 남은 개인정보 탐지 — 이건 **차단 유지**(하드룰 2·4).

    키워드 규칙과 달리 PII는 문맥에 따라 괜찮아지는 종류가 아니고,
    마스킹 누락은 곧 개인정보 유출이므로 경고로 낮추지 않는다.
    """
    _, pii = mask_pii(text)
    return [f"VIOLATION: 개인정보({', '.join(pii)}) 포함"] if pii else []


""" 
필드	             의미
------------------------------------------
memo	            사용자가 입력한 원본 메모
masked	            개인정보를 가린 메모
pii_found	        발견된 개인정보 유형
polished	        LLM이 윤문한 문장
violations	        결과를 숨겨야 하는 차단 사유 (하드룰 위반 — PII·사실추가·검증불가)
warnings	        교사가 확인할 주의 사항 (차단하지 않음 — 키워드 규칙·규정 판정)
generated_pii	    LLM이 새로 생성한 개인정보
validation_status	검증 상태
attempt	            현재 시도 횟수
------------------------------------------
validation_status
    - pending: 아직 검사하지 않음
    - passed: 검증 통과 (warnings가 있어도 passed — 결과는 정상 반환됨)
    - violations_found: 차단 사유 발견 → 결과 숨김
    - unavailable: 검증 시스템 오류 또는 자료 부족 → 결과 숨김
"""
# 처리 과정에서 지속 전달되는 상태값 구조
class RecordState(TypedDict):
    memo: str
    masked: str
    pii_found: List[str]
    polished: str
    violations: List[str]
    warnings: List[str]
    generated_pii: List[str]
    validation_status: Literal["pending", "passed", "violations_found", "unavailable"]
    attempt: int

# 최종적으로 사용자에게 반환할 결과 구조
class RecordOutput(TypedDict):
    masked_memo: str
    pii_found: List[str]
    polished: str
    violations: List[str]
    warnings: List[str]  # 차단하지 않는 주의 사항 — 교사가 보고 판단
    validation_status: Literal["passed", "violations_found", "unavailable"]
    warning: str  # 교사 책임 고지 문구(하드룰 5) — 위 warnings와 다름


class RecordChain:
    def __init__(self):
        # RecordChain 객체가 생성될 때 다음 구성요소를 준비 
        # 규정 저장소 / 규정 검색기 / 언어 모델 
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
        user_input = state["masked"]
        if state["violations"]:
            feedback = "\n".join(f"- {v}" for v in state["violations"])
            user_input = (
                f"[메모]\n{state['masked']}\n\n"
                f"[이전 검증 결과 — 아래 문제를 고쳐 다시 윤문]\n{feedback}"
            )
        messages = POLISH_TPL.build(user_input)
        raw = await self._llm.generate(messages)
        polished, generated_pii = mask_pii(raw.strip())
        pii_found = list(dict.fromkeys([*state["pii_found"], *generated_pii]))
        return {
            **state,
            "polished": polished,
            "pii_found": pii_found,
            "generated_pii": generated_pii,
            "validation_status": "pending",
        }

    async def _step_validate(self, state: RecordState) -> RecordState:
        """③ 사실보존·규정 검증.

        **차단(violations)과 경고(warnings)를 구분한다**(2026-08-03 변경):
        - 차단 = 결과를 숨김. CLAUDE.md 하드룰에 직접 걸리는 것만 —
          PII 생성·잔존(하드룰 2·4), 메모에 없는 사실 추가(하드룰 5),
          검증 시스템 실패(fail-closed 유지).
        - 경고 = 결과는 그대로 주되 교사가 확인. 키워드 규칙과 LLM 규정 판정은
          "학생 본인의 속성"과 "학생이 탐구한 주제"를 구분하지 못해 오탐이 잦고
          (`_rule_warnings` 참고), 최종 기재 책임은 어차피 교사에게 있으므로
          코드가 결과를 숨기는 대신 판단 재료를 넘긴다.
        """
        # 1단계: 결정론적 규칙 — 키워드는 경고, PII는 차단
        warnings: List[str] = _rule_warnings(state["polished"])
        violations: List[str] = _pii_violations(state["polished"])
        if state["generated_pii"]:
            violations.append(
                f"VIOLATION: 윤문 결과에 개인정보({', '.join(state['generated_pii'])}) 생성"
            )
        if violations:
            return {
                **state,
                "violations": violations,
                "warnings": warnings,
                "validation_status": "violations_found",
            }
        state = {**state, "warnings": warnings}

        # 2단계: 원 메모 대비 새로운 사실 추가 여부 확인
        try:
            fact_prompt = f"[메모] {state['masked']}\n[윤문] {state['polished']}"
            fact_raw = (await self._llm.generate(FACT_CHECK_TPL.build(fact_prompt))).strip().upper()
        except Exception:
            logger.warning("사실보존 검증 실패 — 결과를 통과시키지 않습니다.")
            return {
                **state,
                "violations": ["VALIDATION_UNAVAILABLE: 사실보존 검증 실패"],
                "validation_status": "unavailable",
            }

        if fact_raw.startswith("YES"):
            return {
                **state,
                "violations": ["VIOLATION: 메모에 없는 새로운 사실이 추가됨"],
                "validation_status": "violations_found",
            }
        if not fact_raw.startswith("NO"):
            logger.warning("사실보존 검증 응답 형식 오류 — 결과를 통과시키지 않습니다.")
            return {
                **state,
                "violations": ["VALIDATION_UNAVAILABLE: 사실보존 검증 응답 형식 오류"],
                "validation_status": "unavailable",
            }

        # 3단계: 규정 RAG + LLM 검증
        if self._store.count(REGULATION_COLLECTION) == 0:
            logger.warning("regulations 컬렉션이 비어있어 결과를 통과시키지 않습니다.")
            return {
                **state,
                "violations": ["VALIDATION_UNAVAILABLE: 규정 자료 없음"],
                "validation_status": "unavailable",
            }

        try:
            results = self._retriever.retrieve(
                state["polished"], REGULATION_COLLECTION, top_k=3, n_candidates=10
            )
            if not results:
                raise RuntimeError("규정 검색 결과 없음")
            reg_text = "\n".join(r["text"] for r in results[:3])
            prompt = f"[규정]\n{reg_text}\n\n[문장]\n{state['polished']}"
            raw = (await self._llm.generate(VALIDATE_TPL.build(prompt))).strip()
        except Exception:
            logger.warning("규정 검증 실패 — 결과를 통과시키지 않습니다.")
            return {
                **state,
                "violations": ["VALIDATION_UNAVAILABLE: 규정 검증 실패"],
                "validation_status": "unavailable",
            }

        normalized = raw.upper()
        if normalized.startswith("OK"):
            return {**state, "violations": [], "validation_status": "passed"}
        if normalized.startswith("VIOLATION"):
            # LLM 규정 판정도 경고로 강등(2026-08-03) — VALIDATE_TPL이 나열하는 8종
            # 위반 유형 중 상당수는 인덱싱된 기재요령에 근거 조항이 없어(EVAL.md 14절)
            # 이 판정은 검색된 규정이 아니라 프롬프트 문구에 기대고 있다. 키워드 규칙과
            # 같은 이유로 최종 결정은 교사에게 넘긴다.
            return {
                **state,
                "violations": [],
                "warnings": [*state.get("warnings", []), raw.replace("VIOLATION", "WARNING", 1)],
                "validation_status": "passed",
            }
        logger.warning("규정 검증 응답 형식 오류 — 결과를 통과시키지 않습니다.")
        return {
            **state,
            "violations": ["VALIDATION_UNAVAILABLE: 규정 검증 응답 형식 오류"],
            "validation_status": "unavailable",
        }

    # ── 공개 API ────────────────────────────────────────────────────

    async def run(self, memo: str, max_retry: int = 2) -> RecordOutput:
        """메모를 입력받아 윤문 결과를 반환. 차단 사유 발생 시 최대 max_retry 재시도.

        경고(warnings)는 재시도를 유발하지 않는다 — 오탐이 잦은 판정이라
        다시 윤문시키면 정상 문장을 괜히 망가뜨린다. 그대로 반환하고 교사가 본다.
        """
        state: RecordState = {
            "memo": memo,
            "masked": "",
            "pii_found": [],
            "polished": "",
            "violations": [],
            "warnings": [],
            "generated_pii": [],
            "validation_status": "pending",
            "attempt": 0,
        }

        # mask 는 한 번만
        state = self._step_mask(state)

        for attempt in range(max(1, max_retry)):
            state["attempt"] = attempt
            state = await self._step_polish(state)
            state = await self._step_validate(state)
            if state["validation_status"] in ("passed", "unavailable"):
                break

        safe_to_return = state["validation_status"] == "passed"
        return RecordOutput(
            masked_memo=state["masked"],
            pii_found=state["pii_found"],
            polished=state["polished"] if safe_to_return else "",
            violations=state["violations"],
            warnings=state.get("warnings", []),
            validation_status=state["validation_status"],
            warning=WARNING,
        )


_instance: RecordChain = None


# BGEEmbedder/BGEReranker 로딩 비용이 크므로 프로세스당 한 번만 생성
def get_record_chain() -> RecordChain:
    global _instance
    if _instance is None:
        _instance = RecordChain()
    return _instance
