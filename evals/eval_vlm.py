#!/usr/bin/env python
"""VLM(이미지 → 텍스트 추출, POST /exam/extract) 정확도 평가.

data/golden/vlm_extraction_golden.json(40건: text_only 20 + figure 15 + adversarial 5)을
읽어 실제 get_vlm_backend()로 추출한 뒤 세 축을 각각 다른 방식으로 채점한다(설계 근거는
MODEL_SELECTION.md 7절):

1. 텍스트 재현 정확도 — CER/WER(문자/단어 오류율). 정답 원문을 이미 아는 문제라 코드로만
   계산하고 LLM Judge를 쓰지 않는다(jiwer 등 외부 라이브러리도 안 씀, ~20줄 순수 파이썬).
2. figure 카테고리의 "[자료: ...]" 서술 커버리지 — 정답 문자열이 없는 주관적 판단이라
   get_judge_backend()(생성/VLM과 독립된 기존 Judge 축, 새 축 신설 안 함)로 1~5점 채점.
   채점 기준인 figure_summary는 Claude 초안이었으나 2026-08-20 사람 검수 완료(전체
   reviewed=true) — 단, item_golden/structure_golden과 달리 Judge 점수 자체의 신뢰도
   (사람 채점과 kappa 비교)는 측정한 적 없다(아래 print_report의 caveat 참고).
3. adversarial — 정답/해설을 스스로 지어내지 않는지(forbidden_answer_leak, 규칙 기반),
   mask_pii()가 실제 VLM 출력에서도 여전히 이름/전화번호 등을 잡아내는지(pii_labels_expected).

실행마다 실제 OpenAI API(VLM 40회 + Judge 최대 15회) 호출이 발생하므로 정기 자동 실행
대상에 넣지 않았다 — eval_exam.py와 달리 필요할 때 수동 실행한다(README 참고).
"""
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.common.llm import get_judge_backend, get_vlm_backend
from app.common.privacy import mask_pii

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_PATH = os.path.join(ROOT, "data", "golden", "vlm_extraction_golden.json")
GOLDEN_DIR = os.path.join(ROOT, "data", "golden")

_FIGURE_BLOCK = re.compile(r"\[자료\s*:.*?\]", re.DOTALL)
# 2026-08-20 실측: table 유형에서 VLM이 지시한 "[자료: ...]" 대신 마크다운 표를 그대로
# 직접 옮기는 경우가 있었다(수치를 원문 그대로 보존한다는 점에서 프롬프트 취지에 어긋나지
# 않는 결과로 판단 — 표는 그 자체로 이미 "텍스트"라 서술이 아니라 전사가 자연스러움).
# CER 계산 시 이것도 "자료 설명 블록"으로 인정하지 않으면 표를 그대로 옮긴 정확한 응답이
# 오히려 더 나쁜 점수를 받는 역설이 생겨 마크다운 표 블록도 함께 인식한다.
_MD_TABLE_BLOCK = re.compile(r"(?:^\|.*\|$\n?)+", re.MULTILINE)
_FORBIDDEN_LEAK_PATTERNS = ["정답은", "정답:", "정답 :", "따라서 답은", "해설:", "해설 :"]

_JUDGE_PROMPT = (
    "당신은 채점자입니다. 아래 '핵심 사실'이 '학생 서술'에 실제로 담겨 있는지 1~5점으로 "
    "채점하세요. 5점=핵심 수치·경향을 거의 다 담음, 3점=일부만 담음, 1점=핵심을 놓치거나 "
    "서술 자체가 없음. 설명 없이 숫자만 답하세요.\n\n"
    "[핵심 사실]\n{summary}\n\n[학생 서술]\n{description}"
)


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _edit_distance(a: list, b: list) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def cer(ref: str, hyp: str) -> float:
    ref_n, hyp_n = _normalize(ref), _normalize(hyp)
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    return _edit_distance(list(ref_n), list(hyp_n)) / len(ref_n)


def wer(ref: str, hyp: str) -> float:
    ref_w, hyp_w = _normalize(ref).split(" "), _normalize(hyp).split(" ")
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return _edit_distance(ref_w, hyp_w) / len(ref_w)


def strip_figure_block(text: str) -> tuple[str, str | None]:
    """[자료: ...] 블록 또는 마크다운 표 블록을 떼어내고 (나머지 텍스트, 자료 설명 또는 None)을
    반환한다. 어느 쪽도 없으면 자료 설명이 아예 없었다는 뜻(None) — 진짜 실패로 채점."""
    m = _FIGURE_BLOCK.search(text)
    if m:
        return (text[:m.start()] + text[m.end():]).strip(), m.group(0)
    m = _MD_TABLE_BLOCK.search(text)
    if m:
        return (text[:m.start()] + text[m.end():]).strip(), m.group(0)
    return text, None


async def _call_with_retry(coro_fn, *, retries: int = 5, base_wait: float = 20.0):
    """OpenAI TPM(분당 토큰) 레이트리밋 대응. 40건을 순차 호출해도 이미지가 커서
    분당 한도에 걸릴 수 있음(2026-08-20 실측) — 429면 분당 윈도우가 갈릴 때까지 대기 후 재시도."""
    from openai import RateLimitError

    for attempt in range(retries):
        try:
            return await coro_fn()
        except RateLimitError:
            if attempt == retries - 1:
                raise
            wait = base_wait * (attempt + 1)
            print(f"    [429] {wait:.0f}초 대기 후 재시도 ({attempt + 1}/{retries})...")
            await asyncio.sleep(wait)


async def extract_all(entries: list[dict]) -> list[dict]:
    vlm = get_vlm_backend()
    results = []
    for e in entries:
        img_path = os.path.join(GOLDEN_DIR, e["image"])
        with open(img_path, "rb") as f:
            image_bytes = f.read()
        try:
            raw = await _call_with_retry(lambda: vlm.extract_text(image_bytes, "image/png"))
        except Exception as exc:  # noqa: BLE001 — 평가 스크립트: 개별 실패를 기록하고 계속 진행
            raw = ""
            print(f"  [경고] {e['id']} 추출 실패: {exc}")
        results.append({**e, "vlm_output": raw})
        await asyncio.sleep(1.0)  # TPM 한도를 미리 피하기 위한 페이싱(재시도보다 예방이 저렴)
    return results


async def score_figure_judges(scored: list[dict]) -> None:
    judge = get_judge_backend()
    for e in scored:
        if e["category"] != "figure":
            continue
        _, fig_text = strip_figure_block(e["vlm_output"])
        if fig_text is None:
            e["judge_score"] = 1
            e["judge_note"] = "[자료: ...] 블록 자체가 출력에 없음"
            continue
        prompt = _JUDGE_PROMPT.format(summary=e["figure_summary"], description=fig_text)
        try:
            raw = await _call_with_retry(lambda: judge.generate([{"role": "user", "content": prompt}]))
            digits = "".join(ch for ch in raw if ch.isdigit())
            e["judge_score"] = max(1, min(5, int(digits))) if digits else None
        except Exception as exc:  # noqa: BLE001
            e["judge_score"] = None
            e["judge_note"] = f"judge 호출 실패: {exc}"


def score_text_metrics(scored: list[dict]) -> None:
    for e in scored:
        if e["category"] == "figure":
            text_part, _ = strip_figure_block(e["vlm_output"])
        else:
            text_part = e["vlm_output"]
        e["cer"] = cer(e["ground_truth_text"], text_part)
        e["wer"] = wer(e["ground_truth_text"], text_part)


def score_adversarial(scored: list[dict]) -> None:
    for e in scored:
        if e["category"] != "adversarial":
            continue
        out = e["vlm_output"]
        if e["adversarial_kind"] in ("no_leak", "leak_present"):
            leaked = any(p in out for p in _FORBIDDEN_LEAK_PATTERNS)
            if e.get("forbidden_answer_leak"):
                # 이미지에 없는 정답을 스스로 지어내면 안 됨
                e["adversarial_pass"] = not leaked
            else:
                # leak_present: 이미지에 실제로 있던 메모이므로 원문 전사(그대로 옮김)는 허용.
                # 다만 그 메모 없이 별도로 새 해설을 지어냈는지까지는 이 정규식만으로 구분
                # 못하므로 참고 정보로만 기록(하드 실패로 채점하지 않음).
                e["adversarial_pass"] = None
        elif e["adversarial_kind"] == "pii_variant":
            _, found = mask_pii(out)
            found_set = set(found)
            expected = set(e["pii_labels_expected"])
            missing = expected - found_set
            e["adversarial_pass"] = len(missing) == 0
            e["pii_missing_required"] = sorted(missing)
            e["pii_known_limitation_caught"] = sorted(
                set(e.get("pii_labels_known_limitation", [])) & found_set
            )


def print_report(scored: list[dict]) -> None:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n" + "=" * 70)
    print("VLM 추출 정확도 리포트")
    print("=" * 70)

    text_only = [e for e in scored if e["category"] == "text_only"]
    figure = [e for e in scored if e["category"] == "figure"]
    adversarial = [e for e in scored if e["category"] == "adversarial"]

    print(f"\n[1] text_only ({len(text_only)}건) — 문자/단어 오류율")
    print(f"  CER 평균 = {mean([e['cer'] for e in text_only]):.4f}")
    print(f"  WER 평균 = {mean([e['wer'] for e in text_only]):.4f}")
    worst = sorted(text_only, key=lambda e: e["cer"], reverse=True)[:3]
    for e in worst:
        print(f"    최악 사례: {e['id']} CER={e['cer']:.3f}")

    print(f"\n[2] figure ({len(figure)}건) — 텍스트 부분 CER/WER + 자료 서술 커버리지(Judge)")
    print(f"  CER 평균(텍스트 부분) = {mean([e['cer'] for e in figure]):.4f}")
    print(f"  WER 평균(텍스트 부분) = {mean([e['wer'] for e in figure]):.4f}")
    scores = [e.get("judge_score") for e in figure]
    unreviewed = [e["id"] for e in figure if not e.get("reviewed")]
    caveat = (
        f"  ⚠️ figure_summary 검수 안 됨({unreviewed}) — 채점 기준 자체가 아직 미확정이라 참고용"
        if unreviewed
        else "  (figure_summary 전부 사람 검수 완료 — 채점 기준 자체는 신뢰 가능. "
             "단, Judge가 그 기준을 얼마나 일관되게 적용하는지는 별도 검증 안 함 — "
             "kappa 등 신뢰도 측정은 item_golden/structure_golden과 달리 이 골든셋엔 없음)"
    )
    print(f"  자료 서술 커버리지 Judge 평균(1~5) = {mean(scores):.2f}")
    print(caveat)
    missing_block = [e["id"] for e in figure if strip_figure_block(e["vlm_output"])[1] is None]
    if missing_block:
        print(f"  [자료: ...] 블록 자체가 없던 사례: {missing_block}")

    print(f"\n[3] adversarial ({len(adversarial)}건)")
    for e in adversarial:
        if e["adversarial_kind"] in ("no_leak", "leak_present"):
            status = "N/A(참고용)" if e["adversarial_pass"] is None else ("PASS" if e["adversarial_pass"] else "FAIL")
            print(f"  {e['id']} ({e['adversarial_kind']}): {status}")
        elif e["adversarial_kind"] == "pii_variant":
            status = "PASS" if e["adversarial_pass"] else f"FAIL(누락: {e['pii_missing_required']})"
            extra = f", 한계 항목 중 실제 잡힌 것: {e['pii_known_limitation_caught']}" if e["pii_known_limitation_caught"] else ""
            print(f"  {e['id']} (pii_variant): {status}{extra}")

    n_leak_fail = sum(1 for e in adversarial if e["adversarial_kind"] == "no_leak" and e["adversarial_pass"] is False)
    n_pii_fail = sum(1 for e in adversarial if e["adversarial_kind"] == "pii_variant" and e["adversarial_pass"] is False)
    print(f"\n요약: no_leak 위반 {n_leak_fail}건, pii_variant 필수 라벨 누락 {n_pii_fail}건")
    print("=" * 70)


async def main():
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        doc = json.load(f)
    entries = doc["entries"]
    print(f"골든셋 {len(entries)}건 로드 — VLM 추출 시작 (실제 API 호출)...")

    scored = await extract_all(entries)
    score_text_metrics(scored)
    score_adversarial(scored)
    print("figure 카테고리 Judge 채점 중...")
    await score_figure_judges(scored)

    print_report(scored)


if __name__ == "__main__":
    asyncio.run(main())
