#!/usr/bin/env python
"""Phase 5 생기부 모듈 통합 테스트.
마스킹 동작 + 윤문 + 규정 위반 플래그 확인.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:1.5b")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db_record_test")

from app.modules.record import get_record_chain, mask_pii

# ── 테스트 케이스 ────────────────────────────────────────────────────
MASK_CASES = [
    {
        "input": "김철수(010-1234-5678) 수학 시간에 발표 잘 함.",
        "expected_pii": ["전화번호"],
        "desc": "전화번호 마스킹",
    },
    {
        "input": "900101-1234567 학생이 조별 과제에서 리더 역할.",
        "expected_pii": ["주민번호"],
        "desc": "주민번호 마스킹",
    },
    {
        "input": "한국고등학교 2학년 학생, 이메일 student@school.kr로 자료 제출.",
        "expected_pii": ["학교명", "이메일"],
        "desc": "학교명+이메일 마스킹",
    },
    {
        "input": "독서 토론에서 근거 들어 주장함. 다른 의견 수용함.",
        "expected_pii": [],
        "desc": "PII 없는 정상 메모",
    },
]

POLISH_CASES = [
    {
        "memo": "수학 시간에 발표를 잘 함. 친구들이 이해못할 때 도와줌.",
        "desc": "기본 윤문",
    },
    {
        "memo": "조별 과제에서 리더 역할. 의견 조율하고 발표까지 맡아서 함.",
        "desc": "리더십 윤문",
    },
    {
        "memo": "독서 토론에서 근거 들어 주장함. 다른 의견도 잘 수용함.",
        "desc": "토론 활동 윤문",
    },
]


def check(cond: bool) -> str:
    return "✓" if cond else "✗"


def test_masking():
    print("\n[1] PII 마스킹 테스트")
    all_pass = True
    for case in MASK_CASES:
        masked, found = mask_pii(case["input"])
        pii_ok = set(case["expected_pii"]) == set(found)
        no_pii_in_masked = not any(
            kw in masked for kw in ["010-", "900101", "@school.kr", "한국고등학교"]
        )
        ok = pii_ok and (no_pii_in_masked or not case["expected_pii"])
        all_pass = all_pass and ok
        print(f"  {check(ok)} {case['desc']}")
        if not ok:
            print(f"       기대 PII: {case['expected_pii']}, 감지: {found}")
        # 마스킹 결과는 PII가 포함되지 않으므로 로그 출력 가능
        print(f"       마스킹 결과: {masked[:80]}")
    return all_pass


def test_chain():
    print("\n[2] 윤문 Chain 테스트")
    chain = get_record_chain()
    results = []
    for case in POLISH_CASES:
        print(f"\n  --- {case['desc']} ---")
        print(f"  메모  : {case['memo']}")
        out = chain.run(case["memo"])
        print(f"  마스킹: {out['masked_memo']}")
        print(f"  PII   : {out['pii_found']}")
        print(f"  윤문  : {out['polished'][:120]}")
        print(f"  위반  : {out['violations'] if out['violations'] else '없음'}")
        polished_ok = len(out["polished"]) > 10
        results.append(polished_ok)
        print(f"  결과  : {check(polished_ok)}")
    print(out["warning"])
    return all(results)


def main():
    print("=== Phase 5 생기부 모듈 통합 테스트 ===")

    mask_ok = test_masking()
    chain_ok = test_chain()

    print("\n" + "=" * 50)
    print(f"  마스킹 : {check(mask_ok)}")
    print(f"  윤문   : {check(chain_ok)}")
    print(f"  전체   : {check(mask_ok and chain_ok)}")
    print("=" * 50)

    import shutil
    shutil.rmtree("./chroma_db_record_test", ignore_errors=True)


if __name__ == "__main__":
    main()
