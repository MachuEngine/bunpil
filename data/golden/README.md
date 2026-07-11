# data/golden/ 파일 가이드

## 1. 정기 평가용 골든셋 (6개) — eval_exam.py / eval_record.py가 매번 로드

| 파일 | 로드하는 스크립트 | 사람 라벨 필드 | 용도 |
|---|---|---|---|
| retrieval_golden_final.json | eval_exam.py | expected_chunk_id + reviewed | RAG 검색 Recall@5 평가 |
| item_golden.json | eval_exam.py | human_score | 문항 품질(오답매력도 등) Judge 신뢰도 검증 |
| structure_golden.json | eval_exam.py | human_label | 구조 유사도 Judge 신뢰도 검증 |
| masking_golden.json | eval_record.py | pii | 생기부 개인정보 마스킹 평가 |
| hallucination_golden.json | eval_record.py | forbidden | 생기부 환각 탐지 평가 |
| violation_golden.json | eval_record.py | label | 생기부 위반 문구 탐지 평가 |

> retrieval_golden_final.json은 일회성 실험 스크립트(test_topk_recall.py,
> eval_example_retrieval.py)에서도 기준값 측정용으로 재사용된다.

## 2. 일회성 실험/조사 기록 (5개) — 정기 평가에 안 쓰임, 각자 별도 스크립트 전용

| 파일 | 관련 스크립트 | 성격 |
|---|---|---|
| example_question_retrieval_test.json | eval_example_retrieval.py | 예시문제→성취기준 검색 정합성 조사 (라벨링 미완료) |
| structure_golden_contaminated_examples.json | 없음(순수 아카이브) | 언어 오염 사례 보관 — 트러블슈팅/블로그 참고용 |
| _distractor_quality_compare.json | compare_distractor_quality.py | 오답매력도 A/B 실험 결과 |
| _temperature_ab_compare.json | test_temperature_effect.py | temperature 0.7 vs 0.2 A/B 실험 결과 |
| _topk_recall_compare.json | test_topk_recall.py | top_k 3→2 실험 결과 |

## 명명 규칙

- `_`로 시작 = 실험 결과 비교 기록 (골든셋 아님, 사람 라벨링 대상 아님)
- `_contaminated_examples`, `_test` 접미사 = 정기 평가 파이프라인에서 제외된 보조 파일
- 각 골든셋 JSON은 파일 안에 `_schema` 키로 필드 설명·provenance를 자체 문서화한다
