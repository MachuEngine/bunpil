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

## 2. 일회성 실험/조사 기록·아카이브 (13개) — 정기 평가에 안 쓰임, 각자 별도 스크립트 전용

| 파일 | 관련 스크립트 | 성격 |
|---|---|---|
| example_question_retrieval_test.json | eval_example_retrieval.py | 예시문제→성취기준 검색 정합성 조사 (라벨링 미완료) |
| structure_golden_contaminated_examples.json | 없음(순수 아카이브) | 언어 오염 사례 보관 — 트러블슈팅/블로그 참고용 |
| structure_golden_v1_labeled.json | 없음(순수 아카이브) | 유사도 게이트 적용 전 7B 출력 + 사람 라벨 v1 (Judge 신뢰도 v1 측정 기준 데이터, 2026-07-11 동결) |
| structure_golden_v2_pre_retry_fix.json | 없음(순수 아카이브) | 유사도 게이트 적용, tool-calling 재시도 로직 적용 전 14B 출력 v2 중간본 (2026-07-11 동결, human_label 없음) |
| _distractor_quality_compare.json | compare_distractor_quality.py | 오답매력도 A/B 실험 결과 |
| _temperature_ab_compare.json | test_temperature_effect.py | temperature 0.7 vs 0.2 A/B 실험 결과 |
| _topk_recall_compare.json | test_topk_recall.py | top_k 3→2 실험 결과 |
| _structure_judge_eval_results.json | eval_exam.py (구조 Judge 신뢰도 측정 시 수동 실행) | 사람 라벨 vs Judge 점수 대조 raw 결과, 재실행 시 덮어씀 |
| _model_comparison_results.json | compare_models.py | 모델 비교 실험(Qwen2.5-7B/14B, Llama3.1-8B, GPT-4o-mini) 생성·채점 raw 결과 |
| _model_comparison_results_budget1_backup.json | compare_models.py (수동 백업) | 위 모델 비교 실험의 budget=1 원본 결과 백업(EVAL.md 참고) |
| _model_comparison_results_budget5_partial_backup.json | compare_models.py (수동 백업) | 모델 비교 실험 budget=5 재검증 중 부분 실행분 백업(EVAL.md 참고) |
| _ragas_eval_results.json | eval_ragas.py | Faithfulness/Answer Relevancy(Ragas 알고리즘 자체 구현) 측정 raw 결과, 재실행 시 덮어씀 |
| _judge_comparison_results.json | compare_judge_models.py | Judge 모델 비교 실험(qwen2.5:7b/14b vs gpt-5.6-luna/sol) raw 결과, 재실행 시 누적 저장(EVAL.md 참고) |

## 명명 규칙

- `_`로 시작 = 실험 결과 비교 기록 (골든셋 아님, 사람 라벨링 대상 아님)
- `_contaminated_examples`, `_test` 접미사 = 정기 평가 파이프라인에서 제외된 보조 파일
- 각 골든셋 JSON은 파일 안에 `_schema` 키로 필드 설명·provenance를 자체 문서화한다
