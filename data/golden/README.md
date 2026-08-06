# data/golden/ 파일 가이드

## 1. 정기 평가용 골든셋 (4개) — 2026-08-03 생기부 모듈 제거 후 현재 상태

| 파일 | 로드하는 스크립트 | 사람 라벨 필드 | 용도 |
|---|---|---|---|
| retrieval_golden_final.json | eval_exam.py | expected_chunk_id + reviewed | RAG 검색 Recall@5 평가 |
| item_golden.json | eval_exam.py | human_score | 문항 품질(오답매력도 등) Judge 신뢰도 검증 |
| structure_golden.json | eval_exam.py | human_label | 구조 유사도 Judge 신뢰도 검증 — 2026-07-23부터 이 Judge(`get_judge_backend()`)가 런타임 `judge` 노드와 동일 코드이므로, 이 수치가 곧 배포된 judge의 신뢰도(자세한 내용은 MODEL_SELECTION.md 2.5절) |
| masking_golden.json | `tests/test_masker.py` | pii | PII 마스킹 평가(FN=0 강제). **2026-08-03**: 채점 스크립트였던 `eval_record.py`가 생기부 모듈과 함께 삭제되면서, 이 골든셋은 pytest 파라미터화 테스트로 흡수됐다 — `mask_pii()`는 출제 경로(`app/main.py` `_build_spec()`)가 계속 쓰므로 커버리지는 유지 |

> `hallucination_golden.json`/`violation_golden.json`(생기부 전용)은 모듈과 함께
> **2026-08-03 삭제됨** — 더 이상 이 디렉토리에 존재하지 않는다(EVAL.md 14절).
>
> retrieval_golden_final.json은 일회성 실험 스크립트(test_topk_recall.py,
> eval_example_retrieval.py)에서도 기준값 측정용으로 재사용된다.

### 편입 검토 대기 중인 골든셋

| 파일 | 상태 |
|---|---|
| regulations_retrieval_candidates.json | 2026-08-03 작성, 사람 검수 완료(`reviewed: true`), 코퍼스에 실재하는 조항만 근거로 한 10건. 현재 검색기 Recall@5=0.600으로 기존 골든셋(천장 도달, 1.000)보다 변별력 있음 — 정식 편입 여부는 `bunpil_roadmap.md` "남은 작업" 4번 참고, 아직 어떤 스크립트도 로드하지 않음 |

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
| _structure_judge_eval_results.json | **고아 파일(2026-08-04 확인)** — 어떤 현재 코드도 이 파일을 읽거나 쓰지 않음(`eval_exam.py`는 구조 Judge 결과를 콘솔에만 출력, 이 파일명을 참조하는 곳 없음). 이전 버전 `eval_exam.py`가 남긴 것으로 추정되는 raw 결과 스냅샷 — 삭제해도 기능 영향 없음, 삭제 여부는 사람 판단 필요 |
| _model_comparison_results.json | compare_models.py | 모델 비교 실험(Qwen2.5-7B/14B, Llama3.1-8B, GPT-4o-mini) 생성·채점 raw 결과 |
| _model_comparison_results_budget1_backup.json | compare_models.py (수동 백업) | 위 모델 비교 실험의 budget=1 원본 결과 백업(EVAL.md 참고) |
| _model_comparison_results_budget5_partial_backup.json | compare_models.py (수동 백업) | 모델 비교 실험 budget=5 재검증 중 부분 실행분 백업(EVAL.md 참고) |
| _ragas_eval_results.json | eval_ragas.py | Faithfulness/Answer Relevancy(Ragas 알고리즘 자체 구현) 측정 raw 결과, 재실행 시 덮어씀 |
| _judge_comparison_results.json | compare_judge_models.py | Judge 모델 비교 실험(qwen2.5:7b/14b vs gpt-5.6-luna/sol) raw 결과, 재실행 시 누적 저장(EVAL.md 참고) |
| _validate_gate_calibration.json | measure_validate_gate.py | **(신규 2026-08-04)** validate 게이트 임계값 재보정 실측 — 프로덕션 Judge(gpt-5.6-luna)의 실제 점수 분포·임계값별 통과율. 옛 기준 통과율이 6.7%로 사실상 도달 불가였음을 보인 근거 데이터(EVAL.md 15절), 재실행 시 덮어씀 |
| _near_copy_diagnosis.json | diagnose_near_copy.py | **(신규 2026-08-07)** 게이트 실패의 성격 진단 — 생성 문항별 containment·임베딩 코사인과 Judge 점수 대조. 어휘·의미 가설이 모두 반증된 근거 데이터(EVAL.md 20절), 재실행 시 덮어씀 |
| _budget_effect.json | measure_budget_effect.py | **(신규 2026-08-07)** budget=5(프로덕션 조건) 통과율·재시도 소모량 측정. 개선 작업 없이 0.833 달성을 보인 근거 데이터(EVAL.md 21절), 재실행 시 덮어씀 |
| _distractor_diagnosis.json | diagnose_distractor.py | **(신규 2026-08-07)** 현재 스택 생성물의 문항 품질 3기준(정답유일성·오답매력도·근거성) 분해와 선지 원문. 타깃을 오답매력도→정답유일성으로 전환한 근거(EVAL.md 22절), 재실행 시 덮어씀 |

## 명명 규칙

- `_`로 시작 = 실험 결과 비교 기록 (골든셋 아님, 사람 라벨링 대상 아님)
- `_contaminated_examples`, `_test` 접미사 = 정기 평가 파이프라인에서 제외된 보조 파일
- 각 골든셋 JSON은 파일 안에 `_schema` 키로 필드 설명·provenance를 자체 문서화한다
