# data/golden/ 파일 가이드

## 1. 정기 평가용 골든셋 (6개) — 2026-10 문항 품질 골든셋 교체

| 파일 | 로드하는 스크립트 | 사람 라벨 필드 | 용도 |
|---|---|---|---|
| retrieval_golden_final.json | eval_exam.py | expected_chunk_id + reviewed | RAG 검색 Recall@5 평가 |
| item_golden.json | — | human_score | **합성(Claude가 문항·점수 모두 합성), 2026-10 정기 평가 기준에서 item_quality_golden.json으로 대체됨(이력 보존, EVAL.md 28절) — eval_exam.py의 정기 평가는 더 이상 이 파일을 쓰지 않는다** |
| item_quality_golden.json | eval_exam.py, eval_item_quality_runs.py | human_label(기준별 1~5점, 사람 라벨 1인) | 문항 품질(정답유일성·오답매력도·근거성) Judge 신뢰도 검증. 실제 생성 모델 3종(qwen2.5-14b/gpt-6-luna/gemini-3.8-flash) 출력에서 (지문, 모델)마다 객관식 1개씩 뽑아 모델 정보를 블라인드 처리했다(매핑은 `_item_quality_model_map.json`). 95건 중 cannot_judge 4건 제외한 91건이 평가 대상. 2026-10부터 item_golden.json을 대체(EVAL.md 28절) |
| structure_golden.json | eval_exam.py | human_label | 구조 유사도 Judge 신뢰도 검증 — 2026-07-23부터 이 Judge(`get_judge_backend()`)가 런타임 `judge` 노드와 동일 코드이므로, 이 수치가 곧 배포된 judge의 신뢰도(자세한 내용은 MODEL_SELECTION.md 2.5절). 사람 라벨은 qwen 출력 45건 기준(API 모델 출력에서의 구조 Judge 정확도는 미검증) |
| masking_golden.json | `tests/test_masker.py` | pii | PII 마스킹 평가(FN=0 강제). **2026-08-03**: 채점 스크립트였던 `eval_record.py`가 생기부 모듈과 함께 삭제되면서, 이 골든셋은 pytest 파라미터화 테스트로 흡수됐다 — `mask_pii()`는 출제 경로(`app/main.py` `_build_spec()`)가 계속 쓰므로 커버리지는 유지 |
| vlm_extraction_golden.json | evals/eval_vlm.py, evals/eval_vlm_compare.py | figure_summary(기존 f01~f15 15건은 2026-08-20 사람 검수 완료, 신규 75건은 **Claude 합성 초안·사람 검수 전**) | 이미지→텍스트 추출(`/exam/extract`) 정확도 평가. text_only 20 + figure 90(선 29·막대 21·원 20·표 20) + adversarial 5 = **115건**(2026-08 40건→2026-10 확대), 이미지는 `vlm_golden_images/`(`golden_gen/gen_vlm_golden.py`로 생성, 전부 합성 — 실제 스크린샷 미사용). 그림 서술 Judge 채점의 신뢰도(kappa 등)는 2026-10에 검증 완료(아래 2026-10 VLM 모델 선정 작업 기록 섹션, MODEL_SELECTION.md 7절·EVAL.md 29절) |
| ~~vlm_figure_judge_golden.json~~ | evals/eval_vlm_judge_reliability.py | human_label(null — **라벨링 보류했다가 폐기**) | vlm_extraction_golden.json의 figure 15건을 고쳐 재생성해 (VLM 출력, Judge 점수)를 고정해 둔 것. 15건으로는 kappa의 신뢰구간이 너무 넓어 라벨링을 보류했었는데, **2026-10에 아래 vlm_figure_human_labels.json(90건)+vlm_defect_human_labels.json(40건) 하네스로 대체됐다**(EVAL.md 29절, MODEL_SELECTION.md §7) |

> `hallucination_golden.json`/`violation_golden.json`(생기부 전용)은 모듈과 함께
> **2026-08-03 삭제됨** — 더 이상 이 디렉토리에 존재하지 않는다(EVAL.md 14절).
>
> retrieval_golden_final.json은 일회성 실험 스크립트(test_topk_recall.py,
> eval_example_retrieval.py)에서도 기준값 측정용으로 재사용된다.

### 편입 검토 대기 중인 골든셋

| 파일 | 상태 |
|---|---|
| regulations_retrieval_candidates.json | 2026-08-03 작성, 사람 검수 완료(`reviewed: true`), 코퍼스에 실재하는 조항만 근거로 한 10건. 현재 검색기 Recall@5=0.500으로 기존 골든셋(천장 도달, 1.000)보다 변별력 있음 — 정식 편입 여부는 `bunpil_roadmap.md` "남은 작업" 4번 참고, 아직 어떤 스크립트도 로드하지 않음 |

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

## 3. 2026-10 문항 품질·Judge 재선정 작업 기록 — `golden_gen/gen_item_quality_golden.py`, `evals/eval_item_quality_runs.py`, `evals/ci_report.py` 전용

item_quality_golden.json(위 1절)을 만들고 Judge·생성 모델을 재선정하는 과정에서 생긴
입력·중간·결과 파일. 상세 경위는 EVAL.md 28절.

| 파일/디렉토리 | 관련 스크립트 | 사람 라벨 | 성격 |
|---|---|---|---|
| item_quality_inputs.json | gen_item_quality_golden.py | 없음(review 필드는 지문 자체의 사람 검토 상태) | 합성 지문 33개(입력) — 2026-09 합성 입력에서 가져와 2026-10-03에 재구성(변별력 확보를 위해 normal 10개를 hard_content 10개로 교체, 28.1절) |
| item_quality_labeling_sheet.json | gen_item_quality_golden.py export-sheet/import-sheet | 작업 중간 포맷(최종 반영 대상은 item_quality_golden.json) | JSON 직접 편집용 라벨링 작업 시트. 라벨링은 이 시트를 편집한 뒤 import-sheet로 골든셋에 반영 |
| _item_quality_model_map.json | gen_item_quality_golden.py build-labelset | 없음(모델/지문 매핑표) | item_quality_golden.json의 블라인드 id(iq_001…) → 실제 모델/원본 지문 매핑. **라벨링 중에는 참조 금지**(블라인드 유지) |
| _item_quality_runs/*.jsonl | gen_item_quality_golden.py generate | 없음(모델 생성물) | 지문별로 운영 경로를 그대로 실행한 생성 결과, 모델별 1파일(`{model}.jsonl`, smoke 변형 별도). item_quality_golden.json의 원재료 |
| _item_quality_judged/\<judge-slug\>/\<model\>.jsonl | eval_item_quality_runs.py judge | 없음(Judge 채점 결과만, 문항 원문 미포함) | run의 객관식 문항을 Judge 후보별로 채점해 이어서 쌓는 캐시(재호출 없이 재분석 가능) |
| _item_quality_struct_judged/\<judge-slug\>/\<model\>.jsonl | eval_item_quality_runs.py structure-judge | 없음 | run을 `judge_structure()`(런타임 judge_node와 같은 함수)로 재채점한 캐시 — 런타임 게이트 Judge(gpt-5.6-luna)와 다른 Judge의 같은 계열 편향을 대조하는 재료 |
| _judge_selection_round1.json | experiments/compare_judge_models.py | 사람 라벨 1인(structure_golden 45건, qwen 출력만) | Judge 1차 선별 — 5개 후보를 구조 유사도로 2회씩 비교(28.3절) |
| _judge_selection_round2.json | eval_item_quality_runs.py reliability | **사람 라벨 1인, 91건** | Judge 2차 선별 — item_quality_golden.json 라벨과 sonnet/luna 3회 채점을 대조한 기준별 가중 κ·MAE·편향·짝지은 비교(28.4절) |
| _generator_comparison.json | eval_item_quality_runs.py compare-generators | 사람 라벨 포함(위 골든셋에서 조인) | 생성 모델 3종(qwen2.5-14b/gpt-6-luna/gemini-3.8-flash) 규칙·Judge·사람 지표·짝지은 비교 종합(28.6절) |
| _eval_exam_latest.json | evals/eval_exam.py | 사람 라벨 포함(위 골든셋에서 조인) | 정기 평가(`eval_exam.py`) 최신 실행 결과 스냅샷 — `ci_report.py`가 이 파일은 직접 읽지 않고 콘솔 출력만 남김(재현 명령은 EVAL.md 28.11절) |
| _ci_report.json | evals/ci_report.py | 없음(여러 raw 파일을 모아 재계산) | 재실행 없이 저장된 raw 결과(RAGAS·구조 Judge·게이트 재보정 등)만으로 95% 신뢰구간을 다시 계산한 리포트. 원자료가 집계값만 남아 CI를 못 낸 항목(2026-07 Judge/생성 모델 비교, VLM)도 `no_raw_data`에 명시 |

## 4. 2026-10 VLM 모델 선정 작업 기록 — `evals/eval_vlm_compare.py` 전용

vlm_extraction_golden.json(위 1절)을 115건으로 늘리고, 그림 서술 Judge를 검증한 뒤
후보 5종을 비교하는 과정에서 생긴 입력·중간·결과 파일. 상세 경위는 EVAL.md 29절,
MODEL_SELECTION.md §7 "2차 선정".

| 파일/디렉토리 | 관련 서브커맨드 | 사람 라벨 | 성격 |
|---|---|---|---|
| vlm_labeling_sheet.json | export-sheet/import-sheet | 작업 중간 포맷(최종 반영 대상은 vlm_figure_human_labels.json) | 그림 전체를 후보에 블라인드 균등 배정한 라벨링 작업 시트 |
| _vlm_label_map.json | export-sheet | 없음(그림 id → 실제 VLM 후보 매핑) | 블라인드 유지용 — **라벨링 중에는 참조 금지** |
| vlm_figure_human_labels.json | import-sheet | **사람 라벨 1인, 90건**(블라인드) | 그림 서술 Judge 검증 1회차 기준 라벨(29.4절) |
| _vlm_runs/\<model\>.jsonl | extract | 없음(모델 추출 결과) | 후보별 추출 run(이어서 실행 가능), `gpt-6-luna-openai.jsonl`은 OpenAI 직접 경로 재측정분(29.7절) |
| _vlm_runs_archive/qwen3-vl-8b-thinking.jsonl | 없음(아카이브) | 없음 | 생각 버전 실패 증거(그림 61장 중 25장 빈 응답) — instruct로 교체되며 보존(29.3절) |
| _vlm_judged/\<judge-slug\>/\<model\>.jsonl | judge | 없음(Judge 채점 결과만) | run의 그림 서술을 Judge 후보별로 채점한 캐시 |
| _vlm_judge_selection.json | reliability | 사람 라벨 포함(위 골든셋에서 조인) | 1회차 90건 기준 κ·MAE·편향·짝지은 비교(29.4절) |
| _vlm_defect_items.json | make-defects | 없음(결함 주입 서술) | 정상 서술에 결함 6종을 규칙 기반으로 주입한 중간 산출물 |
| vlm_defect_labeling_sheet.json | make-defects | 작업 중간 포맷(최종 반영 대상은 vlm_defect_human_labels.json) | 결함 셋 블라인드 라벨링 작업 시트 |
| vlm_defect_human_labels.json | import-defect-sheet | **사람 라벨 1인, 40건**(블라인드) | 결함 유형별 6건 + 원본 10건, 중간 품질 구간을 채우는 보강 라벨(29.4절) |
| _vlm_judged_defects/\<judge-slug\>.jsonl | judge-defects | 없음(Judge 채점 결과만) | 결함 셋 서술을 Judge 후보별로 채점한 캐시 |
| _vlm_judge_selection_defects.json | reliability-defects | 사람 라벨 포함(위 두 골든셋에서 조인) | 1회차+결함셋 합산 130건 κ·MAE, 결함 유형별·경보 기준별 집계(29.4절) |
| _vlm_comparison.json | compare | 사람 라벨 포함(간접, Judge 채점 경유) | 후보 5종의 CER/WER·서술 품질·adversarial·지연 종합 비교, 기준 대비·후보 간 짝지은 비교(29.5절) |

## 명명 규칙

- `_`로 시작 = 실험 결과 비교 기록 (골든셋 아님, 사람 라벨링 대상 아님)
- `_contaminated_examples`, `_test` 접미사 = 정기 평가 파이프라인에서 제외된 보조 파일
- 각 골든셋 JSON은 파일 안에 `_schema` 키로 필드 설명·provenance를 자체 문서화한다
