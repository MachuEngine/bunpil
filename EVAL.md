# 분필(Bunpil) 평가(Eval) 문서

`scripts/eval_exam.py`, `scripts/eval_record.py`가 측정하는 지표, 골든셋 현황, 실행 방법, 결과 이력을 모아둔 참고 문서.
모델 교체·프롬프트 튜닝 등 평가에 영향을 주는 변경이 있을 때마다 [결과 이력](#4-결과-이력)에 행을 추가한다.

## 1. 평가 항목 전체 목록

### 출제 모듈 (`eval_exam.py`)

| 계층 | 지표 | 방식 | 기준 |
|---|---|---|---|
| 검색 | Recall@5 | 함수 | ≥ 0.8 |
| 검색 | MRR | 함수 | 참고값 |
| 문항 품질 | 정답유일성·오답매력도·근거성 | LLM Judge | 평균 ≥ 4.0 |
| Judge 신뢰도 | Cohen's kappa | 사람 라벨 비교 | ≥ 0.4 |
| Judge 신뢰도 | ±1 일치율 | 사람 라벨 비교 | ≥ 0.7 |
| 구조 유사도 Judge 신뢰도 | difficulty 일치율, overall MAE (LLM Judge) + 문항 개수 일치(코드) | 사람 라벨(STRUCTURE_GOLDEN) 비교 + `len(draft_items)==num_items` | 미정 (부트스트랩 단계) |

> 2026.07 passage_text 리디자인으로 "세트 제약(유형/난이도/커버리지/중복률)" 함수 검증은 폐기되고 위 "구조 유사도 Judge 신뢰도"로 대체됨 (`check_duplicate`/`past_exams` 제거에 따름).
>
> **2026-07-09 count_match 개념 폐기**: "생성 개수가 예시 문제 개수와 일치해야 한다"는 전제 자체가 틀렸음이 발견됨 — 실제로는 개수가 예시와 무관하게 `ExamSpec.num_items`(명시 없으면 기본 5)로 별도 지정된다. count_match는 이제 LLM Judge/사람 라벨 대상이 아니라 `validate_node`가 `len(draft_items)==num_items`로 직접 계산한다. `STRUCTURE_GOLDEN`의 `human_label`·`eval_structure_judge()`·`similarity_judge` 도구 시그니처에서 count_match 전면 제거(자세한 내용은 `data/golden/structure_golden.json`의 `_schema.count_match_deprecated`, `bunpil_roadmap.md` 참고).

### 생기부 모듈 (`eval_record.py`)

| 지표 | 방식 | 기준 |
|---|---|---|
| PII 마스킹 FN율 | 함수 | = 0 |
| 사실 추가율 (키워드) | 함수 | = 0 |
| 사실 추가율 (NLI Judge) | LLM Judge | = 0 |
| 규정 위반 Recall | 함수 | ≥ 0.95 |

### 추가 예정

| 지표 | 도구 | 상태 |
|---|---|---|
| Faithfulness | Ragas | ⬜ 미착수 |
| Answer Relevancy | Ragas | ⬜ 미착수 |

## 2. 골든셋 현황

| 골든셋 | 경로 | 규모 | 비고 |
|---|---|---|---|
| retrieval_golden | `data/golden/retrieval_golden_final.json` | 22개 (reviewed 21개) | 실데이터 기반, 사람 검수. 2026.07 past_exams 참조 8개 제거(30→22) |
| STRUCTURE_GOLDEN | `data/golden/structure_golden.json` | 3개 | Claude 합성 부트스트랩(실제 모델 출력 아님) — eval 스캐폴딩 검증용, 실제 라벨 보강 필요 |
| MASKING_GOLDEN | `scripts/eval_record.py` 내 상수 | 20개 | 합성 |
| VIOLATION_GOLDEN | `scripts/eval_record.py` 내 상수 | 50개 | 위반 25 + 정상 25 |
| HALLUCINATION_GOLDEN | `scripts/eval_record.py` 내 상수 | 20개 | 합성 |
| ITEM_GOLDEN | `scripts/eval_exam.py` 내 상수 | 30개 | human_score 1~5점 분포 |
| example_question_retrieval_test | `data/golden/example_question_retrieval_test.json` | 8개 (reviewed 0개) | 주제어가 아닌 "실제 문제 문장" 스타일 query — standards 컬렉션과의 문체 격차 검증용, 라벨링 대기 |

## 3. 실행 방법

```bash
# 출제 모듈 평가
python scripts/eval_exam.py

# 생기부 모듈 평가
python scripts/eval_record.py
```

Windows 콘솔에서 실행 시 `cp949` 인코딩 오류(`UnicodeEncodeError`)가 날 수 있음 — 실행 전 `chcp 65001` 또는 `set PYTHONIOENCODING=utf-8` 필요.

## 4. 결과 이력

| 날짜 | 모델 | Recall@5 | MRR | 문항품질 | kappa | ±1일치율 | 세트제약/구조Judge |
|---|---|---|---|---|---|---|---|
| 2025.07 | qwen2.5:1.5b | - | - | - | - | - | - |
| 2025.07 | qwen2.5:7b | 0.679 ❌ | 0.494 | 3.68 ❌ | 0.328 ❌ | 0.800 ✅ | 세트제약 ✅ (리디자인 전) |
| 2025.07 | qwen2.5:7b (생기부) | - | - | - | - | - | FN=0✅ NLI=0.1❌ 위반Recall=0.84❌ |
| 2026.07 | (BGE만, LLM 무관) | 0.905 ✅ | 0.659 | - | - | - | past_exams 8개 제거 후 재측정(n=21) |
| 2026.07 | qwen2.5:1.5b | - | - | - | - | - | 구조Judge count 0.667/diff 0.667/MAE 1.333 (STRUCTURE_GOLDEN 3개, 부트스트랩) |
| 2026.07.09 | qwen2.5:7b (JUDGE_TPL 5점 앵커 추가 전) | 0.857 | 0.611 | 오답매력도 2.50, 종합 3.70, 합격률 47% | 0.328 | 0.800 | 구조Judge MAE 2.000 (n=3) |
| 2026.07.09 | qwen2.5:7b (JUDGE_TPL 5점 앵커 추가 후) | 0.905 | 0.659 | 오답매력도 2.83(+0.33), 종합 3.79, 합격률 73%(+26%p) | 0.328(변화없음) | 0.800 | 구조Judge MAE 1.667 (n=3) |
| 2026.07.09 | qwen2.5:7b (agent_node 프롬프트 변경 전, `compare_distractor_quality.py`로 실제 생성) | - | - | 오답매력도 2.500 (n=8, 객관식만) | - | - | - |
| 2026.07.09 | qwen2.5:7b (agent_node 프롬프트 변경 후, `compare_distractor_quality.py`로 실제 생성) | - | - | 오답매력도 2.846 (n=13, 객관식만, +0.346) | - | - | - |

> 모델 교체 또는 프롬프트 튜닝 시마다 행 추가. 2026.07 Recall@5/MRR은 passage_text 리디자인으로 past_exams golden 항목이 제거되며 n이 28→21로 줄어 재측정한 값(검색은 LLM과 무관하므로 모델 열은 해당 없음).
>
> **2026.07.09 전/후 비교 주의사항**: 위 두 행은 20분 간격으로 연속 실행한 것으로, `JUDGE_TPL`(`정답유일성`·`오답매력도`·`근거성`)에 오답매력도=5점 few-shot 예시 1개만 추가한 차이만 있음(생성 프롬프트는 미변경). Recall@5(0.857→0.905)와 구조Judge MAE(2.000→1.667)도 이 변경과 무관한데 함께 흔들려서, 이 실행 간 약 ±0.05~0.3 수준의 자연 노이즈(HNSW 근사검색·LLM 샘플링 변동)가 있는 것으로 보임 — 오답매력도 +0.33도 전부가 few-shot 효과라고 단정하기보다는 방향성 신호로 해석 권장. 합격률이 47%→73%로 크게 뛴 건 여러 문항의 overall이 4.0 문턱을 살짝 넘었기 때문(경계 근처 문항이 많았다는 뜻).
>
> **방법론 오류 정정(2026.07.09)**: 이후 `graph.py` agent_node 프롬프트(생성 측)에도 오답 매력도 지시를 추가하고 같은 방식(`eval_exam.py` 전/후 재실행)으로 검증하려 했으나, `eval_item_quality()`가 채점하는 `ITEM_GOLDEN`은 **스크립트에 하드코딩된 고정 30개 문항**이라 agent_node를 전혀 호출하지 않는다 — 즉 생성 프롬프트를 바꿔도 이 지표엔 원리적으로 반영될 수 없다(실제로 전/후 평균이 2.815로 완전히 동일하게 나와서 발견). 생성 프롬프트 변경 효과는 `scripts/compare_distractor_quality.py`로 별도 검증함(아래 결과 이력 참고).

## 5. 진행 중인 조사

### 예시 문제 문장 → standards 검색 정합성 (2026.07.09)
retrieval_golden_final.json의 쿼리는 성취기준 해설 문체(주제어/서술형)로 만들어졌는데, 실제로는 교사가
시험 문제 문장(passage_text)을 그대로 검색에 쓸 수 있어 문체 격차가 있는지 검증되지 않은 상태였음.

- `scripts/eval_example_retrieval.py` + `data/golden/example_question_retrieval_test.json`(8개, 실제 문제 문장 스타일) 작성
- 라벨링(`expected_chunk_id`/`chunk_preview`/`reviewed`)은 사람이 직접 할 것 — 현재 0/8 라벨링 완료, Recall@5/MRR 정량 비교는 라벨링 후 가능
- 기존 골든셋 재확인 기준값: Recall@5=0.905, MRR=0.659 (n=21)
- 예비 관찰(정량 아님, top-5 reranker score 기준): "누진세에 대한 설명으로 옳은 것은?" 같은 구체적 문제 문장은 상위 후보 score가 -10대까지 낮게 나오는 반면, "소선거구제와 비례대표제의 차이점을 서술하시오." 같은 서술형은 -0.6대로 상대적으로 높음 — 문체보다 "핵심 개념어 포함 여부"가 더 큰 영향일 가능성. 라벨링 후 Recall 격차로 확인 필요
- **다음 실행**: `data/golden/example_question_retrieval_test.json`의 8개 항목에 `chunk_preview`(정답 청크 원문 앞부분)와 `expected_chunk_id`를 채우고 `reviewed: true`로 바꾼 뒤 `python scripts/eval_example_retrieval.py` 재실행 → 기존 골든셋과 Recall@5/MRR 나란히 비교됨

## 6. 개선 계획

| 지표 | 현재 | 목표 | 접근 방법 |
|---|---|---|---|
| ~~Recall@5~~ | 0.905 ✅ | ≥ 0.8 | past_exams 제거 후 이미 달성(2026.07) |
| 오답매력도 | 실제 생성 기준 2.500→2.846(+0.346, n=8→13, 객관식만, 2026.07.09) | ≥ 4.0 | 1단계(Judge 5점 앵커, ITEM_GOLDEN 채점) + 2단계(agent_node에 오답 매력도 지시+예시, 실제 생성 재검증) 둘 다 완료. 방향은 맞으나 목표에는 크게 못 미침 — few-shot을 진짜 멀티턴 tool-call 예시로 강화하거나, validate_item_format에 오답 매력도 최소 기준을 추가하는 등 추가 개입 필요 |
| Cohen's kappa | 0.328 (JUDGE_TPL 변경 전후 동일) | ≥ 0.4 | Judge 5점 앵커 추가만으론 kappa 불변 확인됨(고정 ITEM_GOLDEN 기준) — 근본 원인이 오답매력도 채점 기준 하나가 아닐 가능성, 추가 조사 필요 |
| 구조 Judge 신뢰도 | diff 0.667, MAE 1.333 (1.5b, n=3, count_match는 2026-07-09 개념 폐기로 지표에서 제외) | 미정 | STRUCTURE_GOLDEN 실제 모델(7B+) 라벨 보강(pending 8개 라벨링 대기 중) 후 재측정 |
| 규정 위반 Recall | 0.840 | ≥ 0.95 | 위반 탐지 프롬프트 튜닝 또는 규정 RAG 보강 |
| NLI 사실추가율 | 0.100 | = 0 | 오탐 2건 원인 분석 (골든셋 or 프롬프트 문제) |
