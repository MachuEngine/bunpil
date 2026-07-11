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
| STRUCTURE_GOLDEN | `data/golden/structure_golden.json` | 14개 (라벨링 대기) | count_match 폐기·num_items 도입으로 Claude 부트스트랩 전량 폐기, 실제 qwen2.5:7b 출력으로 전면 재생성(정확히 일치 5·부족 8·초과 1) — human_label 라벨링 대기 |
| MASKING_GOLDEN | `data/golden/masking_golden.json` | 20개 | 합성. 2026-07-09 `scripts/eval_record.py` 하드코딩에서 외부화 |
| VIOLATION_GOLDEN | `data/golden/violation_golden.json` | 50개 | 위반 25 + 정상 25. 2026-07-09 `scripts/eval_record.py` 하드코딩에서 외부화 |
| HALLUCINATION_GOLDEN | `data/golden/hallucination_golden.json` | 20개 | 합성. 2026-07-09 `scripts/eval_record.py` 하드코딩에서 외부화 |
| ITEM_GOLDEN | `data/golden/item_golden.json` | 30개 | human_score 1~5점 분포. 2026-07-09 `scripts/eval_exam.py` 하드코딩에서 외부화 |
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
| 2026.07.11 | qwen2.5:7b (STRUCTURE_GOLDEN 사람 라벨 20개 완성 후 첫 정식 측정) | - | - | - | - | - | 구조Judge diff 일치율 0.900(κ 0.615✅)/overall MAE 2.150·±1 일치 0.300·이진(≥3) κ **-0.103**❌/type_ratio MAE 0.206(r 0.575) — overall은 Judge가 +1.95 체계적 과대평가, 원인 분석 아래 참고 |
| 2026.07.11 | qwen2.5:7b (STRUCTURE_JUDGE_TPL에 rubric 주입 후 재측정) | - | - | - | - | - | 구조Judge overall MAE 1.800·±1 일치 0.500·이진 κ 0.111(여전히❌)·편향 +1.60 / diff κ 0.474(0.615→하락했으나 목표 유지) — 개선됐지만 미달, 잔여 원인은 "원문 복사 감지 실패"(아래 분석) |

> 모델 교체 또는 프롬프트 튜닝 시마다 행 추가. 2026.07 Recall@5/MRR은 passage_text 리디자인으로 past_exams golden 항목이 제거되며 n이 28→21로 줄어 재측정한 값(검색은 LLM과 무관하므로 모델 열은 해당 없음).
>
> **2026.07.09 전/후 비교 주의사항**: 위 두 행은 20분 간격으로 연속 실행한 것으로, `JUDGE_TPL`(`정답유일성`·`오답매력도`·`근거성`)에 오답매력도=5점 few-shot 예시 1개만 추가한 차이만 있음(생성 프롬프트는 미변경). Recall@5(0.857→0.905)와 구조Judge MAE(2.000→1.667)도 이 변경과 무관한데 함께 흔들려서, 이 실행 간 약 ±0.05~0.3 수준의 자연 노이즈(HNSW 근사검색·LLM 샘플링 변동)가 있는 것으로 보임 — 오답매력도 +0.33도 전부가 few-shot 효과라고 단정하기보다는 방향성 신호로 해석 권장. 합격률이 47%→73%로 크게 뛴 건 여러 문항의 overall이 4.0 문턱을 살짝 넘었기 때문(경계 근처 문항이 많았다는 뜻).
>
> **방법론 오류 정정(2026.07.09)**: 이후 `graph.py` agent_node 프롬프트(생성 측)에도 오답 매력도 지시를 추가하고 같은 방식(`eval_exam.py` 전/후 재실행)으로 검증하려 했으나, `eval_item_quality()`가 채점하는 `ITEM_GOLDEN`은 **스크립트에 하드코딩된 고정 30개 문항**이라 agent_node를 전혀 호출하지 않는다 — 즉 생성 프롬프트를 바꿔도 이 지표엔 원리적으로 반영될 수 없다(실제로 전/후 평균이 2.815로 완전히 동일하게 나와서 발견). 생성 프롬프트 변경 효과는 `scripts/compare_distractor_quality.py`로 별도 검증함(아래 결과 이력 참고).

## 5. 진행 중인 조사

### 구조 유사도 Judge 신뢰도 첫 정식 측정 결과 분석 (2026-07-11, n=20)

사람 라벨 20개 완성 후 `judge_structure_one()`(qwen2.5:7b)을 각 엔트리에 재실행해 대조.
raw 결과는 `data/golden/_structure_judge_eval_results.json`.

| 항목 | 결과 | 해석 |
|---|---|---|
| difficulty_match 일치율 / κ | 0.900 / **0.615** | 목표(0.4) 초과 — 난이도 판단은 신뢰 가능 |
| type_ratio_score MAE / r | 0.206 / 0.575 | 방향은 맞으나 Judge가 관대한 편 |
| overall_score MAE | 2.150 | 매우 나쁨 |
| overall_score 이진(≥3) κ | **-0.103** | 우연 이하 — 현재 형태로는 사용 불가 |
| Judge 편향 | +1.95 | 20건 중 19건에서 human보다 높거나 같게 채점 |

**원인(핵심)**: 사람 라벨은 이번에 신설된 rubric(`_schema.overall_score_rubric`)에 따라
**중복 문항·원문 복사·주제 이탈·환각을 감점**하는데, Judge 프롬프트(`STRUCTURE_JUDGE_TPL`)는
"유형 비율·난이도 구성의 구조적 유사도"만 물어봄 — 즉 **사람과 Judge가 서로 다른 것을
채점하고 있음**. 최대 불일치 사례가 전부 이 패턴: str_029/041/025/047(human 1: 완전 중복
문항들, judge 4~5: 유형·난이도는 실제로 일치), str_037(human 0: 언어 오염, judge 3).
Judge가 무능하다기보다 질문이 rubric과 정렬되지 않은 것이 1차 원인 — **다음 단계는
STRUCTURE_JUDGE_TPL에 rubric(중복·복사·주제 이탈 감점)을 주입하고 재측정**. 그래도 κ가
낮으면 그때 Judge 모델 크기(14B) 문제로 넘어감.

**rubric 주입 후 재측정 (2026-07-11, 같은 날)**: `STRUCTURE_JUDGE_TPL`에 0~5 rubric 전문과
"세트 내부 완전 중복 → overall 1" few-shot을 추가하고 20개 재실행.

| 항목 | 주입 전 | 주입 후 | 판정 |
|---|---|---|---|
| overall MAE | 2.150 | **1.800** | 개선 |
| overall ±1 일치 | 0.300 | **0.500** | 개선 |
| overall 이진(≥3) κ | -0.103 | **0.111** | 개선됐으나 목표(0.4) 미달 ❌ |
| Judge 편향 | +1.95 | +1.60 | 개선 |
| difficulty κ | 0.615 | 0.474 | 소폭 하락(목표는 유지 ✅, n=20 노이즈 범위) |

일부 케이스는 정확해짐(str_014: judge 3→0 정답, str_041: 5→3, str_037: 3→2). 그러나
**잔여 최대 불일치 6건(str_012/025/029/047/043/049, human 0~1 vs judge 3~5)은 전부
"생성 문항이 예시 문제의 원문 복사"인 케이스** — rubric에 "원문 단순 복사=1점"이 명시돼
있어도 7B Judge는 생성 문항과 예시 문제 텍스트가 사실상 동일하다는 대조를 해내지 못함.

**다음 단계 제안**: 원문 복사·세트 내 중복은 정성 판단이 아니라 **결정론적 텍스트 유사도
검사로 코드가 감지 가능** — "LLM judges, code decides" 원칙대로 이 감점 요소를 Judge에서
떼어내 코드 게이트(예: `save_item`에 예시 문제·기존 저장 문항과의 유사도 검사)로 옮기면,
Judge는 남은 정성 판단(주제·형식 유사성)만 담당하게 되어 정렬이 쉬워짐. 부수적으로 생성
품질 문제(중복 생성이 최다 감점 사유)도 동시에 개선됨. 그 후에도 κ가 낮으면 Judge 14B 검토.

**→ 유사도 게이트 적용됨 (2026-07-11)**: `save_item`에 문자 bigram 기반 결정론적 검사 추가
(`_check_similarity`, tools.py) — 예시 문제 대비 containment ≥0.90이면 "원문 복사" 거부,
기존 저장 문항 대비 jaccard ≥0.80이면 "세트 내 중복" 거부. 임계값은 라벨링된 골든셋 실측
분포로 결정(완전 복사 1.00 vs 정상 주제-유사 변형 ≤0.73 / 진짜 중복 0.86~1.00 vs 정상 변형
≤0.67). 소급 검증: human 0~1점 복사 케이스(str_012/025/029/047/043) 전부 차단, 중복 세트
(str_035/036/041/049) 전부 차단, 정상 세트 중복 오탐 0. 흥미롭게도 human 4~5점 세트에도
원문 복사 문항이 1개씩 있었는데(라벨링 시 세트 단위 관대 평가) rubric상 감점 대상이므로
차단이 설계에 부합. e2e 확인: 게이트가 복사·오염 문항을 실제로 거부하고 에이전트가 재작성을
시도함. **단, 게이트로 인해 7B의 "가짜 성공"(복사본 반환)이 명시적 실패/재시도로 바뀌면서
생성 성공률이 하락** — 시스템 프롬프트에 복사 금지 명시로 일부 완화했으나, 근본적으로 7B가
"새 문항 창작"을 어려워하는 문제가 드러남(이전엔 복사본이 성공으로 집계돼 가려져 있었음).
구조 Judge 재측정은 게이트 적용 후 생성된 새 골든셋(라벨링 필요)에서 의미가 있음.

**부수 발견**: human 라벨 분포 자체가 낮은 쪽에 몰림(평균 1.95/5, 0~2점이 20개 중 14개) —
rubric 기준으로 보면 현재 7B 생성 품질 자체가 낮다는 뜻이기도 함(중복 생성이 가장 흔한 감점
사유). 구조 Judge 신뢰도와 별개로 "중복 문항 생성" 자체를 줄이는 개입(예: 부분 진행 보존
프롬프트에 기존 문항과의 중복 금지 강화, 또는 save_item에 중복 검사 게이트)이 필요해 보임.

### 예시 문제 문장 → standards 검색 정합성 (2026.07.09)
retrieval_golden_final.json의 쿼리는 성취기준 해설 문체(주제어/서술형)로 만들어졌는데, 실제로는 교사가
시험 문제 문장(passage_text)을 그대로 검색에 쓸 수 있어 문체 격차가 있는지 검증되지 않은 상태였음.

- `scripts/eval_example_retrieval.py` + `data/golden/example_question_retrieval_test.json`(8개, 실제 문제 문장 스타일) 작성
- 라벨링(`expected_chunk_id`/`chunk_preview`/`reviewed`)은 사람이 직접 할 것 — 현재 0/8 라벨링 완료, Recall@5/MRR 정량 비교는 라벨링 후 가능
- 기존 골든셋 재확인 기준값: Recall@5=0.905, MRR=0.659 (n=21)
- 예비 관찰(정량 아님, top-5 reranker score 기준): "누진세에 대한 설명으로 옳은 것은?" 같은 구체적 문제 문장은 상위 후보 score가 -10대까지 낮게 나오는 반면, "소선거구제와 비례대표제의 차이점을 서술하시오." 같은 서술형은 -0.6대로 상대적으로 높음 — 문체보다 "핵심 개념어 포함 여부"가 더 큰 영향일 가능성. 라벨링 후 Recall 격차로 확인 필요
- **다음 실행**: `data/golden/example_question_retrieval_test.json`의 8개 항목에 `chunk_preview`(정답 청크 원문 앞부분)와 `expected_chunk_id`를 채우고 `reviewed: true`로 바꾼 뒤 `python scripts/eval_example_retrieval.py` 재실행 → 기존 골든셋과 Recall@5/MRR 나란히 비교됨

### 출제 에이전트 안정성 개선 (컨텍스트/tool-calling, 2026-07-10 야간 자율 세션)
TROUBLESHOOTING.md의 num_ctx 발견 이후 남은 잔여 tool-calling 실패율(~35~40%)을 줄이기 위한 실험.

- **설명텍스트 금지 지시 위치/강조**: 문장 끝에 한 번만 넣은 baseline은 효과 없었음(temperature A/B에서 0.70으로 무변화 확인). 정체성 선언 직후(초두 효과) + 끝(최신 효과) 두 번, "**매우 중요한 규칙**" 강조로 바꾼 strong 변형은 소표본(n=8, temperature=0.2)에서 턴당 설명텍스트 0.38→0.00으로 완전 제거 — 프로덕션(`graph.py`)에 strong 변형 채택
- **temperature 0.7 vs 0.2**: strong 프롬프트 기준 n=28 A/B. exact_match_rate(생성 개수==num_items) 0.7→14%, 0.2→21%(+7%p) — n=28에서 표준오차 약 ±7%p라 노이즈 범위. 초과생성 비율은 0.7→18%(5/28), 0.2→46%(13/28)로 뚜렷이 악화. **판단: temperature 기본값 0.7 유지**(정확도 개선은 불확실한데 초과생성 부작용은 명확) — 코드에는 파라미터만 추가, 즉시 전환 가능
- **재시도 시 부분 진행 보존**: 기존엔 재시도마다 `init_session()`으로 전체 초기화(이전 시도 문항 폐기). `plan_node`가 요청당 1회만 초기화하고, `agent_node`는 `reset_judge()`만 호출해 문항을 유지, "나머지 N개만 작성" 프롬프트로 이어서 생성하도록 개선. `test_exam.py`로 정성 확인 완료(재시도 후에도 이전 시도 문항이 최종 결과에 유지됨).
  - **원인 조사 결과(2026-07-11)**: `scripts/test_retry_preservation.py`의 old 시뮬레이션이 배치 실행에서 첫 샘플부터 30분+ 걸리는 현상을 `ollama.log`(`/opt/homebrew/var/log/ollama.log`)를 직접 보며 재조사. 프로세스/커넥션 누적, GPU 경합, `_invoke_with_retry()` 무한 재시도 전부 배제됨(로그상 매 턴 10~12초로 정상 처리, task ID 계속 증가) — **버그가 아니라 old 방식(budget=3, 전체 재시도) 자체가 `num_items`가 크고 잘 안 맞는 샘플에서 구조적으로 오래 걸리는 것**으로 결론. 상세 조사 과정은 TROUBLESHOOTING.md 참고
  - **정식 old vs new 정량 비교(n=8)는 시간 제약으로 중단**. 대신 실제 STRUCTURE_GOLDEN 생성 이력을 비공식 비교 근거로 사용: 개선 **이전** 코드로 생성된 14개(str_001~032, 2026-07-10 01:20 이전)는 문항 0개/부족 실패가 다수(8/14 부족), 개선 **이후** 코드로 생성된 6개(str_035~040, 재시도 시 부분 진행 보존 포함 전체 반영)는 **6/6 전부 0개 없이 성공**(정확히 일치 4·근접 초과 1·근접 부족 1) — 통제된 A/B는 아니지만 방향성은 뚜렷함. 정식 비교는 다음 세션에서 old 조건 샘플 수를 줄이거나(예: n=3) budget을 낮춰(예: budget=1) 재시도 권장
- **RAG top_k 3→2 실험**: Recall 0.810→0.762(-0.048, n=21), 기준(0.05) 경계선이라 top_k=3 유지
- **안정성 보강**: 장시간 세션에서 간헐적으로 발생하는 Ollama 스트림 오류에 `_invoke_with_retry()`(graph.py, 최대 2회 재시도) 추가

## 6. 개선 계획

| 지표 | 현재 | 목표 | 접근 방법 |
|---|---|---|---|
| ~~Recall@5~~ | 0.905 ✅ | ≥ 0.8 | past_exams 제거 후 이미 달성(2026.07) |
| 오답매력도 | 실제 생성 기준 2.500→2.846(+0.346, n=8→13, 객관식만, 2026.07.09) | ≥ 4.0 | 1단계(Judge 5점 앵커, ITEM_GOLDEN 채점) + 2단계(agent_node에 오답 매력도 지시+예시, 실제 생성 재검증) 둘 다 완료. 방향은 맞으나 목표에는 크게 못 미침 — few-shot을 진짜 멀티턴 tool-call 예시로 강화하거나, validate_item_format에 오답 매력도 최소 기준을 추가하는 등 추가 개입 필요 |
| Cohen's kappa | 0.328 (JUDGE_TPL 변경 전후 동일) | ≥ 0.4 | Judge 5점 앵커 추가만으론 kappa 불변 확인됨(고정 ITEM_GOLDEN 기준) — 근본 원인이 오답매력도 채점 기준 하나가 아닐 가능성, 추가 조사 필요 |
| 구조 Judge 신뢰도 | diff κ 0.474 ✅ / overall 이진 κ 0.111 ❌ (7B, n=20, rubric 주입 후) | diff κ ≥ 0.4 달성 / overall κ ≥ 0.4 | rubric 주입으로 MAE 2.15→1.80 개선했으나 미달. 잔여 원인은 원문 복사 감지 실패 — 복사·중복 감지를 코드 게이트로 이관 후 재측정 권장(5절 분석 참고) |
| 규정 위반 Recall | 0.840 | ≥ 0.95 | 위반 탐지 프롬프트 튜닝 또는 규정 RAG 보강 |
| NLI 사실추가율 | 0.100 | = 0 | 오탐 2건 원인 분석 (골든셋 or 프롬프트 문제) |
