# 평가 스크립트·데이터셋 맵 (한눈에 보기)

> `EVAL.md`가 "지표·기준·결과 이력"을 다룬다면, 이 문서는 **"이 프로젝트에 평가가
> 몇 층으로 나뉘어 있고, 각 층이 어떤 스크립트·데이터로 무엇을 확인하는지"**를
> 빠르게 찾기 위한 색인이다. 2026-07-23 조사 시점 기준(이후 2026-08-03/08-04 변경사항은
> 각 절에 갱신 표시로 반영).
>
> 결과·개선 스토리 요약은 [EVAL_SUMMARY.md](./EVAL_SUMMARY.md) 참고.
>
> **2026-07-23 디렉토리 재구성**: 원래 20개 스크립트가 전부 `scripts/` 하나에
> 섞여 있었는데, 아래 4계층 분류를 그대로 디렉토리로 옮겼다 —
> `evals/`(①) · `golden_gen/`(골든셋 생성) · `experiments/`(④) · `scripts/`(③ 스모크
> + 인덱싱 인프라만 남음). pytest(②)는 원래부터 `tests/`로 분리돼 있었다.

## 0. 4개 계층 — 먼저 이 구분부터

| 계층 | 위치 | 성격 | LLM 호출 | CI/정기 실행 |
|---|---|---|---|---|
| ① 정기 평가 파이프라인 | `evals/*.py` | 골든셋(사람 라벨) 대비 정량 지표 측정 | O | 모델·프롬프트 바꿀 때마다 수동 실행, 결과를 EVAL.md에 기록 |
| ② pytest 유닛테스트 | `tests/*.py` | 결정론적 로직 검증 (마스킹, 게이트, 순서 등) | 거의 없음(FakeLLM) | `pytest` 실행, 회귀 방지용 |
| ③ 스모크 테스트 | `scripts/test_exam.py`, `test_llm.py`, `test_rag.py`, `test_record.py` | 실제 로컬 Ollama로 파이프라인이 "일단 돌아가는지" 눈으로 확인 | O (로컬) | 구조 변경 직후 수동 1회 |
| ④ 일회성 실험/비교 | `experiments/*.py` | 특정 의사결정 하나를 검증하고 끝(A/B, 파라미터 튜닝) | O | 이미 실행 완료, 결과는 `data/golden/_*.json`에 아카이브 |

②는 "코드가 항상 이렇게 동작해야 한다"는 회귀 테스트, ①은 "모델/프롬프트 품질이
기준을 넘는가"라는 품질 게이트, ③은 "배선이 끊기지 않았는가"라는 연결 확인, ④는
"이 결정을 왜 이렇게 내렸는가"의 증거 기록 — 성격이 다 달라서 겹치는 것처럼 보여도
중복이 아니다.

---

## 1. ① 정기 평가 파이프라인 — `evals/`

```bash
python evals/eval_exam.py      # 출제 모듈
```

> `evals/eval_record.py`는 2026-08-03 생기부 모듈 제거와 함께 삭제됨(아래 "생기부 모듈" 절은
> 삭제 당시 기록으로만 보존).

### 출제 모듈 — `evals/eval_exam.py` (+ 공용 로직 `evals/eval_lib.py`)

| 지표 | 방식 | 골든셋 | 기준 |
|---|---|---|---|
| Recall@5 / MRR | 함수 | `retrieval_golden_final.json` (22개, 22개 reviewed) | Recall ≥ 0.8 |
| 문항 품질(정답유일성·오답매력도·근거성) | LLM Judge | `item_golden.json` (30개, human_score) | 평균 ≥ 4.0 |
| Judge 신뢰도(문항 품질) | 사람 라벨 대비 kappa/±1 일치율 | 〃 | kappa ≥ 0.4, ±1 ≥ 0.7 |
| 구조 유사도 Judge 신뢰도 | 사람 라벨 대비 MAE/일치율 (LLM Judge, `judge_structure()`) + 문항 개수 일치(코드) | `structure_golden.json` (45개, human_label) | 참고값, 별도 pass/fail 게이트 없음 — overall 이진 kappa 게이트는 2026-07-24 폐기 결정(EVAL.md 6절) |

> **2026-07-23부터**: `structure_golden.json`을 채점하는 `judge_structure()`가
> `app/modules/exam/judge.py`에 있고, 이게 런타임 `graph.py`의 `judge_node`가
> 호출하는 **바로 그 함수**다(이전엔 서로 다른 코드 경로였음 — 이 문서 하단 §7 참고).
> 즉 이 표의 "구조 유사도 Judge 신뢰도" 수치가 곧 배포된 judge의 신뢰도.

### ~~생기부 모듈 — `evals/eval_record.py`~~ — **2026-08-03 삭제됨** (아래는 삭제 전 기록)

| 지표 | 방식 | 골든셋 | 기준 |
|---|---|---|---|
| PII 마스킹 FN율 | 함수 | `masking_golden.json` (20개) | = 0 |
| 사실 추가율(키워드) | 함수 | `hallucination_golden.json` (20개) | = 0 |
| 사실 추가율(NLI Judge) | LLM Judge | 〃 | = 0 |
| 규정 위반 Recall | 함수 | `violation_golden.json` (50개: 위반25+정상25) | ≥ 0.95 |
| regulations RAG Recall@5/MRR | 함수 | (내장 쿼리) | 참고값 |

### 출제 모듈 RAG 품질 — `evals/eval_ragas.py`

Ragas 패키지 대신 Faithfulness/Answer Relevancy 알고리즘을 직접 구현(의존성 충돌
회피, EVAL.md 8절). 실제 `get_exam_graph()`를 호출해 생성된 문항을 채점하므로
①에 속하지만 `eval_exam.py`와 별도 스크립트로 분리돼 있다. 결과는
`data/golden/_ragas_eval_results.json`에 누적.

**신뢰도(kappa 등) 3개 지표는 실행할 때마다 LangSmith Experiments에도 자동
기록**된다(`evals/langsmith_experiments.py` 공용 유틸). 회차별 최신 수치는
LangSmith 탭이 더 정확한 소스 — EVAL.md는 "왜 이렇게 바꿨는지" 서사 기록용.

---

## 2. 골든셋을 만드는 스크립트 — `golden_gen/` (평가 자체가 아니라 데이터 생성)

| 스크립트 | 만드는 파일 | 용도 |
|---|---|---|
| `gen_structure_golden.py` | `data/golden/structure_golden.json` | 실제 qwen2.5 출력으로 구조 유사도 골든셋 생성(초안, human_label은 사람이 채움) |
| `gen_golden_retrieval.py` | `data/golden/retrieval_golden.json` | ChromaDB 컬렉션에서 검색 골든셋 초안 생성(`reviewed: false`, 사람 검수 필요) |

---

## 3. 인덱싱 스크립트 — `scripts/` (평가 아님, RAG 컬렉션 구축)

| 스크립트 | 역할 |
|---|---|
| `index_regulations.py` | `data/regulations/` PDF → regulations 영구 컬렉션 (idempotent) |
| `index_standards.py` | `data/standards/` 텍스트/PDF → standards 영구 컬렉션 (idempotent) |

---

## 4. ③ 스모크 테스트 (`scripts/test_*.py` — pytest 아님, 그냥 python 스크립트)

| 스크립트 | 확인 내용 |
|---|---|
| `test_llm.py` | LLM 백엔드 연결(응답이 오는지)만 확인. 내용 정확도는 범위 밖 |
| `test_rag.py` | 인덱싱→검색→rerank 배선 확인 |
| `test_exam.py` | 출제 그래프(`plan→agent→judge→validate`) 전체 흐름을 실제 로컬 모델로 1회 실행해 확인. 2026-07-23 judge 분리 반영해 docstring 갱신됨, 2026-08-04 `load_dotenv()`/`init_langsmith_project()` 호출 추가(LANGSMITH_GUIDE.md 3.3.1절) |

> `test_record.py`는 2026-08-03 생기부 모듈 제거와 함께 삭제됨.

이름이 `tests/`의 pytest 파일들과 비슷해서 헷갈리기 쉬운데, 이쪽은 **실제 로컬
Ollama를 호출하는 수동 스모크 확인**이고 `tests/`는 **FakeLLM/결정론적 로직만
보는 pytest 회귀 테스트**라 역할이 다르다. 이번 judge 분리처럼 그래프 노드 구조가
바뀌면 재실행 대상은 이쪽(`test_exam.py`) + `tests/test_exam_validation.py` 둘 다.

## 4-1. ② pytest 유닛테스트 (`tests/`)

`pytest`로 일괄 실행. 대부분 LLM 호출 없이 순수 로직/결정론적 게이트만 검증:

| 파일 | 검증 대상 |
|---|---|
| `test_masker.py` | PII 마스킹 순수 로직. **2026-08-03 변경**: `eval_record.py` 삭제로 고아가 된 MASKING_GOLDEN 20건을 파라미터화 테스트로 흡수 |
| `test_exam_tool_gates.py` | save_item/record_score/discard_item 결정론적 게이트 |
| `test_exam_validation.py` | validate_node 재시도 피드백 생성 (judge 분리 반영해 similarity_judge 참조 제거됨) |
| `test_exam_input_privacy.py` | 출제 입력이 LLM 호출보다 먼저 마스킹되는지 |
| `test_bm25.py` | **(신규, 2026-08-03)** BM25 순수 로직 유닛테스트 |
| `test_rag_store.py` | Chroma 익명 텔레메트리 비활성화 확인 |
| `test_runpod_backend.py` | RunPod 작업 중복 제출 방지 |
| `test_api_security.py` | 인증/요청 크기 경계 |

> `test_record_chain_rules.py`/`test_record_chain_safety.py`는 2026-08-03 생기부 모듈 제거와 함께 삭제됨.

---

## 5. ④ 일회성 실험/비교 스크립트 — `experiments/` (이미 실행 완료, 결과는 아카이브)

정기 평가 파이프라인에 안 쓰이고, 각자 특정 질문 하나에 답하기 위해 한 번(또는
A/B 몇 회) 실행되고 끝난 스크립트들. 결과는 `data/golden/_*.json`(밑줄 접두사)에
남아있다.

| 스크립트 | 검증했던 질문 | 결과 파일 |
|---|---|---|
| `compare_models.py` | 생성 모델 비교(Qwen2.5-7B/14B, Llama3.1-8B, GPT-4o-mini) | `_model_comparison_results*.json` |
| `compare_judge_models.py` | Judge 모델 비교(로컬 qwen2.5 vs OpenAI gpt-5.6-luna/sol) | `_judge_comparison_results.json` |
| `compare_distractor_quality.py` | agent_node 오답매력도 프롬프트 변경 전/후 효과 | `_distractor_quality_compare.json` |
| `test_temperature_effect.py` | temperature 0.7 vs 0.2가 tool-calling 성공률에 미치는 영향 | `_temperature_ab_compare.json` |
| `test_retry_preservation.py` | 재시도 시 부분 진행 보존 개선 전/후 비교 | (`_retry_preservation_compare.json`, 미생성) |
| `test_topk_recall.py` | top_k 3→2 축소 시 Recall 저하 폭 | `_topk_recall_compare.json` |
| `measure_validate_gate.py` | **(신규 2026-08-04)** validate 게이트 임계값이 Judge 실제 점수 분포 대비 적절한가 — 옛 기준 통과율 6.7%로 도달 불가 확인 | `_validate_gate_calibration.json` |
| `eval_example_retrieval.py` | "실제 문제 문장" 스타일 쿼리가 standards 검색과 잘 맞는지 | `data/golden/example_question_retrieval_test.json` (라벨링 미완료, 판정 보류 상태) |

### ⚠️ 재실행 시 주의 — judge 분리로 stale해진 부분

`compare_distractor_quality.py`, `test_temperature_effect.py`,
`test_retry_preservation.py`의 `run_old()`는 자체 시스템 프롬프트에 "작성이
끝나면 **similarity_judge 도구**를 호출해 평가하라"는 지시를 담고 있다. 그런데
2026-07-23 커밋(`11de6ed`)에서 `similarity_judge`가 `TOOLS`(`app/modules/exam/tools.py`)에서
완전히 제거됐다. import는 깨지지 않지만(에이전트가 그 이름의 도구를 애초에
`bind_tools`로 받지 못하므로 조용히 무시됨), **조기 종료 로직(`judged` 플래그)이
더 이상 트리거되지 않아** 매 시도 14턴을 다 채울 때까지 도는 비효율이 생긴다.
결과(`get_draft_items()`)는 여전히 유효하지만, 이 스크립트들을 **다시 실행할
계획이 있다면** 프롬프트 지시를 `submit_for_review`(무인자 종료 신호)로 갱신부터
해야 한다. 이미 끝난 실험이라 당장 조치 불필요 — 재실행 직전에만 손보면 됨.

---

## 6. 파일 점검 결과 — "안 쓰는 파일"이 있는가?

**스크립트 자체는 전부 정상.** `evals/`·`golden_gen/`·`experiments/`·`scripts/`
전체 20개 import 시도 결과 에러 없음(재배치 후 재검증 완료 — cross-directory
import 7곳은 `sys.path.insert`를 추가해 수정, §7 참고).
`check_duplicate`/`past_exams`/`count_match` 같은 폐기된 개념에 대한 참조는 전부
"과거에 이랬다"는 docstring/주석뿐이고 실제 코드에서 호출하는 곳은 없음(정상
아카이브 기록).

| 대상 | 상태 | 판단 |
|---|---|---|
| `structure_golden_v1_labeled.json` | 코드에서 로드하는 곳 없음 | 의도적 아카이브(`data/golden/README.md`에 명시) — 삭제해도 기능엔 영향 없으나, 트러블슈팅/블로그 참고용으로 보존 결정된 파일. 지울지는 판단 필요 사안 |
| `structure_golden_v2_pre_retry_fix.json` | 〃 | 〃 |
| `structure_golden_contaminated_examples.json` | 〃 | 〃 |
| `example_question_retrieval_test.json` | `experiments/eval_example_retrieval.py`가 로드하지만 `reviewed: false`라 채점에선 자동 제외됨(라벨링 미완료 상태로 방치) | 완료할지 폐기할지 결정 필요 — 현재는 "진행 중도 아니고 끝나지도 않은" 애매한 상태 |
| `compare_distractor_quality.py` / `test_temperature_effect.py` / `test_retry_preservation.py` | §5의 stale 프롬프트 이슈 있음 | 삭제 대상 아님(의사결정 기록으로서 가치 있음), 단 재실행 전 프롬프트 갱신 필요 |
| 나머지 전체 | 각자 문서화된 역할 있음, 중복 없음 | 정리 불필요 |

**결론: 삭제해야 할 완전히 죽은 파일은 없음.** 다만 위 3개 항목은 "왜 존재하는지"가
README/EVAL.md에 이미 적혀 있긴 해도 실제로 쓰이진 않는 상태라, 언젠가
정리(삭제 또는 최종 결론 내리기)할지 여부는 사람 판단이 필요해 표로만 남겨둠 —
이 세션에서 임의로 지우지 않음.

---

## 7. 이번 세션에서 있었던 두 가지 변경

### 7.1 judge 분리(2026-07-23, `11de6ed`)가 각 계층에 미친 영향

- **① 정기 평가**: `structure_golden.json` 채점 로직이 `eval_lib.py`에서
  `app/modules/exam/judge.py`로 이동 — 로직 자체는 동일(순수 relocation, diff
  확인함), 신규 실행 없이도 "지금까지의 구조 Judge 신뢰도 수치"는 여전히 유효.
  단, **그래프 흐름 자체가 바뀌었으므로**(self-judge 제거 → 별도 judge 노드)
  end-to-end 스모크(`scripts/test_exam.py`)와 `tests/test_exam_validation.py`는
  이미 이번 커밋에 포함되어 갱신·통과 확인됨.
- **② pytest**: `tests/test_exam_validation.py`에서 `similarity_judge` 참조 제거,
  나머지 파일은 영향 없음.
- **③ 스모크**: `scripts/test_exam.py` docstring 갱신 완료.
- **④ 일회성 실험**: §5에서 정리한 3개 스크립트가 stale — 재실행 전 갱신 필요.

### 7.2 디렉토리 재구성(2026-07-23, 이 문서 §0 참고)

`scripts/` 하나에 20개가 몰려 있던 걸 4계층 기준 디렉토리(`evals/`, `golden_gen/`,
`experiments/`, `scripts/`)로 분리. 이동 자체는 `git mv`로 히스토리 보존, 아래
7개 파일은 다른 디렉토리로 옮긴 스크립트를 이름만으로 import하고 있었어서
`sys.path.insert`를 한 줄씩 추가해 고쳤다(패키지화는 하지 않음 — 실행 커맨드가
`python evals/eval_exam.py`처럼 지금과 동일한 형태를 유지하도록 최소 변경):

| 파일 | 새로 필요해진 import |
|---|---|
| `evals/eval_ragas.py` | `golden_gen/gen_structure_golden.py`의 `PASSAGE_SAMPLES` |
| `experiments/compare_models.py` | 〃 |
| `experiments/test_temperature_effect.py` | 〃 |
| `experiments/test_retry_preservation.py` | 〃 |
| `experiments/compare_judge_models.py` | `evals/eval_lib.py` |
| `experiments/compare_distractor_quality.py` | 〃 |
| `experiments/eval_example_retrieval.py` | 〃 |

이동 후 20개 스크립트 전체를 (main 블록 제외) import 재검증해 에러 없음을
확인했다. `EVAL.md`/`README.md`/`MODEL_SELECTION.md`/`LANGSMITH_GUIDE.md`/
`DESIGN.md`의 **현재 상태를 설명하는 부분**(실행 커맨드, "현재 이 스크립트가
담당" 류 서술)은 새 경로로 갱신했고, `EVAL.md`·`bunpil_roadmap.md`의 **날짜가
찍힌 과거 기록(결과 이력·완료 항목)은 당시 실제 경로(`scripts/...`)를 그대로
보존**했다 — 두 문서 모두 "과거 기록은 지우지 않고 그대로 둔다"는 정책을 이미
명시하고 있어서다.
