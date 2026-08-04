# 분필(Bunpil) 코드 리뷰 체크리스트 (2026-08-04 갱신)

> 처음부터 끝까지 다시 리뷰한다고 가정하고, 의존성 순서(하위 → 상위)로 정리. 목표는
> "전부 읽기"가 아니라 핵심 구조를 설명할 수 있는 수준. 프론트엔드는 이번 라운드
> 범위 밖(제외). `bunpil_roadmap.md`의 "참고 — 코드 리뷰 대상 파일" 표는 예전
> 스냅샷이라 신뢰하지 말 것 — 진행 상황은 이 문서로 추적한다.

> **이번 갱신에서 바뀐 것**: ① 생기부(record) 모듈 전체 삭제(4단계 통째로 제거)
> ② `scripts/eval_*.py`·`scripts/langsmith_experiments.py`는 실제로 전부
> `evals/`에 있음(경로 오류) ③ `compare_models.py`·`gen_structure_golden.py`·
> `gen_golden_retrieval.py`·`eval_example_retrieval.py`도 `scripts/`가 아니라
> `experiments/`·`golden_gen/`(경로 오류) ④ `lexical.py`·`judge.py`·
> `eval_trajectory.py`·`compare_reranker.py` 신규 파일 4개 추가.

> **체크 표시 기준**: 예전 로드맵 표의 ✅는 2026-07-22에 "실제로 코드 리뷰가 전부
> 끝난 상태가 아님"으로 정정된 적이 있다(신뢰할 수 없는 상태로 방치됐던 표시). 같은
> 실수를 반복하지 않기 위해, 이 문서는 **서술 근거(누가 언제 무엇을 확인했는지)가
> 있는 항목만 체크**한다. "파일이 존재한다"·"오늘 이 파일을 수정했다"는 체크 근거가
> 아니다 — 리뷰(핵심 구조 설명 가능 여부 확인)와 수정/작성은 별개다.

## 1단계 — 기반 유틸 / LLM 공통 모듈

- [ ] `app/common/llm/base.py` — LLM 어댑터들의 공통 베이스
- [ ] `app/common/llm/backends/chat_runpod.py` — 왜 `BaseChatModel`을 직접 상속했는가, `_agenerate` vs `_generate` 차이
- [ ] `app/common/llm/backends/chat_openai.py` — RunPod 방식(직접 구현)과 왜 다르게(langchain_openai 래핑) 구현했는지 비교
- [ ] `app/common/llm/backends/ollama.py` — 설정값
- [ ] `app/common/llm/backends/openai.py` — 설정값
- [ ] `app/common/llm/backends/runpod.py` — 설정값
- [ ] `app/common/llm/factory.py` — 환경변수 분기 (흐름만 파악, ~10줄)
- [ ] `app/common/llm/tracing.py` — dev/prod LangSmith 프로젝트 자동 분기 로직 (18줄, `main.py` 맨 앞줄에서 호출)
- [ ] `app/common/llm/prompts.py` — `PromptTemplate` 공통 클래스, 여러 TPL이 어떻게 상속/사용하는지
- [ ] `app/common/privacy.py` — **(신규 추가)** `mask_pii()`. 생기부 모듈 삭제(2026-08-03) 후 유일한 사용처는 `app/main.py`의 `_build_spec()`(출제 경로, 모델 호출 **전** 마스킹). 원래 `record/masker.py`가 재노출하던 걸 이제 직접 참조

## 2단계 — RAG 공통 모듈

- [ ] `app/common/rag/store.py` — ChromaDB 컬렉션 구조. **2026-08-03 변경**: `all_documents()` 추가(BM25 인덱스 구축용), `query()`가 이제 `id`도 반환(dense·BM25 순위 조인 키)
- [ ] `app/common/rag/retriever.py` — 2단계 검색 흐름. **2026-08-03 대폭 변경**: dense-only → 하이브리드(BM25+dense, RRF 융합), `n_candidates` 기본값 20→10. 이 리뷰 라운드에서 가장 우선순위 높은 파일 — MODEL_SELECTION.md §5, EVAL.md §12·13과 함께 볼 것
- [ ] `app/common/rag/lexical.py` — **(신규, 2026-08-03)** BM25 역색인. `retriever.py`와 함께 봐야 함 — 왜 BGE-M3 `sparse_vecs` 대신 BM25 직접 구현을 택했는지(재인덱싱·스키마 변경 회피)가 핵심
- [ ] `app/common/rag/embedder.py` — BGE 모델 래퍼 (흐름만). `tokenize()` 메서드가 BM25용으로 추가됨(임베딩 계산 없이 토크나이저만 재사용)
- [ ] `app/common/rag/reranker.py` — BGE 리랭커 래퍼 (흐름만)
- [ ] `app/common/rag/singleton.py` — exam 모듈이 쓰는 lazy-singleton 패턴 (생기부 삭제로 현재 사용처는 exam뿐)
- [ ] `app/common/rag/parser.py` — 문서 파싱/청킹 로직
- [ ] `app/common/rag/telemetry.py` — ChromaDB 익명 텔레메트리 비활성화 override (10줄, 흐름만 파악)

## 3단계 — 출제(exam) 모듈

- [ ] `app/modules/exam/state.py` — `ExamSpec`, `DraftItem`, `ExamState` 타입 정의 (다른 파일 이해의 전제)
- [ ] `app/modules/exam/tools.py` — `@tool` 데코레이터, `_ctx` 공유 상태, 언어 게이트(`_check_korean`). **2026-08-03 변경**: `search_regulations` 도구 제거(도구 7→6개) — 선언("교육과정 법령 검색")과 실제 코퍼스(생기부 문서뿐) 불일치가 원인, 파일 상단 주석에 경위 있음
- [ ] `app/modules/exam/judge.py` — **(신규, 2026-07-23)** `STRUCTURE_JUDGE_TPL`·`judge_structure()`. self-judge(에이전트 자기채점) 폐기 후 런타임 `judge` 노드와 오프라인 eval이 공유하는 채점 함수를 분리한 파일 — 검증-배포 불일치 해소가 핵심 배경
- [ ] `app/modules/exam/graph.py` — LangGraph 노드 구조(`plan→agent→judge→validate`), 재시도 구조(부분 진행 보존). `judge` 노드가 도구가 아니라 그래프 노드로 분리된 배경은 `judge.py`와 함께 볼 것
- [ ] `app/modules/exam/llm.py` — 모델별 초기화(`num_ctx`, `temperature` 파라미터화 등)
- [ ] `scripts/test_exam.py` — `graph.py`/`tools.py`를 실제로 어떻게 호출해서 쓰는지 보여주는 사용 예시. 이론만 보고 끝내지 말고 이어서 보기

## ~~4단계 — 생기부(record) 모듈~~ — **2026-08-03 전체 삭제**

`app/modules/record/*`, `scripts/test_record.py`, `evals/eval_record.py` 전부 제거됨. 사유: 검증 규칙 6종 중
규정 근거가 확인된 건 1개뿐이고, 그 1개조차 사회 교과 문장(예: "정치적 다원주의 개념을 조사해 발표함")을
오탐하면서 정작 규정이 금지하는 문장은 놓쳤다. 하드룰 1 때문에 실제 데이터 검증도 불가능해 범위에서
제외했다. **코드는 git 이력에 남아있고, 조사·측정 과정은 EVAL.md 14절 "결말 — 생기부 모듈 제거"에
상세히 기록돼 있음** — 리뷰 대신 그 절을 읽는 게 더 빠름.

`mask_pii()`(1단계 `app/common/privacy.py`)만 출제 경로에서 유지됨.

## 5단계 — 앱 진입점

- [ ] `app/main.py` — FastAPI 엔드포인트 등록. **2026-08-03 변경**: `/record` 엔드포인트·`RecordRequest` 제거, 하드룰 3 관련 주석 정리. `/exam`·`/exam/stream` 중복 제거, 실제 SSE 노드 단위 스트리밍은 그대로

## 6단계 — 평가(eval) 스크립트

> **경로 정정**: 아래 전부 `scripts/`가 아니라 **`evals/`**에 있음.

- [ ] `evals/eval_exam.py` — retrieval/item/structure eval 통합, `STRUCTURE_JUDGE_TPL`(3점 앵커 반영). **2026-08-03**: `eval_retrieval()`이 하드코딩하던 `n_candidates=20` 제거, `retrieve()` 기본값을 따르도록 변경(검증-배포 불일치 방지)
- [ ] `evals/eval_lib.py` — golden 로더·judge 템플릿/함수 공용 모듈 (`eval_exam.py`에서 분리됨)
- [x] ~~`evals/eval_record.py`~~ — **삭제됨** (4단계 참고, 리뷰 대상에서 소멸)
- [x] `evals/eval_ragas.py` — 2026-07-12 리뷰 완료 기록 있음(`bunpil_roadmap.md`) — `build_sample()`→`faithfulness_one()`/`answer_relevancy_one()`→`run_langsmith_experiments()` 흐름 확인, 사소한 죽은 코드 1건·의도된 중복 방어 1건만 발견(둘 다 동작 무관, 미수정)
- [ ] `evals/langsmith_experiments.py` — 위 eval 스크립트가 공통으로 쓰는 LangSmith Experiments 기록 유틸
- [ ] `evals/eval_trajectory.py` — **(신규, 2026-08-03)** 산출물이 아니라 **과정**을 재는 스크립트. LangSmith 트레이스를 읽어 도구 호출 오류/거부율 분리, 재시도 원인(형식 실패 vs Judge 판단 불일치) 분류. `--since` 옵션의 필요성(신·구 아키텍처 트레이스 혼재)과 타임존 처리(로컬 자정 해석 후 UTC 정규화 — 2026-08-04 수정)가 리뷰 포인트. EVAL.md 11·11.1절, LANGSMITH_GUIDE.md 3.3·3.3.1절 참고

## 7단계 — 실험/도구 스크립트 (가벼운 리뷰)

> **경로 정정**: `compare_models.py`·`eval_example_retrieval.py`·`compare_judge_models.py`는 `experiments/`,
> `gen_structure_golden.py`·`gen_golden_retrieval.py`는 `golden_gen/`. `index_*`·`test_rag.py`·`test_llm.py`만 `scripts/`가 맞음.

- [ ] `experiments/compare_models.py` — 모델 비교 실험 설계(생성 모델/Judge 모델 분리 방식, `get_judge_backend()` 고정 설계와 직결)
- [ ] `experiments/compare_reranker.py` — **(신규, 2026-08-03)** 리랭커 ablation + `n_candidates` 스윕. "후보를 늘릴수록 리랭커가 나빠진다"는 반직관적 결과의 근거 스크립트 — `retriever.py` 리뷰 시 반드시 같이 볼 것
- [ ] `golden_gen/gen_structure_golden.py` — STRUCTURE_GOLDEN 생성 스크립트
- [ ] `golden_gen/gen_golden_retrieval.py` — 검색 골든셋 생성
- [ ] `experiments/eval_example_retrieval.py` — 검색 정합성 평가
- [ ] `scripts/index_standards.py` — RAG 인덱싱 (standards 컬렉션)
- [ ] `scripts/index_regulations.py` — RAG 인덱싱 (regulations 컬렉션). **주의**: 생기부 삭제 후 이 컬렉션은 런타임 미사용, **검색 eval 전용**(retrieval_golden 22건 중 10건이 여기 속해 유지)
- [ ] `scripts/test_rag.py` — RAG 파이프라인 테스트
- [ ] `scripts/test_llm.py` — LLM 백엔드 테스트

## 8단계 — 테스트 스위트 (신규 추가 — 오늘 변경 범위가 커서 별도 명시)

- [ ] `tests/test_bm25.py` — **(신규)** BM25 순수 로직 유닛테스트 8건 — 가짜 토크나이저로 조사 분리·idf 부호·빈 코퍼스 등 경계 케이스 검증
- [ ] `tests/test_masker.py` — **변경**: `eval_record.py` 삭제로 고아가 된 MASKING_GOLDEN 20건을 파라미터화 테스트로 흡수(FN=0 강제 유지)
- [ ] `tests/test_api_security.py` — **변경**: `/record` 기준 테스트를 전부 `/exam`·`/exam/stream` 기준으로 이전
- [ ] `tests/test_exam_input_privacy.py` — **변경**: 생기부 전용이던 `test_record_backends_are_not_langchain_traceable` → `test_plain_backends_are_not_langchain_traceable`로 일반화
- [x] ~~`tests/test_record_chain_rules.py`~~, ~~`tests/test_record_chain_safety.py`~~ — **삭제됨** (리뷰 대상에서 소멸)
- [ ] `tests/conftest.py`, `tests/test_exam_tool_gates.py`, `tests/test_exam_validation.py`, `tests/test_rag_store.py`, `tests/test_runpod_backend.py` — 변경 없음(이번 라운드 무관, 참고용)

## 스킵 가능 (코드보다 결과가 핵심인 1회성 A/B 스크립트 — `bunpil_roadmap.md`/`EVAL.md`에 결과 이미 서술됨)

- [ ] `experiments/compare_distractor_quality.py`
- [ ] `experiments/compare_judge_models.py`
- [ ] `experiments/test_temperature_effect.py`
- [ ] `experiments/test_topk_recall.py`
- [ ] `experiments/test_retry_preservation.py`

## 순서 원칙

하위 유틸(LLM 백엔드) → RAG 공통 → 도메인 모듈(exam, 상태 정의부터 사용 예시까지) → 앱 진입점 → 평가 스크립트 → 실험 스크립트 → 테스트.

## 참고

* 이번 갱신에서 새로 추가된 파일: `app/common/rag/lexical.py`, `app/modules/exam/judge.py`, `evals/eval_trajectory.py`, `experiments/compare_reranker.py`, `tests/test_bm25.py`
* 이번 갱신에서 완전히 제거된 것: `app/modules/record/` 전체, `scripts/test_record.py`, `evals/eval_record.py`, `tests/test_record_chain_*.py`, VIOLATION/HALLUCINATION 골든셋, 프론트 `RecordTab`
* `app/modules/exam/__init__.py`, `app/common/{llm,rag}/__init__.py`, `app/common/llm/backends/__init__.py`는 단순 재노출용이라 제외 (`app/modules/record/__init__.py`는 모듈 자체가 삭제되어 항목 소멸)
* 프론트엔드(`frontend/`)는 이번 라운드 범위 밖
