# 분필(Bunpil) 개발 로드맵

## 진행 상태 요약

- ✅ 완료: 배포, LangSmith 트레이싱, 골든셋 구축, eval 스크립트 실데이터 전환, Judge/생성 모델 분리, 7B 전환 및 첫 eval 실행, 코드 리뷰(`chat_runpod.py`, `graph.py`·`tools.py`·`store.py`+`retriever.py`·`chain.py`·`main.py`), 출제 모듈 passage_text 리디자인(2026.07, FEEDBACK_DRIVEN_REDESIGN_v2.md), 리디자인 후속 코드 리뷰 지적사항 5건 정리(2026.07, 69ebdee)
- ⬜ 남은 작업: 아래 "남은 작업" 목록 참고 (우선순위 순)

---

## ✅ 완료

### 배포
- EC2 + RunPod 서버리스 + Caddy HTTPS

### 모니터링 — LangSmith
- LangSmith 연동 + `@traceable` 데코레이터 적용 (`eval_exam.py`, `eval_record.py`)

### Eval 체계 — 골든셋 구축
| 골든셋 | 규모 | 비고 |
|---|---|---|
| retrieval_golden | 28개 | 실데이터 기반, 사람 검수 완료 |
| MASKING_GOLDEN | 20개 | 합성 |
| VIOLATION_GOLDEN | 50개 | 위반 25 + 정상 25 |
| HALLUCINATION_GOLDEN | 20개 | 합성 |
| ITEM_GOLDEN | 30개 | human_score 1~5점 분포 |

### Eval 체계 — 스크립트 업그레이드
- `eval_exam.py`, `eval_record.py` 실데이터 기반으로 업그레이드
- Judge/생성 모델 분리 (`OLLAMA_MODEL` / `OLLAMA_JUDGE_MODEL`)
- 7B 전환 및 첫 eval 실행 완료 (결과는 [EVAL.md](./EVAL.md) 참고)

### 코드 리뷰
- `chat_runpod.py`

### 출제 모듈 입력 방식 리디자인 (2026.07, FEEDBACK_DRIVEN_REDESIGN_v2.md)
- PDF 업로드 + 유형/난이도/개수 드롭다운 → `passage_text` 붙여넣기 단일 입력으로 전면 교체
- `check_duplicate`/`past_exams` 컬렉션 완전 제거 (2028 수능 개편으로 과목별 구조 무의미해짐), `similarity_judge`(구조 유사도 LLM Judge)로 대체
- retrieval_golden: past_exams 참조 8개(`ret_023`~`ret_030`) 제거 → 30개 → 22개(standards 12 + regulations 10), Recall@5/MRR 재측정 완료(0.905/0.659, n=21 — 이전 0.679/0.494보다 상승, past_exams가 상대적으로 검색 난도 높았던 것으로 추정)
- STRUCTURE_GOLDEN 부트스트랩 3개(str_001~003) 추가 — Claude가 만든 합성 데이터, eval 스캐폴딩 검증용(`eval_structure_judge()` 동작 확인 완료)
- standards 컬렉션 조회 도구(`search_standards`) 재도입
- 로컬 Ollama `num_predict` 캡 누락 버그 수정 (RunPod과 동일하게 2048 캡 — 폭주 생성 방지)

### 리디자인 후속 코드 리뷰 지적사항 5건 정리 (2026.07, 커밋 69ebdee)
- `/exam/stream`을 `graph.stream(stream_mode="updates")`로 전환해 실제 노드 단위(plan/agent/validate, 재시도 포함) 진행 이벤트 전송. 프론트(`ExamTab.tsx`)는 가짜 `LOADING_STEPS` 스테퍼 대신 `fetch`+`ReadableStream`으로 실제 SSE 소비
- `main.py`의 `/exam`·`/exam/stream` 중복 로직(truncation·spec 구성·그래프 실행)을 헬퍼로 추출
- `app/common/rag/singleton.py` 신설 — `exam/tools.py`·`record/chain.py`가 store·embedder·reranker·retriever를 공유하는 lazy-singleton으로 통합 (이전엔 서로 다른 패턴이었음)
- 앱 코드에서 안 쓰이던 `RAGStore.create_temp_collection`/`delete_collection`과 이를 시뮬레이션하던 `test_rag.py` 스텝 제거

---

## ⬜ 남은 작업 (우선순위 순)

1. **프론트엔드 프로덕션 배포** — Next.js 전환(Gradio `app/ui.py` 대체) 이후 `Dockerfile`·`docker-compose.yml`·`Caddyfile`에 `frontend/` 빌드·서빙이 반영되지 않아, 현재 배포 파이프라인만으로는 EC2에서 UI 접근 불가(FastAPI API만 서빙됨, 2026-07-08 확인). Vercel 등 별도 호스팅 + `BACKEND_URL`로 EC2 연결, 또는 EC2 상시 `next start` 프로세스 + Caddy 경로별 프록시 추가 중 택1
2. **오답매력도 개선** — 2026-07-09 완료: 1단계 `JUDGE_TPL`에 오답매력도=5점 few-shot 앵커 추가(ITEM_GOLDEN 기준 2.50→2.83). 2단계 `graph.py` agent_node 시스템 프롬프트에 오답 매력도 지시+예시 추가. **주의**: `eval_exam.py` 전/후 재실행으로 2단계를 검증하려 했으나 `eval_item_quality()`가 채점하는 `ITEM_GOLDEN`은 하드코딩된 고정 문항이라 agent_node 변경이 반영될 수 없음(방법론 오류, EVAL.md 4절 정정 사항 참고) — `scripts/compare_distractor_quality.py`로 실제 생성 기반 재검증: 오답매력도 2.500→2.846(+0.346, n=8→13, 객관식만). 방향은 맞으나 목표(4.0)엔 크게 못 미침, 추가 개입 필요(EVAL.md 6절 개선계획 참고)
3. **STRUCTURE_GOLDEN 실제 모델 라벨 보강** — 현재 3개(str_001~003)는 Claude 합성 부트스트랩. `scripts/gen_structure_golden.py`로 신규 passage 다수 생성 시도 — budget=1/3 모두 문항 0개 실패율이 높아(qwen2.5:7b tool-calling 안정성 문제, blog_draft.md "배운 것" 7번 참고) 여러 라운드(str_004~023)를 거쳐 문항 0개 결과는 전부 제외하고 **실제 출력이 나온 8개(str_005/007/008/011/015/017/018/020)**를 `data/golden/structure_golden_pending.json`에 확보(라벨 없음, 기존 `structure_golden.json` 미변경). 사람 라벨링(`human_label`) 후 수동 병합 필요 — 라벨링은 대신하지 않음
4. **생기부 모듈 eval 개선**
   - 규정 위반 Recall 0.840 → 0.95 목표 (위반 탐지 프롬프트 튜닝 또는 규정 RAG 보강)
   - NLI 사실추가율 오탐 2건 원인 분석 (골든셋 검수 or Judge 프롬프트 개선)
5. **모델 비교 실험** — Qwen2.5-7B vs GPT-3.5 vs Ollama 소형 모델
   - 동일 골든셋으로 3개 모델 eval 실행
   - 정량 비교 결과로 Qwen 채택 근거 확보
   - GPT-3.5는 API 비용 발생, 비교 후 즉시 종료
6. **Ragas 연동 + LangSmith Experiments 연동**
   - Faithfulness, Answer Relevancy 지표 추가 (`eval_ragas.py` 신규 스크립트)
   - eval 실행 시 결과가 LangSmith Experiments에 자동 기록되도록 연동
   - 모델/프롬프트 변경 시 Experiments 탭에서 결과 비교 가능
   - EVAL.md 결과 이력 수동 업데이트 → LangSmith 자동 기록으로 전환
   - **완료 직후 코드 리뷰 1건 추가**: `eval_ragas.py`는 완전 신규 스크립트라 "핵심 구조를 설명할 수 있는 수준" 원칙상 리뷰 필요. STRUCTURE_GOLDEN용 스크립트나 모델 비교 실험 코드는 기존 `graph.py`/`eval_exam.py` 호출 재사용 수준이라 작성하면서 바로 이해되므로 별도 리뷰 라운드 불필요 — `eval_ragas.py` 하나만 핵심으로 본다.
7. **GitHub Actions CI** — eval 자동화
8. **문서화 및 포트폴리오 정리**

---

## 참고 — 코드 리뷰 대상 파일

"전부 읽기"가 아니라 **핵심 구조를 설명할 수 있는 수준**이 목표.

| 파일 | 핵심 이해 포인트 | 상태 |
|---|---|---|
| `app/common/llm/backends/chat_runpod.py` | 왜 BaseChatModel을 직접 상속했는가, `_agenerate` vs `_generate` 차이 | ✅ |
| `app/modules/exam/graph.py` | LangGraph 노드 구조, 각 노드의 역할과 연결 (리디자인 이후 구조로 재검토 완료) | ✅ |
| `app/modules/exam/tools.py` | `@tool` 데코레이터, `_ctx` 공유 상태 문제, RAG 싱글턴 통합 | ✅ |
| `app/common/rag/store.py` + `retriever.py` | ChromaDB 컬렉션 구조, 2단계 검색 흐름, 죽은 임시 컬렉션 코드 제거 | ✅ |
| `app/modules/record/chain.py` | LCEL 파이프 구조, 하이브리드 위반 탐지 순서, RAG 싱글턴 통합 | ✅ |
| `app/main.py` | `/exam`·`/exam/stream` 중복 제거, 실제 SSE 노드 단위 스트리밍 | ✅ |
| `scripts/eval_ragas.py` | Faithfulness/Answer Relevancy 산출 방식, LangSmith Experiments 연동 구조 (Ragas 연동 작업 완료 직후 리뷰 예정, 신규 스크립트라 우선 리뷰 대상) | ⬜ |

### 흐름만 파악하면 되는 파일
- `app/common/llm/factory.py` — 환경변수 분기, 10줄
- `app/common/rag/embedder.py` / `reranker.py` — BGE 모델 래퍼
- `app/modules/record/masker.py` — 정규식 PII 마스킹
- `app/main.py` — FastAPI 엔드포인트 등록

---

## 보류 중인 아이디어 (우선순위 낮음, 추후 검토)

이전 로드맵에서 계획했으나 현재 "남은 작업" 목록엔 없는 항목들. 필요해지면 위 우선순위 목록에 편입.

### 모니터링 — 메트릭/대시보드
- FastAPI 미들웨어 메트릭 (`prometheus-fastapi-instrumentator`)
- Grafana 대시보드 (latency P50/P95, 오류율, 일별 요청 수)

### 컨텍스트 엔지니어링 체계화
- 현재 문제: 프롬프트가 `tools.py` 인라인에 하드코딩
- 목표: `app/prompts/*.yaml`로 버전 분리, 버전별 eval 점수 비교

```yaml
# exam_v1.yaml (예시)
version: "1.0"
system: |
  당신은 고등학교 사회 교사입니다...
user_template: |
  단원: {unit}
  성취기준: {standards}
```

### RAG 고도화
- **HyDE**: 질문으로 가상 문서를 먼저 생성한 뒤 검색
- **Multi-query retrieval**: 하나의 질문을 여러 각도로 변환해 검색
- **컨텍스트 압축**: 가져온 문서 청크를 LLM으로 요약 후 주입

### 기타
- 스트리밍 응답 (FastAPI `StreamingResponse` + SSE)
- 모델 워밍업 (`lifespan`에서 서버 시작 시 모델 로드)

### 포트폴리오 정리 시 포함할 것
- README에 아키텍처 다이어그램
- LangSmith 트레이스 스크린샷
- Grafana 대시보드 스크린샷 (구축 시)
- eval 결과 수치 ([EVAL.md](./EVAL.md) 참고)
- 기술 블로그 초안: "소형 LLM으로 RAG 시스템 만들기"
