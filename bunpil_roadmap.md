# 분필(Bunpil) 개발 로드맵

## 진행 상태 요약

- ✅ 완료: 배포, LangSmith 트레이싱, 골든셋 구축, eval 스크립트 실데이터 전환, Judge/생성 모델 분리, 7B 전환 및 첫 eval 실행, 코드 리뷰 일부(`chat_runpod.py`, `graph.py`)
- 🔄 진행 중: 코드 리뷰 (`tools.py` → `store.py`+`retriever.py` → `chain.py`)
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
- `graph.py`

---

## 🔄 진행 중

- 코드 리뷰: `tools.py` → `store.py` + `retriever.py` → `chain.py`

---

## ⬜ 남은 작업 (우선순위 순)

1. **코드 리뷰 완료** — `tools.py` → `store.py`+`retriever.py` → `chain.py` (면접 준비 직결)
2. **Recall@5 개선** — 현재 0.679 → 목표 0.8 (리트리버/임베딩/청킹 튜닝)
3. **오답매력도 개선** — 현재 2.43 (출제 프롬프트 튜닝)
4. **eval_record.py 7B로 재실행** — 1.5b 기준 통과였으나 7B 결과 미확인
5. **모델 비교 실험** — Qwen2.5-7B vs GPT-3.5 vs Ollama 소형 모델
   - 동일 골든셋으로 3개 모델 eval 실행
   - 정량 비교 결과로 Qwen 채택 근거 확보 → 포트폴리오 서사에 활용
   - GPT-3.5는 API 비용 발생, 비교 후 즉시 종료
6. **Ragas 연동 + LangSmith Experiments 연동**
   - Faithfulness, Answer Relevancy 지표 추가 (`eval_ragas.py` 신규 스크립트)
   - eval 실행 시 결과가 LangSmith Experiments에 자동 기록되도록 연동
   - 모델/프롬프트 변경 시 Experiments 탭에서 결과 비교 가능
   - EVAL.md 결과 이력 수동 업데이트 → LangSmith 자동 기록으로 전환
7. **GitHub Actions CI** — eval 자동화
8. **문서화 및 포트폴리오 정리**

---

## 참고 — 코드 리뷰 대상 파일

"전부 읽기"가 아니라 **면접에서 설명할 수 있는 수준**이 목표.

| 파일 | 핵심 이해 포인트 | 상태 |
|---|---|---|
| `app/common/llm/backends/chat_runpod.py` | 왜 BaseChatModel을 직접 상속했는가, `_agenerate` vs `_generate` 차이 | ✅ |
| `app/modules/exam/graph.py` | LangGraph 노드 구조, 각 노드의 역할과 연결 | ✅ |
| `app/modules/exam/tools.py` | `@tool` 데코레이터, `_ctx` 공유 상태 문제 | 🔄 |
| `app/common/rag/store.py` + `retriever.py` | ChromaDB 컬렉션 구조, 2단계 검색 흐름 | ⬜ |
| `app/modules/record/chain.py` | LCEL 파이프 구조, 하이브리드 위반 탐지 순서 | ⬜ |

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
