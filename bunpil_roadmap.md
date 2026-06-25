# 분필(Bunpil) 개발 로드맵

## 전체 진행 순서

```
1단계: 배포 완료
2단계: 코드 리뷰 (핵심 파일 이해)
3단계: 모니터링 추가
4단계: 컨텍스트 엔지니어링 체계화
5단계: Eval CI 자동화
6단계: 성능 개선
7단계: 문서화 및 포트폴리오 정리
```

---

## 1단계: 배포 완료

### 체크리스트

- [ ] EC2 인스턴스에 Docker + docker-compose 설치
- [ ] `.env` 파일 작성 (`.env.example` 참고)
  - `LLM_BACKEND=runpod`
  - `RUNPOD_ENDPOINT_ID=...`
  - `RUNPOD_API_KEY=...`
  - `CHROMA_PERSIST_DIR=/data/chroma_db`
- [ ] EBS 볼륨 마운트 확인 (`/data/chroma_db`)
- [ ] `docker-compose up -d` 실행
- [ ] `/health` 엔드포인트 응답 확인
- [ ] Caddyfile 도메인 설정 (`your-domain.com` 교체)
- [ ] Caddy 실행 및 HTTPS 인증서 발급 확인
- [ ] RunPod 엔드포인트 실제 요청 테스트

### 현재 알려진 배포 이슈

- `Caddyfile`에 `your-domain.com` 하드코딩 → 실제 도메인으로 교체 필요
- `docker-compose.yml` 포트가 `7860`인데 `main.py`는 FastAPI (Next.js 연동 구조) → 포트 정합성 확인
- `CORS`에 `http://localhost:3000`만 허용 → 프로덕션 프론트엔드 도메인 추가 필요
- 첫 요청 시 BGE 모델 로딩으로 수십 초 지연 발생 → 추후 `lifespan` 워밍업으로 해결

---

## 2단계: 코드 리뷰 (핵심 파일 이해)

"전부 읽기"가 아니라 **면접에서 설명할 수 있는 수준**이 목표

### 깊게 봐야 하는 파일 (우선순위 순)

| 파일 | 핵심 이해 포인트 |
|---|---|
| `app/common/llm/backends/chat_runpod.py` | 왜 BaseChatModel을 직접 상속했는가, `_agenerate` vs `_generate` 차이 |
| `app/modules/exam/graph.py` | LangGraph 노드 구조, 각 노드의 역할과 연결 |
| `app/modules/exam/tools.py` | `@tool` 데코레이터, `_ctx` 공유 상태 문제 |
| `app/common/rag/store.py` + `retriever.py` | ChromaDB 컬렉션 구조, 2단계 검색 흐름 |
| `app/modules/record/chain.py` | LCEL 파이프 구조, 하이브리드 위반 탐지 순서 |

### 흐름만 파악하면 되는 파일

- `app/common/llm/factory.py` — 환경변수 분기, 10줄
- `app/common/rag/embedder.py` / `reranker.py` — BGE 모델 래퍼
- `app/modules/record/masker.py` — 정규식 PII 마스킹
- `app/main.py` — FastAPI 엔드포인트 등록

### 볼 필요 없는 파일

- `scripts/` — 실행해보면 됨
- `frontend/` — 별도 영역으로 취급

---

## 3단계: 모니터링 추가 ← 현재 최대 약점

### 3-1. LangSmith 트레이싱 (1~2일)

LangGraph 전체 흐름이 자동으로 기록됨 — 코드 변경 최소

```bash
# .env에 추가
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=bunpil
```

확인 사항:
- [ ] LangSmith 가입 및 API 키 발급
- [ ] `.env`에 위 3줄 추가
- [ ] 실제 요청 후 LangSmith 대시보드에서 trace 확인
- [ ] 각 노드 latency, token 수 확인

### 3-2. FastAPI 미들웨어 메트릭 (2~3일)

```python
# 추가할 것들
- 요청 latency (엔드포인트별)
- 오류율
- token 사용량 (RunPod 응답에서 추출)
```

도구: `prometheus-fastapi-instrumentator` 라이브러리 (2줄 설치)

### 3-3. Grafana 대시보드 (1일)

- EC2에 Prometheus + Grafana docker-compose로 추가
- 핵심 패널: latency P50/P95, 오류율, 일별 요청 수

---

## 4단계: 컨텍스트 엔지니어링 체계화

### 현재 문제

프롬프트가 `tools.py` 인라인에 하드코딩되어 있음

```python
# 지금 (나쁜 예)
prompt = f"다음 단원에 대한 시험 문제를 출제하세요: {unit}"
```

### 목표 구조

```
app/
  prompts/
    exam_v1.yaml      ← 버전 관리
    exam_v2.yaml
    record_v1.yaml
```

```yaml
# exam_v1.yaml
version: "1.0"
system: |
  당신은 고등학교 사회 교사입니다...
user_template: |
  단원: {unit}
  성취기준: {standards}
```

- [ ] `prompts.py` → YAML 분리
- [ ] 버전별 eval 점수 비교 스크립트 작성
- [ ] 프롬프트 변경 시 Recall@5, LLM Judge 점수 자동 비교

---

## 5단계: Eval CI 자동화

### 목표

PR 올릴 때마다 eval이 자동 실행되고 점수가 코멘트로 달림

```
PR 생성
  → GitHub Actions 트리거
  → eval_exam.py 실행
  → Recall@5, LLM Judge 점수 출력
  → PR 코멘트에 결과 자동 게시
```

### 추가할 것

- [ ] `.github/workflows/eval.yml` 작성
- [ ] Ragas 연동 (faithfulness, answer_relevancy 표준 메트릭)
- [ ] 점수 기준선(baseline) 설정 — 이하면 PR 실패 처리

---

## 6단계: 성능 개선

### RAG 고도화

- **HyDE**: 질문으로 가상 문서를 먼저 생성한 뒤 검색 → 검색 품질 향상
- **Multi-query retrieval**: 하나의 질문을 여러 각도로 변환해 검색
- **컨텍스트 압축**: 가져온 문서 청크를 LLM으로 요약 후 주입

### 스트리밍 응답

- FastAPI `StreamingResponse` + SSE
- Next.js `useChat` 또는 `EventSource`
- 체감 UX 크게 향상, 면접 단골 주제

### 모델 워밍업

```python
# app/main.py에 추가
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_record_chain()  # 서버 시작 시 모델 로드
    yield
```

---

## 7단계: 문서화 및 포트폴리오 정리

- [ ] README에 아키텍처 다이어그램 추가
- [ ] LangSmith 트레이스 스크린샷
- [ ] Grafana 대시보드 스크린샷
- [ ] eval 결과 수치 (Recall@5=1.0, LLM Judge 점수)
- [ ] 기술 블로그 초안: "소형 LLM으로 RAG 시스템 만들기"

---

---

# 오늘~내일 할 수 있는 것

## 오늘 (3~4시간)

### 목표: 배포 환경 준비 완료

**1. .env 파일 작성 (30분)**
```bash
cp .env.example .env
# 아래 값 채우기
LLM_BACKEND=runpod
RUNPOD_ENDPOINT_ID=실제값
RUNPOD_API_KEY=실제값
CHROMA_PERSIST_DIR=/data/chroma_db
```

**2. Caddyfile 도메인 교체 (10분)**
```
# your-domain.com → 실제 도메인으로 교체
```

**3. main.py CORS 수정 (10분)**
```python
allow_origins=[
    "http://localhost:3000",
    "https://your-frontend-domain.com"  # 추가
]
```

**4. EC2에서 docker-compose up (1시간)**
```bash
git pull
docker-compose up -d
docker-compose logs -f  # 로그 확인
curl http://localhost:7860/health
```

**5. RunPod 엔드포인트 실제 테스트 (30분)**
```bash
# /record 엔드포인트로 간단한 요청
curl -X POST http://localhost:7860/record \
  -H "Content-Type: application/json" \
  -d '{"memo": "학생이 수업에 적극적으로 참여함"}'
```

---

## 내일 (3~4시간)

### 목표: 핵심 코드 이해 + LangSmith 연동

**1. 핵심 파일 코드 리뷰 (2시간)**

순서대로, 각 파일 읽으면서 모르는 것 바로 물어보기:
1. `chat_runpod.py` — BaseChatModel 구조
2. `graph.py` — LangGraph 노드 흐름
3. `store.py` + `retriever.py` — RAG 파이프라인

**2. LangSmith 연동 (1시간)**
```bash
# LangSmith 가입 후 API 키 발급
# .env에 추가
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=bunpil

# docker-compose restart
docker-compose down && docker-compose up -d

# 요청 한 번 날리고 LangSmith 대시보드 확인
```

**3. 첫 번째 trace 확인 (30분)**
- LangSmith에서 `/exam` 요청 trace 열기
- 각 노드 실행 시간, 입출력 확인
- "이 숫자가 뭘 의미하는가" 파악

---

## 이번 주 끝날 때 목표 상태

```
✓ EC2 + RunPod 배포 완료
✓ /health, /record, /exam 엔드포인트 실제 동작
✓ LangSmith에 trace 찍히는 것 확인
✓ 핵심 파일 3개 이상 설명 가능한 수준으로 이해
```
