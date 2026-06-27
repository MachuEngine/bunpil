# 분필 (bunpil)

고등학교 사회 교사용 AI 어시스턴트. 두 가지 기능을 제공합니다.

- **문항 출제** — 지문 PDF 업로드 → 유형·난이도·성취기준 지정 → 1문항 자동 출제
- **생기부 다듬기** — 교사 관찰 메모 → PII 마스킹 → 학교생활기록부 문체 교정 → 규정 위반 플래그

> 포트폴리오 목적 + 지인 사회 교사 1인 실사용.

---

## 아키텍처

```
브라우저
  │
  ▼
Gradio UI (app/ui.py)
  │
  ├─ 출제 모듈 ─── LangGraph ReAct Agent
  │                  │
  │                  │  [도구 — 모두 LLM 없는 순수 계산/검색/저장]
  │                  ├─ search_passages       성취기준 RAG 검색
  │                  ├─ search_regulations    교육과정 법령 RAG 검색
  │                  ├─ get_past_item_examples 기출 스타일 참조
  │                  ├─ validate_item_format  형식 자기교정
  │                  ├─ save_item             에이전트가 직접 생성한 문항 저장
  │                  ├─ record_score          에이전트 자체 품질 평가 기록
  │                  └─ check_duplicate       기출 중복 유사도 검사
  │
  └─ 생기부 모듈 ── Chain (수동 루프)
                     ├─ mask_pii     regex 기반, 모델 호출 전 처리
                     ├─ polish       Few-shot LLM 문체 교정
                     └─ validate     규칙 기반 + RAG 규정 검증

LLM 백엔드
  개발:       Ollama (qwen2.5:1.5b, 로컬)
  프로덕션:   RunPod 서버리스 (Qwen2.5-7B-Instruct, vLLM)

RunPod Tool Calling 흐름
  ChatRunPod → apply_chat_template(tools=...) → vLLM
  → <tool_call> 태그 파싱 → AIMessage(tool_calls=[...]) → LangGraph ReAct 루프
```

### ReAct 에이전트 설계 원칙

에이전트(LLM)가 추론과 문항 생성을 **직접** 담당합니다. 도구는 검색·저장·검증의 **순수 계산**만 수행하며 내부 LLM 호출이 없습니다. 이를 통해 도구 내부에 LLM을 중첩하는 안티패턴을 제거했습니다.

```
에이전트 실행 흐름 (1문항 기준)
search_passages → [선택: get_past_item_examples, search_regulations]
→ validate_item_format (형식 오류 시 자기수정 후 재검증)
→ save_item → record_score → check_duplicate
                                      └─ 호출 즉시 루프 종료 (중복 save_item 방지)
```

### 동시성 설계

- **요청 간 세션 격리**: 출제 요청별 컨텍스트를 `contextvars.ContextVar`로 분리. `asyncio.to_thread` + `contextvars.copy_context()`로 worker 스레드에 전파.
- **이벤트 루프 비블로킹**: `/exam`, `/record` 엔드포인트 모두 `asyncio.to_thread`로 LangGraph·Chain 실행. FastAPI 이벤트 루프 점유 없음.

### 업로드 지문 처리

PDF 텍스트를 에이전트 프롬프트에 직접 삽입합니다. 임시 컬렉션 생성과 BGE-M3 임베딩 단계를 거치지 않으므로 인덱싱 대기 시간이 없습니다.

```
기존: PDF → BGE-M3 임베딩 → ChromaDB 임시 컬렉션 → RAG 검색
현재: PDF → 텍스트 추출(최대 4000자) → 에이전트 프롬프트에 직접 포함
```

---

## 스택

| 구분 | 기술 |
|---|---|
| 백엔드 | FastAPI (GET /health) |
| 에이전트 | LangGraph (ReAct) |
| 생기부 체인 | LangChain (수동 루프) |
| 벡터스토어 | ChromaDB |
| 임베딩 | BGE-M3 (CPU) |
| 리랭킹 | BGE-reranker-base (CPU) |
| LLM 서빙 | Ollama (개발) / RunPod vLLM (프로덕션) |
| UI | Gradio 4.x |
| 배포 | AWS EC2 t3.medium + EBS + RunPod 서버리스 + Caddy HTTPS |

---

## 빠른 시작 (로컬)

### 1. 환경 설정

```bash
git clone https://github.com/MachuEngine/bunpil.git
cd bunpil

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # 필요 시 값 수정
```

### 2. Ollama 모델 설치

```bash
# Ollama 설치: https://ollama.com
ollama pull qwen2.5:1.5b   # 개발용 (빠른 테스트)
# ollama pull qwen2.5:7b   # 품질 향상 시
```

### 3. RAG 데이터 인덱싱

```bash
# data/ 경로에 PDF를 넣은 뒤 아래 순서대로 실행
.venv/bin/python scripts/index_regulations.py   # 생기부 기재요령·훈령
.venv/bin/python scripts/index_past_exams.py    # 수능·모평 기출 (사탐)
.venv/bin/python scripts/index_standards.py     # 사회과 교육과정 성취기준
```

> 이미 적재된 파일은 자동 스킵 (idempotent). 처음 한 번만 실행하면 됩니다.

### 4. UI 실행

```bash
ollama serve &                    # 로컬 LLM 서버 (별도 터미널)
.venv/bin/python app/ui.py
# → http://localhost:7860
```

---

## 데이터

| 컬렉션 | 경로 | 출처 | 용도 |
|---|---|---|---|
| `regulations` | `data/regulations/` | 학교생활기록부 종합지원포털 | 생기부 규정 위반 검증 + 출제 시 교육과정 법령 참조 |
| `past_exams` | `data/past_exams/` | 한국교육과정평가원 | 출제 시 기출 중복 체크 + 스타일 참조 |
| `standards` | `data/standards/` | 국가교육과정정보센터(NCIC) | 출제 시 성취기준 검색 |

> **저작권**: 수능·모평 기출은 참고용 인덱싱만 허용 — 재배포·노출 금지.

---

## 디렉토리 구조

```
bunpil/
├── app/
│   ├── common/
│   │   ├── llm/          # LLM 추상화 (OllamaBackend / RunPodBackend / ChatRunPod)
│   │   └── rag/          # PDF 파싱, 임베딩, 리랭킹, ChromaDB
│   ├── modules/
│   │   ├── exam/         # 출제 모듈 (LangGraph ReAct Agent, 7개 도구)
│   │   └── record/       # 생기부 모듈 (수동 루프 Chain)
│   ├── main.py           # FastAPI (GET /health)
│   └── ui.py             # Gradio UI
├── data/
│   ├── regulations/      # 생기부 기재요령, 작성·관리지침
│   ├── past_exams/       # 수능·모평 기출 PDF (사탐 과목)
│   └── standards/        # 사회과 교육과정 PDF
├── scripts/
│   ├── index_regulations.py  # regulations 컬렉션 인덱싱
│   ├── index_past_exams.py   # past_exams 컬렉션 인덱싱
│   ├── index_standards.py    # standards 컬렉션 인덱싱
│   ├── test_llm.py           # LLM 추상화 레이어 검증
│   ├── test_rag.py           # RAG 파이프라인 검증
│   ├── test_exam.py          # 출제 모듈 통합 테스트
│   ├── test_record.py        # 생기부 모듈 통합 테스트
│   ├── eval_exam.py          # 출제 평가 (Recall@5, MRR, LLM Judge)
│   └── eval_record.py        # 생기부 평가 (마스킹 FN, 사실추가율, 위반 Recall)
├── runpod_handler/       # RunPod 서버리스 핸들러 (Qwen2.5-7B vLLM)
├── deploy/               # EC2·Caddy·빌링알람 프로비저닝 스크립트
├── Dockerfile
├── docker-compose.yml
└── Caddyfile
```

---

## 검증

### 검증 구조

| 레이어 | 스크립트 | 목적 | 실행 시점 |
|---|---|---|---|
| 기능 검증 | `test_*.py` | 파이프라인이 에러 없이 동작하는가 | 개발 중 수시 |
| 품질 평가 | `eval_*.py` | 얼마나 잘 하는가 (수치 지표) | 모델 교체 시 |

### 현재 검증 환경

- **LLM**: `qwen2.5:1.5b` (Ollama 로컬) — 로직 검증 전용
- **품질 평가**: RunPod `Qwen2.5-7B` 연결 후 수행 예정

### 기능 검증 결과 (qwen2.5:1.5b)

| 테스트 | 항목 | 결과 |
|---|---|---|
| `test_rag.py` | PDF 파싱·청킹·임베딩·ChromaDB 저장/검색 | ✅ |
| `test_rag.py` | 검색 + BGE-reranker 재정렬 | ✅ |
| `test_llm.py` | Ollama 응답 수신 | ✅ |
| `test_llm.py` | local → RunPod 백엔드 전환 | ✅ |
| `test_exam.py` | 지문 업로드 → 에이전트 문항 생성 → 저장 → 중복 검증 흐름 | ✅ |
| `test_record.py` | PII 마스킹 4케이스 (전화번호·주민번호·학교명·이메일) | ✅ |
| `test_record.py` | 관찰 메모 → 생기부 문체 교정 | ✅ |
| `test_record.py` | 교사 책임 고지 출력 | ✅ |

> 1.5b 모델로 생성된 문항 품질(문장·정확도)은 낮을 수 있음. 파이프라인 로직 검증 목적.

### 품질 평가 지표 (RunPod 7B 연결 후 수행)

**출제 모듈**

```bash
.venv/bin/python scripts/eval_exam.py
```

| 지표 | 기준 |
|---|---|
| Recall@5 | ≥ 0.80 |
| MRR | 참고값 |
| 유형·난이도·성취기준 제약 | 통과 |
| LLM Judge 종합평균 | ≥ 4.0 / 5 |

**생기부 모듈**

```bash
.venv/bin/python scripts/eval_record.py
```

| 지표 | 기준 | 결과 |
|---|---|---|
| PII 마스킹 FN율 | = 0 | 0.000 ✓ |
| 키워드 사실추가율 | = 0 | 0.000 ✓ |
| 규정 위반 Recall | ≥ 0.95 | 1.000 ✓ |

> 생기부 평가는 규칙 기반(결정론적) 지표 비중이 높아 소형 모델에서도 유효.

### 프로덕션 검증 결과 (RunPod Qwen2.5-7B, RTX A5000)

| 항목 | 결과 |
|---|---|
| 에이전트 tool calling (ChatRunPod → vLLM) | ✅ |
| 1문항 출제 (save_item → record_score → check_duplicate) | ✅ |
| validate_item_format 자기교정 루프 | ✅ |
| RAG 인덱싱 (3개 컬렉션) | ✅ regulations 510 / past_exams 124 / standards 573 청크 |
| EBS 영구 저장 | ✅ 컨테이너 재시작 후 재인덱싱 불필요 |
| 업로드 PDF 인덱싱 제거 | ✅ 텍스트 직접 삽입으로 대기 시간 제거 |
| 추론 속도 (1문항) | ~2–3분 (RTX A5000, min workers=1) |

---

## 보안 원칙

- 실제 학생 데이터 미사용 — 전부 합성/익명
- PII 마스킹은 모델 호출 **이전**에 수행
- 사용자 입력(메모·업로드 지문) **비저장** (업로드 PDF는 메모리에서만 처리 후 폐기)
- 로그·캐시에 **PII 기록 금지**
- 생기부: 메모에 없는 사실 **추가 금지**. 출력에 교사 책임 고지 표시

---

## 배포 (프로덕션)

```
브라우저 → Caddy (HTTPS) → EC2 t3.medium (Gradio + ChromaDB) → RunPod 서버리스 (Qwen2.5-7B)
                                    │
                              EBS 10GB (ChromaDB 영구 저장)
```

### RunPod 서버리스 설정

```bash
# 1. 핸들러 이미지 빌드 & 푸시
cd runpod_handler
docker build -t <your-dockerhub>/bunpil-runpod:latest .
docker push <your-dockerhub>/bunpil-runpod:latest

# 2. RunPod 콘솔 → Serverless → New Endpoint → 이미지 URL 입력
# 3. 워커 설정: min workers=1 (콜드스타트 방지), max workers=4 (병렬 출제 시)
# 4. 발급된 Endpoint ID를 .env에 입력
# LLM_BACKEND=runpod
# RUNPOD_API_KEY=...
# RUNPOD_ENDPOINT_ID=...
```

### EC2 배포 (Docker Hub 이미지 사용)

```bash
# EC2 (Ubuntu 22.04 t3.medium) 내부에서
docker pull jongmin0826/bunpil-app:latest

# EBS 볼륨 마운트 (처음 한 번)
sudo mkfs.ext4 /dev/nvme1n1
sudo mkdir -p /data/chroma_db
echo '/dev/nvme1n1 /data/chroma_db ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
sudo mount -a

# 컨테이너 실행
docker run -d --name bunpil \
  -p 7860:7860 \
  --env-file /home/ubuntu/.env \
  -v /data/chroma_db:/data/chroma_db \
  jongmin0826/bunpil-app:latest

# RAG 인덱싱 (처음 한 번 — EBS에 영구 저장됨)
docker exec bunpil python scripts/index_regulations.py
docker exec bunpil python scripts/index_past_exams.py
docker exec bunpil python scripts/index_standards.py
```

### 빌링 알람

```bash
bash deploy/billing_alarm.sh   # 월 $10 초과 시 이메일 알람
```

---

## 환경변수

`.env.example` 참고. 시크릿은 `.env`에만 보관 — 커밋 금지.

| 변수 | 설명 | 기본값 |
|---|---|---|
| `LLM_BACKEND` | `local` 또는 `runpod` | `local` |
| `OLLAMA_MODEL` | 로컬 개발 모델명 | `qwen2.5:1.5b` |
| `RUNPOD_API_KEY` | RunPod API 키 | — |
| `RUNPOD_ENDPOINT_ID` | RunPod 엔드포인트 ID | — |
| `CHROMA_PERSIST_DIR` | ChromaDB 저장 경로 | `/data/chroma_db` (EC2) / `./chroma_db` (로컬) |

---

## 월 운영비 (1인 기준)

| 항목 | 비용 |
|---|---|
| EC2 t3.medium | ~$30 |
| RunPod 서버리스 (추론만 과금, min workers=1) | ~$5–15 |
| EBS 10GB | ~$1 |
| **합계** | **~$36–46** |

데모/개발 중에는 EC2를 필요할 때만 켜서 절감 가능. min workers=0으로 설정 시 RunPod 비용 대폭 절감 (단, 콜드스타트 30–60초 발생).
