# 분필 (bunpil)

고등학교 사회 교사용 AI 어시스턴트. 두 가지 기능을 제공합니다.

- **출제 도우미** — 지문 업로드 → 유형·난이도·성취기준 제약을 만족하는 문항 세트 자동 생성
- **생기부 윤문** — 교사 관찰 메모 → PII 마스킹 → 생기부 문체 윤문 → 규정 위반 플래그

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
  │                  ├─ search_passages   (ChromaDB + BGE-M3 + BGE-reranker)
  │                  ├─ generate_item     (Few-shot LLM)
  │                  ├─ judge_item        (LLM as Judge)
  │                  └─ check_duplicate   (기출 유사도)
  │
  └─ 생기부 모듈 ── LCEL Chain
                     ├─ mask_pii          (regex, 모델 호출 전)
                     ├─ polish            (Few-shot LLM)
                     └─ validate          (규칙 기반 + RAG + LLM)

LLM 백엔드
  개발: Ollama (qwen2.5:1.5b, 로컬)
  프로덕션: RunPod 서버리스 (Qwen2.5-7B, vLLM)
```

---

## 스택

| 구분 | 기술 |
|---|---|
| 백엔드 | FastAPI (비동기) |
| 에이전트 | LangGraph (ReAct) / LangChain LCEL |
| 벡터스토어 | ChromaDB |
| 임베딩 | BGE-M3 (CPU) |
| 리랭킹 | BGE-reranker-base (CPU) |
| LLM 서빙 | Ollama (개발) / RunPod vLLM (프로덕션) |
| UI | Gradio 4.x |
| 배포 | AWS EC2 + RunPod 서버리스 + Caddy HTTPS |

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

### 3. UI 실행

```bash
python app/ui.py
# → http://localhost:7860
```

---

## 디렉토리 구조

```
bunpil/
├── app/
│   ├── common/
│   │   ├── llm/          # LLM 추상화 (OllamaBackend / RunPodBackend)
│   │   └── rag/          # PDF 파싱, 임베딩, 리랭킹, ChromaDB
│   ├── modules/
│   │   ├── exam/         # 출제 모듈 (LangGraph ReAct Agent)
│   │   └── record/       # 생기부 모듈 (LCEL Chain)
│   ├── main.py           # FastAPI (GET /health)
│   └── ui.py             # Gradio UI
├── scripts/
│   ├── test_exam.py      # 출제 모듈 통합 테스트
│   ├── test_record.py    # 생기부 모듈 통합 테스트
│   ├── eval_exam.py      # 출제 평가 (Recall@5, MRR, LLM Judge)
│   └── eval_record.py    # 생기부 평가 (마스킹 FN, 사실추가율, 위반 Recall)
├── runpod_handler/       # RunPod 서버리스 핸들러 (Qwen2.5-7B vLLM)
├── deploy/               # EC2·Caddy·빌링알람 프로비저닝 스크립트
├── Dockerfile
├── docker-compose.yml
└── Caddyfile
```

---

## 평가 지표

### 출제 모듈

```bash
python scripts/eval_exam.py
```

| 지표 | 기준 | 1.5b 결과 |
|---|---|---|
| Recall@5 | ≥ 0.80 | 1.000 ✓ |
| MRR | 참고값 | 0.850 |
| 세트 제약 (유형·난이도·커버리지) | 전체 통과 | ✓ |
| LLM Judge 종합평균 | ≥ 4.0 / 5 | 3.21 (7B에서 재평가) |

### 생기부 모듈

```bash
python scripts/eval_record.py
```

| 지표 | 기준 | 결과 |
|---|---|---|
| PII 마스킹 FN율 | = 0 | 0.000 ✓ |
| 키워드 사실추가율 | = 0 | 0.000 ✓ |
| 규정 위반 Recall | ≥ 0.95 | 1.000 ✓ |

---

## 보안 원칙

- 실제 학생 데이터 미사용 — 전부 합성/익명
- PII 마스킹은 모델 호출 **이전**에 수행
- 사용자 입력(메모·업로드 지문) **비저장**
- 로그·캐시에 **PII 기록 금지**
- 생기부: 메모에 없는 사실 **추가 금지**. 출력에 교사 책임 고지 표시

---

## 배포 (프로덕션)

```
브라우저 → Caddy (HTTPS) → EC2 t3.small (Gradio + ChromaDB) → RunPod 서버리스 (Qwen2.5-7B)
```

### RunPod 서버리스 설정

```bash
# 1. 핸들러 이미지 빌드 & 푸시
cd runpod_handler
docker build -t <your-dockerhub>/bunpil-runpod:latest .
docker push <your-dockerhub>/bunpil-runpod:latest

# 2. RunPod 콘솔 → Serverless → New Endpoint → 이미지 URL 입력
# 3. 발급된 Endpoint ID를 .env에 입력
# LLM_BACKEND=runpod
# RUNPOD_API_KEY=...
# RUNPOD_ENDPOINT_ID=...
```

### EC2 배포

```bash
# AWS CLI 설정 후
bash deploy/ec2_setup.sh

# EC2 내부에서
git clone https://github.com/MachuEngine/bunpil.git && cd bunpil
cp .env.example .env && nano .env   # RunPod 키 입력
docker compose up -d --build

# HTTPS (도메인 DNS가 서버 IP를 가리킨 후)
bash deploy/caddy_setup.sh
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
| `CHROMA_PERSIST_DIR` | ChromaDB 저장 경로 | `./chroma_db` |

---

## 월 운영비 (1인 기준)

| 항목 | 비용 |
|---|---|
| EC2 t3.small | ~$15 |
| RunPod 서버리스 (추론만 과금) | ~$1–5 |
| EBS 스토리지 | ~$1 |
| **합계** | **~$17–21** |

데모/개발 중에는 EC2를 필요할 때만 켜서 절감 가능.
