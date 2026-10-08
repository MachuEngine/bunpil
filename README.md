<div align="center">

# 분필 (bunpil)

**고등학교 사회 교사를 위한 AI 문항 출제 워크플로**

![Skills](https://skillicons.dev/icons?i=python,fastapi,typescript,nextjs,tailwind,docker,react,aws)

![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-30363D?logo=langgraph&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-2B2B2B?logo=ollama&logoColor=white)
![RunPod](https://img.shields.io/badge/RunPod-5D29F0)
![vLLM](https://img.shields.io/badge/vLLM-1B76C4?logo=vllm&logoColor=white)
![LangSmith](https://img.shields.io/badge/LangSmith-1B2733)

[아키텍처](#아키텍처) · [핵심 지표](#핵심-지표) · [설계 결정](#설계-결정) · [개발하며 찾은 문제](#개발하며-찾은-문제) · [한계](#한계) · [빠른 시작](#빠른-시작) · [문서](#문서)

</div>

---

예시 문제를 붙여넣으면 유형과 난이도 구성이 비슷한 새 문항 세트를 만드는 **LangGraph 평가-수정 루프 워크플로**입니다. 흐름은 코드가 고정하고 생성 단계만 도구 호출 루프로 두며, 통과 여부는 코드 게이트와 생성 모델과 분리된 Judge가 판단합니다.

> [!NOTE]
> 2026-10에 포트폴리오로 마무리한 프로젝트입니다. 지인 교사 1인이 실제 수업에 썼고, 지금은 서비스를 내린 상태입니다. 배포 구성은 [docs/DEPLOY.md](./docs/DEPLOY.md)의 절차로 다시 띄울 수 있습니다.

<img width="1173" height="562" alt="분필 웹 UI 실행 화면" src="https://github.com/user-attachments/assets/e82129c1-e4f2-4e49-8cf9-cb3a8c7aebcd" />

## 아키텍처

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/architecture-dark.svg">
  <img src="./assets/architecture-light.svg" alt="분필 아키텍처. 위쪽 런타임 경로에서는 교사 입력이 FastAPI를 거쳐 PII 마스킹 후 출제 그래프(LangGraph)로 들어가고, agent는 생성 LLM(Qwen2.5-14B)을, judge 노드는 별도 런타임 Judge(gpt-5.6-luna)를 호출한다. 캡처 이미지는 마스킹 전에 VLM(gpt-6-luna)으로 보내고 추출 텍스트를 마스킹한다. 아래쪽 오프라인 평가에서는 평가 하네스가 같은 그래프 코드로 후보 모델의 문항을 만들고, 오프라인 Judge(claude-sonnet-5.5)가 채점하며, 그 Judge는 사람 라벨로 검증한다.">
</picture>

- **런타임**: 모든 텍스트는 모델 호출 전에 `mask_pii()`로 마스킹합니다. 캡처 이미지만 원본을 마스킹할 수 없어, VLM이 텍스트로 옮긴 뒤 마스킹합니다.
- **생성과 채점 분리**: 문항을 쓰는 모델과 채점하는 Judge가 다릅니다. 통과, 재시도, 문항 개수는 `validate` 노드의 코드가 정합니다.
- **오프라인 평가**: 평가 하네스는 운영과 같은 그래프 코드로 문항을 만들고, 그 채점에 쓰는 Judge는 사람 라벨로 먼저 검증합니다.
- 성취기준 검색(RAG)은 검색을 끄고 비교한 실험에서 품질 기여가 없어 2026-10에 생성 경로에서 뺐습니다.

노드 흐름, 도구 4개, RunPod 백엔드 같은 상세 구조는 [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)에 있습니다.

## 핵심 지표

2026-10에 사람 라벨을 다시 달고 모델을 다시 고르면서 잰 최신 값입니다.

> **읽는 법**: κ(카파)는 LLM 채점과 사람 채점이 얼마나 일치하는지를 0~1로 나타낸 값이고, 이 프로젝트의 기준은 0.4입니다. `[하한, 상한]`은 95% 신뢰구간입니다. 하한까지 기준을 넘으면 "확정", 점추정치만 넘으면 "미확정"으로 적었습니다. CER은 글자 단위 오류율로, 낮을수록 좋습니다.

| 지표 | 결과 | 측정 조건 |
|---|---|---|
| Judge 신뢰도 (가중 κ) | 정답유일성 **0.86** [0.74, 0.94] 확정<br>오답매력도 **0.70** [0.60, 0.77] 확정<br>근거성 0.56 [0.35, 0.73] 미확정 | 실제 생성 문항 91건에 단 사람 라벨, 오프라인 Judge claude-sonnet-5.5 |
| 생성 모델 재선정 | 게이트 통과 **12% → 100%**<br>세트당 시간 7.2분 → 38초 | 합성 지문 33개, 재시도 최대 5회. Qwen2.5-14B → gpt-6-luna |
| 생성 문항 품질 (사람 채점, 5점 만점) | gpt-6-luna 정답유일성 **4.55**, 오답매력도 **3.76**, 근거성 **4.79** | 지문 33개에서 뽑은 객관식 33문항 |
| VLM 재선정 | 그림 서술 실패 **30% → 0%**<br>그림 문항 CER **0.143 → 0.035** | 그림 90건과 텍스트 20건. gpt-4o-mini → gpt-6-luna |
| VLM 그림 서술 Judge 신뢰도 | κ **0.89** [0.82, 0.93] | 사람 라벨 130건(결함을 넣은 서술 40건 포함), gpt-5.6-luna |
| RAG 제거 실험 | 품질 차이 최대 0.07점, 모두 판정 불가 → 생성 경로에서 제거 | 지문 33개 × 검색 켬/끔 × 2회 |
| PII 마스킹 누락률 | **0** | 골든셋 20건, 매 커밋 CI에서 검사 |

- 생성 모델은 gemini-3.8-flash도 게이트를 100% 통과했습니다. 품질 차이는 대부분 판정 불가였고, 정답 오류와 사실 오류가 더 적고(통계적으로 유의하지는 않음) 비용이 6.6배 낮은 gpt-6-luna를 골랐습니다(세트당 $0.0034).
- 재선정 결과는 평가 결론입니다. 운영 전환은 하지 않았고, 코드 기본값은 생성 Qwen2.5-14B, 런타임 Judge gpt-5.6-luna, VLM gpt-6-luna입니다.
- 비교 방법과 전체 수치는 [MODEL_SELECTION.md](./MODEL_SELECTION.md), 시간순 원본 기록은 [EVAL.md](./EVAL.md) 28~30절에 있습니다.

## 설계 결정

1. **LLM은 판단하고 코드가 결정합니다.** 유형 비율, 난이도, 종합 품질처럼 의미를 봐야 하는 항목은 Judge가 점수를 매기고, 통과와 재시도, 문항 개수, 언어, 형식은 코드가 판정합니다.
2. **생성 모델과 Judge를 런타임까지 분리했습니다.** 처음에는 생성 모델이 자기 출력을 채점했고, 오프라인 평가만 외부 Judge를 썼습니다. 지금은 런타임과 오프라인 평가가 같은 채점 코드를 쓰고, Judge 호출이 실패하면 조용히 넘어가지 않고 멈춥니다.
3. **Judge도 사람 라벨로 검증합니다.** 같은 두 후보라도 과제에 따라 순위가 뒤집혔습니다. 문항 품질은 sonnet(κ 0.87 vs luna 0.78)이, 그림 서술은 luna(κ 0.89 vs sonnet 0.63)가 더 정확했습니다.
4. **기여를 확인하지 못한 기능은 뺍니다.** 근거 규정을 찾지 못한 생기부 윤문 모듈과, 꺼 보니 품질 차이가 없던 RAG를 제거했습니다.

## 개발하며 찾은 문제

| 문제 | 원인 | 해결 |
|---|---|---|
| 몇 달간 쌓은 Judge 신뢰도 수치가 실제 배포 경로를 재지 않았음 | 오프라인 평가는 외부 Judge를, 런타임은 생성 모델의 자기 채점을 쓰고 있었음 | 별도 `judge` 노드를 두고 두 경로가 같은 채점 코드를 쓰도록 통일 |
| 모델 비교 사전 점검에서 두 API 모델의 차이가 전혀 보이지 않음 | 당시 Judge가 사람보다 오답매력도를 0.61점 후하게 줘 변별력이 없었음. 옛 κ 0.468도 Claude가 합성한 라벨 기준이었음 | 실제 생성 문항 95개에 블라인드로 사람 라벨을 달고 Judge부터 다시 고름 |
| 생성 모델 비교 33세트 내내 검색 도구가 실패했는데 드러나지 않음 | 로컬 벡터 DB 경로가 배포용이었고, 도구 예외가 메시지로만 모델에 돌아가 로그에 남지 않았음 | 경로 자동 전환, 도구 예외를 로그와 집계에 남김. qwen은 검색 도구 없이 다시 돌려 결론 유지를 확인 |

나머지 사례(중국어 오염, `num_ctx` 초과, OpenAI 400 오류 등)는 [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md#엔지니어링-하이라이트)에 있습니다.

## 한계

- 사람 라벨은 라벨러 1인이 달아, 라벨러 간 일치도를 재지 못했습니다.
- 평가 지문은 합성이고 객관식만 라벨링했으며, 학생 기준 난이도 '상' 문항이 없습니다.
- 근거성 Judge κ는 신뢰구간 하한이 기준 아래라 미확정입니다.
- 요청 분석(문항 수, 형식 판정)과 해설은 생성 모델을 그대로 쓰고, 따로 고르거나 정확도를 재지 않았습니다.
- VLM 평가 그림 중 75건은 합성 초안이라 사람 검수 전이고, 중간 품질 검증에 쓴 결함은 인공 주입입니다.
- RunPod가 2026-07부터 크레딧 소진으로 멈춰, 그 뒤의 변경은 프로덕션 생성 경로에서 검증하지 못했습니다.

## 빠른 시작

로컬 Ollama만으로 돌리는 최소 절차입니다. 환경변수 전체와 배포 절차는 [docs/DEPLOY.md](./docs/DEPLOY.md)를 보세요.

```bash
git clone https://github.com/MachuEngine/bunpil.git && cd bunpil
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

ollama pull qwen2.5:14b   # 생성 모델. JUDGE_BACKEND=local이면 Judge로도 재사용

# 터미널 1: API (포트 8765)
BUNPIL_API_KEY=change_me LLM_BACKEND=local OLLAMA_MODEL=qwen2.5:14b JUDGE_BACKEND=local \
  .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8765

# 터미널 2: 웹 UI (포트 3000)
cd frontend && npm install
BUNPIL_API_KEY=change_me BACKEND_URL=http://localhost:8765 npm run dev
```

브라우저에서 http://localhost:3000 으로 접속합니다. 유닛 테스트는 LLM 호출 없이 몇 초 안에 끝납니다.

```bash
.venv/bin/python -m pytest tests/ -q
```

## 문서

| 문서 | 내용 |
|---|---|
| [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) | 노드와 도구 구조, 설계 원칙, 엔지니어링 사례, 데이터, 디렉토리 구조 |
| [MODEL_SELECTION.md](./MODEL_SELECTION.md) | 생성 모델, Judge, VLM 선정 근거와 전체 비교표 |
| [EVAL_SUMMARY.md](./EVAL_SUMMARY.md) | 평가 체계 요약과 지표 의미 |
| [EVAL.md](./EVAL.md) | 회차별 실험 원본 기록 (시간순) |
| [docs/EVAL_DETAILS.md](./docs/EVAL_DETAILS.md) | 이전 README에 있던 평가 상세와 옛 기록 |
| [docs/DEPLOY.md](./docs/DEPLOY.md) | 배포 절차, 운영비, 환경변수 |
| [DESIGN.md](./DESIGN.md) | 초기 설계 문서 |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | 장애 진단 기록 |
