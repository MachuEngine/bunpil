# 분필(Bunpil) 개발 로드맵

## 진행 상태 요약

> **2026-08-03 범위 축소 — 생기부 모듈 제거**: 검증 규칙의 규정 근거를 추적한 결과 인덱싱된
> 교육부 기재요령에 종교·외모·추측 조항이 한 건도 없었고(EVAL.md 14절), 키워드가 사회 교과
> 문장을 오탐하는 반면 규정이 실제 금지하는 문장은 놓쳤다. 하드룰 1 때문에 실제 데이터로
> 검증할 수도 없어 **출제 단일 모듈로 범위를 좁혔다.** 아래 생기부 관련 항목·열린 이슈는
> 이력으로 보존하되 **더 이상 진행 대상이 아니다.**

- ✅ 완료: 배포(현재 RunPod는 크레딧 소진으로 일시 비활성 — 아래 "남은 작업" 참고), LangSmith 트레이싱, 골든셋 구축, eval 스크립트 실데이터 전환, 출제 모듈 passage_text 리디자인, GitHub Actions 경량 CI, 생성 모델 7B→14B 승격, Judge 모델 gpt-5.6-luna 채택, 출제 성취기준 사용자 입력 제거(2026-07-21), README/DESIGN/MODEL_SELECTION 문서 갱신(2026-07-22), **런타임 self-judge 폐기 → 별도 judge 노드로 생성·Judge 모델 완전 분리(2026-07-23, 아래 상세)**, Agent Trajectory Eval 신규(2026-08-03) + 재측정 완료(2026-08-04 — "한도 소진"은 오진, 실제로는 배선 버그 3건이었음. EVAL.md 11.1절), 하이브리드 검색 도입 + 리랭커 조사(2026-08-03, BM25+dense RRF & `n_candidates` 20→10 — 전체 Recall@5 **1.000** 첫 달성, regulations MRR 0.667→0.753)
- ✅ 완료(2026-08-04 추가): **validate 게이트 임계값 재보정** — "에이전트가 정상적으로 마치는 경우가 거의 없다"는 관측을 추적한 결과 `submit_for_review`는 정상이었고, `overall_score >= 4`·`type_ratio >= 0.7` 게이트의 실측 통과율이 6.7~8.9%로 사실상 도달 불가였음을 확인(2026-07-06 도입 후 한 번도 재검토 안 됨). `>=3`·`>=0.5`로 재보정해 42.2~48.9%로 정상화. 근거: `experiments/measure_validate_gate.py`, EVAL.md 15절
- ✅ 완료(2026-08-07 추가): **스모크 테스트 조건 수정 → 게이트 통과율 첫 런타임 측정** — `test_exam.py`가 예시(객관식1+서술형1)와 `num_items=1`의 조합 때문에 Judge 루브릭상 `overall=2`가 확정이라 **어떤 임계값으로도 통과 불가**한 조건이었다(회귀 감지 도구로 기능 못 함). 예시를 유형 균일(객관식2)로 바꿔 개수 디커플링은 유지한 채 게이트를 도달 가능하게 하고, 실패 진단을 배선/개수/게이트 3단계로 분리. 재측정 결과 개수 달성 6/6, 게이트 통과 **3/6**(첫 관측) — 같은 측정치에 옛 기준 적용 시 1/6이라 재보정 효과가 런타임에서도 확인됨. EVAL.md 18·19절
- ✅ 완료(2026-08-07 추가): **프로덕션 조건 통과율 확인 — 개선 작업 불필요 판정** — 통과율을 올리려 검토한 두 안(save_item 어휘 임계값 조정 / 임베딩 유사도 게이트)이 실측으로 **모두 반증**됐고(containment 최대 0.600으로 임계값 조정은 잡을 대상이 없음, 코사인은 통과·실패 구간이 겹침 — EVAL.md 20절), 이어서 프로덕션 조건(`budget=5`)을 재보니 **코드 변경 없이 통과율 0.833**(스모크 조건 0.500). 통과 5건 중 3건이 재시도로 회복됐고 전부 `overall=3`이라 **재보정이 없었다면 0.000**이었다. 낮은 스모크 통과율은 빡빡한 테스트 조건의 산물이지 프로덕션 품질 문제가 아니었음. EVAL.md 21절
- 🔄 진행 중: 코드 리뷰(아래 "참고 — 코드 리뷰 대상 파일" 표는 예전 스냅샷 — 신뢰 금지, 진행 상황은 [CODE_REVIEW_CHECKLIST.md](./CODE_REVIEW_CHECKLIST.md)로 추적)
- ⬜ 남은 작업: 성능 개선 미달 지표 3건 + 포트폴리오 정리 (아래 "남은 작업" 참고)

---

## ✅ 완료

### 배포
- EC2 + RunPod 서버리스 + Caddy HTTPS

### 모니터링 — LangSmith
- LangSmith 연동 + `@traceable` 데코레이터 적용 (`eval_exam.py`, `eval_record.py`)
- **dev/prod 프로젝트 자동 분리**(2026-07-12, 2026-07-13 기준 정정): `app/common/llm/tracing.py`의
  `init_langsmith_project()`가 `LANGCHAIN_PROJECT` 기본값('bunpil') 유지 시 `LLM_BACKEND`
  기준으로 자동 분기 — `local`(Ollama, 순수 로컬 개발)만 `-dev`, 나머지(`runpod`·`openai` 등
  실제 서빙 가능한 백엔드)는 전부 `-prod`. 최초 구현은 "runpod만 prod"였는데, `openai`가
  이미 `factory.py`에 정식 백엔드로 등록돼 있고(GPT-4o-mini가 모델 비교 실험에서 유력
  후보로 나옴, EVAL.md 7절) 향후 실제로 이 백엔드로 서빙할 가능성이 있어 `_PROD_BACKENDS`
  상수로 판단 기준을 넓힘. `LANGCHAIN_PROJECT`에 'bunpil'이 아닌 값을 직접 설정하면
  접미사 없이 그대로 사용(override)

### Eval 체계 — 골든셋 구축
| 골든셋 | 규모 | 비고 |
|---|---|---|
| retrieval_golden | 28개 | 실데이터 기반, 사람 검수 완료 |
| MASKING_GOLDEN | 20개 | 합성. `data/golden/masking_golden.json`(2026-07-09 하드코딩→외부화) |
| VIOLATION_GOLDEN | 50개 | 위반 25 + 정상 25. `data/golden/violation_golden.json`(2026-07-09 하드코딩→외부화) |
| HALLUCINATION_GOLDEN | 20개 | 합성. `data/golden/hallucination_golden.json`(2026-07-09 하드코딩→외부화) |
| ITEM_GOLDEN | 30개 | human_score 1~5점 분포. `data/golden/item_golden.json`(2026-07-09 하드코딩→외부화) |

> **2026-07-09 골든셋 전면 외부화**: 그동안 `eval_exam.py`/`eval_record.py`에 파이썬 리스트로 하드코딩돼 있던 ITEM_GOLDEN/MASKING_GOLDEN/HALLUCINATION_GOLDEN/VIOLATION_GOLDEN을 `retrieval_golden_final.json` 방식과 동일하게 전부 `data/golden/*.json`으로 분리(각 파일에 `_schema` 포함). 스크립트는 로더 함수(`_load_item_golden()`, `_load_golden()`)로 읽음.

### Eval 체계 — 스크립트 업그레이드
- `eval_exam.py`, `eval_record.py` 실데이터 기반으로 업그레이드
- Judge/생성 모델 분리 (`OLLAMA_MODEL` / `OLLAMA_JUDGE_MODEL`)
- 7B 전환 및 첫 eval 실행 완료 (결과는 [EVAL.md](./EVAL.md) 참고)

### 코드 리뷰
- `chat_runpod.py`

### 출제 모듈 입력 방식 리디자인 (2026.07, docs/history/FEEDBACK_DRIVEN_REDESIGN_v2.md)
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

### 프론트엔드 프로덕션 배포 (2026-07-12)
- `frontend/Dockerfile`(멀티스테이지, `next.config.ts`에 `output: "standalone"` 추가) + `docker-compose.yml`에 `frontend` 서비스 추가(3000 포트, `BACKEND_URL=http://app:8765`로 컨테이너 간 통신) + `Caddyfile`을 8765(FastAPI 직접)에서 3000(Next.js)으로 전환 — `/api/*` 프록시는 `frontend/app/api/*/route.ts`가 컨테이너 내부에서 이미 담당하므로 Caddy는 frontend 하나만 보면 됨
- 로컬 Docker로 검증: 이미지 빌드 성공, standalone 서버 기동 후 `/` 200 응답, 백엔드 미기동 상태에서 `/api/record` 호출 시 503 JSON 정상 반환(프록시 배선 확인) — `docker compose config`로 compose 파일 문법 검증 완료
- **한계**: 실제 EC2/AWS 접근 권한(SSH 키·AWS CLI 자격증명)이 이 작업 환경에 없어 라이브 프로덕션 배포·접속 URL 확인은 수행하지 못함. 사용자가 EC2에서 `git pull && docker compose up -d --build` 실행 필요 (frontend 이미지를 Docker Hub에 별도로 올리는 방식을 쓴다면 README '배포 (프로덕션)' 절 참고)
- README '배포 (프로덕션)' 다이어그램·EC2 배포 절(`docker run` 수동 배포 시 `docker network create` 필요) 갱신
- **추가 검증(2026-07-12, 사용자 요청)**: 최초 검증은 컨테이너 포트(3000)에 직접 접근한 것이라 Caddy를 경유하는 전체 경로(브라우저 → Caddy → 3000 → 컨테이너 → 8765)는 검증되지 않았다는 지적을 받아, 로컬에 Caddy를 설치(`brew install caddy`)해 **커밋된 Caddyfile을 도메인/로그 경로 두 줄만 로컬용으로 바꿔 그대로 실행**하고 `docker compose up -d --build`로 실제 app+frontend 컨테이너를 띄워 전체 경로를 재현. 이 과정에서 **이번 작업과 무관한 기존 버그 2건**을 발견해 함께 수정:
  - `requirements.txt`에 `python-multipart`가 빠져 있어 `/exam`,`/exam/stream`의 `Form(...)` 라우트 등록 시점에 FastAPI가 즉시 `RuntimeError`를 던지고 앱이 기동조차 안 됨(로컬 `.venv`에서도 동일하게 재현 — Docker만의 문제가 아니라 기존부터 있던 의존성 누락). `requirements.txt`에 `python-multipart==0.0.20` 추가로 해결
  - `frontend/Dockerfile`의 `HEALTHCHECK`가 `http://localhost:3000`을 썼는데, Alpine 리졸버가 `localhost`를 IPv6(`::1`)로 먼저 시도하면서 IPv4에만 바인딩된 Next.js에 연결 실패 → 서비스는 정상인데 컨테이너가 계속 unhealthy로 보임. `127.0.0.1`로 명시해 해결
  - 수정 후 Caddy(HTTPS, 로컬 CA) → frontend(3000) → app(`http://app:8765`, 컨테이너명 DNS) 경로로 실제 `POST /api/record` 요청을 보내 app 컨테이너에서 RAG 임베더가 그 요청 때문에 로드되기 시작하는 것(직전엔 없던 onnxruntime/telemetry 로그가 요청 직후에만 등장)을 확인해 **전체 경로가 실제로 연결됨을 검증**. 로컬 Ollama가 컨테이너 내부에서 `localhost:11434`로 접근 불가해 실제 생성 완료까지는 확인 못 했지만(이 세션의 배포 검증 범위 밖 — 프로덕션은 RunPod 백엔드라 무관), 라우팅 자체는 확인 완료

### 모델 비교 실험 (2026-07-12, Qwen3.5-9B는 2026-07-13 추가)
- Qwen2.5-7B/14B, Llama3.1-8B, GPT-4o-mini 4개를 동등 조건(동일 passage_text 15개, 고정
  Judge=qwen2.5:7b, budget=1)으로 비교. OpenAI는 `langchain_openai.ChatOpenAI`를 얇게 감싼
  `app/common/llm/backends/chat_openai.py`로 연동(RunPod처럼 httpx 직접 구현 대신 — 이미
  검증된 공식 통합 재사용, 사용자 확인 후 결정)
- 결과: 실패율 Qwen2.5-7B 40%·Qwen2.5-14B 0%·Llama3.1-8B 60%·GPT-4o-mini 0%, 평균 생성시간
  50.3s/158.1s/59.4s/**18.1s**, 개수 충족률 0.15/0.76/0.11/**0.97** — **GPT-4o-mini가 안정성
  지표를 전부 석권**(EVAL.md 7절 상세 비교표·raw 데이터 참고). Llama3.1-8B는 이 ReAct
  tool-calling 워크플로우와 궁합이 안 좋아 채택 후보에서 제외 권장. 오답매력도는 4개 모델
  전부 2.3~2.9로 목표(4.0) 미달 — 모델 문제가 아니라 프롬프트 설계 한계로 보임
- **2026-07-13 Qwen3.5-9B 추가 측정**(사용자 요청): 필자 지식(2026-01 기준)에 없던 최근
  출시 모델이라 실행 전 Ollama 라이브러리 페이지로 실존·태그를 먼저 확인 후 동일 조건으로
  비교. 결과: 실패율 26.7%(4/15), 평균 생성시간 **171.1s(5개 중 최저 속도)**, 개수 충족률
  0.55, 정답유일성 4.81(5개 중 최고). 품질은 우수하지만 속도가 가장 느리고 실패율도
  GPT-4o-mini·Qwen2.5-14B(둘 다 0%)에 못 미쳐 로컬 대안으로도 뚜렷한 우위 없음 — 채택
  근거 약함, 현재로선 우선순위 아님
- **최종 결정**: budget=1 조건은 재시도 없는 하한선일 뿐이라 실제 프로덕션 조건(`budget=5`)으로
  재검증한 결과, 7B와 14B의 속도 격차가 사실상 사라지고(258.8s vs 260.2s) 14B가 실패율·개수
  충족률에서 뚜렷이 우수해 **Qwen2.5-14B로 승격**(2026-07-15, 커밋 `8301b6e`, 이후 `22a2628`로
  실제 전환). GPT-4o-mini는 budget=5에서 실패율 상승·과다생성이 드러나 보류. 상세는 EVAL.md
  7.1절, 채택 근거 요약은 [MODEL_SELECTION.md](./MODEL_SELECTION.md) 참고.

### Judge 모델 비교 실험 (2026-07-17)
- 배경: `get_judge_backend()`(`app/common/llm/factory.py`)가 `LLM_BACKEND`와 무관하게 항상
  Ollama(qwen2.5:7b)를 반환하도록 고정돼 있었음 — 로컬 오픈모델의 judge 신뢰도(구조Judge
  이진 kappa 미달 등)가 몇 달째 한계로 지적돼 옴. "생성 모델"이 아니라 **"judge 모델"만**
  바꿔서 같은 사람 라벨(ITEM_GOLDEN 30개 human_score, STRUCTURE_GOLDEN 45개 human_label)
  대비 신뢰도를 비교하는 `scripts/compare_judge_models.py` 신규 작성(`compare_models.py`와
  반대 축 — 생성 없이 재채점만). `JUDGE_BACKEND=openai` 분기 추가(기본값 `local`이라 미설정
  시 기존과 동일하게 동작), `.env`/`.env.example`에 `JUDGE_BACKEND`/`OPENAI_JUDGE_MODEL` 추가.
- **결과** (`qwen2.5-7b` vs `gpt-5.6-luna`, 생성은 동일·재채점만 다름, raw는
  `data/golden/_judge_comparison_results.json`):
  - 문항 품질(JUDGE_TPL, n=30, human_avg 3.33 동일): kappa **0.328→0.595**(거의 2배),
    exact 0.267→0.300, llm_avg 3.83→**4.27**(편향 +0.50→**+0.94**, GPT가 오히려 더 후하게
    채점), ±1일치율은 qwen이 더 높음(0.800→0.733) — kappa는 "합격선(≥3) 통과 여부"의
    이진 판단 일치도라 GPT가 더 정확하지만, 절대 점수 자체는 더 관대함(똑똑한 모델=덜
    후한 채점이 아니었음)
  - 구조 유사도(STRUCTURE_JUDGE_TPL, n=45): difficulty_match 0.867→0.933,
    **overall_score MAE 1.689→0.600** — GPT용으로 프롬프트를 전혀 손대지 않고 qwen 버릇
    교정용 few-shot(3점 앵커 등)을 그대로 재사용했는데도, 몇 달간 튜닝한 qwen2.5-**14B**
    최고 기록(MAE 1.185, 2026-07-12, 위 STRUCTURE_GOLDEN 재보정 절 참고)보다도 뚜렷이 낮음
- **해석**: judge 신뢰도가 로컬 오픈모델(7B~14B)의 근본적 한계였다는 가설을 뒷받침하는
  결과. 다만 n=30/45로 표본이 여전히 작고, GPT의 절대 점수 편향이 qwen보다 큰 상충 신호가
  있어 "완전히 신뢰 가능한 judge"보다는 "qwen보다 확실히 나은 judge" 정도로 해석 권장.
- **최종 결정(2026-07-21)**: 비용(한 사이클 $0.10 수준)은 걸림돌이 아니었고, kappa/MAE
  히스토리 단절이라는 트레이드오프를 감수하고 **gpt-5.6-luna를 Judge 기본값으로 채택**
  (`JUDGE_BACKEND=openai` 기본화). 생성은 비용·데이터 로컬 처리 우선으로 로컬 Qwen2.5-14B
  유지 — 두 역할의 우선순위가 달라 서로 다른 백엔드로 분리됐다. 근거 요약은
  [MODEL_SELECTION.md](./MODEL_SELECTION.md) 참고.

### 런타임 self-judge 폐기 → 생성·Judge 모델 완전 분리 (2026-07-23)

**배경**: 위 Judge 모델 비교(2026-07-17)·채택(2026-07-21) 당시, `JUDGE_BACKEND`는
**오프라인 eval 스크립트에만** 영향을 줬다. 실제 런타임(출제 그래프)에서 구조 유사도
판단은 생성 에이전트 자신이 `similarity_judge` 도구로 스스로 채점했다(self-judge) —
이 self-judge의 신뢰도는 사람 라벨과 한 번도 대조된 적이 없었다. 이걸 측정해보려고
`self_judge_result`/`self_judge_passed` 필드를 STRUCTURE_GOLDEN에 캡처하는 계측을
잠시 도입(2026-07-22)하고 45개 재생성을 시도했으나, 로컬 하드웨어(M5 24GB) 기준
항목당 소요 시간 편차가 너무 커서(2분~30분+) 전량 재생성이 비현실적이었다.

이 과정에서 **더 근본적인 문제**를 발견: "검증에 쓰는 Judge"(오프라인
`get_judge_backend()`)와 "실제 배포된 judge"(런타임 self-judge)가 애초에 서로 다른
코드 경로였다(검증-배포 불일치). self-judge 신뢰도를 측정하는 것보다, 애초에 생성
모델과 Judge 모델을 런타임에서도 분리해 **같은 judge를 검증·배포 양쪽에서 쓰는 것**이
더 근본적인 해결책이라고 판단해 self-judge 신뢰도 측정 작업은 중단하고(STRUCTURE_GOLDEN은
세션 시작 전 상태로 원복) 이 아키텍처 변경(옵션 B)으로 전환했다.

**적용**:
- `app/modules/exam/judge.py` 신규 — `STRUCTURE_JUDGE_TPL`·`judge_structure()`를
  `scripts/eval_lib.py`에서 이곳으로 이동, 런타임·오프라인 eval이 이 함수를 공유
- `tools.py`: `similarity_judge` 도구 제거, 무인자 종료 신호 `submit_for_review` 추가
- `graph.py`: `judge` 노드 신규(`plan→agent→judge→validate`), `agent_node`에서 자기채점
  로직 전부 제거
- `JUDGE_BACKEND`가 이제 **프로덕션 앱 실행에도 적용**됨(이전엔 eval 전용) — 기본값
  `openai`에서 키 없음/호출 실패 시 fail-fast(조용한 로컬 폴백 없음, 사용자 결정)
- `eval_lib.py`/`eval_exam.py`/`gen_structure_golden.py`의 self-judge 계측 코드는 전부
  되돌림(더 이상 필요 없음 — 오프라인 `eval_structure_judge()` 결과가 곧 런타임 judge
  신뢰도이므로)

**트레이드오프**: 매 문항 세트 생성마다 `passage_text`(PII 마스킹됨, 저작권은 별개)가
OpenAI로 전송됨 — 사용자 확인 후 수용. 로컬 전용 처리가 필요하면 `JUDGE_BACKEND=local`.

상세 설계·코드 대조는 근거는 [MODEL_SELECTION.md](./MODEL_SELECTION.md) 2절,
README "모델 선정" 절 참고.

### 하이브리드 검색 도입 (BM25 + dense, 2026-08-03)

**배경**: 검색이 도입 이래 dense 단독이었고, 그 결정 자체에 문서화된 근거가 없었음
(MODEL_SELECTION.md 5절). 생기부 규정 위반 탐지의 "RAG가 관련 규정을 상위로 못 올림"
약점을 키워드 규칙으로 우회해온 상태였음(위 열린 이슈 2번).

**적용**: `app/common/rag/lexical.py` 신규(BM25 역색인) + `retriever.py`에 RRF 융합.
**새 모델·재인덱싱·스키마 변경·새 의존성 전부 없음** — BGE-M3에 딸린 토크나이저를
빌려 한국어 조사를 분리하고(형태소 분석기 불필요), ChromaDB에 이미 있는 청크 텍스트로
메모리에 인덱스를 세운다. `RAG_HYBRID=false`로 롤백 가능.

**결과**: 전체 MRR 0.789→**0.814**, standards MRR 0.892→**0.938**, 회귀 없음 →
기본값 `true`로 채택(사용자 결정). **다만 주 목표였던 regulations MRR 0.667은 변화
없음** — 대신 병목이 검색기가 아니라 **리랭커 쪽**임이 드러나 후속 조사로 이어짐(아래).
상세는 [EVAL.md](./EVAL.md) 12절.

### 리랭커 조사 → `n_candidates` 20→10 (2026-08-03)

**배경**: 위 하이브리드 도입 후 "병목이 리랭커"라는 결론이 나와, 모델 교체
(`bge-reranker-v2-m3`, ~2.3GB)를 검토하기 전에 **비용 0인 것부터** 측정
(`experiments/compare_reranker.py` 신규).

**발견**: 통념과 반대로 **후보를 많이 줄수록 나빠졌다** — `n_candidates`
10/20/30/50에서 Recall@5가 1.000/0.955/0.955/0.909. `bge-reranker-base`가 후보가
늘수록 오답을 상위로 잘못 올리는 것으로 보임. 골든 22건 전수 대조에서 10이 20보다
**개선 3건·악화 0건**이었고, 리랭커 처리 쌍이 절반이라 **속도도 2배**(25s→13s).

**적용**: `retrieve()`의 `n_candidates` 기본값 20→10(실질 변경은 출제 모듈 — 생기부는
원래 10을 명시해 쓰고 있었음). `eval_lib.py`가 하드코딩하던 `n_candidates=20`도 제거해
eval이 프로덕션 기본값을 따르게 함(judge에서 겪은 검증-배포 불일치 재발 방지).

**결과**: 전체 Recall@5 **1.000**(첫 만점)·MRR **0.854**, regulations Recall@5
**1.000**·MRR **0.753** — 하이브리드 착수 시 세운 주 목표를 여기서 달성. 리랭커
ablation도 함께 측정해 MODEL_SELECTION.md 4절의 "기여도 미측정" 항목을 닫음
(기여는 MRR +0.034~0.045, Recall엔 영향 없음). 모델 교체는 목표를 넘겨 **보류**.
상세는 [EVAL.md](./EVAL.md) 13절.

### Agent Trajectory Eval 신규 (2026-08-03)

**배경**: 기존 eval은 전부 최종 산출물(문항 품질·구조 유사도·검색 Recall)만 채점했고,
"에이전트가 그 결과에 어떻게 도달했는가"(도구 오호출, 재시도 사유)는 LangSmith UI에서
트레이스를 하나씩 눈으로 펼쳐볼 수만 있었지 집계된 적이 없었다.

**적용**: `evals/eval_trajectory.py` 신규 — **앱 코드 무변경**(LangGraph가 이미 자동으로
남기는 노드/도구 run만 읽음). 도구 호출을 `error`/`rejected`(가드레일 정상 동작)/
`empty_result`로 **분리** 집계하고, `validate` 노드의 `validation_feedback`을 형식·절차
실패 vs Judge 판단 불일치로 분류한다. 문구 매칭이 깨지면 `unclassified`로 드러나게 했다.

**첫 실행에서 발견**: 최근 30일 조회 결과에 2026-07-23 리팩터로 **제거된**
`similarity_judge` 도구 호출이 30건 잡혀, 신·구 아키텍처 트레이스가 한 통계에 섞여
있었음을 확인 → `--since YYYY-MM-DD` 옵션 추가. 부수적으로 `record_score` 거부율
74%(존재하지 않는 `item_id`로 호출)라는 후보 이슈를 발견했으나, 아키텍처 혼재 때문에
현재 코드의 문제인지 미확정.

**남은 과제**: LangSmith 무료 플랜 조회 한도 소진(2026-08-03)으로 `--since 2026-07-23`
재측정 미완 — 한도 복구 후 신 아키텍처만 놓고 재산출 필요. 상세는 [EVAL.md](./EVAL.md) 11절,
사용법은 [LANGSMITH_GUIDE.md](./LANGSMITH_GUIDE.md) 3.3절 참고.

---

## ⬜ 남은 작업 (우선순위 순)

### 📌 현재 열린 항목 (2026-08-04 기준 정리)

생기부 모듈 제거로 "교사 확인 대기" 항목 3건이 사라졌고, 마지막 blocker로 알던 LangSmith 한도도 2026-08-04에 **오진으로 판명**(실제로는 배선 버그)돼 해소됐다. 현재 외부 요인으로 막힌 항목은 RunPod 크레딧뿐이다.

| # | 항목 | 상태 | 막힌 이유 / 다음 행동 |
|---|---|---|---|
| 1 | **Agent Trajectory 재측정** | ✅ 완료(2026-08-04) · ⬜ 표본 확대 남음 | **"LangSmith 한도 소진"은 오진이었다** — 실제 원인은 `test_exam.py`에 `load_dotenv()`가 없어 API 키가 안 실린 것(ingest 직접 호출 시 202 정상). 배선 3건(dotenv·`init_langsmith_project()`·`--since` 타임존)을 고치고 재측정 완료: 제거된 도구(`similarity_judge`)가 안 잡히고 `submit_for_review`가 처음 잡힘, 재시도 원인이 format→**judgment 전량**으로 바뀜. 표본도 **6세션(도구 호출 91건)으로 확대 완료**. 결과: `record_score` item_id 혼동 가설은 **기각**(거부율 0.500으로 `save_item` 0.625보다 낮음 — 11절의 74%는 오염 표본 탓), 재시도 원인은 **`judgment`/`both`뿐이고 `format` 0건**(Judge 게이트 단독 병목), malformed tool-call은 0건이 아니라 **18.4%**로 재관측. 상세는 EVAL.md 11.1절 |
| 2 | ~~**`record_score` self-judge + Judge payload 불일치**~~ | ✅ **완료(2026-08-06)** | ① 자기채점을 게이트에서 제외(08-05) 후 **도구 자체를 제거**(08-06) — 게이트에서 빠진 뒤에도 문항당 4턴 중 1턴을 쓰고 있었고 14턴 한도가 실제 병목이었다(목표 개수별 달성률 3개 0.421/5개 0.100/7개 0.000이 이 산술과 일치). ② Judge 입력 정규화를 공유 함수 `judge_structure()`에 넣어 런타임·오프라인이 구조적으로 어긋날 수 없게 함. 상세는 EVAL.md 17절 |
| 2-1 | **정답유일성 4.125 / 목표 4.0** (구 오답매력도 항목에서 2026-08-07 전환) | ✅ 목표 달성(2026-08-07) · ⬜ 잔여 개선 여지 | 기존 열린 이슈(아래 1번 항목). `agent_node` few-shot이 텍스트 지시문뿐이라 실효성이 약할 수 있음 — 진짜 멀티턴 tool-call 예시로 강화하거나 `validate_item_format`에 최소 기준 게이트 추가 검토 |
| 3 | **코드 리뷰 전수 확인** | 🔄 진행 중 | 하단 "참고 — 코드 리뷰 대상 파일" 표는 예전 스냅샷이라 신뢰 금지. 체크리스트는 [CODE_REVIEW_CHECKLIST.md](./CODE_REVIEW_CHECKLIST.md)로 이전(2026-08-04, 경로 오류 6곳 정정 + 신규 파일 5개 반영). 2026-08-03에 `tools.py`·`retriever.py`·`chain.py`(삭제됨)를 실제로 훑으며 도구-코퍼스 불일치를 발견한 것이 부분 진행분 |
| 4 | **더 어려운 검색 골든셋** | ⬜ 미착수 | `retrieval_golden_final.json` 22건은 Recall@5 **1.000**으로 천장 도달 — 이걸로는 추가 개선을 측정할 수 없다. 2026-08-03 신설한 `regulations_retrieval_candidates.json` 10건은 **0.500**이라 아직 여유가 있으니, 우선 이 10건을 정식 골든셋에 편입하는 것부터 검토 |
| 5 | **리랭커 모델 교체(`bge-reranker-v2-m3`)** | ⏸️ 보류 | `n_candidates` 조정만으로 목표를 넘겨 착수 안 함(~2.3GB 다운로드 + CPU 추론 부담). 필요해지면 `BGE_RERANK_MODEL`만 바꿔 `experiments/compare_reranker.py`로 재측정 |
| 6 | **RunPod 재가동** | ⏸️ 크레딧 소진 | 2026-07-22부터 일시 중단. 설정은 그대로라 크레딧 충전 시 바로 복구 가능 |
| 7 | **포트폴리오 정리** | 🔄 진행 중 | 아키텍처 다이어그램(README mermaid·SVG 갱신 완료), LangSmith 트레이스 스크린샷(1번 해소 후), 기술 블로그 초안(`blog_draft.md`) |

> **생기부 모듈과 함께 종료된 항목**(더 이상 추적 안 함): 규정 위반 Recall 0.95 목표,
> 위반 오탐 후보 14건 교사 검수, 키워드 규칙 존폐 결정, 종교·외모 조항 근거 확인.
> 경위는 EVAL.md 14절 "결말 — 생기부 모듈 제거".

---

### 🔧 코드 리뷰용 잔여 이슈 요약 (2026-07-14 재확인)

아래 2건은 목표 미달로 **실제로 열려있는 이슈**. 나머지(출제 에이전트 안정성, Ragas/LangSmith
연동)는 결정·구현이 이미 확정/완료돼 재검토 대상이 아님(하단 상세 항목 2·5번은 참고용 기록).

> **STRUCTURE_GOLDEN 구조 Judge "overall 이진(≥3) κ ≥ 0.4" 목표는 2026-07-24 폐기 결정**(사용자 확인)
> — 몇 달간 few-shot 튜닝을 거쳐도 0.000~0.178 사이를 벗어나지 못했고, 대신 이미 계산 중인
> difficulty_match 일치율·overall MAE로 충분하다고 판단. `eval_structure_judge()`는 애초에 이
> binary kappa를 계산하지 않는 상태였음(코드 확인 완료) — 되살리지 않기로 함. 최신 측정치(14B+
> gpt-5.6-luna, n=45, 2026-07-24): difficulty_match 일치율 0.933, overall MAE **0.644**(역대 최저,
> EVAL.md §4 참고).

1. **정답유일성 미달 (3.375 / 목표 4.0)** — 2026-08-07 타깃 전환 (구 "오답매력도" 항목)

   > **전환 사유**: 추적하던 "오답매력도 2.846"은 **7B 생성 + qwen Judge**(2026-07-09) 시절
   > 값인데 그 뒤 생성 모델(14B, 07-15)·Judge(gpt-5.6-luna, 07-21)가 **둘 다 교체**됐고
   > 재측정이 없었다. 현재 스택으로 실측하니 **오답매력도 3.75**(목표 근접), 최약점은
   > **정답유일성 3.375**였다. 8건 중 3건이 **실제로 정답이 2개**인 파손 문항이었고
   > (부정형 문항에서 "정확히 하나만 조건 만족"을 미검증), 이를 차단하는 장치가
   > 파이프라인에 없다 — `validate_item_format`은 형식만 보고, Judge의 `정답유일성`은
   > 오프라인 eval 전용이라 런타임 게이트에 안 쓰인다. 근거: EVAL.md 22절,
   > `experiments/diagnose_distractor.py`

   <details><summary>구 오답매력도 항목 기록 (2026-07-09, 펼치기)</summary>
   - 코드: `app/modules/exam/graph.py` `agent_node`(123~125행, 오답 매력도 지시문) /
     `evals/eval_exam.py` `JUDGE_TPL`(117행~, 채점 few-shot)
   - 현상: 1단계(Judge 앵커)+2단계(agent_node 지시) 둘 다 적용해도 실제 생성 기준
     2.500→2.846(+0.346)에 그침. 목표까지 갭이 큼 — **위 전환 사유대로 이 수치는 현재 스택 기준이 아님**
   - 리뷰 시 볼 것: agent_node few-shot이 진짜 멀티턴 tool-call 예시가 아니라 텍스트 지시문뿐이라
     실효성이 약할 수 있음(EVAL.md 6절). `validate_item_format`에 오답매력도 최소 기준 게이트를
     추가하는 방향도 검토 후보

   </details>

2. **생기부 규정 위반 Recall 미달 + 알려진 RAG 우회 리스크 (0.927 / 목표 0.95)**
   - 코드: `app/modules/record/chain.py` `_rule_violations`(39행~, 키워드 3종: `_RULE_BACKGROUND`
     33행/`_RULE_RELIGION_POLITICS` 34행/`_RULE_APPEARANCE` 36행) / `_step_validate`(105행~,
     112행에서 `self._retriever.retrieve(..., REGULATION_COLLECTION, top_k=3)`로 RAG 검증) /
     `app/modules/record/prompts.py` `VALIDATE_TPL`(40행~)
   - 현상: 규칙 기반 키워드 + VALIDATE_TPL few-shot으로 0.840→0.927까지 끌어올렸지만, **이건
     `search_regulations`류 RAG 검색 자체(가정환경·종교 규정 청크가 검색 상위에 안 잡히는 문제)를
     고친 게 아니라 키워드 목록으로 우회한 것**(EVAL.md 398~406행). `VIOLATION_GOLDEN` 50개 안에서만
     잘 작동하고, 골든셋에 없는 새로운 표현(다른 단어로 가정환경/종교/정치성향을 언급하는 문장 등)은
     규칙도 못 잡고 RAG도 관련 규정을 못 찾아 놓칠 위험이 있음
   - 리뷰 시 볼 것: `chain.py:112`의 retrieve 호출이 왜 관련 규정 청크를 상위로 못 올리는지
     (임베딩 매칭 방식? 청킹? — 이 문서 위쪽 RAG 청킹 재설계 작업과 연결지어 재조사 가치 있음)
   - **2026-08-03 갱신**: 하이브리드 검색(BM25+dense) 도입 + `n_candidates` 20→10으로
     **RAG 검색 자체는 크게 개선됨** — regulations Recall@5 0.900→**1.000**,
     MRR 0.667→**0.753**, 전체 Recall@5 **1.000**(EVAL.md 12·13절). 리랭커 조사 과정에서
     "후보를 많이 줄수록 리랭커가 나빠진다"는 반직관적 원인이 드러난 것이 핵심이었음
   - **⚠️ 2026-08-03 진단 정정 — 이 이슈의 원인은 검색이 아니었음**: 골든셋을 만들려고
     regulations 코퍼스 472청크를 전수 스캔한 결과 **'종교'·'외모/용모/신체'·'추측' 조항이
     코퍼스에 0건**이었다(EVAL.md 14절). 즉 "검색이 상위로 못 올린" 게 아니라 **조항 자체가
     인덱싱돼 있지 않다** — 하이브리드도 리랭커 교체도 고칠 수 없는 문제였다.
     `_rule_violations` 키워드는 "검색 약점을 우회한 것"이 아니라 **코퍼스에 없는 내용을
     대신 담고 있던 유일한 근거**였다. ('가정환경'은 `부모(친인척 포함)의 사회･경제적 지위`
     라는 다른 표현으로 실재 — 공유 토큰이 없어 BM25로도 안 잡히는 어휘 격차 사례)
   - **다음 결정 필요**: 종교·외모·추측 조항이 담긴 문서를 `data/regulations/`에 추가해
     인덱싱할지, 아니면 규칙 기반 판정으로 두고 "이 유형은 RAG 근거 없이 규칙으로만
     판정된다"를 문서에 명시할지. 후보 골든셋 10건은
     `data/golden/regulations_retrieval_candidates.json`에 작성해뒀음(사람 검수 대기,
     현재 검색기 Recall@5=0.600으로 변별력 확인)

---

1. **오답매력도 개선** — 2026-07-09 완료: 1단계 `JUDGE_TPL`에 오답매력도=5점 few-shot 앵커 추가(ITEM_GOLDEN 기준 2.50→2.83). 2단계 `graph.py` agent_node 시스템 프롬프트에 오답 매력도 지시+예시 추가. **주의**: `eval_exam.py` 전/후 재실행으로 2단계를 검증하려 했으나 `eval_item_quality()`가 채점하는 `ITEM_GOLDEN`은 하드코딩된 고정 문항이라 agent_node 변경이 반영될 수 없음(방법론 오류, EVAL.md 4절 정정 사항 참고) — `scripts/compare_distractor_quality.py`로 실제 생성 기반 재검증: 오답매력도 2.500→2.846(+0.346, n=8→13, 객관식만). 방향은 맞으나 목표(4.0)엔 크게 못 미침, 추가 개입 필요(EVAL.md 6절 개선계획 참고)
2. **STRUCTURE_GOLDEN 실제 모델 라벨 보강** — 2026-07-10 전면 재구성 완료. count_match(생성 개수가 예시 문제 개수와 일치해야 한다는 전제) 자체가 설계 오류였음이 확인되어, 기존 str_001~003(Claude 합성 부트스트랩)과 count_match 기반 시도 전부 폐기. `ExamSpec.num_items` 도입(개수는 예시와 무관하게 별도 지정) 후 `scripts/gen_structure_golden.py`로 재생성하는 과정에서 **로컬 Ollama의 `num_ctx` 기본값(4096)이 멀티턴 ReAct 루프에서 쉽게 초과되어 컨텍스트가 잘리고 모델 응답이 깨지는 문제를 발견**(성공률 6%까지 급락) → `app/modules/exam/llm.py`에 `num_ctx=16384` 명시로 수정, 동일 passage 재현 테스트로 확인(4096: 0/5 → 16384: 5/5). 수정 후 정상 성공률(~35~40%) 회복, 총 34개 passage 시도로 **문항 0개가 아닌 14개 확보**(정확히 일치 5·부족 8·초과 1 — count_match 판정용으로 다양성 확보) `data/golden/structure_golden.json`에 저장(human_label 비워둠). 사람 라벨링 필요 — 라벨링은 대신하지 않음
   - **2026-07-12 확대**: 이후 20개까지 사람 라벨링이 끝나 Judge 신뢰도 측정에 실사용(EVAL.md 5절). n=20 과적합 위험을 줄이기 위해 새 주제 25개(str_052~076)를 추가 생성해 **45개로 확대**(생성만, human_label은 null로 비워둠 — 사람 라벨링 대기 중). budget=1 기준 0문항 실패율 28%(7/25) 확인 → budget=3 재시도 + 지속 실패 2건은 다른 주제로 교체해 최종 0문항 없이 45개 확보
   - **2026-07-12 라벨링 완료 + n=45 재측정**: 신규 25개 사람 라벨링 완료(전체 45개), bigram 반복도 proxy로 일관성 재검토해 3건(str_076/072/052) 재조정. 동일 C안 프롬프트로 n=45 재측정한 결과 diff κ 0.231→**0.509(목표 최초 달성)**, Pearson r -0.073→0.143(음수→양수)로 순위·이진 지표는 전부 개선(n=20 노이즈였을 가능성 시사)했으나, Judge의 관대한 편향(judge−human +1.178, human avg 2.04 vs judge avg 3.22)은 표본 확대로도 안 풀림 — overall 이진 κ는 여전히 0.4 미달(EVAL.md 5·6절 참고)
   - **2026-07-12 편향 원인 분석 + 3점 앵커 추가**: human 점수 구간별 judge 평균을 분석해 "편향이 아니라 판단 애매 시 3점으로 회피 수렴"하는 패턴을 확인(judge 분포 58%가 3점, few-shot 점수 분포가 {1,1,2,4,5}로 3점 예시 부재). `STRUCTURE_JUDGE_TPL`에 3점 앵커 few-shot 1개 + "애매해도 3점으로 회피 금지" 지시 추가 후 3회 반복 재측정: MAE·±1·편향·이진κ·Pearson r·Spearman ρ 전부 개선(편향 +1.178→+0.978, 이진κ 0.126→0.167), 저점 few-shot 시행착오 때 같은 상관관계 붕괴 없음. diff κ만 변동성 커짐(3회 중 1회 0.4 미달, 평균 0.424는 유지) — 롤백하지 않고 유지 결정(EVAL.md 5·6절 참고)
3. **출제 에이전트 안정성 개선 (컨텍스트/tool-calling, 2026-07-10 야간 자율 세션)** — TROUBLESHOOTING.md 참고. 세부 내용:
   - **1단계(컨텍스트 관리)**: agent_node 시스템 프롬프트에 "설명 텍스트 금지" 지시 추가, 위치를 정체성 선언 직후+끝 두 번(강조)으로 배치한 strong 변형이 소표본(n=8)에서 설명텍스트 턴을 0.38→0.00으로 제거해 채택. temperature 파라미터화(get_langchain_model, ChatOllama/ChatRunPod 둘 다) 후 0.7 vs 0.2 A/B(n=28, exact_match 기준): 0.7→14%, 0.2→21%(+7%p, n=28 표본 노이즈 범위), 초과생성 비율은 0.7→18%, 0.2→46%로 뚜렷이 악화 — **temperature 기본값 0.7 유지 결정**(노이즈+부작용 고려, 즉시 0.2 전환 가능하도록 파라미터만 추가)
   - **3단계(재시도 구조 개선)**: 재시도마다 init_session()으로 전체 초기화하던 것을 plan_node(요청당 1회)로 옮기고, agent_node는 reset_judge()만 호출해 기존 문항을 유지. 부족분만 "나머지 N개만 작성" 프롬프트로 이어서 생성(_build_system_prompt). **구현 완료, test_exam.py로 정성 확인**(재시도해도 이전 문항 유지됨). 정식 old vs new 정량 비교(test_retry_preservation.py)는 old 시뮬레이션이 배치에서 샘플당 30분+ 걸려 시간상 중단 — `ollama.log` 직접 확인으로 프로세스 누적/GPU 경합/무한 재시도는 배제, old 방식(budget=3 전체 재시도) 자체의 구조적 느림으로 결론(TROUBLESHOOTING.md 참고). 대신 STRUCTURE_GOLDEN 생성 이력을 비공식 근거로 사용: 개선 전 14개는 부족 실패 8건, 개선 후 6개는 0개 없이 전부 성공 — 방향성은 뚜렷하나 통제된 비교는 다음 세션 과제(더 작은 n·낮은 budget으로 재시도 권장)
   - **4단계(RAG top_k 실험)**: search_standards/search_regulations의 top_k 3→2 축소 시 Recall 0.810→0.762(-0.048, n=21) — 기준(0.05)에 근접한 경계선이라 **top_k=3 유지 결정**(노이즈 고려)
   - **안정성 보강**: 장시간 세션에서 간헐적으로 발생하는 "No data received from Ollama stream" 오류에 대응해 _invoke_with_retry()(graph.py) 추가 — LLM 호출 실패 시 자동 재시도(최대 2회, 2초 간격)
4. **생기부 모듈 eval 개선 (2026-07-12 야간 자율 세션 — EVAL.md 5·6절 참고)**
   - 규정 위반 Recall: 규칙 기반 키워드 3종(가정환경/종교·정치/외모) + 비교 표현 근접 정규식 + VALIDATE_TPL 위반유형 한정으로 0.840 → 0.920~1.000(3회 평균 0.927) 개선. 잔여 FP 1건·부정적 낙인 카테고리 LLM 판단 변동은 과적합 위험으로 보류. **⚠️ 알려진 리스크**: RAG 검색 자체의 약점(가정환경/종교 규정이 검색 상위에 안 잡힘)을 고친 게 아니라 규칙+프롬프트로 우회한 것 — 골든셋에 없는 새 표현 유형은 놓칠 수 있음(EVAL.md 5절 참고)
   - NLI 사실추가율: 원인 분석(문체 다듬기 vs 사실 추가 경계 모호) 후 사용자가 경계 사례를 직접 검토해 기준 확정("정도부사·평가어 중첩=NO / 구체 행위·결과 신규 서술=YES") → NLI_TPL few-shot 보강, 0.050~0.200(변동) → 0.000~0.050(3회 재측정, 안정화)로 개선
5. **Ragas 연동 + LangSmith Experiments 연동** (2026-07-12 완료, 5단계 전부)
   - ✅ 1단계: LangSmith dev/prod 프로젝트 자동 분리 완료(`app/common/llm/tracing.py`)
   - ✅ 2단계: `eval_ragas.py` 작성 완료 — **Ragas 패키지는 의존성 충돌(langchain-community
     제거된 경로를 무조건 import하는 상류 버그, GitHub vibrantlabsai/ragas #2741·#2745)로
     사용 포기**, 알고리즘만 직접 구현(사용자 확인 후 결정). 구현 중 qwen2.5:7b 언어 오염
     버그가 이 스크립트의 자체 LLM 호출에서도 재현되는 것을 발견해 필터 추가, 기존 문항
     생성물에서도 `_check_korean()` 임계값(한자 비율 5%)이 못 잡는 낮은 비율 오염 사례를
     새로 발견(후속 과제로 기록). 첫 측정: Faithfulness 0.600(n=5), Answer Relevancy
     0.631(n=5) — EVAL.md 8절 참고
   - ✅ 3단계: eval 실행 시 결과가 LangSmith Experiments에 자동 기록되도록 연동 완료. Judge 기반 3개(문항 품질/구조 유사도/RAG Faithfulness·Answer Relevancy)만 정식 연동(사용자 확인, 결정론적 함수 지표 3개는 제외). `scripts/langsmith_experiments.py` 공용 유틸 신설, golden JSON을 매 실행마다 LangSmith Dataset으로 동기화(삭제 후 재생성). 3개 데이터셋 전부 실제 실행으로 검증(EVAL.md 9절 참고) — 검증 중 로컬 Ollama가 한 요청에서 정상 대비 약 1000배 느려지는 시스템 레벨 이상을 관찰(장시간 다중 모델 전환에 따른 것으로 추정, 코드 버그 아님, 참고 기록만)
   - ✅ 4단계: EVAL.md 상단에 "Judge 3개 지표는 LangSmith Experiments가 최신 소스, EVAL.md는 마일스톤/의사결정 기록용" 안내 추가 완료. 과거 회차별 기록은 그대로 보존
   - ✅ 5단계: `eval_ragas.py` 코드 리뷰 완료 — 버그 없음, 사소한 죽은 코드 1건·의도된 중복 방어 1건만 확인(둘 다 동작에 영향 없어 수정 안 함). 리뷰 포인트는 아래 "참고 — 코드 리뷰 대상 파일" 표 참고
6. **GitHub Actions CI** (2026-07-14 완료 — 경량 CI만 도입, eval 자동화는 도입 안 하기로 결정. 상세는 DESIGN.md 9절 참고)
   - **경량 CI** (`.github/workflows/ci.yml`, 매 push/PR 블로킹, 모델 호출 없음): 백엔드 import
     스모크테스트 + `tests/`에 `mask_pii`/`_rule_violations` 순수 로직 pytest 유닛테스트 신규 작성 +
     프론트 `lint`/`build`
   - **eval 자동화는 도입 안 함**: 설계 검토 중 `get_judge_backend()`(factory.py)가 `LLM_BACKEND`
     무관하게 항상 Ollama 하드코딩임을 발견(`compare_models.py` 고정 Judge 설계 때문) — GitHub
     러너엔 GPU/Ollama가 없어 `eval_exam.py`의 Judge 채점이 CI에서 실행 불가하고, 우회하려면 Judge
     백엔드 분기 코드 추가 + 기존 Ollama 고정 Judge 이력과 단절되는 문제. 배포 자체도 아직 수동인데
     그 앞단만 자동화하는 것도 순서가 안 맞고, eval 점수 자체의 실행별 변동성(STRUCTURE_GOLDEN κ 등)도
     자동 게이트에 부적합 — 이 규모(1인+지인 실사용)엔 로컬 수동 실행 + EVAL.md 기록 유지가 더 적합.
     `eval_exam.py`/`eval_record.py`/`eval_ragas.py`는 변경 없음. self-hosted runner도 인프라 부담
     대비 이득 적어 제외
7. **문서화 및 포트폴리오 정리** — README/DESIGN.md/`MODEL_SELECTION.md` 최신 모델 결정 반영 완료(2026-07-22). 남은 것: 아키텍처 다이어그램(README에 mermaid로 이미 있음, 스크린샷 별도 불필요할 수 있음) 확인, LangSmith 트레이스 스크린샷, 기술 블로그 초안(`blog_draft.md` 진행 상태 확인 필요)

---

## 참고 — 코드 리뷰 대상 파일

"전부 읽기"가 아니라 **핵심 구조를 설명할 수 있는 수준**이 목표.

> **2026-07-22 상태 정정**: 아래 ✅ 표시는 예전 리뷰 세션 기록이며, 실제로 코드 리뷰가
> 전부 끝난 상태가 아님(사용자 확인). 어느 파일이 재검토 대상인지는 별도 md 문서로
> 추적할 예정 — 그 전까지 이 표를 "완료 보증"으로 신뢰하지 말 것.

| 파일 | 핵심 이해 포인트 | 상태 |
|---|---|---|
| `app/common/llm/backends/chat_runpod.py` | 왜 BaseChatModel을 직접 상속했는가, `_agenerate` vs `_generate` 차이 | ✅ |
| `app/modules/exam/graph.py` | LangGraph 노드 구조, 각 노드의 역할과 연결 (리디자인 이후 구조로 재검토 완료) | ✅ |
| `app/modules/exam/tools.py` | `@tool` 데코레이터, `_ctx` 공유 상태 문제, RAG 싱글턴 통합 | ✅ |
| `app/common/rag/store.py` + `retriever.py` | ChromaDB 컬렉션 구조, 2단계 검색 흐름, 죽은 임시 컬렉션 코드 제거 | ✅ |
| `app/modules/record/chain.py` (**2026-08-03 삭제됨** — 모듈 자체가 제거되어 리뷰 대상에서 소멸, 코드는 git 이력에만 존재) | 수동 루프 구조(LCEL 파이프 아님 — `run()`이 `_step_mask/_step_polish/_step_validate`를 for 루프로 직접 호출), 하이브리드 위반 탐지 순서, RAG 싱글턴 통합 | ✅(당시 기록, 현재 무관) |
| `app/main.py` | `/exam`·`/exam/stream` 중복 제거, 실제 SSE 노드 단위 스트리밍 | ✅ |
| `evals/eval_ragas.py` | (2026-07-12 리뷰 완료) `build_sample()`이 실제 그래프 호출+RAG 검색으로 (question, context, answer) 구성 → `faithfulness_one()`(주장 분해 후 컨텍스트 대조) / `answer_relevancy_one()`(역질문 생성 후 임베딩 코사인 유사도) → `run_langsmith_experiments()`가 Dataset 동기화 후 `evaluate()`로 두 함수를 evaluator로 래핑. 버그는 없었고, 사소한 죽은 코드(도달 불가능한 `else 0.0` 폴백) 1건과 의도된 중복 방어 로직 1건만 확인(둘 다 동작에 영향 없어 수정 안 함) | ✅ |

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

### 문항 품질 Judge에 "예시 문제 표절/패러프레이즈" 전용 지표 추가 (2026-07-17 논의, 결정 보류)
- **현재 상태**: 이 개념은 이미 부분적으로 존재함 — `STRUCTURE_JUDGE_TPL`(구조 유사도 Judge,
  `evals/eval_exam.py`)의 `overall_score` 채점 기준에 "예시 문제를 그대로 복사한 경우"·
  "표현만 바꿔 사실상 같은 것을 묻는 패러프레이즈 반복"이 감점 요소로 이미 포함돼 있음
  (`eval_exam.py:199-201`). 다만 이게 유형/난이도 구조, 환각, 언어오염, 주제이탈과
  **하나의 0~5 overall_score에 뭉쳐 있어 표절 여부만 따로 뽑아볼 수 없음**.
- **정답유일성/오답매력도/근거성(`JUDGE_TPL`, ITEM_GOLDEN 기반) 쪽엔 추가 불가**: `item_golden.json`
  스키마에 원본 예시 문제(passage_text)가 애초에 연결돼 있지 않아, 비교 대상 자체가 없음.
- **검토할 방향 두 가지**:
  1. `STRUCTURE_JUDGE_TPL`의 `overall_score`에서 "원문 유사도/표절" 항목을 별도 필드로 분리
     — 단, STRUCTURE_GOLDEN 45개를 이 새 차원으로 사람이 다시 라벨링해야 함(kappa/MAE
     재보정과 동일한 수준의 재작업).
  2. LLM Judge 대신 **임베딩 코사인 유사도** 같은 결정론적 함수로 원문-생성 문항 간 유사도를
     따로 계산 — Judge 신뢰도 재보정 부담 없이 이 축만 분리 가능.
- 사용자가 나중에 "현 상태 유지 vs 개선" 여부를 다시 판단할 예정. 착수 전 재확인 필요.

### 포트폴리오 정리 시 포함할 것
- README에 아키텍처 다이어그램
- LangSmith 트레이스 스크린샷
- Grafana 대시보드 스크린샷 (구축 시)
- eval 결과 수치 ([EVAL.md](./EVAL.md) 참고)
- 기술 블로그 초안: "소형 LLM으로 RAG 시스템 만들기"
