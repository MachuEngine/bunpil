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
- 최종 채택은 사용자 결정 대기(비용·외부 API 의존 트레이드오프 고려 필요)

---

## ⬜ 남은 작업 (우선순위 순)

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
6. **GitHub Actions CI** — eval 자동화
7. **문서화 및 포트폴리오 정리**

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
| `scripts/eval_ragas.py` | (2026-07-12 리뷰 완료) `build_sample()`이 실제 그래프 호출+RAG 검색으로 (question, context, answer) 구성 → `faithfulness_one()`(주장 분해 후 컨텍스트 대조) / `answer_relevancy_one()`(역질문 생성 후 임베딩 코사인 유사도) → `run_langsmith_experiments()`가 Dataset 동기화 후 `evaluate()`로 두 함수를 evaluator로 래핑. 버그는 없었고, 사소한 죽은 코드(도달 불가능한 `else 0.0` 폴백) 1건과 의도된 중복 방어 로직 1건만 확인(둘 다 동작에 영향 없어 수정 안 함) | ✅ |

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
