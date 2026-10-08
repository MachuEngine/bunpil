# 분필 아키텍처 상세

> README에서 옮긴 아키텍처 설명입니다(2026-10-08). 핵심 요약과 최신 지표는 [README](../README.md)를 보세요.

## 구현 현황

| 영역 | 상태 |
|---|---|
| 출제 모듈 (평가-수정 루프 워크플로, 자기교정 게이트) | ✅ 완료 |
| **생성 모델 ↔ Judge 모델 완전 분리** (2026-07-23) | ✅ 완료 |
| 검색(ChromaDB + BM25 하이브리드 + BGE-M3 + BGE 리랭커) | ✅ 완료 — ⛔ **2026-10-07부터 생성 경로에서 제외**, 검색 평가(Recall@5·MRR)용으로만 유지 |
| 평가 체계 (사람 라벨 골든셋 + LangSmith Experiments) | ✅ 완료 |
| 궤적 평가 (트레이스 집계 — 도구 실패 분포·재시도 원인) | ✅ 완료 — 현재 아키텍처 6세션 재측정(2026-08-04) |
| CI (GitHub Actions 경량 파이프라인) | ✅ 완료 |
| 배포 구성 (EC2 + RunPod 서버리스 + Caddy HTTPS) | ✅ 구성 완료 · ⏸️ 현재 내려 둠(EC2 삭제, RunPod 크레딧 소진) — `deploy/` 절차로 다시 띄울 수 있음 |
| 문항 품질 (오프라인 지표, 런타임 게이트 아님) | 🔄 사람 라벨 91건 기준 Judge κ(sonnet-5.5): 정답유일성 **0.86** 확정 · 오답매력도 **0.70** 확정 · 근거성 **0.56** 미확정 — 2026-10 생성 모델 재비교에서 gpt-6-luna를 평가 결론으로 선정(운영 전환은 하지 않음). 상세는 [품질 평가](EVAL_DETAILS.md#품질-평가) |
| 검색 골든셋 난이도 | ⛔ 기존 22건이 Recall@5 1.000으로 천장 도달. 검색이 생성 경로에서 빠져(2026-10-07) 골든셋은 더 늘리지 않음 |
| 코드 리뷰 전수 확인 | 🔄 진행 중 |

> 열린 항목의 상태·다음 행동은 [bunpil_roadmap.md](../bunpil_roadmap.md) "현재 열린 항목" 표에 정리돼 있습니다.

프로젝트의 특징 세 가지:

- **로컬 ↔ 프로덕션 전환 가능한 LLM 추상화** — 개발은 Ollama(로컬), 프로덕션은 RunPod 서버리스(vLLM). 환경변수 하나로 전환
- **"LLM이 판단하고, 코드가 결정한다"** — 품질과 유사도 판단은 LLM이 맡고, 통과 여부·재시도·개수·언어 검사는 결정론적 코드가 맡습니다 ([설계 원칙](#설계-원칙))
- **평가 기반 개발** — 사람이 라벨링한 골든셋으로 검색·생성·마스킹 품질을 수치로 추적

평가와 관련한 기록은 네 문서로 나눠 두었습니다.

| 문서 | 담는 내용 |
|---|---|
| [EVAL_SUMMARY.md](../EVAL_SUMMARY.md) | 평가 체계 요약 (지표 의미, 개선 스토리) |
| [EVAL.md](../EVAL.md) | 회차별 원본 기록 (시간순 실험 로그) |
| [LANGSMITH_GUIDE.md](../LANGSMITH_GUIDE.md) | LangSmith Experiments 연동 |
| [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) | 삽질 기록 |

<!-- 제거된 모듈 배경: 제품 설명을 먼저 읽도록 섹션 끝으로 옮김 -->
<details>
<summary><b>생기부 윤문 모듈을 왜 걷어냈는가 (2026-08-03, 펼치기)</b></summary>

원래 이 프로젝트는 **출제**와 **생기부 윤문** 두 모듈이었습니다. 생기부 모듈은 관찰 메모를 받아 PII를 마스킹하고 문체를 다듬은 뒤 기재 규정 위반을 검증하는 체인이었고, 규정 위반 Recall 0.927 / F1 0.962라는 수치도 있었습니다.

**그 수치의 근거를 추적하다 걷어냈습니다.** 검증 규칙(종교·정치성향, 외모, 추측 등 키워드 6종)이 어느 조항에서 나왔는지 확인하려고 교육부 기재요령 PDF **원문 262,678자를 전수 검색**했더니 `종교`·`신앙`·`외모`·`용모`·`추측`이 **한 번도 나오지 않았습니다**. 규칙의 실제 출처는 규정이 아니라 합성 골든셋의 라벨이었고, 그 골든셋으로 채점해 0.927이 나온 순환 구조였습니다.

더 결정적인 건 오작동 방향이었습니다 — 사회과가 가르치는 주제어를 그대로 막고 있었습니다:

| 결과 | 문장 |
|---|---|
| 🚫 차단 | 사회 수업에서 **정치적** 다원주의 개념을 조사해 발표함 |
| 🚫 차단 | **가정환경**에 따른 교육 격차를 주제로 보고서를 작성함 |
| ✅ 통과 | 아버지가 대기업 임원이라 경제에 관심이 많음 ← **기재요령 p24 위반** |

잡아야 할 건 놓치고 놓아줘야 할 건 잡았습니다. [하드룰 1](#설계-원칙)(실제 학생 데이터 미사용) 때문에 합성 데이터로만 검증할 수 있어 실제 현장에 적용해 보지도 못한 상태였고, 규칙의 옳고 그름을 가려 줄 도메인 근거도 없었습니다. **검증할 수 없는 기능을 포트폴리오에 남기는 것보다 걷어내는 쪽이 정직하다고 판단**했습니다.

조사·측정 기록은 [EVAL.md](../EVAL.md) 14절에 남아 있고, 코드는 git 이력에 있습니다. PII 마스킹(`app/common/privacy.py`)은 출제 경로가 계속 사용하므로 골든셋 20건과 함께 유지됩니다.

</details>

---

## 아키텍처

| 구분 | 기술 |
|---|---|
| 백엔드 | FastAPI (비동기) |
| 프론트엔드 | Next.js (`frontend/`) |
| 에이전트 | LangGraph (평가-수정 루프 워크플로) |
| 검색(RAG, 생성 경로 미사용) | ChromaDB + BGE-M3 임베딩 + **BM25 하이브리드** + BGE 리랭커 (모두 CPU). 의미 기반 검색과 단어 기반 검색을 따로 돌린 뒤, 두 순위표를 RRF라는 표준 기법으로 하나로 합칩니다. **2026-10-07부터 `agent` 노드가 이 검색을 쓰지 않습니다** — 검색 평가(Recall@5·MRR)용으로만 유지합니다 |
| 생성 LLM 서빙 | Ollama (개발) / RunPod 서버리스 vLLM (프로덕션) |
| Judge LLM | 런타임 구조 게이트: OpenAI gpt-5.6-luna(기본) / Ollama(대안). claude-sonnet-5.5 교체는 2026-10 재선정 결론으로만 남기고 운영 전환은 하지 않음. 오프라인 정기 평가(2026-10부터): anthropic/claude-sonnet-5.5 — 모두 생성 백엔드와 독립 |
| 트레이싱 | LangSmith. 2026-07-24부터 PII 마스킹을 거친 뒤에 한해, 프로덕션 API 서버 기록도 켤 수 있게 열어 뒀습니다([하드룰 3](#설계-원칙) "사용자 입력 비저장"의 예외) |
| 배포 | AWS EC2 t3.medium + EBS + RunPod 서버리스 + Caddy HTTPS |

### 출제 모듈 — 평가-수정 루프 워크플로 + 분리된 Judge

분필을 "ReAct 에이전트"가 아니라 "LangGraph 평가-수정 루프 워크플로"로 부르는 건, LLM 판단에 맡기는 범위를 계속 줄여 온 이력과 맞추기 위해서입니다. 2026-07-23에 생성 모델이 자기 출력을 스스로 채점하던 self-judge 도구를 없애고 별도 `judge` 노드로 분리했고, 2026-08-06에는 self-judge의 흔적이던 `record_score` 도구까지 제거해 채점을 코드 밖 Judge LLM과 코드 게이트로 완전히 넘겼습니다. 지금 LLM이 스스로 반복하며 진행하는 부분은 `agent` 노드 안에서 도구를 호출해 문항을 쓰는 생성 단계뿐이고, `plan → agent → judge → validate → (미달 시) 재시도`라는 흐름 자체는 LangGraph가 코드로 고정합니다. 생성 단계에 도구 호출 루프를 남겨 둔 이유는 검증 결과(형식 오류·중복 등)를 도구 응답으로 즉시 돌려줘 모델이 같은 턴에서 고치게 하기 위해서이고, 그 대가로 인자가 깨진 도구 호출(malformed tool-call)이나 재시도 턴 한도 초과로 생성이 실패하는 경우가 있습니다([엔지니어링 하이라이트](#엔지니어링-하이라이트) 참고).

에이전트(생성 LLM)가 추론과 문항 생성을 **직접** 담당하고, 도구 4개는 저장·검증의 **순수 계산**만 수행합니다(도구 내부 LLM 호출 없음 — LLM을 도구 안에 중첩하는 안티패턴 제거). 구조 유사도 채점만은 도구가 아니라 그래프의 별도 `judge` 노드가 담당하며, 생성 모델과 **완전히 다른 LLM 백엔드**를 호출합니다.

| 도구 | 역할 |
|---|---|
| `validate_item_format` | 선지 4개·①②③④ 형식 등 결정론적 형식 검증 (오류 시 수정 지침 반환 → 자기교정) |
| `save_item` | 문항 저장 — 세 검사를 모두 통과해야 저장합니다. ① 한국어 오염(한자 비율 ≥5% 또는 한글 부재 시 거부) ② 예시 문제 베끼기(연속한 두 글자 묶음이 예시와 90% 이상 겹치면 거부) ③ 세트 내 중복(두 문항의 단어 집합이 80% 이상 겹치면 거부) |
| `discard_item` | 승인 불가 문항을 ID로 폐기 |
| `submit_for_review` | 문항 세트 작성 완료 신호(인자 없음) — 이후 채점은 `judge` 노드가 수행 |

> **`search_regulations` 제거(2026-08-03)**: "교육과정 법령·지침을 검색한다"고 선언해 놓고 실제로는 `regulations` 컬렉션(= **생기부 문서 둘뿐**)을 조회하고 있었습니다. *"사회 문항 출제 시 유의사항"* 을 물으면 *"교외상은 기재할 수 없음"*, *"민주주의 문항 교육과정 준수"* 를 물으면 *"공직선거법에 따라 투표에 참가하는 경우 출석 인정 범위는…"* 이 돌아왔습니다. 문항 출제에 무관한 텍스트가 매번 컨텍스트에 주입된 셈입니다(궤적 eval에서 이 도구 호출 28건이 전부 `ok`로 집계). 당시 교육과정은 `search_standards`가 담당하므로 중복이자 오염원이라 제거했습니다 — `search_standards` 자체도 이후 2026-10-07에 생성 경로에서 빠졌습니다(바로 아래 참고).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/exam-loop-dark.svg">
  <img src="../assets/exam-loop-light.svg" alt="문항 생성 루프. agent 노드의 LLM이 validate_item_format, save_item 순서로 도구를 호출하고, 형식 오류나 저장 거부 사유는 ToolMessage로 LLM에 돌아가 다음 턴을 정한다. 목표 개수를 저장하면 submit_for_review로 judge 노드에 넘긴다. 교체가 필요하면 discard_item을 호출하고, 형식이 깨진 응답은 연속 3회까지 다시 요청한다.">
</picture>

> `search_standards`는 2026-10-07부터 생성 경로에서 빠져 루프는 `validate_item_format`부터 시작합니다(아래 참고).

### RAG를 생성 경로에서 뺀 이유

처음엔 "생성이 참고할 근거가 있어야 한다"는 생각으로 검색(ChromaDB + BGE-M3 + BM25
하이브리드 + 리랭커)을 구축했고, 하이브리드 전환으로 검색 자체의 Recall@5를
0.955→1.000까지 끌어올렸습니다([모델 선정](EVAL_DETAILS.md#모델-선정)). 그런데 검색이 실제로 생성
품질에 기여하는지는 한 번도 직접 떼어본 적이 없었습니다.

그래서 떼어봤습니다. 운영 생성 모델(gpt-6-luna)로 지문 33개를 검색을 켜고 끈 두
조건 × 2회 생성해 사람 수준 Judge(claude-sonnet-5.5)로 짝 비교했습니다
(`experiments/ablate_retrieval.py`). 정답유일성·오답매력도·근거성 차이는 전부
"판정 불가"(95% 신뢰구간이 0을 포함, 최대 0.07점 차이로 같은 조건의 반복 간 잡음과
비슷한 크기)였습니다. 검색이 실제로 하는 일은 문항의 `standard` 칸(화면의
"성취기준: ...")을 채우는 것뿐이었고(기재율 87% vs 0%), 그 코드 78개는 전부
코퍼스에 실재했지만 문항 품질과는 무관했습니다.

그 표시를 쓸 데가 있는지도 확인했습니다 — 사용자(지인 교사)는 이 표시를 쓰지
않습니다. 대안으로 "근거 확인 버튼"을 검토했지만 기각했습니다. 교육과정 문서(성취
기준·내용 요소)에는 지문에 등장하는 구체적 사실(예: 헌법재판소 권한 목록, 합성
자료의 통계 수치)이 없어 사실 확인 자체가 불가능하고, 쓰임새도 불확실했습니다.

**결정**: 생성 경로에서 RAG를 뺐습니다(`TOOLS` 5→4개, `standard` 필드·화면 표시
제거, 런타임이 더는 FlagEmbedding을 불러오지 않음). 검색 모듈·인덱스·검색 평가
(Recall@5·MRR)와 실험 재현용 `search_standards` 함수는 지우지 않고 유지했습니다 —
임베딩(BGE-M3)·리랭커(BGE-reranker) 선정도 이제 필요 없어졌습니다
([모델 선정](EVAL_DETAILS.md#모델-선정) MODEL_SELECTION.md §3~5). 부수적으로, 이 실험을 준비하며
로컬 벡터 DB 경로 문제로 검색이 실패해도 도구 오류 메시지로 삼켜져 드러나지 않는다는
것도 발견해 함께 고쳤습니다([엔지니어링 하이라이트](#엔지니어링-하이라이트) 참고).
상세 수치와 재현 명령은 [EVAL.md](../EVAL.md) 30절 참고.

세트 전체는 LangGraph 그래프가 관리합니다. `agent`가 `submit_for_review`로 제출하면 `judge` 노드가 **생성 모델과 완전히 분리된 Judge 백엔드**(`get_judge_backend()`)로 구조 유사도를 채점합니다. 그다음 `validate` 노드가 문항 개수가 맞는지, Judge 점수가 기준값(threshold)을 넘었는지를 코드로 판정합니다.

기준에 못 미치면 최대 5회까지 `agent`로 되돌아갑니다. 이때 **이미 만든 문항은 그대로 두고 부족한 개수만 이어서 작성**합니다(부분 진행 보존).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/exam-graph-dark.svg">
  <img src="../assets/exam-graph-light.svg" alt="LangGraph 상태 흐름. START에서 plan, agent, judge를 거쳐 validate가 문항 개수와 Judge 점수 기준값을 판정한다. 미달이고 budget이 남으면 agent로 재시도하고, 통과하면 END(통과)로, budget을 다 쓰면 문항은 반환하되 통과 표시 없이 END로 간다.">
</picture>

<details>
<summary><b>왜 <code>judge</code> 노드를 따로 뒀는가 (2026-07-23 변경, 펼치기)</b></summary>

원래는 `agent`가 `similarity_judge`라는 도구로 **자기 출력을 스스로 채점**했습니다(self-judge). 그런데 이 self-judge의 신뢰도는 사람 라벨과 한 번도 대조된 적이 없었습니다.

정작 EVAL.md에 몇 달간 쌓아온 "구조 Judge 신뢰도" 수치는 전부 **오프라인** eval 스크립트(`get_judge_backend()` 재호출)를 잰 값이었습니다. "검증에 쓰는 Judge"와 "실제 배포된 Judge"가 서로 다른 코드였던 셈입니다(검증-배포 불일치).

해결책은 self-judge의 신뢰도를 측정하는 게 아니라, **애초에 같은 Judge를 검증과 배포 양쪽에서 쓰는 것**이었습니다. `similarity_judge` 도구를 없애고 별도 `judge` 노드를 추가해, 런타임과 오프라인 eval(`evals/eval_lib.py`)이 `app/modules/exam/judge.py`의 `judge_structure()`를 그대로 공유하도록 통합했습니다. 이제 EVAL.md의 구조 Judge 신뢰도 수치가 곧 실제 배포된 Judge의 신뢰도입니다.

**트레이드오프**: `JUDGE_BACKEND=openai`(기본값)에서는 문항 세트를 만들 때마다 `passage_text`가 OpenAI로 전송됩니다. PII는 마스킹되지만 저작권이 있는 교사 지문일 수 있습니다.

API 키가 없거나 호출이 실패하면 조용히 폴백하지 않고 그대로 실패합니다(fail-fast). 신뢰도가 검증되지 않은 Judge로 게이트를 통과시키는 문제를 반복하지 않기 위해서입니다. 전부 로컬에서 처리하려면 `JUDGE_BACKEND=local`로 바꾸면 됩니다.

</details>

### 입력 PII 마스킹

교사가 붙여넣은 예시 문제에 학생 이름·연락처가 섞여 들어올 수 있어, **모델 호출보다 먼저** 마스킹합니다(`app/main.py` `_build_spec()`). 마스킹된 텍스트만 에이전트와 Judge, LangSmith로 흘러갑니다. 그래서 프로덕션 트레이싱을 켤 수 있었습니다([하드룰 3](#설계-원칙)의 예외).

### API와 스트리밍

- **`POST /exam/stream`** (SSE) — UI가 사용하는 기본 경로. `graph.stream(stream_mode="updates")`로 LangGraph 노드 완료 시점마다 진행 이벤트를 전송합니다. POST 요청이라 브라우저 네이티브 `EventSource`(GET 전용) 대신 프론트엔드가 `fetch` + `ReadableStream`을 수동 파싱합니다.
- **`POST /exam`** (JSON 단발) — 같은 로직을 쓰는 대안 경로입니다. `curl`처럼 브라우저가 아닌 클라이언트를 위한 것입니다
- **`POST /exam/extract`** — 예시 문제 캡처 이미지(png/jpeg/webp, 5MB까지)를 받아 **VLM**(이미지를 읽는 LLM)으로 텍스트를 추출하고 마스킹해 반환합니다. 그래프와는 무관한 별도 경로이며, 추출된 텍스트를 `/exam/stream`에 그대로 넘기는 건 교사 몫입니다.
- **`POST /exam/explain`** — 생성된 문항 하나를 받아 정답 해설을 만듭니다. '해설 보기' 버튼을 누를 때만 호출합니다
  - 서버는 문항을 저장하지 않습니다. 브라우저가 다시 보내고, 서버는 모델 호출 전에 마스킹합니다(하드룰 2·3)
- **`GET /health`** — 서버가 살아 있는지만 확인합니다(인증 불필요)

문항 다운로드(학생용·교사용 PDF/TXT/PNG)는 서버를 거치지 않고 브라우저에서 만듭니다.

### 이미지로 예시 문제 입력하기

텍스트를 옮겨 적는 대신 화면 캡처를 붙여넣을 수 있습니다. 입력창에 이미지를 `Ctrl+V`로 붙여넣거나 "이미지 첨부" 버튼으로 파일을 고르면 VLM(기본 OpenAI, `VLM_BACKEND`)이 이미지를 읽어 텍스트로 옮깁니다. 발문과 `<보기>`, 선지는 원문 그대로 옮기고, 표·그래프·지도처럼 글자가 아닌 자료는 `[자료: ...]`로 풀어 씁니다.

추출된 텍스트는 입력창에 자동으로 채워지지만 생성이 바로 시작되지는 않습니다. 교사가 확인하고 고친 뒤 직접 "문항 생성"을 눌러야 합니다.

> ⚠️ **이 경로만 마스킹 순서가 다릅니다.**
>
> 다른 LLM 호출은 전부 개인정보 마스킹 뒤에 이뤄지지만, 여기서는 원본 이미지가 마스킹 전에
> VLM으로 먼저 갑니다. 이미지 자체를 마스킹할 방법이 없어, 텍스트로 뽑은 뒤에야 마스킹할 수
> 있기 때문입니다. **학생 개인정보가 찍힌 캡처는 넣지 마세요.**
>
> 이미지 원본은 디스크에 저장하지 않고 요청을 처리하는 동안에만 다룹니다(예외 범위와 트레이싱
> 차단 방식은 [DESIGN.md](../DESIGN.md) 6절).

> 정확도는 **CER**(글자 단위 오류율, 낮을수록 좋음)로 쟀습니다. 2026-10 재선정에서 모델을
> gpt-4o-mini에서 **gpt-6-luna**로 바꾸면서, 글자만 있는 문제는 0.010→0.000, 표·그래프가
> 낀 문제는 0.143→0.035로 줄었습니다(그림 90건+텍스트 20건, `evals/eval_vlm_compare.py`).
> 전환 근거는 [MODEL_SELECTION.md](../MODEL_SELECTION.md) §7 "2차 선정" 참고.

```
data: {"status": "truncated", "msg": "입력이 길어 앞부분만 반영되었습니다."}  # 8,000자 초과 시만
data: {"status": "format_notice", "msg": "예시의 OX 형식은 지원하지 않아 4지 선다 형식으로 만들었습니다."}  # 미지원 형식일 때만
data: {"status": "progress",  "msg": "준비 중..."}
data: {"status": "progress",  "msg": "AI가 문항을 생성하고 있습니다. 수 분 소요됩니다..."}
data: {"status": "progress",  "msg": "생성된 문항의 구조적 유사도를 채점하고 있습니다..."}
data: {"status": "progress",  "msg": "채점 결과가 기준을 통과했는지 확인하고 있습니다..."}
data: {"status": "progress",  "msg": "문항 세트를 다시 생성하고 있습니다 (2번째 시도)..."}  # 재시도(최대 5회)마다
data: {"status": "done",      "items": [...], "validation_passed": true, "truncated": false, "format_notice": ""}
data: {"status": "error",     "msg": "요청을 처리하지 못했습니다."}  # 내부 상세는 노출하지 않음
```

<details>
<summary><b>동시성 설계 (펼치기)</b></summary>

- **요청 간 세션 격리**: 출제 요청별 컨텍스트를 `contextvars.ContextVar`로 분리. `asyncio.to_thread` + `contextvars.copy_context()`로 worker 스레드에 전파.
- **이벤트 루프 비블로킹**: `/exam`은 `asyncio.to_thread`로 LangGraph 실행. `/exam/stream`은 `graph.stream()`(동기 제너레이터)을 executor 스레드에서 돌리며 `asyncio.Queue`로 이벤트만 이벤트 루프에 전달.
- **동시 요청 제한**: `asyncio.Semaphore(2)`로 전역 동시 처리 슬롯을 2개로 제한(GPU 백엔드 과부하 방지). 슬롯 획득 실패(0.05초 타임아웃) 시 429 반환.

</details>

### LLM 백엔드 — ChatRunPod ↔ RunPodBackend

LangGraph 에이전트는 BaseChatModel 인터페이스만 알면 되고, RunPod과의 실제 HTTP 통신은 별도 레이어가 전담합니다. (Judge 백엔드는 이 경로를 타지 않는 별개 인터페이스 — `LLMBackend.generate()`, `app/common/llm/base.py`.)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/runpod-backend-dark.svg">
  <img src="../assets/runpod-backend-light.svg" alt="RunPod 백엔드 계층. agent 노드가 BaseChatModel 어댑터인 ChatRunPod를 호출하고, ChatRunPod는 메시지와 tool_calls를 변환해 RunPodBackend.generate_chat()을 부른다. 요청 분석과 해설은 FastAPI가 RunPodBackend.generate()를 바로 호출한다. RunPodBackend는 HTTPS /run, /status로 RunPod 서버리스(vLLM, Qwen2.5-14B-AWQ)와 통신한다.">
</picture>

RunPodBackend는 비동기 `/run`으로 작업을 한 번만 제출한 뒤 동일한 `job_id`를 폴링해, 긴 생성(멀티턴 도구 호출 루프)도 중복 실행 없이 안전하게 기다립니다.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/runpod-polling-dark.svg">
  <img src="../assets/runpod-polling-light.svg" alt="RunPod 작업 폴링 흐름. POST /run으로 제출해 job id를 받고, 5초 간격으로 최대 60회 GET /status를 조회한다. COMPLETED면 output을 반환하고, IN_QUEUE나 IN_PROGRESS면 대기 후 다시 조회한다. 제출 응답이 없으면 재제출 없이 실패, FAILED나 CANCELLED면 RuntimeError, 60회를 넘기면 TimeoutError로 끝난다.">
</picture>

> `/run` 제출 응답을 받지 못하면 job 자체는 이미 실행 중일 수 있으므로, 무작정 재제출하지 않고 명확한 예외로 상위 로직에 알립니다.

---

## 설계 원칙

**1. LLM이 판단하고, 코드가 결정한다.**
LLM의 자기 채점은 "기록"까지만 쓰입니다. 그 값으로 무엇을 할지는 전부 결정론적 코드가 정합니다.

| 대상 | 누가 정하나 | 그렇게 나눈 이유 |
|---|---|---|
| 구조 유사도 (유형·난이도) | 점수는 별도 Judge LLM(`judge` 노드), 기준값은 코드 | 정성 판단은 LLM이 낫고, 합격선은 코드가 안정적 |
| 문항 개수 | 코드 (`len(items) == num_items`) | LLM Judge에 맡겼다가 설계 오류 발견 후 이관 |
| 언어 (한국어) | 코드 (`save_item` 한글 비율 게이트) | 중국어 오염 문항을 저장 전 차단 |
| 재시도 여부 | 코드 (`budget` 루프 — 재시도 허용 횟수) | LLM에 재시도 판단을 맡기면 수량 제어가 깨짐 |

**2. 도구는 순수 계산만, Judge는 도구가 아니라 별도 노드.**
도구 호출 루프의 도구 내부에는 LLM 호출이 없습니다. 구조 유사도 채점은 그래프의 별도 `judge` 노드가 담당하며, 여기서만 생성 모델과 다른 LLM 백엔드를 호출합니다(왜 분리했는지는 [아키텍처](#출제-모듈--평가-수정-루프-워크플로--분리된-judge) 참고).

**3. 보안 하드룰 (예외 없음).**
본문에서 "하드룰 N"으로 부르는 것이 아래 네 가지입니다.

| # | 하드룰 | 설명 |
|---|---|---|
| 1 | 실제 학생 데이터 미사용 | 골든셋을 포함해 전부 합성 또는 익명 데이터입니다 |
| 2 | PII 마스킹은 모델 호출 **이전** | 마스킹을 거치지 않은 텍스트는 어떤 모델에도 가지 않습니다 |
| 3 | 사용자 입력 비저장 | 예시 문제는 요청을 처리하는 동안 메모리에만 있습니다. 단 2026-07-24부터 마스킹을 거친 뒤의 LangSmith 트레이싱은 허용합니다 |
| 4 | 로그·캐시에 PII 금지 | 마스킹 전 원문은 어디에도 남기지 않습니다 |

여기에 더해 Next.js에서 FastAPI로 가는 서버 간 요청에는 API 키 인증을 겁니다.

**4. 평가 기반 개발.**
골든셋은 코드에 하드코딩하지 않고 전부 `data/golden/*.json`으로 관리합니다(파일별 용도는 [data/golden/README.md](../data/golden/README.md)). 모델이나 프롬프트를 바꿀 때마다 [EVAL.md](../EVAL.md)에 결과 이력을 남깁니다.

---

## 엔지니어링 하이라이트

소형 오픈소스 LLM(7B, 이후 14B로 승격)으로 출제 워크플로를 만들고, 2026-10에 평가 체계를 다시 세우며 겪은 문제와 해결 과정입니다. 상세 진단 기록은 [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) 참고.

| 문제 | 진단 | 해결 |
|---|---|---|
| 골든셋 생성 성공률이 약 37%에서 갑자기 **6%로 급락**, 모델이 중국어·스페인어 섞인 응답 | 로컬 Ollama가 `num_ctx` 기본값 **4096**으로 실행 중이었음(모델은 32K 지원). 생성 단계의 멀티턴 도구 호출 루프(당시 ReAct로 부름) + RAG 검색 결과 누적이 몇 턴 만에 한도를 초과 → 컨텍스트가 잘리며 시스템 프롬프트 유실. 동일 케이스 재현: 4096에서 0/5문항 → 16384에서 5/5문항 | `num_ctx=16384` 명시. vLLM(프로덕션)은 모델 네이티브 값을 쓰므로 **로컬 개발 환경에만 있던 설정 격차**였음 — dev/prod parity의 실례 |
| 생성 문항의 **45%에 중국어 오염** — "한국어로만 응답" 지시에도 발생 | LangSmith 트레이스 100건 정량 분석: 오염 출력의 입력 크기 중앙값 11,263자 vs 정상 8,009자 — 입력이 길수록 오염 확률이 오르는 **확률적 현상**. 오염된 문항이 재시도 프롬프트에 그대로 실려 다음 시도까지 번지는 연쇄 경로도 확인 | `save_item`에 결정론적 한국어 게이트(한글 부재 또는 한자 비율 ≥5% 시 저장 거부 + 재작성 피드백). 기존 오염 사례 9건 소급 판정에서 수동 분류와 100% 일치 |
| "생성 개수 = 예시 문제 개수" 전제로 만든 count_match 검증이 실제 요구사항과 불일치 | 개수는 예시와 무관하게 사용자가 지정하는 값(`num_items`)이어야 함 — **골든셋 라벨링 직전에 설계 전제 자체가 틀렸음을 발견** | count_match를 LLM Judge에서 제거하고 `len(items)==num_items` 코드 검증으로 이관. 골든셋 전면 재생성 |
| 생성 프롬프트를 개선했는데 eval 수치가 **전혀 안 변함** | eval의 문항 품질 평가는 하드코딩된 고정 30문항을 채점하는 구조 — 생성 코드를 아무리 바꿔도 이 지표에 반영될 수 없었음 | 실제로 문항을 새로 생성해 채점하는 별도 검증 스크립트 작성. "eval이 있는가"와 "내가 바꾼 코드가 그 eval이 실제로 거쳐 가는 경로에 있는가"는 별개 |
| **검증-배포 불일치**: 몇 달간 쌓은 "구조 Judge 신뢰도"는 오프라인 `get_judge_backend()`를 잰 값인데, 런타임은 생성 모델이 `similarity_judge` 도구로 자기 출력을 채점(self-judge)하고 있었음 | self-judge는 사람 라벨과 대조된 적이 없고, 채점 기준(루브릭)도 함수 설명문 한 줄뿐이라 오프라인 Judge보다 신뢰도가 낮을 가능성이 높았음. "검증한 것"과 "배포된 것"이 다른 코드 경로였다는 뜻 | 생성 모델과 Judge를 완전 분리. `similarity_judge` 제거 후 별도 `judge` 노드가 오프라인 eval과 **같은 함수**로 채점 — EVAL.md 수치가 곧 배포된 Judge의 신뢰도 |
| 재시도마다 이전 시도의 문항까지 전부 폐기 → `num_items`가 클수록 성공률 급락 | 재시도 구조가 세트 전체 재생성 방식이었음 | **부분 진행 보존**: 재시도 시 저장된 문항은 유지하고 "나머지 N개만 작성" 프롬프트로 이어서 생성. 개수 기준으로 적용 전 14건 중 부족 실패 8건 → 적용 후 6건 전부 목표 근접 달성(통제 실험은 아닌 생성 이력 기반 비교) |
| 사전 점검에서 API 모델 2종이 **변별 지문 0/33** — 두 모델의 차이가 전혀 안 보임 | 당시 Judge(gpt-5.6-luna)가 대부분 4~5점을 줌. 사람 라벨 91건과 대조하니 오답매력도를 **+0.61점** 후하게 채점하고 있었음(근거성 +0.43). 이전 κ 0.468도 Claude가 합성한 라벨 30건 기준이었음 | 실제 생성 문항 95개에 사람 라벨(블라인드, 91건 채점)을 달아 Judge를 다시 고름. 오프라인 Judge를 claude-sonnet-5.5로 교체(정답유일성 κ 0.87 vs 0.78, 오답매력도 0.67 vs 0.53, 둘 다 짝지은 비교 CI가 0 미포함) |
| gpt-6-luna로 생성하면 일부 지문이 `400 No tool output found for function call`로 **통째로 실패** | 모델이 인자 JSON이 깨진 도구 호출을 내면 langchain-openai는 이를 `invalid_tool_calls`로 분리하지만, 다음 요청에는 다시 직렬화해 보냄. `agent_node`는 정상 `tool_calls`에만 응답해 "응답 없는 도구 호출"이 남았고 OpenAI 계열 API가 이를 거부함. Ollama는 관대해서 몇 달간 드러나지 않음 | `invalid_tool_calls`에도 오류 ToolMessage로 응답(노드 순서·재요청 흐름은 그대로). 실패했던 지문을 재실행해 400 없이 통과 확인, 회귀 테스트 추가 |
| 합답형·자료형 문항의 정답유일성을 Judge가 판단할 근거가 없음 | `judge_one()`이 Judge에게 발문·선지·정답만 보내고 `<보기>`·자료(stimulus)를 빠뜨림. 기존 골든셋 30건은 전부 `<보기>` 없는 단순 객관식이라 드러나지 않음 | stimulus가 있으면 Judge 입력에 포함(없는 문항은 기존과 같은 입력). 새 평가셋은 절반이 합답형·자료형 |
| main CI가 **2026-09-24부터 연속 실패** | `FlagEmbedding==1.2.11`이 내부에서 `peft`를 import하지만 의존성에 선언하지 않음. 로컬 `.venv`에는 우연히 설치돼 있어 로컬 테스트는 통과 | `requirements.txt`에 `peft==0.19.1` 고정. 새로 끌어오는 패키지 없음, CI 289개 통과 확인 |
| VLM 그림 서술 Judge 사람 라벨 90건이 **5점·1점에만 몰려** κ 0.95가 나왔음 — 중간 품질을 구분하는지는 검증된 적이 없었음 | 라벨 분포를 보니 2~3점이 0건. 핵심 수치 오류처럼 "중간 정도로 나쁜" 서술에도 Judge가 높은 점수를 줄 가능성을 이 데이터로는 배제할 수 없었음 | 결함 40건(핵심 수치 변경·단위 삭제 등)을 인공 주입해 중간 구간을 채움. 합산 130건에서 κ가 0.89로 내려갔고, **핵심 수치 오류를 약하게 보는 것**(사람 2.0점을 luna는 3.1점)을 발견해 품질 감시 경보 기준을 "3점 이하"로 다시 잡음 |
| `qwen3-vl:8b` 기본 태그로 VLM 추출을 돌리면 그림 61장 중 **25장이 빈 응답** | 이 태그는 생각(thinking) 버전이라, 출력 한도 2048토큰을 생각 과정에 다 써서 본문이 안 나옴(`think=false` 지정도 무시됨) | `qwen3-vl:8b-instruct`로 교체 — 추출에는 생각 과정이 필요 없음 |
| 2026-10 생성 모델 비교(33세트) 내내 로컬 검색 백엔드가 **고장나 있었는데 드러나지 않음**(검색 제거 실험을 준비하다 발견) | 로컬 `.env`의 `CHROMA_PERSIST_DIR`이 배포 경로(`/data`)라 `RAGStore` 초기화가 `OSError`로 실패했는데, `agent_node`가 도구 실행 예외를 메시지로만 돌려줘 로그에 남지 않았음 | 로컬 경로 자동 전환 헬퍼(`evals/local_env.py`)를 평가·실험 스크립트 16개에 적용, 도구 실행 예외를 이름·타입만 warning으로 로그, 생성 하네스에 세트별 `tool_errors` 집계 추가 |

---

## 데이터

| 컬렉션 | 경로 | 출처 | 용도 |
|---|---|---|---|
| `regulations` | `data/regulations/` | 학교생활기록부 종합지원포털 | **검색 eval 전용** — 생기부 모듈 제거(2026-08-03) 후 런타임에서는 조회하지 않으나, `retrieval_golden_final.json` 22건 중 10건이 이 컬렉션이라 Recall@5 히스토리 연속성을 위해 유지 |
| `standards` | `data/standards/` | 국가교육과정정보센터(NCIC) | **검색 eval 전용**(2026-10-07부터) — `search_standards` 도구가 TOOLS에서 빠져 런타임 생성 경로는 조회하지 않으나, Recall@5·MRR 검색 평가와 실험 재현용으로 유지 |

> `past_exams` 컬렉션(수능·모평 기출)은 리디자인 때 완전히 제거했습니다. `check_duplicate`를 폐기했고, 2028 수능 개편으로 과목별 구조 자체가 의미를 잃었기 때문입니다.

**여기 있는 데이터는 전부 공개 자료입니다.** 실제 학생 데이터는 어떤 형태로도 넣지 않으며, 평가용 골든셋은 모두 합성 또는 익명 데이터입니다([설계 원칙 3](#설계-원칙)). 교사가 붙여넣는 예시 문제(`passage_text`)는 ChromaDB에 적재하지 않고 요청을 처리하는 동안에만 다룹니다.

---

## 디렉토리 구조

```
bunpil/
├── app/
│   ├── common/
│   │   ├── llm/          # LLM 추상화 (OllamaBackend / RunPodBackend / OpenAIBackend / ChatRunPod / OpenAIVLMBackend)
│   │   └── rag/          # PDF 파싱, 임베딩, 리랭킹, ChromaDB
│   ├── modules/
│   │   ├── exam/         # 출제 모듈 — graph.py(LangGraph) / tools.py(도구 4개) / judge.py(Judge 채점 함수)
│   └── main.py           # FastAPI (/exam/stream · /exam · /exam/extract · /exam/explain · /health)
├── frontend/             # Next.js UI
├── data/
│   ├── regulations/      # 생기부 기재요령·훈령 (검색 eval 전용 — 런타임 미사용)
│   ├── standards/        # 사회과 교육과정 PDF
│   └── golden/           # 골든셋 JSON — 정기 평가용 5종 + 실험 아카이브
│                         # (파일별 용도·라벨 필드는 data/golden/README.md 참고)
├── evals/                # 품질 평가 — eval_exam.py / eval_ragas.py (+ 공용 eval_lib.py)는 정기 실행
│                         # eval_trajectory.py는 산출물이 아닌 과정(궤적) 집계 — LangSmith 트레이스만 읽음
│                         # eval_vlm.py(2026-08-20)·eval_vlm_compare.py(2026-10, 모델 선정용)는
│                         # 실행마다 실제 VLM·Judge API를 호출해 비용이 들어 정기 자동 실행 대상에
│                         # 넣지 않음 — 필요할 때 수동 실행(MODEL_SELECTION.md 7절, EVAL.md 29절)
├── golden_gen/           # 골든셋 생성 도구 — gen_structure_golden.py / gen_golden_retrieval.py /
│                         # gen_vlm_golden.py(합성 시험 문제 이미지, 2026-08-20 40장 PIL 렌더링 →
│                         # 2026-10 그림 90장으로 확대해 총 115장)
├── experiments/          # 일회성 실험·비교 기록 (compare_*.py 등, 결과는 data/golden/_*.json에 아카이브)
├── tests/                # 유닛 테스트 410개 (LLM 호출 없음 — pytest tests/)
├── scripts/
│   ├── index_*.py        # RAG 컬렉션 인덱싱
│   └── test_*.py         # 실제 로컬 모델로 파이프라인 배선 확인 (스모크 테스트)
├── runpod_handler/       # RunPod 서버리스 핸들러 (Qwen2.5-14B-AWQ vLLM)
├── deploy/               # EC2·Caddy·빌링알람 프로비저닝 스크립트
├── assets/               # README·문서 구성도 SVG (라이트/다크 쌍), 원본 JSON은 assets/diagram-src/
├── docs/                 # 상세 문서(ARCHITECTURE·EVAL_DETAILS·DEPLOY), 평가 대시보드(eval_overview.html)
├── Dockerfile
├── docker-compose.yml
└── Caddyfile
```

---

