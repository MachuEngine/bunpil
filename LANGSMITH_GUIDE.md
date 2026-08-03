# LangSmith 보는 법 & 팁

> 이 프로젝트에서 LangSmith를 언제·어떻게 켜고, 웹 UI에서 뭘 봐야 하는지 정리한 실전 가이드.
> 설계 이유(dev/prod 분리 등)는 코드 주석에 이미 있으므로, 여기서는 "화면 어디를 보면 되는가"에 집중한다.

## 1. 가장 먼저 알아야 할 것 — 트레이싱은 옵트인

**출제 모듈은 2026-07-24부터 프로덕션에서도 트레이싱 가능.** `passage_text`·생성 문항은
실존 인물 정보가 아니고 PII 마스킹(하드룰 2)도 LLM 호출 전에 이미 거치므로, 관측성 확보를
위해 사용자 승인 하에 하드룰 3의 예외로 지정됐다(CLAUDE.md 참고). `.env`에
`LANGCHAIN_TRACING_V2=true`를 켜두면 `app/main.py`가 `init_langsmith_project()`를 호출해
실제 API 서버(`agent_node`의 ChatOllama/ChatRunPod, `judge_node`)도 트레이싱된다. 기본값은
`false`(옵트인) — 켜려면 `LANGCHAIN_API_KEY`도 `.env`에 필요.

합성 골든셋만 다루는 `evals/`/`experiments/` 스크립트는 원래대로 각 진입점에서 tracing을
별도로 초기화한다(셸에서 직접 실행할 때만):

```bash
LANGCHAIN_TRACING_V2=true LANGCHAIN_API_KEY=your_key python evals/eval_exam.py
```

`LANGCHAIN_API_KEY`는 [smith.langchain.com](https://smith.langchain.com) 로그인 후
Settings → API Keys에서 발급. `.env`에 미리 넣어두면 매번 셸에 안 쳐도 되지만,
그러면 실수로 로컬 개발 중에도 계속 트레이싱될 수 있으니 이 프로젝트는 의도적으로
**필요할 때만 셸 변수로 켜는 방식**을 쓴다(`.env.example`의 기본값은 `false`).

## 2. 프로젝트 이름 — bunpil-dev vs bunpil-prod

`app/common/llm/tracing.py`의 `init_langsmith_project()`가 스크립트 시작 시 자동으로 분기한다:

| 실행 시 `LLM_BACKEND` | 실제 LangSmith 프로젝트명 |
|---|---|
| `local` (Ollama) | `bunpil-dev` |
| `runpod` / `openai` | `bunpil-prod` |

`LANGCHAIN_PROJECT`를 `.env`에서 건드리지 않았다면(기본값 `bunpil`) 위처럼 자동으로
`-dev`/`-prod` 접미사가 붙는다. **로컬에서 eval을 돌렸다면 LangSmith 왼쪽 프로젝트
목록에서 `bunpil-dev`를 먼저 찾을 것** — `bunpil`이나 `bunpil-prod`엔 아무것도 없어서
"안 되나?" 하고 헷갈리기 쉬운 지점.

> `LANGCHAIN_PROJECT`에 `bunpil`이 아닌 값을 직접 설정하면 이 자동 분기를 건너뛰고
> 그 값을 그대로 쓴다(override).

## 3. 웹 UI — 두 개의 다른 화면을 구분할 것

### 3.1 Traces 탭 — "이번 실행에 무슨 일이 있었나" (디버깅용)

프로젝트(`bunpil-dev` 등) 클릭 → Traces 탭. LangGraph 노드(`plan`/`agent`/`judge`/`validate`)와
그 안의 LLM 호출·tool 호출(`search_standards`, `save_item`, `submit_for_review` 등)이
트리로 펼쳐진다. 각 노드를 클릭하면 실제로 모델에 들어간 프롬프트 전문과 응답을 그대로
볼 수 있다. 2026-07-23부터 `judge` 노드는 생성 모델과 별개인 `get_judge_backend()`를
호출하므로, 트레이스에서도 생성 호출과 채점 호출이 별개 항목으로 보인다.

**언제 쓰나**: "왜 이 문항이 이렇게 나왔지", "재시도가 왜 3번이나 돌았지" 같은 개별
실행을 파고들 때. `eval_exam.py`/`compare_models.py` 등을 실행한 직후 확인하면 된다.

**필터 팁**: 트레이스마다 `metadata`에 `model`(`OLLAMA_MODEL` 값)·`backend`(`LLM_BACKEND` 값)가
붙어 있다(`evals/eval_lib.py:28`) — 여러 모델을 번갈아 비교 실행했다면 이 메타데이터로
필터링해서 "이건 14B 실행분만" 식으로 골라볼 수 있다.

### 3.2 Datasets & Experiments 탭 — "이번 변경이 나아졌나" (품질 추적용)

`evals/langsmith_experiments.py`가 골든셋 JSON을 LangSmith Dataset으로 동기화해둔
것이 3개 있다:

| Dataset 이름 | 골든셋 소스 | 무엇을 채점하나 |
|---|---|---|
| `bunpil-item-quality-judge` | `data/golden/item_golden.json` (30개) | 정답유일성·오답매력도·근거성 |
| `bunpil-structure-judge` | `data/golden/structure_golden.json` (45개, human_label 채워진 것만) | 구조 유사도(overall/type_ratio/difficulty) — `get_judge_backend()`로 채점. 2026-07-23부터 런타임 `judge` 노드와 완전히 같은 코드(`app/modules/exam/judge.py`)라 이 수치가 곧 배포된 judge의 신뢰도 |
| `bunpil-rag-quality` | `gen_structure_golden.PASSAGE_SAMPLES` 기준 실제 생성 | Faithfulness / Answer Relevancy |

`eval_exam.py`/`eval_ragas.py`를 트레이싱 켜고 실행할 때마다 각 Dataset에
**새 Experiment(실행 회차)**가 하나씩 쌓인다(`experiment_prefix`: `item-quality-judge` /
`structure-judge` / `rag-quality`). Dataset 페이지에서 여러 Experiment를 체크박스로
선택하면 **회차별 점수가 표(컬럼)로 나란히** 나온다 — "3점 앵커 few-shot 넣기 전/후
kappa가 어떻게 바뀌었나"를 EVAL.md에 손으로 옮겨 적지 않고 여기서 바로 비교할 수 있다.

> **주의**: `item-quality-judge`/`structure-judge`의 Dataset은 **실행할 때마다 삭제 후
> 재생성**된다(`sync_dataset()`, 골든셋 JSON이 항상 단일 진실 공급원이라는 원칙 때문 —
> 사람이 라벨을 재검토해 바꿨는데 Dataset이 옛날 값을 들고 있는 불일치를 막기 위함).
> 즉 **과거 Experiment 자체는 남지만, Dataset 예제 목록은 항상 "최신 골든셋"** 이라는
> 뜻 — Dataset 항목 개수가 실행마다 달라 보여도 버그가 아니다.

### 3.3 트레이스를 눈으로 보지 않고 집계하기 — `eval_trajectory.py`

3.1의 Traces 탭은 **개별 실행 하나**를 파고들 때 쓴다. 반대로 "지난 한 달 동안 도구가
몇 번이나 거부됐나", "재시도 원인이 형식 문제였나 Judge 판단 문제였나" 같은 **분포**를
보려면 트레이스를 하나씩 열어선 답이 안 나온다.

```bash
python evals/eval_trajectory.py --since 2026-07-23
python evals/eval_trajectory.py --days 30 --json /tmp/traj.json
```

LangSmith API를 읽기만 하고 모델은 호출하지 않는다(비용 없음, 단 무료 플랜 조회 한도는
소모). 집계 대상은 `LANGCHAIN_PROJECT` 환경변수가 가리키는 프로젝트이며, `--project`로
직접 지정할 수도 있다(`bunpil-dev` / `bunpil-prod`).

> **`--since`를 쓰는 이유**: 2026-07-23 self-judge 폐기로 `similarity_judge` 도구가
> 사라지고 `submit_for_review`가 도입됐다. 그냥 `--days 30`으로 조회하면 그 리팩터
> 전후 트레이스가 한 통계에 섞여 "지금 코드의 성능"을 잘못 읽게 된다(실제로 첫 실행에서
> 이 혼재가 발견됐다 — EVAL.md 11절). **아키텍처를 바꾼 날짜를 `--since`로 주는 습관**을
> 들일 것.

지표 정의와 해석은 EVAL.md 11절 참고. 집계는 트레이스의 **카운트·카테고리만** 뽑고
`passage_text`·생성 문항 원문은 콘솔에도 JSON에도 쓰지 않는다(하드룰 4).

## 4. 자주 헷갈리는 것들

- **결정론적 지표(Recall@5, PII 마스킹, 키워드 사실추가율)는 LangSmith에 없다** — 이 3개는
  프롬프트/모델 변화에 흔들리지 않는 함수 지표라 Experiments 연동 대상에서 의도적으로
  제외됐다(EVAL.md 9절). 이 수치는 스크립트 콘솔 출력과 EVAL.md에서만 확인 가능.
- **EVAL.md vs LangSmith, 뭘 믿어야 하나**: Judge 기반 3개 지표(문항품질/구조유사도/RAG
  품질)는 **LangSmith Experiments가 최신 소스**(매 실행 자동 기록, 사람이 옮겨 적다 생기는
  오류 없음). EVAL.md는 그 3개의 세세한 회차 로그보다 "왜 이 프롬프트를 이렇게 바꿨는지"
  같은 **의사결정 서사** 기록용으로 쓴다(EVAL.md 상단 안내 참고).
- **Judge 모델을 gpt-5.6-luna로 바꾼 뒤 실행하면?** — `JUDGE_BACKEND`는 트레이싱 프로젝트
  분기(`LLM_BACKEND` 기준)와는 무관하다. 즉 로컬에서 `LLM_BACKEND=local`로 생성 모델은
  Ollama를 쓰면서 `JUDGE_BACKEND=openai`로 채점만 gpt-5.6-luna를 써도 프로젝트는 그대로
  `bunpil-dev`에 남는다 — Judge 백엔드 교체가 dev/prod 분류에 영향을 주지 않는다.
- **아무 프로젝트에도 안 보임**: `LANGCHAIN_TRACING_V2=true`를 셸에 안 넣고 실행했을 가능성이
  제일 흔하다. 스크립트 콘솔에 `"LangSmith 트레이싱: 활성화됨"`이 찍혔는지 먼저 확인
  (`eval_exam.py:184`).

## 5. 관련 코드 위치

| 파일 | 역할 |
|---|---|
| `app/common/llm/tracing.py` | dev/prod 프로젝트 자동 분기 (`init_langsmith_project()`) |
| `evals/langsmith_experiments.py` | Dataset 동기화 공용 유틸(`sync_dataset`, `identity_target`) |
| `evals/eval_trajectory.py` | 트레이스 집계(도구 신뢰도·재시도 원인·궤적 형태) — 3.3절, EVAL.md 11절 |
| `evals/eval_exam.py` `run_langsmith_experiments()` | item-quality-judge / structure-judge 등록 |
| `evals/eval_ragas.py` `run_langsmith_experiments()` | rag-quality 등록 (매 실행 실제 생성 후 채점) |
| `evals/eval_lib.py` `_TRACE_META` | 트레이스에 붙는 `model`/`backend` 메타데이터 |
| `app/modules/exam/judge.py` | 런타임 `judge` 노드와 오프라인 eval이 공유하는 채점 함수(`judge_structure()`) |
