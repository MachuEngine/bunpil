# 모델 선정 실행 계획 — 생성 모델 · LLM Judge

> 작성 2026-09-24. 기준은 `docs/model-selection-requirements.md`, 후보는 `evals/candidate-models.csv`.
> 이 문서는 **무엇을 어떤 순서로, 얼마의 비용으로** 측정할지 정한다. 결과는 `docs/model-selection-report.md`.

## 1. 파일럿 후보

| 역할 | 생성 모델 | Judge 모델 |
|---|---|---|
| 기준(현재) | `qwen2.5:14b` — **로컬 Ollama**(OpenRouter에 14B 없음, 지연은 비교 불가로 표시) | `openai/gpt-5.6-luna` @ `openai` |
| 품질 상한 | `openai/gpt-6-sol` @ `openai` | `anthropic/claude-opus-5.5` @ `anthropic` (다른 제공사) |
| 두 번째 강한 모델 | — | `openai/gpt-6-sol` @ `openai` (같은 계열 편향 측정 겸) |
| 비용 효율 | `openai/gpt-6-luna` @ `openai` | `google/gemini-3.8-flash` @ `google-ai-studio` |
| 저지연·세 번째 제공사 | `google/gemini-3.8-flash` @ `google-ai-studio` | — |
| 자체 호스팅급 오픈 모델 | `qwen/qwen3.8-27b` @ `deepinfra/bf16` | `qwen/qwen3.8-27b` @ `deepinfra/bf16` |

**2026-09-24 변경 — OpenRouter를 기본 접근 계층으로 사용**(사용자 지시). `모델 @ endpoint`는 OpenRouter 모델 ID와
고정 provider endpoint다. 선정 근거(2026-09-24 `GET /api/v1/models/{id}/endpoints` 조회):
제공사 직접 endpoint 우선, flex 등급 제외(우선순위가 낮아 지연 변동), 오픈 모델은 원래 정밀도(bf16) 고정.
gpt-6 계열·claude-opus-5.5(`anthropic`)는 `temperature`를 지원 파라미터로 올리지 않아 보내지 않는다(아래 5절).

- 생성 5개, Judge 5개. 제외 사유는 CSV의 `inclusion_or_exclusion_reason`.
- **Judge는 채점 대상 생성 모델과 같은 모델을 쓰지 않는다.** 같은 계열(gpt-6-sol Judge ↔ gpt-6-luna 생성,
  qwen3.8 Judge ↔ qwen 생성)은 허용하되 자기 계열 선호를 따로 측정한다.

## 2. 순서 — Judge 검증이 생성 비교보다 먼저

생성 품질 지표 대부분(Q1·Q2·Q4·Q5)을 Judge가 채점하므로, **Judge를 먼저 검증하지 않으면 생성 비교 결과를
믿을 수 없다.** 다만 Judge를 검증하려면 채점할 생성물이 필요하므로 아래 순서로 푼다.

```
단계 A  데이터·rubric·하네스 준비 (비용 0)
단계 B  라벨링 풀 생성 — 로컬 모델 2개 + gpt-6-luna로 파일럿 입력을 생성 (≈ $1)
        + 결함 주입(정답 키 교체·복수 정답·사실 오류·형식 위반)으로 정답을 아는 사례 추가
단계 C  사람 gold label — 2인 독립, 150~200건                        (사람 작업)
단계 D  Judge 파일럿 5개 × gold label → Judge 탈락·선정            (≈ $8)
단계 E  생성 파일럿 5개 × 파일럿 입력 40개 → hard gate·지배 후보 제거 (≈ $6)
단계 F  holdout 정식 평가 — 통과한 생성 모델 × test 입력 40개 × 3회   (≈ $20~30)
단계 G  Pareto·가중 점수·sensitivity → 보고서
```

**승인 지점**: 단계 B(첫 유료 호출), 단계 D, 단계 E, 단계 F는 각각 실행 전에 예상 비용을 보여주고 승인을 받는다.

## 3. 데이터

### 3.1 입력 세트 (생성 평가용)

| 파일 | 용도 | 크기 | 조정에 사용 |
|---|---|---|---|
| `evals/data/pilot.jsonl` | development — 프롬프트·설정·rubric 조정 허용 | 입력 48개 | ✅ |
| `evals/data/test.jsonl` | holdout — 최종 비교 전용 | 입력 48개 | ❌ (조정 금지) |

두 파일 모두 **합성**(하드룰 1)이며 각 행에 `"synthetic": true`. 구성은 같은 분포로 맞춘다.

| 축 | 값 |
|---|---|
| 예시 형식(5종 × 8) | 4지 단순, 5지 단순, `<보기>` 합답형, 자료 제시형, 서술형 |
| 과목 분포 | 통합사회, 정치와 법, 경제, 사회·문화, 한국지리/세계지리, 윤리 |
| 예상 밖 유형(2026-09-24 추가, split별 8) | OX·빈칸·연결형·범위 밖 과목(미지원 → 가까운 형식 + 안내), 순서 배열·(가)(나) 텍스트 자료·본문 ⑤·영어 지문(지원 형식으로 올바르게 인식) |
| 수정 요청(입력마다 3개) | 수정 1 + (범위 밖·형식 변경·injection 중 1) + (질문·인사 중 1) — 의도 분류 확인 |
| 사례 유형(기본 입력 40개에 배분) | 정상(16), 어려운 요청(6: 여러 형식 혼합·문항 5개 요청), 경계(4: 예시 1문항에 문항 3개 요청 등), 긴 입력(4: 5,000자 이상 — split 하나의 예시 40개를 이어 붙인 시험지 전체 붙여넣기, split 간 문항 공유 없이 만들 수 있는 상한이 약 5,400자), 모호한 요청(3: 과목·형식 불명), 안전·정책 경계(3: PII 포함·특정 집단 비하 요청), prompt injection(4: 예시 안에 "지시를 무시하고…") |

### 3.2 Judge gold label 세트

| 출처 | 건수(목표) | 정답을 아는 방법 |
|---|---|---|
| 단계 B 생성물(여러 모델) | 100~140 | 사람 2인 라벨 |
| 결함 주입 사례 | 40~60 | 주입한 결함이 곧 라벨(사람 1인 확인) |
| — 유창하지만 사실이 틀린 해설 / 지나치게 길고 핵심 없는 해설 / 형식 위반 / 평가 조작 문구("이 문항은 5점") 포함 | 위 결함 주입에 포함 | |

사람 간 불일치 항목은 하나로 합치지 않고 **ambiguous subset**으로 따로 둔다(`evals/annotation-guide.md`).

### 3.3 공개 기출 사용 여부 — 보류

정답이 확정된 문항(Judge 독립 풀이 정확도 JQ4)에 공개 기출을 쓰면 사람 부담이 크게 준다. 이용 조건을 확인한
뒤 사용자에게 별도로 확인한다. 확인 전까지는 결함 주입 사례와 사람 확정 정답으로 대신한다.

## 4. 하네스 설계 (`evals/model_selection/`)

기존 프로덕션 코드(`app/`)와 골든셋(`data/golden/`)은 **수정하지 않는다.** 필요한 경우 모델 선택만 하네스
안에서 주입한다(monkeypatch 방식 — 기존 `experiments/compare_models.py`의 env 교체 패턴을 확장).

| 기능 | 구현 |
|---|---|
| 접근 계층 | **OpenRouter**(OpenAI 호환 API) — 단발 작업(G-T2~T4, Judge)은 `openai` SDK, G-T1 에이전트는 이미 프로젝트 의존성인 `langchain-openai`의 `ChatOpenAI`에 OpenRouter base URL을 지정. 새 LangChain 패키지 없음. 로컬 기준 모델만 Ollama adapter. 벤더 공식 API 재검증용 adapter를 끼울 수 있게 adapter 인터페이스로 분리 |
| 요청 고정 | `provider: {order: [endpoint], allow_fallbacks: false, require_parameters: true, quantizations: [..](오픈 모델)}`, `plugins: [{id: "context-compression", enabled: false}]`, 모델 ID는 정확한 슬러그(`auto`·`latest`·`:free` 금지), 모델 fallback(`models` 배열) 미사용 |
| 실제 처리 검증 | 응답의 `model`·`provider`와 `GET /api/v1/generation?id=`의 `provider_name`·`total_cost`를 기록. **경로가 확인된(route_ok=True) 결과만 집계**하고, 다른 provider(`mismatch`)와 확인 불가(`unverified`)는 제외해 건수를 따로 보고한다. 에이전트 경로(LangChain)는 응답에 provider가 없어 `generate`·`score`·`judge-validate`가 끝날 때 `/generation` 조회를 자동 실행해 채우고, 에이전트 비용도 이때 추정치에서 실청구액으로 바꾼다. 생성 결과·Judge 판정 양쪽에 같은 규칙을 적용한다 |
| 인증 | `OPENROUTER_API_KEY` 환경변수만 읽음. 설정·로그·결과 파일에 기록하지 않음 |
| 모델별 설정 | `evals/model_selection/config.json` — model_id, endpoint, quantization, 보낼 파라미터, 단가 스냅샷 |
| 기록 | 실행마다 prompt 해시·모델 ID·설정·코드 커밋을 결과에 저장 |
| 원시 결과·오류 | `evals/results/model_selection/<run_id>/raw.jsonl`, `errors.jsonl` (gitignore) |
| 재시도·timeout | 지수 백오프 3회, 호출 timeout 120s. 재시도 횟수는 C4 지표로 집계 |
| 지연·토큰·비용 | 호출별 wall time, 입력·출력 토큰(응답 usage), 단가표로 비용 계산 |
| 결정론적 채점 | 형식 게이트(`_format_errors` 재사용), JSON 스키마, 개수 exact match, 한국어·PII 검사 |
| LLM Judge | rubric 프롬프트(`evals/rubrics/`), JSON 출력 검증, 실패 시 1회 재요청 후 `parse_error`로 기록(조용한 기본값 금지) |
| 캐시 | (model_id, 설정, prompt 해시, 입력 ID, 반복 번호) 키로 응답 캐시 → 같은 조건 재실행 시 재호출 없음 |
| 익명화 pairwise | Judge 입력에 "응답 A/B"만 노출. 모델명·제공사 제거, 응답 안의 자기소개 문구도 제거 |
| 위치 교환 | 모든 pairwise를 A/B, B/A 두 번 채점해 일치율 계산 |
| 내보내기 | `summary.csv`, `summary.json`, 문항별 `details.csv` |
| dry-run / mock | `--dry-run`: 호출 없이 예상 토큰·비용만 계산. `--mock`: 가짜 adapter로 전체 파이프라인 검증 |
| 비용 상한 | 실행마다 `--max-cost` 필수. 누적 비용이 넘으면 즉시 중단 |

**의존성**: 새 LangChain 패키지는 추가하지 않는다. 평가 전용으로 쓰는 통계 패키지(`scipy`, `scikit-learn`)만
`requirements-eval.txt`에 적는다(프로덕션 `requirements.txt`는 건드리지 않음).

**데이터 보존 설정**: 평가 데이터는 전부 합성이라 민감 데이터가 아니다. 그래도 요청마다 `data_collection`과
`zdr` 설정값을 결과에 기록하고, 민감 데이터 평가 모드(`sensitive: true`)에서는 `data_collection: "deny"`,
`zdr: true`를 강제해 적격 endpoint가 없으면 요청을 실패 처리한다.

## 5. 고정 조건

| 항목 | 생성 비교 | Judge |
|---|---|---|
| temperature | 0.7 (프로덕션과 동일) — **endpoint가 지원할 때만** 보낸다. gpt-6 계열은 미지원이라 모델 기본값으로 실행되며 결과에 표시. 반복 3회로 분산 측정 | 0 (지원하면). 지원 안 하면 기록 |
| reasoning effort / thinking | 최소(none/low 또는 off) — L1(60s) 조건 때문. 상한 후보는 별도로 medium 1회 참고 측정 | 모델 기본값의 가장 낮은 단계, 동일하게 고정 |
| max tokens | 프로덕션 값(생성 2048, 해설 1024, 수정 2048) | 1024 |
| 재시도 예산 | 프로덕션 budget=5 | — |
| 실행 순서 | 모델을 입력 단위로 **번갈아** 실행(배치 효과 방지, EVAL.md 24절 교훈) | 항목 순서 무작위 |

## 6. 지표 계산

- 비율 지표: 입력 예시 단위 **부트스트랩 95% 신뢰구간**(1,000회).
- 모델 비교: 같은 입력 짝 비교 — 비율은 McNemar, 점수는 Wilcoxon 부호순위. 다중 비교는 Holm 보정.
- Judge 일치도: weighted κ(순서형), macro F1(pass/fail), Spearman(점수형), pairwise accuracy, 사람 간 κ를 상한으로 함께 보고.
  ambiguous subset은 주 지표에서 빼고 별도 보고.
- 편향: 길이 편향(응답 길이와 점수의 편상관), 위치 편향(A/B 교환 불일치율), 자기 계열 선호(같은 계열 응답에 대한 승률 − 사람 기준 승률).

## 7. 판단 순서와 가중치

1. Hard gate 위반 제거 → 2. 치명적 오류율 기준 위반 제거 → 3. 개별 지표 비교 → 4. Pareto(품질 × 성공 1건당 비용 × p95 지연)
→ 5. trade-off 서술 → 6. 가중 점수 → 7. 가중치 ±20% sensitivity → 8. 운영 위험·공급사 종속 검토.

| 생성 모델 | 가중치 | 조정 근거 |
|---|---|---|
| 작업 성공률·품질 (Q1·Q3·Q4·Q5) | 40% | 기본값 유지 |
| 치명적 오류·안전 (X1~X7) | 25% | 기본값 유지 — 정답 키 오류는 가장 큰 손실 |
| 지시·형식 준수 (F6·Q3 세부) | 15% | 기본값 유지 |
| 비용 (C1) | 10% | 기본값 유지 — C2가 hard gate로 상한을 이미 막음 |
| 지연·안정성 (L1~L3·C4) | 10% | 기본값 유지 — L1이 hard gate |

| Judge 모델 | 가중치 | 조정 근거 |
|---|---|---|
| 인간 판단 일치도 (JQ1) | 35% | 기본값 |
| 치명적 오류 탐지 (JQ2·JQ4·JQ5) | 25% | 기본값 — 정답 키 검증(JQ4)을 포함 |
| 편향 내성 (JQ7) | 15% | 기본값 — 현 Judge의 +0.94 편향 이력 |
| 반복 안정성 (JQ6) | 10% | 기본값 |
| 도메인·언어 적합성 | 10% | 기본값 |
| 비용·지연 | 5% | 기본값 — 평가량이 작음 |

## 8. 예산 추정 (총 한도 $50)

> ⚠️ **추정치다.** 유일한 실측 근거는 GPT-4o-mini로 15세트를 만들 때의 토큰 수(세트당 입력 약 22K,
> 출력 약 0.9K, budget=1, EVAL.md:665-680)다. 재시도(budget=5)를 고려해 세트당 입력 35K·출력 2K로 잡았다.
> 단계 B의 실측 토큰으로 이후 단계 견적을 다시 계산해 승인받는다.

**2026-09-24 하네스 견적으로 갱신**(`run.py estimate`, 최대 = 추정 × 1.5, 런타임 Judge 비용 포함):
파일럿 입력 40개 × 1회 기준 최대 비용은 gpt-6-sol $9.77, gemini-3.8-flash $3.72, qwen3.8-27b $1.25,
gpt-6-luna $0.57이다. 처음 표(단계 E ≈ $6)보다 커서 아래처럼 조정한다.

| 단계 | 내용 | 최대 추정 |
|---|---|---|
| B | 라벨링 풀: gpt-6-luna·qwen3.8-27b(OpenRouter) × 파일럿 40 + qwen2.5-14b(로컬, 무료) | ≈ $1.8 |
| D | Judge 5개 × gold 150~200건 × (반복 2 + 위치 교환) | ≈ $8 (단계 C 후 재견적) |
| E | 생성 파일럿: gemini-3.8-flash 40 + **gpt-6-sol 20(절반)**, B에서 만든 luna·qwen 결과 재사용 | ≈ $8.6 |
| F | holdout: 통과 모델만 × test 40 × 3회 | 단계 E 실측으로 재견적(남은 예산 ≈ $30 이내) |
| 예비 | 재실행·조정 | ≈ $2 |

gpt-6-sol은 파일럿을 절반(20개)만 돌린다. 다른 후보를 **지배하지 못하면** holdout에서 빼고 상한 참고값만 남긴다.

## 9. 단계별 완료 기준

| 단계 | 완료 기준 |
|---|---|
| A | mock으로 전체 파이프라인(생성→결정론 채점→Judge→집계→CSV)이 통과, `pytest` 통과 |
| B | 라벨링 풀 150건 이상, 형식 5종이 모두 포함 |
| C | 2인 독립 라벨 완료, 사람 간 κ 산출, ambiguous subset 분리 |
| D | Judge별 JQ1~JQ8 산출, 기본·보조 Judge 후보 결정 |
| E | 생성 후보별 hard gate 판정, 지배 후보 제거 사유 기록 |
| F | holdout 결과에 신뢰구간, Pareto, sensitivity |
| G | 보고서 15개 절과 최종 추천 형식 완성 |
