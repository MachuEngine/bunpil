# 사회 교사용 AI 어시스턴트 — 설계 스펙

> 포트폴리오 + 지인(고등학교 사회 교사) 실사용을 위한 LLM 서비스.
> 본 문서는 Claude Code 빌드 브리프(`CLAUDE.md`)이자 설계 요약본으로 사용한다.

---

## 1. 개요

- **목적**: 고등학교 사회 교사의 (1) 문제 출제, (2) 생기부 작성을 돕는 대화형 웹 서비스
- **사용 맥락**: 포트폴리오(Agent·RAG·평가·배포 실습) + 지인 교사 1인 실사용
- **모듈**: ② 출제 도우미(Agent), ③ 생기부 윤문 도우미(Chain)
- **보류**: ① 입시 상담 모듈 (최신성·데이터 부담 커서 1차 범위 제외)
- **데이터 원칙**: 실제 학생 데이터 미사용(전부 합성/익명), 사용자 입력 비저장(stateless), 공개 자료 + 소량 큐레이션

---

## 2. 아키텍처

### 모듈 ② 출제 도우미 — ReAct Agent (LangGraph)

과제 정의: *"교사가 붙여넣은 예시 문제의 구성(개수·유형·난이도)을 그대로 반영한 새 문항 세트 작성"*.
(2026.07 리디자인, `FEEDBACK_DRIVEN_REDESIGN_v2.md` — 실사용 교사 피드백: PDF 업로드+유형/난이도/개수
드롭다운 대신 ChatGPT처럼 예시 문제를 붙여넣는 사용 패턴이 실제와 더 맞았음. 2028 수능 개편으로 과목별
구조가 곧 무의미해질 `past_exams`/`check_duplicate`도 이때 완전히 제거)

```
예시 문제 붙여넣기(passage_text) → Agent(ReAct) ↔ 도구 → similarity_judge 구조 유사도 자체 평가
  → 코드가 threshold 판정 → 미달 시 세트 전체 재시도(budget) → 교사 검토 세트
```

**도구(Tools)**

| 도구 | 역할 | 구현 |
|---|---|---|
| 성취기준 검색 | `search_standards` — 성취기준 원문 검색 | ChromaDB + Rerank |
| 법령 검색 | `search_regulations` — 교육과정 준수 사항 검색 | ChromaDB + Rerank |
| 형식 검증 | `validate_item_format` — 문항 형식 자기교정 | 함수 |
| 저장 | `save_item` — 검증 통과 문항 저장 | 함수 |
| 자체 채점 | `record_score` — 품질 자체 평가 기록 | 함수 |
| 구조 유사도 판단 | `similarity_judge` — 예시 문제와의 구조 유사도 자체 평가 | LLM as Judge |

**State**

```
spec:                    { passage_text(예시 문제 원문), standards(성취기준, 선택), num_items(생성 개수, 기본 5) }
draft_items:             [ { 문항, 유형, 난이도, judge_score, 상태 } ]
similarity_judge_result: { type_ratio_score, difficulty_match, overall_score }
budget:                  남은 재시도 횟수 (세트 전체 단위, 무한루프 방지)
```

**Agent(LLM)가 판단하는 것**: 문항 세트 작성, 형식 자기수정, 구조 유사도 자체 평가(`similarity_judge` 호출 — 유형 비율·난이도·종합 유사도만).
**코드가 판단하는 것**: 문항 개수 일치 여부(`len(draft_items) == spec["num_items"]`), `similarity_judge` 결과의 threshold 통과 여부, 재시도 여부.
→ "판단은 LLM, 통과/재시도 결정은 코드"라는 원칙은 리디자인 이후에도 그대로 유지.

> **2026-07-09 정정**: 문항 개수는 예시 문제(`passage_text`)의 문항 수와 무관하게 `num_items`로 별도 지정된다(사용자가 자연어로 명시하지 않으면 기본값 5, `main.py`가 LLM 판단으로 추출). 초기엔 "생성 개수가 예시 문제 개수와 일치해야 한다"는 전제로 `count_match`를 LLM Judge가 판단했으나, 이 전제 자체가 실제 설계와 맞지 않아 폐기 — 개수 일치는 이제 LLM Judge가 아니라 코드가 직접 검증한다.

### 모듈 ③ 생기부 윤문 도우미 — 검증 Chain (LCEL)

```
관찰 메모 입력 → 개인정보 마스킹 → 윤문 생성 → 규정 검증 → 출력 + 책임 고지
                                                  └ 위반 시 윤문으로 재시도
```

- **입력**: 교사가 직접 작성한 관찰 메모만 (학생 작성·제출분 금지)
- **윤문**: 생성이 아닌 "다듬기" — 메모에 없는 사실 추가 금지
- **규정 검증**: 규정 RAG로 학교명 노출·과장·금지표현 대조
- **출력**: 교사 최종 책임 고지(보조수단) 명시

---

## 3. 기술 스택

| 구분 | 선택 |
|---|---|
| 백엔드 | FastAPI (비동기) |
| 오케스트레이션 | LangGraph(출제 agent) / LCEL(생기부 chain) |
| 벡터스토어 | ChromaDB + Rerank (BGE-reranker) |
| 임베딩 | BGE-M3 |
| LLM 서빙 | vLLM + Qwen2.5 (프로덕션) |
| 평가용 모델 | Ollama 소형 / OpenAI GPT-3.5 (합성 데이터 비교 전용) |
| 검증 | LLM as a Judge |
| 프롬프트 | Few-shot / CoT |
| 프론트엔드 | Next.js (frontend/) |

※ 임베딩·리랭킹은 앱 팟(CPU)에서 수행(소규모 코퍼스라 충분), **생성·추론만 서버리스 GPU 호출** → GPU 호출 최소화.

---

## 4. 데이터

| 항목 | 출처 | 방법 | 비고 |
|---|---|---|---|
| 생기부 기재요령 | 학교생활기록부 종합지원포털(star.moe.go.kr) 자료실 | PDF 다운로드 | 규정 RAG 핵심 |
| 학생부 작성·관리 지침(훈령) | 동 포털 | PDF | 규정 RAG |
| 사회과 성취기준 | 국가교육과정정보센터(NCIC) | 문서 조회 | `search_standards` RAG |
| 교사가 붙여넣은 예시 문제 | 교사 런타임 입력(`passage_text`) | 0 | ChromaDB 미적재, 프롬프트에만 사용 후 폐기 |
| 윤문 Few-shot 예시 | 직접 합성 | 가상 시나리오 | 실데이터 금지 |
| 규정 위반 테스트 문장 | 직접 합성 | 위반 일부 심기 | 평가용 |

**공통 파이프라인**: PDF 수집 → PyMuPDF 파싱 → 청킹 → BGE-M3 임베딩 → ChromaDB (메타데이터: 출처·연도 태깅)

**⛔ 절대 수집 금지**: 실제 학생 생기부, 실제 내신 답안, 식별 가능한 학생 정보.

---

## 5. 평가 설계

**원칙**: ① LLM Judge는 사람 라벨과 먼저 일치율 검증 ② 정량(함수)/정성(Judge·사람) 분리 ③ 실제 컬렉션 기반 골든셋.

### 출제 Agent

| 계층 | 지표 | 판정 | 통과 기준(시작값) |
|---|---|---|---|
| 검색 | Recall@5, MRR | 함수 | R@5 ≥ 0.8 |
| 문항 | 정답 유일성·오답 매력도·근거성 | LLM Judge | 5점 척도 평균 ≥ 4.0 (보정 후 확정) |
| 구조 유사도 | type_ratio_score·difficulty_match·overall_score (LLM Judge) + 문항 개수 일치(코드) | LLM Judge(개수 제외) + 코드(개수) | 미정 (부트스트랩 단계) |
| 과정 | 평균 반복수·미충족 실패율·latency | 함수 | 예산 내 수렴 |
| 종단 | 수정 없는 교사 채택률 | 사람 | 북극성 |

검색 골든셋(`data/golden/retrieval_golden_final.json`): `standards` / `regulations` 실제 컬렉션에서 샘플링한 22개 청크(reviewed 21개). `scripts/gen_golden_retrieval.py`로 초안 생성 후 검수. (2026.07 리디자인으로 `past_exams` 참조 8개 제거, 30→22)

### 생기부 Chain (안전 지표 우선)

| 우선 | 지표 | 판정 | 통과 기준 |
|---|---|---|---|
| 🔴 | 마스킹 누락률(FN) | 함수 | 0 |
| 🔴 | 사실 추가율(메모에 없는 내용) | LLM Judge(NLI식) | 0 |
| 🔴 | 규정 위반 검출 Recall/F1 | 함수 | Recall ≥ 0.95 |
| 🟡 | 문체 적합성 | LLM Judge | 5점 척도 평균 ≥ 4.0 |
| 🟢 | 교사 채택률·수정량 | 사람 | 북극성 |

**모델 채택 근거**: 위 평가셋을 Qwen2.5 vs GPT-3.5 vs Ollama로 돌려 정량 비교 → Qwen 채택 근거 확보.

**골든셋 현황**: 출제 검색 22개(21개 검수 완료) + STRUCTURE_GOLDEN 14개(2026-07-09 num_items 아키텍처로 전면 재생성, 실제 qwen2.5:7b 출력, 라벨링 대기) / 생기부(위반문장 50 + 마스킹 20 + 메모→윤문 20). 모든 골든셋은 `data/golden/*.json`으로 외부화(하드코딩 금지).

> **2026-07-09 num_ctx 발견**: STRUCTURE_GOLDEN 재생성 중 로컬 Ollama가 기본 `num_ctx=4096`으로 돌고 있어(모델은 32K 네이티브 지원) 멀티턴 ReAct 루프의 검색 결과 누적이 몇 턴 만에 컨텍스트를 초과시키고, 컨텍스트가 잘리며 모델이 시스템 프롬프트를 잃고 응답이 깨지는 문제를 확인함 → `app/modules/exam/llm.py`의 `ChatOllama`에 `num_ctx=16384` 명시로 수정. 동일 passage 재현 테스트로 확인(4096: 0/5문항 → 16384: 5/5문항). RunPod(vLLM)는 `max_model_len` 미지정 시 모델 네이티브 값을 쓰므로 로컬 개발 환경에만 있던 격차로 추정.

---

## 6. 보안 · 개인정보 (Claude Code는 반드시 준수)

- 개인정보 **마스킹은 입력 단계**에서, 외부/모델 호출 전에 수행
- 사용자 입력(생기부 메모·교사가 붙여넣은 예시 문제)은 **비저장 처리** — 영구 저장은 공개 코퍼스뿐
- **로그·캐시에 PII 금지**
- 실데이터 미사용, 전부 합성
- ChromaDB **영구 컬렉션은 공개 자료(규정·성취기준)만**. 교사가 붙여넣은 예시 문제(`passage_text`)는 ChromaDB에 전혀 적재되지 않고 요청 처리 중 프롬프트에만 사용된 후 폐기. 학생 개인정보는 어디에도 미적재
- 생기부 출력에 "교사 최종 책임(보조수단)" 고지 표시

---

## 7. 배포 (확정: AWS EC2(앱) + RunPod 서버리스(GPU))

**구성: 앱은 AWS EC2 상시 가동, GPU 추론만 RunPod 서버리스**

```
브라우저 ─→ [앱] AWS EC2 (t3.small, CPU)            ─→ [GPU] RunPod Serverless
              FastAPI + Agent + ChromaDB                   Qwen2.5 7B / vLLM
              · PII 마스킹 후 추론 호출                     · 쓸 때만 과금, 유휴 시 0
              · ChromaDB는 EBS 볼륨에 저장                  · 콜드스타트 수초~수십초
```

- **앱 = AWS EC2** (t3.small, ~2GB): FastAPI·agent·ChromaDB 구동 (UI는 Next.js, `frontend/`). ChromaDB는 EBS 볼륨에 영구 저장. IAM·보안그룹·SSH·Docker 표준 배포 절차를 따른다.
- **GPU = RunPod Serverless**: 추론만 요청당 과금, 유휴 시 0. 비싼 GPU 비용만 pay-per-use.
- **HTTPS**: Caddy 리버스 프록시로 자동 발급(+도메인) → 표준 배포 실습 포함.
- 요청 흐름: 브라우저 → EC2(마스킹·오케스트레이션) → RunPod 서버리스 호출 → 응답. 앱 로직 stateless, Chroma만 EBS 영구.
- **billing alarm 필수**: EC2 종량제라 예산 알람 설정. t3는 CPU burst throttle 있으니 임베딩 인덱싱은 한 번에 몰아서.
- **에이전트×서버리스 주의**: 출제 ReAct는 한 요청에 LLM을 여러 번 호출 → 첫 호출만 콜드스타트, 세션 중 워커 warm 유지로 후속 호출은 빠름. 긴 세션은 GPU 워밍 고려.

**운영비 (1인 사용 추정 / 월)**

| 항목 | 비용 |
|---|---|
| EC2 t3.small (상시) | ~$15 |
| RunPod 서버리스 GPU (추론) | ~$1–5 |
| 스토리지(EBS·볼륨 수 GB) | ~$1 |
| **합계** | **~$17–21** |

- 데모·개발 단계는 EC2를 필요할 때만 켜서 더 절감 가능.
- 비용을 더 낮추려면 앱을 Lightsail($5~12 정액)이나 저가 VPS로 이전 가능(단 AWS 학습가치 ↓). 컨테이너화돼 있어 이전은 소규모.

**배포 실습 사다리**: Docker 이미지화 → docker-compose(로컬 검증) → EC2 배포(SSH·보안그룹) → Caddy 리버스 프록시·HTTPS → GitHub Actions 경량 CI(완료, 9절 참고)

---

## 8. 빌드 순서 (MVP)

1. **출제 모듈** — 데이터 부담 0(교사가 예시 문제 붙여넣기), RAG·Judge·Recall@5 바로 적용 → 가장 빠른 데모
2. **생기부 모듈** — 규정 RAG + 마스킹 + 사실보존 검증
3. **배포** — AWS EC2(앱) + RunPod 서버리스(GPU)

**Claude Code 활용 가이드**: 보일러플레이트(스캐폴딩·Docker·UI·글루)는 위임, 배포 단계(EC2·보안그룹·SSH·Caddy HTTPS·RunPod 서버리스 설정)는 학습 목적상 단계 설명 들으며 진행. 본 스펙을 컨텍스트로 제공할 것.

---

## 9. 결정 완료 / 남은 선택

**확정**
- 호스팅: **AWS EC2(앱) + RunPod 서버리스(GPU)**
- 모델: Qwen2.5 7B (서버리스 GPU에 적합, 시작값)
- 운영비: 월 ~$17–21 (1인 기준)
- **GitHub Actions CI** (2026-07-14 결정·구현 완료): 코드 회귀 확인용 **경량 CI만 도입**, LLM
  eval 자동화는 도입하지 않기로 결정
  - **경량 CI** (`.github/workflows/ci.yml`): 매 push/PR 블로킹. 백엔드 import 스모크테스트 +
    순수 로직 유닛테스트(`tests/`, `mask_pii`·`_rule_violations`) + 프론트 lint/build. 모델 호출
    없음(무료·수 분 내 완료)
  - **eval 자동화를 뺀 이유**: 설계 검토 중 `app/common/llm/factory.py`의 `get_judge_backend()`가
    `LLM_BACKEND`와 무관하게 항상 로컬 Ollama를 반환하도록 하드코딩돼 있음을 발견 — 이는
    `compare_models.py`(여러 생성 모델 비교 시 채점 기준을 고정하기 위한 의도된 설계)를 위한 것.
    GitHub Actions 러너엔 GPU/Ollama가 없어 `eval_exam.py`의 Judge 채점 부분(문항 품질·구조유사도·
    신뢰도)이 CI에서 원천적으로 실행 불가하고, 이를 우회하려면 Judge 백엔드 분기 코드를 추가해야
    하는데 그러면 CI가 매기는 점수(OpenAI Judge)가 지금까지 EVAL.md·로드맵에 쌓아온 Ollama 고정
    Judge 기록과 다른 잣대가 돼 추세 비교가 끊김. 더불어 배포 자체가 아직 수동
    (`git pull && docker compose up`)이라 그 앞단 eval만 자동화하는 것도 순서가 안 맞고, eval
    점수는 실행마다 변동성이 있어(STRUCTURE_GOLDEN κ 등) 자동 게이트로 쓰기에도 부적합 — 종합적으로
    이 규모(1인+지인 실사용)엔 지금까지처럼 로컬 수동 실행 + EVAL.md 기록이 더 적합하다고 판단.
    `eval_exam.py`/`eval_record.py`/`eval_ragas.py`는 변경 없이 로컬 실행 스크립트로 유지
  - self-hosted runner(로컬 Ollama 호출)도 인프라 유지 부담 대비 이득이 적어 검토 후 제외

**나중에 정해도 되는 것**
- 비용 절감 시 앱을 Lightsail/저가 VPS로 이전 (AWS 학습가치 ↓)
- 모델 14B 확장 여부 (품질 부족 시)
