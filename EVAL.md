# 분필(Bunpil) 평가(Eval) 문서

`scripts/eval_exam.py`, `scripts/eval_record.py`가 측정하는 지표, 골든셋 현황, 실행 방법, 결과 이력을 모아둔 참고 문서.
모델 교체·프롬프트 튜닝 등 평가에 영향을 주는 변경이 있을 때마다 [결과 이력](#4-결과-이력)에 행을 추가한다.

## 1. 평가 항목 전체 목록

### 출제 모듈 (`eval_exam.py`)

| 계층 | 지표 | 방식 | 기준 |
|---|---|---|---|
| 검색 | Recall@5 | 함수 | ≥ 0.8 |
| 검색 | MRR | 함수 | 참고값 |
| 문항 품질 | 정답유일성·오답매력도·근거성 | LLM Judge | 평균 ≥ 4.0 |
| Judge 신뢰도 | Cohen's kappa | 사람 라벨 비교 | ≥ 0.4 |
| Judge 신뢰도 | ±1 일치율 | 사람 라벨 비교 | ≥ 0.7 |
| 세트 제약 | 유형/난이도/커버리지/중복률 | 함수 | 100% |

### 생기부 모듈 (`eval_record.py`)

| 지표 | 방식 | 기준 |
|---|---|---|
| PII 마스킹 FN율 | 함수 | = 0 |
| 사실 추가율 (키워드) | 함수 | = 0 |
| 사실 추가율 (NLI Judge) | LLM Judge | = 0 |
| 규정 위반 Recall | 함수 | ≥ 0.95 |

### 추가 예정

| 지표 | 도구 | 상태 |
|---|---|---|
| Faithfulness | Ragas | ⬜ 미착수 |
| Answer Relevancy | Ragas | ⬜ 미착수 |

## 2. 골든셋 현황

| 골든셋 | 경로 | 규모 | 비고 |
|---|---|---|---|
| retrieval_golden | `data/golden/retrieval_golden_final.json` | 28개 | 실데이터 기반, 사람 검수 완료 |
| MASKING_GOLDEN | `scripts/eval_record.py` 내 상수 | 20개 | 합성 |
| VIOLATION_GOLDEN | `scripts/eval_record.py` 내 상수 | 50개 | 위반 25 + 정상 25 |
| HALLUCINATION_GOLDEN | `scripts/eval_record.py` 내 상수 | 20개 | 합성 |
| ITEM_GOLDEN | `scripts/eval_exam.py` 내 상수 | 30개 | human_score 1~5점 분포 |

## 3. 실행 방법

```bash
# 출제 모듈 평가
python scripts/eval_exam.py

# 생기부 모듈 평가
python scripts/eval_record.py
```

Windows 콘솔에서 실행 시 `cp949` 인코딩 오류(`UnicodeEncodeError`)가 날 수 있음 — 실행 전 `chcp 65001` 또는 `set PYTHONIOENCODING=utf-8` 필요.

## 4. 결과 이력

| 날짜 | 모델 | Recall@5 | MRR | 문항품질 | kappa | ±1일치율 | 세트제약 |
|---|---|---|---|---|---|---|---|
| 2025.07 | qwen2.5:1.5b | - | - | - | - | - | - |
| 2025.07 | qwen2.5:7b | 0.679 ❌ | 0.494 | 3.68 ❌ | 0.328 ❌ | 0.800 ✅ | ✅ |
| 2025.07 | qwen2.5:7b (생기부) | - | - | - | - | - | FN=0✅ NLI=0.1❌ 위반Recall=0.84❌ |

> 모델 교체 또는 프롬프트 튜닝 시마다 행 추가

## 5. 개선 계획

| 지표 | 현재 | 목표 | 접근 방법 |
|---|---|---|---|
| Recall@5 | 0.679 | ≥ 0.8 | 리트리버/청킹 튜닝 |
| 오답매력도 | 2.43 | ≥ 4.0 | 출제 프롬프트 튜닝 |
| Cohen's kappa | 0.328 | ≥ 0.4 | Judge 프롬프트 채점 기준 구체화 |
| 규정 위반 Recall | 0.840 | ≥ 0.95 | 위반 탐지 프롬프트 튜닝 또는 규정 RAG 보강 |
| NLI 사실추가율 | 0.100 | = 0 | 오탐 2건 원인 분석 (골든셋 or 프롬프트 문제) |
