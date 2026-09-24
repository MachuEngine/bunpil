# Judge 모델 검증 기준 (judge rubric)

> LLM Judge 후보를 사람 gold label과 비교해 **어떤 Judge를 믿을지** 정하는 기준이다.
> Judge가 무엇을 채점하는지는 `generation-rubric.md`, 사람 라벨 작성법은 `../annotation-guide.md`.

## 1. Judge에게 주는 입력과 출력 형식

- 입력: 예시 문제(마스킹됨), 평가 대상(문항·해설·수정 결과), `generation-rubric.md`의 판정 문항.
  **생성 모델의 이름·제공사는 넣지 않는다.** 응답 본문에 모델 자기소개 문구가 있으면 제거한다.
- 출력: JSON 스키마 고정(structured outputs). 각 판정은 `"yes" | "no" | "unsure"`와 한 줄 근거.
  파싱에 실패하면 1회 재요청하고, 그래도 실패하면 `parse_error`로 기록한다 — **기본값으로 채우지 않는다**
  (기존 `judge.py`는 파싱 실패 시 조용히 0점을 넣어 왔다).
- temperature 0(지원하는 endpoint만). 지원하지 않으면 그 사실을 결과에 기록한다.

## 2. 측정 지표

| ID | 지표 | 계산 | 비고 |
|---|---|---|---|
| JQ1a | pass/fail macro F1 | 문항별 "치명적 오류 없음=pass"를 사람 합의 라벨과 비교 | label 불균형 대응 |
| JQ1b | weighted Cohen's κ | 순서형 판정(예 > 판단 불가 > 아니오)에 quadratic weight | 사람 간 κ를 상한으로 함께 보고 |
| JQ1c | pairwise accuracy | 사람이 고른 쪽과 Judge가 고른 쪽 일치율(동등·판단 불가 제외분 별도 보고) | |
| JQ1d | Spearman ρ | 점수형 항목(진단용 I7)만 | |
| JQ2 | 치명적 오류 탐지 recall | 사람이 X1~X4로 표시한 항목 중 Judge가 잡은 비율 | 가장 중요한 단일 지표 |
| JQ3 | 정상 응답 오탈락률 | 사람이 pass로 본 항목을 Judge가 fail로 본 비율 | |
| JQ4 | 독립 풀이 정확도 | 정답을 숨기고 풀게 한 답 = 확정 정답 | 결함 주입 사례·확정 정답 문항 |
| JQ5 | 형식 위반 탐지율 | 일부러 I4·I5를 어긴 사례를 "아니오"로 본 비율 | |
| JQ6a | 반복 안정성 | 같은 입력 2회 판정 일치율 | |
| JQ6b | 위치 교환 일치율 | pairwise A/B ↔ B/A 결과가 논리적으로 일치하는 비율 | |
| JQ7a | 길이 편향 | 응답 길이와 Judge 선호의 상관 − 사람 선호와의 상관 | |
| JQ7b | 문체 편향 | 같은 내용을 장황체·간결체로 바꾼 쌍에서 장황체 승률 | 결함 주입 세트 |
| JQ7c | 자기 계열 선호 | 같은 계열 생성물에 대한 Judge 승률 − 사람 승률 | gpt-6-sol↔gpt-6-luna, qwen↔qwen |
| JQ8 | 판단 불가 처리 | ambiguous subset에서 `unsure` 비율 vs 명확한 항목에서 `unsure` 비율 | |
| JQ9 | 출력 형식 실패율 | parse_error ÷ 호출 수 | hard gate ≤ 1% |
| JQ10 | 평가당 비용·지연 | OpenRouter 기록 기준 | |

## 3. 해석 규칙

- **percent agreement만 보고하지 않는다.** 모든 일치 지표에 사람 간 일치도를 같이 적는다 —
  사람끼리 κ가 0.5인 항목에서 Judge κ 0.5는 "사람 수준"이다.
- ambiguous subset(사람 2인이 불일치하고 합의에서도 판단 불가로 남은 항목)은 JQ1·JQ2·JQ3 계산에서
  제외하고 JQ8에만 쓴다.
- 신뢰구간은 항목 단위 부트스트랩 95%(1,000회). n이 60 미만인 지표는 "방향성 참고"로 표시한다.

## 4. Judge 합격 조건 (requirements 기준)

| 조건 | 기준 |
|---|---|
| 출력 형식 실패율(JQ9) | ≤ 1% (hard gate) |
| 치명적 오류 탐지 recall(JQ2) | ≥ 0.8 |
| 정상 응답 오탈락률(JQ3) | ≤ 0.15 |
| weighted κ(JQ1b) | ≥ 0.4, 그리고 사람 간 κ의 80% 이상 |
| 독립 풀이 정확도(JQ4) | ≥ 0.85 |
| 형식 위반 탐지율(JQ5) | ≥ 0.8 |
