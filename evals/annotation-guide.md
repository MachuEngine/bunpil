# 사람 라벨 작성 지침 (annotation guide)

> Judge 후보를 검증할 **gold label**을 만드는 방법이다. 평가자 2인(본인 + 1인)이 **서로의 라벨을 보지 않고**
> 독립적으로 매긴 뒤, 불일치 항목만 합의한다. 채점 기준은 `rubrics/generation-rubric.md`.

## 1. 데이터 원칙

- 평가 데이터는 **전부 합성 또는 공개 자료**다(CLAUDE.md 하드룰 1). 실제 학생 정보가 보이면 라벨링을 멈추고 알린다.
- 평가자는 **어떤 모델이 만든 출력인지 모른다.** 파일에는 `output_id`만 있고 모델 정보는 별도 매핑 파일에 있다.
- 라벨 파일에 개인 식별 정보(평가자 실명 등)를 쓰지 않는다. 평가자는 `rater_1`, `rater_2`로 표기한다.

## 2. 작업 순서

1. 파일 세 개를 받는다(단계 B에서 생성).
   - `evals/data/labeling/<batch>.jsonl` — 판정할 문항(예시 문제·문항·해설). **읽기 전용**
   - `evals/data/labeling/<batch>.calibration.jsonl` — 보정 세트 10건(둘이 같이 먼저 매긴다, gold 제외)
   - `evals/data/labeling/<batch>.labels.rater_N.jsonl` — **내 라벨 템플릿**. 문항마다 한 줄이고 빈 칸만 채운다
2. 항목마다 예시 문제 → 생성 문항 → 해설 순서로 읽는다.
3. `generation-rubric.md` 2절(I1~I7)과 3절(E1~E4)을 판정한다. 각 판정은 `yes` / `no` / `unsure`.
4. `no`·`unsure`에는 한 줄 근거를 쓴다(예: "②도 정답 — 배타성이 없음 = 비배제성").
5. 항목마다 **확신도**(1=추측, 2=어느 정도 확신, 3=확신)를 매긴다.
6. pairwise 항목은 `A` / `B` / `tie` / `both_bad` / `unsure` 중 하나를 고른다.
7. 확인이 필요하면 교과서·교육과정 원문을 찾아봐도 된다. **다른 평가자의 라벨이나 LLM 출력은 보지 않는다.**
8. 중간중간 오타를 검사한다(허용값 밖의 값, `no`·`unsure`인데 근거가 빈 칸 등을 줄 번호와 함께 알려준다):
   ```bash
   .venv/bin/python -m evals.model_selection.run check-labels --labels evals/data/labeling/<batch>.labels.rater_1.jsonl
   ```

## 3. 라벨 형식

한 줄이 평가자 1명의 라벨 1건이다. 최초 라벨은 **덮어쓰지 않는다** — 합의 결과는 별도 행으로 추가한다.

```json
{"output_id": "lp-0007-i1", "rater": "rater_1", "round": "initial",
 "labels": {"I1": "yes", "I2": "no", "I3": "yes", "I4": "yes", "I5": "yes", "I6": "yes", "I7": 3,
            "E1": "yes", "E2": "yes", "E3": "no", "E4": "yes"},
 "reasons": {"I2": "②도 정답", "E3": "④ 설명 누락"},
 "critical": ["X2"], "confidence": 3, "cannot_judge": false,
 "minutes_spent": 3, "labeled_at": "2026-09-30"}
```

합의 행은 `"round": "consensus"`, `"rater": "consensus"`, `"resolution": "agreed" | "ambiguous"`를 추가한다.

## 4. 불일치 처리

1. 두 평가자의 `initial` 라벨이 판정 문항 하나라도 다르면 **불일치 항목**이다.
2. 두 사람이 근거를 비교해 합의한다. 합의되면 `resolution: "agreed"`.
3. 근거를 비교해도 의견이 갈리거나 교육과정상 해석이 여럿이면 `resolution: "ambiguous"`로 둔다.
   **억지로 하나의 정답을 정하지 않는다.** ambiguous 항목은 Judge 주 지표에서 빠지고 "판단 불가 처리(JQ8)" 평가에만 쓴다.

## 5. 품질 관리

- 배치 시작 전에 **보정 세트 10건**을 둘이 같이 매기고 기준을 맞춘다(이 10건은 gold label에서 제외).
- 전체의 10%는 1주 뒤 같은 평가자가 다시 매겨 **자기 일치도**를 잰다.
- 평가자 간 일치도(κ)를 판정 문항별로 보고한다. κ < 0.4인 문항은 rubric 문구가 모호하다는 신호다 — 문구를 고치고 그 문항은 다시 매긴다.

## 6. 예상 작업량

| 대상 | 건수 | 1건당 | 1인 합계 |
|---|---|---|---|
| 문항+해설 판정 | 150 | 약 3분 | 약 7.5시간 |
| pairwise | 60 | 약 2분 | 약 2시간 |
| 보정 세트 | 10 | 약 3분 | 30분 |
