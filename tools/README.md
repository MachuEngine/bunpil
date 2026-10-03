# tools/

사람이 수동으로 쓰는 보조 도구 모음. 평가 파이프라인이 자동으로 호출하지 않는다.

## labeling.html — 문항 품질 라벨링

`golden_gen/gen_item_quality_golden.py build-labelset`이 만든
`data/golden/item_quality_golden.json`에 사람이 직접 라벨(정답유일성·오답매력도·근거성·
학생난이도·판단불가·근거)을 다는 도구. 외부 의존성·CDN 없는 단일 HTML 파일 — 브라우저로
파일을 더블클릭해 열면 오프라인에서 그대로 동작한다.

### 사용법

1. `tools/labeling.html`을 브라우저로 연다(더블클릭 또는 `open tools/labeling.html`).
2. "파일 선택"으로 `data/golden/item_quality_golden.json`을 고른다.
3. 한 화면에 한 문항씩 라벨링한다. 기준표는 화면 상단에서 항상 펼쳐 볼 수 있다(접기 가능).
   정답유일성·오답매력도·근거성 중 하나라도 2점 이하이거나 "판단 불가"를 선택하면 근거를
   채워야 다음 문항으로 넘어갈 수 있다.
4. 작업 중 진행 상황은 브라우저 localStorage에 자동 저장된다(파일 이름 + 문항 id 기준).
   브라우저/탭을 닫았다가 같은 파일을 다시 선택하면 이어서 라벨링할 수 있다. 단, 이 저장은
   **브라우저 하나에 묶여 있다** — 다른 브라우저나 기기에서 열면 처음부터 다시 시작된다.
5. "JSON 내보내기"를 누르면 `item_quality_golden.labeled.json`이 다운로드된다. 미완료
   문항이 남아 있으면 개수를 알려주지만, 그래도 내보내기는 항상 허용한다.

### 내보낸 파일을 원본에 반영하는 법

내보낸 `item_quality_golden.labeled.json`으로 `data/golden/item_quality_golden.json`을
**통째로 교체**한다(예: `mv` 또는 파일 덮어쓰기).

```
mv ~/Downloads/item_quality_golden.labeled.json data/golden/item_quality_golden.json
```

`golden_gen/gen_item_quality_golden.py build-labelset`은 `item_quality_golden.json`에
이미 채워진 `human_label`이 하나라도 있으면 실행을 중단하는 보호 장치
(`_has_existing_labels`)가 있다 — build-labelset을 다시 돌려서 라벨을 유실할 위험은
없지만, 반대로 이미 라벨링을 마친 파일 위에서 build-labelset을 다시 실행하려면(지문을
다시 뽑는 등) 먼저 해당 파일을 지우거나 백업해야 한다.

### 라벨러는 모델을 알 수 없다(블라인드)

`item_quality_golden.json`의 entries에는 생성 모델 정보가 없다(라벨링 편향 방지).
모델 매핑은 `data/golden/_item_quality_model_map.json`에 별도로 있으니, 라벨링이
끝나기 전에는 열어보지 않는다.

## JSON 직접 편집 방식 (labeling.html 대신)

브라우저 도구 없이 JSON 파일을 텍스트 에디터로 직접 편집해 라벨을 달 수도 있다.
`item_quality_golden.json`은 `passage_text`·`stimulus`의 줄바꿈이 `"\n"`으로 그대로
보여 읽기 불편하므로, 줄 단위 배열로 풀고 라벨 칸을 맨 위로 올린 별도 시트 파일을
오간다.

### 사용법

1. 시트 생성:
   ```
   .venv/bin/python golden_gen/gen_item_quality_golden.py export-sheet
   ```
   `data/golden/item_quality_labeling_sheet.json`이 만들어진다(라벨은 전부 빈 값).
   이미 라벨이 채워진 시트가 있으면 덮어쓰지 않고 중단한다(라벨 유실 방지).
2. 생성된 시트를 에디터로 열어 각 문항의 `"라벨"` 블록만 채운다. 맨 앞 `"_안내"`에
   편집 방법과 기준표(정답유일성·오답매력도·근거성·학생난이도)가 들어 있다. 정답유일성·
   오답매력도·근거성 중 하나라도 2점 이하이거나 `"판단불가"`를 `true`로 두면 `"근거"`를
   채워야 한다.
3. 골든셋에 반영:
   ```
   .venv/bin/python golden_gen/gen_item_quality_golden.py import-sheet
   ```
   검증(점수 범위, 근거 필수 조건, id 존재 여부)에 걸리면 어떤 문항·어떤 이유로
   막혔는지만 출력하고(문항 원문은 출력하지 않는다) 아무것도 반영하지 않는다. 통과하면
   `data/golden/item_quality_golden.json`의 `human_label`만 갱신하고, 아직 점수가
   비어 있는 문항 수를 알려준다.
4. 중간 저장도 된다 — 시트를 일부만 채운 채로 `import-sheet`를 실행하면 채운 부분까지
   반영되고, 나머지는 다음에 다시 같은 시트를 편집해 `import-sheet`를 또 실행하면 된다.
   시트 경로가 다르면 `--sheet PATH`로 지정한다.
