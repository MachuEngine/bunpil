# 분필(Bunpil) 피드백 드리븐 리디자인 — 최종 확정안

> 실제 교사 사용 패턴(PDF 업로드+드롭다운 대신 예시 문제 텍스트 붙여넣기, ChatGPT 사용 습관과 동일)을
> 근거로 출제 모듈 입력 방식을 전면 재설계. Claude Code 구현 핸드오프용 문서.
>
> **✅ 구현 완료 (2026-07-06~07)**: 아래 6절 체크리스트 전부 반영. STRUCTURE_GOLDEN은 Claude 합성
> 부트스트랩 3개만 있고 실제 모델 라벨 보강은 아직 미완료 — 최신 현황은 `DESIGN.md`/`README.md`/`EVAL.md`/
> `bunpil_roadmap.md` 참고.

---

## 1. 배경

- 실제 사용 교사 피드백: PDF 업로드 + 유형/난이도/개수 드롭다운 설정이 번거로움. 실제로는 ChatGPT 쓰듯 예시 문제를 텍스트로 붙여넣고 "이런 식으로 만들어줘"라고 요청하는 패턴
- `state.py`에 이미 존재하지만 어떤 엔드포인트도 연결하지 않았던 `passage_text` 필드를 붙여넣기 입력용으로 활용
- 2028 수능 개편(9과목 사회탐구 → 통합사회)으로 `past_exams`의 과목별 구조 자체가 곧 무의미해짐 — 이번 기회에 `past_exams` 컬렉션과 `check_duplicate`를 스코프에서 완전히 제거

---

## 2. Before / After 아키텍처

### Before

```
POST /exam
  pdf: UploadFile (지문 업로드, 필수)
  unit: str (드롭다운 선택)
  num_mc, num_sa, num_hard, num_med, num_easy: int (사용자가 직접 지정)
  standards: str
    │
    ▼
parse_pdf → chunk_document → 임베딩 → 세션 임시 컬렉션 적재
    │
    ▼
ExamSpec{ unit, num_items, type_dist, difficulty_dist, standards }
    │
    ▼
plan → agent(슬롯별 ThreadPoolExecutor 병렬 생성, 문항당 최대 14회 반복)
     → validate(요청 분포·개수 함수 검증)
     → should_retry(budget=3)
     └ check_duplicate(past_exams 컬렉션, 임베딩 유사도, threshold 0.8)
```

### After

```
POST /exam
  passage_text: str (예시 문제 원문 붙여넣기, 필수, 최대 8,000자)
  standards: str (선택 유지)
    │
    ▼
전처리 없음 — 원문 그대로 프롬프트 주입 (길이 초과 시 truncate + 안내)
    │
    ▼
ExamSpec{ passage_text, standards }   ← unit/num_items/type_dist/difficulty_dist 필드 제거
    │
    ▼
plan → agent(passage_text 분석 → 세트 전체 한 번에 생성 → similarity_judge 호출)
     → validate(similarity_judge 결과를 Python이 threshold 판정)
     → should_retry(세트 통째로 재시도, budget=5)
```

**표절(중복) 체크: 완전히 제거.** `check_duplicate`, `past_exams` 컬렉션 관련 로직 전체 삭제, 대체 로직 없음.

---

## 3. 확정 사항 요약

| # | 항목 | 확정 내용 |
|---|---|---|
| 1 | candidate 개수 / 재시도 단위 | 슬롯별 병렬 생성 폐지 → **세트 전체 단위**로 재시도, `budget=5` |
| 2 | UI 컨트롤 범위 | 문항 개수·유형(객관식/서술형) 비율·난이도 분포 입력 **전부 제거**. `passage_text` 하나만 필수 입력 |
| 3 | validate 통과 기준 | `similarity_judge`(LLM Judge) 출력값을 **Python이 threshold로 판정** — LLM은 판단, 코드는 결정이라는 하이브리드 원칙 유지 |
| 4 | `passage_text` 길이 제한 | 현재 스택(Qwen2.5-7B, 32K 네이티브 context) 기준 **최대 8,000자**, 초과 시 앞부분 truncate + "입력이 길어 앞부분만 반영되었습니다" 안내 |
| 5 | 표절(중복) 체크 | **완전 제거**. `check_duplicate` 삭제 유지, 대체 로직 없음 |

---

## 4. 코드 변경 상세

### 4-1. `app/main.py`

```python
# Before
@app.post("/exam")
async def exam(
    pdf: UploadFile = File(...),
    unit: str = Form(...),
    num_mc: int = Form(5),
    num_sa: int = Form(2),
    num_hard: int = Form(2),
    num_med: int = Form(3),
    num_easy: int = Form(2),
    standards: str = Form(""),
):
    pdf_bytes = await pdf.read()
    doc = parse_pdf(tmp_path)
    chunks = chunk_document(doc)
    col = store.create_temp_collection()
    embeddings = embedder.embed([c["text"] for c in chunks])
    store.add_chunks(col, chunks, embeddings)
    ...

# After
MAX_PASSAGE_LENGTH = 8000

@app.post("/exam")
async def exam(
    passage_text: str = Form(...),
    standards: str = Form(""),
):
    truncated = len(passage_text) > MAX_PASSAGE_LENGTH
    text = passage_text[:MAX_PASSAGE_LENGTH] if truncated else passage_text

    spec: ExamSpec = {
        "passage_text": text,
        "standards": std_list,
    }
    graph = get_exam_graph()
    state = await asyncio.to_thread(graph.invoke, {"spec": spec, "budget": 5})

    yield evt({"status": "done", "items": ..., "truncated": truncated})
```

- PDF 파싱/청킹/세션 임시 컬렉션 생성 로직 **엔드포인트에서 전부 제거** (단, `parse_pdf`/`chunk_document` 함수 자체는 `scripts/index_past_exams.py` 외 다른 스크립트에서 쓰일 수 있으니 — `past_exams` 제거로 이 스크립트 자체도 폐기 대상인지 확인 필요, 4-5 참고)

### 4-2. `app/modules/exam/state.py`

```python
# 제거되는 필드
num_items, type_dist, difficulty_dist

# 그대로 활용 (기존 미사용 필드)
passage_text: str

# 신규 필드
similarity_judge_result: dict  # {"count_match": bool, "type_ratio_score": float, "difficulty_match": bool, "overall_score": int}
```

### 4-3. `app/modules/exam/graph.py`

**agent 노드 — 시스템 프롬프트 변경**

```python
system_prompt = (
    "다음은 교사가 참고용으로 제시한 예시 문제입니다.\n\n"
    f"[예시 문제]\n{spec['passage_text']}\n\n"
    "위 예시의 문항 수, 유형(객관식/서술형) 구성, 난이도 수준을 그대로 파악하여 "
    "동일한 개수·구성·난이도의 새 문항 세트를 작성하세요.\n\n"
    "문항 세트 작성이 끝나면 similarity_judge 도구를 호출해 "
    "예시 문제와의 구조적 유사도를 스스로 평가하세요."
)
```

**agent 노드 — 실행 구조 변경**

```python
# Before: ThreadPoolExecutor로 슬롯별 병렬 생성 (_run_item, _build_target_pairs)
# After: 단일 세션에서 세트 전체 생성 (병렬 처리 로직 제거)

def agent_node(state: ExamState) -> dict:
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=passage_text)]
    for _ in range(14):  # 기존 반복 한도 유지
        response = llm.invoke(messages)
        messages.append(response)
        if not getattr(response, "tool_calls", []):
            break
        for tc in response.tool_calls:
            fn = tool_map.get(tc["name"])
            result = str(fn.invoke(tc["args"])) if fn else f"Unknown tool: {tc['name']}"
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            if tc["name"] == "similarity_judge":
                # judge 결과를 state에 저장 후 루프 종료
                break
    return {"agent_messages": messages, "similarity_judge_result": ...}
```

**should_retry — 판정 로직 변경**

```python
# Before: 요청 분포/개수를 함수로 검증
# After: similarity_judge 결과를 threshold로 판정

def should_retry(state: ExamState) -> str:
    judge = state.get("similarity_judge_result", {})
    passed = (
        judge.get("count_match", False)
        and judge.get("type_ratio_score", 0) >= 0.7
        and judge.get("difficulty_match", False)
        and judge.get("overall_score", 0) >= 4
    )
    if passed:
        return "end"
    elif state["budget"] > 0:
        return "agent"
    else:
        return END
```

**제거 대상**
- `_build_target_pairs` (슬롯 생성 로직)
- `_run_item` + `ThreadPoolExecutor` 병렬 처리
- `check_duplicate` 호출 및 관련 분기

### 4-4. `app/modules/exam/tools.py`

| 도구 | 처리 |
|---|---|
| `search_passages` | **제거** (업로드 지문 검색용이었음 — 입력 지문 자체가 사라짐) |
| `check_duplicate` | **제거** |
| `get_past_item_examples` | **제거** (`past_exams` 컬렉션 삭제에 따라 종속 제거) |
| `search_regulations` | 유지 |
| `validate_item_format` | 유지 |
| `save_item` | 유지 (세트 전체를 한 번에 저장하도록 인터페이스 조정 필요) |
| `record_score` | 유지 |
| **신규**: `similarity_judge` | `passage_text`와 생성된 문항 세트를 비교해 구조화된 JSON 반환 |

### 4-5. 정리/폐기 대상 스크립트·데이터

- `scripts/index_past_exams.py` — `past_exams` 컬렉션 자체가 제거되므로 **폐기**
- `data/past_exams/` 디렉토리 — 더 이상 참조 안 됨
- `data/golden/retrieval_golden_final.json`의 `ret_023`, `ret_024`(past_exams 참조 항목) — **삭제 필요**

### 4-6. 프론트엔드 (`frontend/`)

- 문항 개수·유형·난이도 입력 UI 전부 제거
- PDF 업로드 드롭존 제거
- `passage_text` 붙여넣기용 텍스트 영역 하나로 교체
- 8,000자 제한 안내 문구 추가 (truncate 발생 시 표시)

---

## 5. 평가 인프라(`EVAL.md`) 영향

| 평가 항목 | 상태 |
|---|---|
| Recall@5 | 범위 축소 — `past_exams` 관련 golden 항목(`ret_023`, `ret_024`) 제거, `standards`/`regulations` 검색만 남음 |
| 문항 품질(정답 유일성·오답 매력도·근거성) | 변화 없음 |
| 기존 ITEM_GOLDEN kappa | 변화 없음 |
| **신규**: 구조 Judge 신뢰도 | `similarity_judge` 결과(count_match, type_ratio_score, difficulty_match, overall_score)를 사람 라벨과 대조하는 **신규 골든셋 필요** (가칭 `STRUCTURE_GOLDEN`) |
| 기존 세트 제약 함수 검증 (분포·커버리지·중복률) | **제거** — 구조 Judge 신뢰도 검증으로 대체 |
| 표절 체크 Precision/Recall | **계획 자체 제거** |

---

## 6. Claude Code 구현 체크리스트

- [x] `app/main.py`: `/exam`, `/exam/stream` 엔드포인트를 `passage_text` 단일 입력으로 변경
- [x] `app/modules/exam/state.py`: `ExamSpec`/`ExamState`에서 `num_items`/`type_dist`/`difficulty_dist` 제거, `passage_text`/`similarity_judge_result` 반영
- [x] `app/modules/exam/tools.py`: `search_passages`/`check_duplicate`/`get_past_item_examples` 제거, `similarity_judge` 신규 구현
- [x] `app/modules/exam/graph.py`: agent 노드 단일 세션 생성으로 재작성(`ThreadPoolExecutor`/`_build_target_pairs`/`_run_item` 제거), `should_retry`를 `similarity_judge_result` 기반으로 변경, `budget` 기본값 5로 변경
- [x] `scripts/index_past_exams.py`, `data/past_exams/` 폐기
- [x] `data/golden/retrieval_golden_final.json`에서 `past_exams` 참조 항목 제거
- [x] `frontend/`: 입력 UI를 텍스트 붙여넣기 단일 필드로 교체
- [ ] 신규 `STRUCTURE_GOLDEN` 골든셋 구축 (예시 문제 + 사람이 매긴 구조 정답 + overall_score 라벨) — Claude 합성 부트스트랩 3개만 있음, 실제 모델 라벨 보강 미완료(상단 안내 참고)
- [x] `EVAL.md`, `README.md`, `bunpil_roadmap.md`에 위 변경사항 반영

---

## 7. 아키텍처 원칙 재확인

이번 리디자인 이후에도 분필의 핵심 원칙은 그대로 유지됨:

> **LLM 판단은 agent 노드 안에서만, 노드 간 라우팅(validate → should_retry)은 결정론적 Python 코드.**

`similarity_judge`가 LLM Judge라는 점은 새롭지 않음 — 기존 문항 품질 평가(정답 유일성·오답 매력도 등)에서 이미 쓰던 패턴과 동일한 구조. 다만 이제 **"판단 대상"이 개별 문항에서 세트 전체의 구조적 유사도로 확장**된 것뿐이며, "판단은 LLM, 통과/재시도 결정은 코드"라는 경계는 변하지 않음.
