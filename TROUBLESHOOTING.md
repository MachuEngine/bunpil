# TROUBLESHOOTING — 출제 모듈 tool-calling / 컨텍스트 이슈 (2026-07-09~10)

STRUCTURE_GOLDEN 골든셋을 실제 qwen2.5:7b 출력으로 재생성하는 과정에서 겪은
tool-calling 실패·컨텍스트 문제의 증상·진단·조치를 기록. 같은 증상을 다시
만나면 이 문서부터 확인할 것.

## 증상

`agent_node`(ReAct 루프)가 문항을 0개 생성하고 끝나는 경우가 자주 발생. 특히:
- 재시도(budget)를 끄고 1회차만 보면 실패율이 매우 높음(6개 중 5개, 약 17% 성공)
- 특정 시점부터는 재시도(budget=3)를 줘도 성공률이 6% 수준까지 폭락

## 원인 1 — qwen2.5:7b의 tool-calling 안정성 자체가 낮음

`save_item`까지는 정상적으로 도구를 호출하다가, 다음 턴부터 실제 tool_call 대신
"이제 record_score를 호출하겠습니다..." 같은 일반 텍스트(간혹 중국어·스페인어·
프랑스어가 섞인 응답)로만 응답하는 경우가 반복 관찰됨. `graph.py`의
`if not getattr(response, "tool_calls", []): break` 조건에 걸려 루프가 조기
종료되고, 그 시점까지 저장된 문항만 남는다(대부분 0개).

**진단 방법**: `graph.invoke()`로 얻은 `state["agent_messages"]`를 순회하며
각 메시지의 `tool_calls`와 `content`를 직접 출력해서 어느 턴에서 끊겼는지 확인.

```python
state = graph.invoke({...})
for m in state["agent_messages"]:
    print(type(m).__name__, getattr(m, "tool_calls", None))
    print(str(getattr(m, "content", ""))[:400])
```

**현재 대응**: 코드 버그가 아니라 모델 한계로 판단, 프로덕션은 `budget=5`
재시도로 완화(`main.py` `_run_exam`). budget을 늘려도 성공률 자체가 크게
개선되지는 않음(아래 원인 2와 얽혀 있었음 — 원인 2 수정 후 재확인 필요).

## 원인 2 (핵심) — 로컬 Ollama의 num_ctx 기본값(4096)이 모델 네이티브 컨텍스트(32K)보다 훨씬 좁음

STRUCTURE_GOLDEN 재생성 성공률이 어느 시점부터 6%까지 폭락한 원인을 추적한 결과:

- `ollama ps` → `CONTEXT` 열이 `4096`으로 표시됨
- `curl localhost:11434/api/show -d '{"model":"qwen2.5:7b"}'` →
  `model_info.qwen2.context_length: 32768` — 모델 자체는 32K 지원
- 별도로 `num_ctx`를 지정한 적이 없어 Ollama가 기본값(4096)으로 로드

`agent_node`의 멀티턴 ReAct 루프는 `search_standards`/`search_regulations`
검색 결과(도구 하나당 top_k=3, 각 300~400자 — 두 도구를 한 턴에 같이 부르면
최대 6개 청크)가 매 검색마다 대화 기록에 그대로 누적된다. 여기에 시스템
프롬프트(오답 매력도 지시 + num_items 지시로 최근에 더 길어짐), 모델이 도구
호출 사이에 쓰는 장황한 서술까지 겹치면 **단 몇 턴 만에 4096 토큰을 넘긴다**.
컨텍스트가 넘치면 앞부분(시스템 프롬프트 포함)이 잘려나가고, 모델이 지시를
잃은 채 응답이 깨지는 것으로 보인다.

**재현/검증**: 동일 passage_text(지방분권, num_items=5)로

| num_ctx | budget | 결과 |
|---|---|---|
| 4096 (기본값) | 1 | 0/5문항, 응답에 중국어·스페인어·프랑스어 뒤섞임 |
| 16384 | 1 | 5/5문항, 정상 |

**조치**: `app/modules/exam/llm.py`의 `ChatOllama` 생성자에 `num_ctx=16384`
명시(커밋 `36d831e`).

**프로덕션(RunPod)은 원래 문제 없었을 가능성 높음**: `runpod_handler/handler.py`의
vLLM 초기화(`LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.90)`)는
`max_model_len`을 지정하지 않는다 — 이 경우 vLLM은 모델 설정의 네이티브 값(32K)을
그대로 쓰므로, 이 버그는 **로컬 Ollama 개발 환경에만 존재했을 가능성이 크다**
(직접 RunPod에서 재현 확인은 못 함, 코드 상 추정).

## 원인 1·2의 관계

원인 2(컨텍스트 잘림)를 고치기 전까지는 "재시도(budget)를 늘려도 성공률이
안 오른다"는 관찰(원인 1 섹션)이 사실이었지만, 이는 컨텍스트 문제가 섞여
있었기 때문일 가능성이 크다. num_ctx 수정 후 budget=1로 32개 재생성했을 때
성공률이 6%→37.5%로 회복되어 "정상적인" 수준(기존에 관찰했던 35~40%대)으로
돌아왔다 — 즉 **원인 2는 원인 1을 악화시키는 요인이었지, 별개의 문제가
아니었다.** 원인 1(모델 자체의 tool-calling 불안정성, ~35~40% 실패율)은
컨텍스트를 고쳐도 남아있는 잔여 문제다.

## 남은 문제 (해결 안 됨, 후속 조치 대상)

- 원인 1의 잔여 실패율(~35~40%대 tool-calling 실패)은 컨텍스트 수정으로도
  없어지지 않음 — 별도 완화책 필요(temperature 조정, 프롬프트에 "설명 텍스트
  금지" 지시 추가, 재시도 시 부분 진행 보존 등 — 진행 중인 후속 작업 참고)
- `search_standards`/`search_regulations`의 검색 결과 크기(top_k=3, 300~400자)
  자체가 컨텍스트 성장의 큰 부분을 차지 — 줄이면 Recall에 영향이 있을 수 있어
  별도 측정 필요

## 관련 커밋/파일

- `app/modules/exam/llm.py` — num_ctx=16384 수정 (`36d831e`)
- `app/modules/exam/graph.py` — agent_node 시스템 프롬프트, validate_node
- `app/modules/exam/tools.py` — search_standards/search_regulations top_k
- `scripts/gen_structure_golden.py` — 문제 재현에 사용한 생성 스크립트
- `blog_draft.md` "배운 것" 7·8·9번 — 같은 내용을 블로그용으로 정리
- `EVAL.md`, `bunpil_roadmap.md` — STRUCTURE_GOLDEN 재생성 결과·발견 사항 기록
