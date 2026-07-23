# 분필(Bunpil) — 모델·검색 방식 선택 근거

> "왜 이 모델/방식을 썼는가"에 바로 답할 수 있도록 6개 항목(생성·Judge·임베딩·리랭커·
> 검색 방식·백엔드 추상화)을 표 위주로 정리한 문서. 실험 원본 데이터·회차별 로그는
> [EVAL.md](./EVAL.md) 4·7·7.1·9절, 결정 이력은 [bunpil_roadmap.md](./bunpil_roadmap.md)에
> 있고, 이 문서는 결론만 압축한다. **근거가 코드·문서 어디에도 없는 항목은 "명시적
> 근거 없음"이라고 그대로 적었다** — 없는 근거를 지어내지 않았다.
>
> 실험 스크립트: `experiments/compare_models.py`(생성), `experiments/compare_judge_models.py`(Judge).
> 코드 근거 확인 대상: `.env.example`, `app/common/llm/*`, `app/common/rag/*`,
> `evals/eval_*.py`.

---

## 한눈 요약표

| # | 항목 | 현재 선택 | 핵심 근거 | 상태 |
|---|---|---|---|---|
| 1 | 생성 모델 | **Qwen2.5-14B** (로컬 Ollama / RunPod AWQ 4bit·vLLM) | budget=5(실서비스 조건) 재검증에서 7B 대비 속도 손해 없이 실패율·개수충족률 뚜렷 우수 | ✅ 확정 |
| 2 | Judge 모델 | **gpt-5.6-luna** (OpenAI) | qwen2.5:7b 대비 kappa 0.328→0.595, 구조 MAE 1.689→0.600 | ✅ 확정 + 런타임 적용(2026-07-23) |
| 3 | 임베딩 모델 | **BGE-M3** (CPU) | 모델 자체 비교 근거 없음. CPU 배치 이유만 명시("소규모 코퍼스+GPU는 생성 전용") | ⚠️ 미검증 채택 |
| 4 | 리랭커 모델 | **BGE-reranker-base** (CPU) | 상동 — 비교 근거 없음, CPU 배치 이유만 명시 | ⚠️ 미검증 채택 |
| 5 | 검색 방식 | **Dense-only** (BGE dense → ChromaDB cosine → rerank) | 명시적 근거 없음. BGE-M3가 sparse도 지원하지만 코드가 dense만 사용 | ⚠️ 하이브리드 미검토 |
| 6 | LLM 백엔드 추상화 | `LLMBackend` ABC + factory(local/runpod/openai) | 로컬↔프로덕션 무중단 전환, Judge/생성 독립 교체 | ✅ 확정, 단 코드·문서 기본값 불일치 1건 발견(§6) |

---

## 1. 생성(Generation) 모델

| 항목 | 내용 |
|---|---|
| 현재 선택 | **Qwen2.5-14B** — 로컬 개발은 Ollama, 프로덕션은 RunPod 서버리스(**AWQ 4bit 양자화**, vLLM). `.env.example`의 `OLLAMA_MODEL=qwen2.5:14b` |
| 선택 근거 | 실서비스 조건인 budget=5(재시도 5회) 재검증에서 7B 대비 **속도 손해 없이**(둘 다 260초대로 수렴 — 7B는 재시도가 잦아 속도 이점을 스스로 상쇄) 실패율(0% vs 6.7%)·개수 충족률(1.036 vs 0.649)에서 뚜렷이 우수 |
| 실측 수치 | budget=1→5 비교(`experiments/compare_models.py`, 아래 표) |
| 대안으로 검토했던 것 | Qwen2.5-7B(기존 기본값) / Llama3.1-8B(전 지표 최저, 제외) / GPT-4o-mini(budget=1엔 최고였으나 budget=5서 실패율 0%→13.3%·개수 과다생성 결함 발견) / Qwen3.5-9B(정답유일성 최고지만 budget=5 속도 621s로 열위) |
| 현재 미해결/보류 중인 결정 | 없음(확정). 단 GPT-4o-mini는 "다음 우선순위 후보"로 문서상 열려는 있음(채택 아님) |
| 트레이드오프 | 로컬 14B는 GPT류 API 대비 비용 없음·교사 지문이 외부로 안 나감이 강점. RunPod은 **RTX A5000 24GB에 float16 14B가 안 들어가 AWQ 4bit 채택**(`runpod_handler/handler.py:52`, 순수 메모리 제약) — AWQ 자체가 다른 양자화 방식보다 나은지 비교한 기록은 없음("명시적 근거 없음") |

**budget=1 → budget=5 비교 결과**

| 모델 | 실패율(0문항) | 평균 생성시간 | 개수 충족률 |
|---|---|---|---|
| Qwen2.5-7B | 40% → **6.7%** | 50.3s → 258.8s | 0.15 → 0.649 |
| Qwen2.5-14B | 0% → 0% | 158.1s → **260.2s** | 0.76 → **1.036** |
| Llama3.1-8B | 60% → 26.7% | 59.4s → 289.4s | 0.11 → 0.336 |
| GPT-4o-mini | 0% → 13.3% | **18.1s → 66.5s** | 0.97 → 1.768(과다생성) |
| Qwen3.5-9B | 26.7% → 0% | 171.1s → **621.2s**(최저 속도) | 0.55 → 1.395 |

---

## 2. 평가(Judge) 모델

| 항목 | 내용 |
|---|---|
| 현재 선택 | **gpt-5.6-luna** (OpenAI). `.env.example`의 `JUDGE_BACKEND=openai`(기본값), `OPENAI_JUDGE_MODEL=gpt-5.6-luna`. 2026-07-23부터 오프라인 eval뿐 아니라 **런타임 `judge` 노드에도 동일 함수 공유**(`app/modules/exam/judge.py`) |
| 선택 근거 | 로컬 qwen2.5(7B/14B)로 몇 달간 few-shot·rubric 튜닝해도 목표(κ≥0.4) 미달 지속 → 생성물 고정, Judge만 교체하는 비교 실험. gpt-5.6-luna가 문항품질 kappa 0.328→**0.595**, 구조유사도 overall MAE 1.689→**0.600**(qwen2.5-14B 몇 달 튜닝 최고 기록 1.185보다도 낮음) |
| 실측 수치 | 아래 비교표 + **2026-07-24 정기평가 재측정**(`evals/eval_exam.py`, 런타임과 동일 코드): 문항품질 κ=**0.468**✅(첫 목표 달성), ±1 일치율=0.700✅, 종합평균=4.06✅(첫 목표 달성). 구조유사도 difficulty_match 일치율=0.933, overall MAE=**0.644**(역대 최저) — EVAL.md §4 최신 행 |
| 대안으로 검토했던 것 | qwen2.5:7b(기존 기본값) / qwen2.5:14b(현재도 `JUDGE_BACKEND=local` 전환 시 대안 경로로 코드에 남아있음) |
| 현재 미해결/보류 중인 결정 | **없음 — 이미 확정+런타임 적용 완료.** (2026-07-21 채택 결정, 2026-07-23 아키텍처 분리로 검증-배포 일치까지 완료. `bunpil_roadmap.md`의 열린 이슈 목록에도 더 이상 없음 — "qwen2.5:7b vs gpt-5.6-luna"를 아직 열린 질문으로 알고 있다면 이미 지난 결정임) |
| 트레이드오프 | 절대 점수 편향이 qwen(+0.50)보다 큼(gpt-5.6-luna +0.94, 더 후하게 채점) — 신뢰도 개선폭이 편향 증가분보다 크다고 판단해 감수. 비용은 사이클당 ~$0.10로 무시 가능. 과거 EVAL.md kappa/MAE 히스토리가 전부 qwen 기준이라 교체 후 수치는 재보정 기준선으로 다시 쌓임(직접 비교 불가). `passage_text`가 매 문항 세트 생성마다 OpenAI로 전송됨(PII는 마스킹되나 저작권 있는 지문 자체는 전송 대상) — 호출 실패 시 fail-fast(조용한 로컬 폴백 금지) |

**qwen2.5:7b vs gpt-5.6-luna 재채점 비교** (`experiments/compare_judge_models.py`, 2026-07-17, 동일 생성물 n=30/45)

| 지표 | qwen2.5:7b | gpt-5.6-luna |
|---|---|---|
| 문항품질 kappa | 0.328 | **0.595** |
| 문항품질 ±1 일치율 | **0.800** | 0.733 |
| 구조유사도 difficulty_match | 0.867 | 0.933 |
| 구조유사도 overall MAE | 1.689 | **0.600** |

**왜 생성 모델과 다른 백엔드를 쓰는가**: 생성은 로컬 우선(비용·데이터 로컬 처리), Judge는 신뢰도 우선 — 두 역할의 우선순위가 다르다는 판단이 그대로 서로 다른 채택 결론으로 이어진 사례(`OLLAMA_MODEL`/`OLLAMA_JUDGE_MODEL`은 별도 env var, 미설정 시 후자가 전자로 폴백).

---

## 3. 임베딩 모델 (RAG)

| 항목 | 내용 |
|---|---|
| 현재 선택 | **BGE-M3** (`BAAI/bge-m3`, `.env.example`의 `BGE_EMBED_MODEL`), `FlagEmbedding.BGEM3FlagModel`로 **CPU**에서 실행 |
| 선택 근거 | **모델 자체에 대한 명시적 비교 근거 없음** — README/DESIGN.md는 "임베딩 = BGE-M3"를 기술 스택 표에 결정으로만 기재, 다른 임베딩 모델(OpenAI text-embedding, E5 등)과 비교한 실험 기록은 코드·문서 어디에도 없음. CPU **배치**(GPU 아님) 이유만 명시적: "소규모 코퍼스라 CPU로 충분, 생성·추론만 서버리스 GPU 호출"(DESIGN.md 3절) |
| 실측 수치 | Recall@5=**0.955**, MRR=**0.789**(n=22, 2026-07-24 최신) — 단 이건 **임베딩+리랭커 결합 파이프라인** 수치라 임베딩 단독 기여도를 분리한 ablation은 없음 |
| 대안으로 검토했던 것 | 없음(비교 기록 자체가 없음) |
| 현재 미해결/보류 중인 결정 | "왜 BGE-M3인가"는 사실상 미검증 채택 상태 — 대안 비교가 이뤄진 적이 없음 |
| 트레이드오프 | BGE-M3는 dense+sparse+multi-vector를 동시 지원하는 모델이지만, `app/common/rag/embedder.py`는 `dense_vecs`만 취해서 씀 — 모델의 하이브리드 잠재력을 실제로는 안 쓰고 있음(§5와 연결). CPU 실행은 GPU 비용은 아끼지만 인덱싱/쿼리 속도의 GPU 대비 정량 비교는 기록 없음 |

---

## 4. 리랭커 모델 (RAG)

| 항목 | 내용 |
|---|---|
| 현재 선택 | **BGE-reranker-base** (`BAAI/bge-reranker-base`, `.env.example`의 `BGE_RERANK_MODEL`), `FlagEmbedding.FlagReranker`로 CPU 실행. query-passage 쌍을 cross-encoder로 재채점 |
| 선택 근거 | 임베딩과 동일 — **모델 자체 비교 근거 없음**, CPU 배치 이유만 문서화됨(§3과 동일 근거) |
| 실측 수치 | §3과 동일 결합 수치(Recall@5=0.955/MRR=0.789) — **리랭커 유무 비교(ablation)는 측정된 적 없음**, 순수 기여도 불명 |
| 대안으로 검토했던 것 | 없음 |
| 현재 미해결/보류 중인 결정 | 리랭커의 실제 기여도(dense-only 대비 얼마나 개선하는지)가 한 번도 정량 측정된 적 없음 |
| 트레이드오프 | top-20 후보 전부를 pairwise cross-encoder로 채점 → 정확하지만 임베딩 단독 검색보다 느림(정성적 설명만 코드 주석에 있고, 정량 latency 비교 기록은 없음) |

---

## 5. 검색 방식 (Retrieval)

| 항목 | 내용 |
|---|---|
| 현재 선택 | **Dense-only 2단계**: BGE-M3 dense 임베딩 → ChromaDB HNSW **cosine** 유사도로 top-20 후보(`n_candidates`) → BGE reranker로 top-`k` 재정렬. **BM25/키워드 기반 sparse 검색 없음, 하이브리드 아님** |
| 선택 근거 | **명시적 근거 없음** — dense-only로 하겠다는 결정 자체가 문서화된 적이 없고 그냥 그렇게 구현됨. `app/common/rag/store.py:query()`는 `col.query(query_embeddings=...)`만 호출(sparse 인자 없음), `embedder.py`는 BGE-M3가 반환하는 `sparse_vecs`를 애초에 버림(직접 코드 확인) |
| 실측 수치 | Recall@5=0.955/MRR=0.789(n=22, standards+regulations 합산, 2026-07-24) · regulations 단독 Recall@5=0.900/MRR=0.667(n=10, 참고용). 실사용 설정: `top_k=3`(출제·생기부 공통), `n_candidates`는 출제 20(기본값) vs 생기부 10(명시적으로 다르게 지정 — `chain.py:194`) |
| 대안으로 검토했던 것 | `top_k` 자체는 3 vs 2 실험이 있음 — 2로 줄이면 Recall 0.810→0.762(-0.048, n=21), 기준(0.05)에 근접한 경계선이라 **3 유지 결정**(`bunpil_roadmap.md` 4단계). **하이브리드(BM25+dense) 자체는 검토된 적이 없음** — 문서·코드·로드맵 어디에도 논의 흔적 없음 |
| 현재 미해결/보류 중인 결정 | 하이브리드 검색 도입 여부 — "미해결"이라기보다 **미착수**. 생기부 규정 위반 탐지에서 "RAG 검색이 관련 규정 청크를 상위로 못 올리는" 알려진 약점이 열린 이슈로 남아있고(`bunpil_roadmap.md` 열린 이슈), 하이브리드가 후보 해법 중 하나로 거론될 법하지만 실제 계획으로 잡힌 적은 없음 |
| 트레이드오프 | 구현 단순함 vs 정확한 키워드 매칭(법령 조항 번호·고유명사 등)엔 sparse/BM25가 유리할 수 있는 경로를 아예 안 씀. 생기부 규정 위반 탐지 약점과 인과관계가 있을 수 있으나 **검증된 연결은 아님**(추정) |

---

## 6. LLM 백엔드 추상화 (local/runpod/openai 스위칭)

| 항목 | 내용 |
|---|---|
| 현재 선택 | `LLMBackend`(ABC, `generate()` 단일 메서드) 인터페이스 + `get_llm_backend()`/`get_judge_backend()` factory(`app/common/llm/factory.py`). `LLM_BACKEND`/`JUDGE_BACKEND` env var로 `local`(Ollama)/`runpod`/`openai` 3분기. 출제 모듈 ReAct 에이전트는 별도로 LangChain `BaseChatModel` 어댑터(`ChatRunPod`, `ChatOllama`)를 씀(tool-calling 필요, `LLMBackend`로는 불가) |
| 선택 근거 | "로컬 ↔ 프로덕션 무중단 전환"(README "빠른 시작"), Judge를 생성 모델과 독립적으로 교체 가능하게(§2.1, `OLLAMA_MODEL`과 `OLLAMA_JUDGE_MODEL` 분리와 동일한 설계 의도) |
| 실측 수치 | 해당 없음 — 구조 선택이라 정량 지표 없음 |
| 대안으로 검토했던 것 | 기록 없음(처음부터 이 패턴으로 설계된 것으로 보임) |
| 현재 미해결/보류 중인 결정 | **코드-문서 불일치 1건 발견**: `factory.py:23`의 `os.getenv("JUDGE_BACKEND", "local")`은 env var 미설정 시 **`local`**로 폴백하는데, `.env.example`은 `JUDGE_BACKEND=openai`를 "기본값"이라 명시하고 있음. 실사용엔 `.env.example`을 복사한 `.env`에 값이 이미 채워져 있어 영향 없지만, 그 줄을 지우거나 `JUDGE_BACKEND` 자체를 빼면 코드는 조용히 `local`로 폴백함 — `factory.py`의 주석("미설정 시 기존과 동일하게 항상 Ollama")도 gpt-5.6-luna 채택 이전 동작을 그대로 설명하는 **stale 주석** |
| 트레이드오프 | 얇은 인터페이스(메서드 1개)라 백엔드 추가가 쉽지만, LangGraph tool-calling 때문에 `LLMBackend`와 `BaseChatModel` **두 인터페이스가 공존**(이중 구현 비용) — `RunPodBackend`(순수 클래스)와 `ChatRunPod`(LangChain 어댑터)가 같은 RunPod 엔드포인트를 서로 다른 계약으로 감쌈 |

---

## 포트폴리오 발표 시 예상 질문

| 예상 질문 | 참고할 표 |
|---|---|
| "왜 GPT를 안 쓰고 Qwen2.5를 생성 모델로 썼나요?" | §1 표 — 특히 budget=5 비교표(속도 동률화·GPT-4o-mini의 budget=5 결함) |
| "생성 모델과 평가(Judge) 모델을 왜 다른 백엔드로 분리했나요?" | §2 표 + "왜 다른 백엔드를 쓰는가" 단락 — 검증-배포 불일치 해소가 핵심 |
| "RAG는 하이브리드 검색인가요, dense만 쓰나요?" | §5 표 — dense-only임을 정직히 답하고, 근거가 왜 없는지("그냥 그렇게 구현됨")까지 설명 가능 |
| "Judge 모델이 믿을 만한지 어떻게 검증했나요?" | §2 "실측 수치" 행(kappa/MAE) + EVAL.md §4 최신 측정 |
| "이 프로젝트에서 아직 확신이 안 서거나 미해결인 결정이 있다면?" | §5(하이브리드 검색 미착수), §6(factory.py 기본값-문서 불일치) — 둘 다 이번에 코드 직접 확인으로 찾아낸 실제 갭 |

---

## 공통 제약 (전 항목에 적용)

- **하드룰 1 (실제 학생 데이터 미사용)**: 생성 모델 비교는 교사 입력 예시 문제 지문(개인정보 아님), Judge 비교는 ITEM_GOLDEN/STRUCTURE_GOLDEN(합성/공개 성취기준 기반). RAG 골든셋(retrieval_golden_final.json)도 공개 자료 청크 기반.
- **RunPod 서버리스는 컴퓨트 시간 과금** — "속도가 느리면 곧 비용 증가"가 생성 모델 선정에 실질적 제약으로 작용(§1, budget=5 속도 동률 확인이 중요했던 이유).
- **GPT/OpenAI 백엔드의 용도 확장**: 생성 모델 비교 실험은 합성 데이터 전용으로 유지. Judge는 2026-07-23부터 프로덕션 런타임에도 gpt-5.6-luna를 실제 사용 — "로컬 우선"은 생성 모델엔 유지되지만 Judge엔 신뢰도를 우선한 의도적 예외.
