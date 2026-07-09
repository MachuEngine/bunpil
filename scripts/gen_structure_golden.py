#!/usr/bin/env python
"""STRUCTURE_GOLDEN 후보 생성 스크립트.

실제 출제 그래프(agent_node, qwen2.5:7b)를 그대로 호출해 passage_text 별로
"진짜 모델이 생성한" generated_items를 뽑는다. 라벨링(human_label)은 하지 않음 —
이 스크립트는 모델 출력 생성까지만 담당하고, 사람이 이후 human_label을 채워
data/golden/structure_golden.json에 병합한다.

기본 budget=1(재시도 없음): STRUCTURE_GOLDEN은 구조 Judge의 신뢰도를 검증하는
골든셋이므로, 원래는 검증 통과까지 재시도한 "이미 걸러진" 결과가 아니라 모델의
1회차 원본 출력(구조가 안 맞는 경우 포함)을 담는 게 목적이었다. 다만 budget=1로
6개 중 5개가 문항 0개로 끝난 사례가 나와(qwen2.5:7b tool-calling 안정성 문제,
blog_draft.md 참고) --budget으로 재시도 횟수를 조정할 수 있게 열어둔다.

--budget N: graph budget (기본 1)
--only id1,id2,...: 지정한 id만 재생성 (기본: 전체). 기존 출력 파일에서 같은
                     id만 교체하고 나머지는 유지(병합).

출력: data/golden/structure_golden_pending.json (기존 structure_golden.json은 건드리지 않음)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:7b")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db")

from app.modules.exam import ExamSpec, get_exam_graph
from app.modules.exam.tools import init_session

_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "structure_golden_pending.json")

# 기존 str_001~003(부트스트랩)과 겹치지 않는 다양한 구성의 예시 문제.
# 문항 수·유형 비율·난이도 구성을 의도적으로 다양하게 섞어, 모델이 지문 구조를
# 얼마나 잘 따르는지(혹은 못 따르는지) 그대로 드러나게 한다.
PASSAGE_SAMPLES = [
    {
        "id": "str_005",
        "passage_text": (
            "1. 공공재의 특성으로 옳은 것은?\n"
            "① 배제성과 경합성을 모두 가진다 ② 비배제성과 비경합성을 가진다 "
            "③ 시장에서 최적으로 공급된다 ④ 무임승차 문제가 발생하지 않는다\n\n"
            "2. 시장 실패 상황에서 정부가 개입하는 방식을 두 가지 이상 서술하시오."
        ),
        "standards": ["시장 실패와 정부의 역할"],
    },
    {
        "id": "str_007",
        "passage_text": (
            "1. 문화 변동의 외재적 요인에 해당하는 것은?\n"
            "① 발명 ② 발견 ③ 직접 전파 ④ 자극 전파\n\n"
            "2. 다음 중 문화 접변의 결과로 옳지 않은 것은?\n"
            "① 문화 동화 ② 문화 병존 ③ 문화 융합 ④ 문화 고립\n\n"
            "3. 아노미 현상이 나타나는 원인으로 옳은 것은?\n"
            "① 문화 지체 ② 급격한 사회 변동으로 인한 규범 혼란 ③ 문화 사대주의 ④ 문화 상대주의\n\n"
            "4. 다음 중 하위문화에 해당하지 않는 것은?\n"
            "① 청소년 문화 ② 지역 문화 ③ 반문화 ④ 주류 문화"
        ),
        "standards": ["문화 변동의 요인과 양상"],
    },
    {
        "id": "str_008",
        "passage_text": (
            "1. 헌법재판소가 위헌법률심판을 통해 보호하고자 하는 헌법상 원리를 서술하시오.\n\n"
            "2. 헌법소원심판과 위헌법률심판의 차이점을 청구 주체와 대상을 중심으로 서술하시오."
        ),
        "standards": ["헌법재판소의 기능과 역할"],
    },
    {
        "id": "str_010",
        "passage_text": (
            "1. 근로기준법상 최저임금제도의 목적으로 가장 적절한 것은?\n"
            "① 기업의 이윤 극대화 ② 근로자의 최저 생활 보장 ③ 물가 상승 유도 ④ 노동조합 결성 제한"
        ),
        "standards": ["근로자의 권리와 노동법"],
    },
    {
        "id": "str_011",
        "passage_text": (
            "1. 직접세와 간접세를 구분하는 기준으로 옳은 것은?\n"
            "① 세율의 높고 낮음 ② 납세자와 담세자의 일치 여부 ③ 국세와 지방세의 구분 ④ 과세 대상의 종류\n\n"
            "2. 간접세가 소득 역진적이라는 평가를 받는 이유를 서술하시오."
        ),
        "standards": ["조세의 종류와 원칙"],
    },
    {
        "id": "str_012",
        "passage_text": (
            "1. 국제법상 조약이 국내에서 법적 효력을 갖기 위한 절차로 옳은 것은?\n"
            "① 대통령 서명만으로 충분하다 ② 국회의 동의 없이도 발효된다 ③ 헌법에 따라 국회 비준 동의가 필요할 수 있다 ④ 조약은 국내법과 무관하다\n\n"
            "2. 조약과 국내법이 충돌할 때 이를 조정하는 원리를 서술하시오."
        ),
        "standards": ["국제법의 법원과 국내 적용"],
    },
    {
        "id": "str_013",
        "passage_text": (
            "1. (하) 선거구 법정주의를 채택하는 목적으로 가장 적절한 것은?\n"
            "① 특정 정당에 유리한 선거구 획정 허용 ② 게리맨더링 방지 ③ 선거 비용 절감 ④ 투표율 상승\n\n"
            "2. (중) 게리맨더링이 대의 민주주의에 미치는 부정적 영향을 서술하시오.\n\n"
            "3. (상) 다음 사례에서 나타난 선거구 획정의 문제점을 분석하여 서술하시오."
        ),
        "standards": ["선거의 기본 원칙과 선거구 제도"],
    },
    {
        "id": "str_014",
        "passage_text": (
            "1. 사회보험에 해당하는 것은?\n"
            "① 국민기초생활보장제도 ② 의료급여 ③ 국민연금 ④ 긴급복지지원\n\n"
            "2. 공공부조의 특징으로 옳은 것은?\n"
            "① 사전 보험료 납부가 필수이다 ② 소득 재분배 효과가 크다 ③ 수혜 대상에 소득 제한이 없다 ④ 가입이 강제되지 않는다"
        ),
        "standards": ["사회보장제도의 유형과 특징"],
    },
    {
        "id": "str_015",
        "passage_text": (
            "1. 탄소중립 정책이 산업 구조에 미치는 영향과, 이에 대응하기 위한 정부의 정책 수단을 두 가지 이상 서술하시오."
        ),
        "standards": ["지속가능발전과 환경 정책"],
    },
    {
        "id": "str_016",
        "passage_text": (
            "1. 죄형법정주의의 내용으로 옳지 않은 것은?\n"
            "① 관습형법 금지 원칙 ② 유추해석 금지 원칙 ③ 소급효 금지 원칙 ④ 유추해석을 통한 처벌 허용\n\n"
            "2. 죄형법정주의가 인권 보장에 기여하는 방식을 서술하시오."
        ),
        "standards": ["법치주의와 죄형법정주의"],
    },
    {
        "id": "str_017",
        "passage_text": (
            "1. 문화 상대주의와 윤리 상대주의를 구분하는 설명으로 옳은 것은?\n"
            "① 둘 다 보편 윤리를 부정한다 ② 문화 상대주의는 문화적 차이 이해를, 윤리 상대주의는 도덕 판단의 상대성을 강조한다 "
            "③ 문화 상대주의는 인권 침해도 정당화한다 ④ 윤리 상대주의는 문화 다양성과 무관하다\n\n"
            "2. 다음 사례에서 문화 상대주의적 관점의 한계를 서술하시오."
        ),
        "standards": ["문화 상대주의와 보편 윤리"],
    },
    {
        "id": "str_018",
        "passage_text": (
            "1. 정당의 기능으로 가장 적절한 것은?\n"
            "① 사법부 견제 ② 여론을 수렴해 정책으로 전환 ③ 행정부 대체 ④ 언론 통제"
        ),
        "standards": ["정당의 의의와 기능"],
    },
    {
        "id": "str_019",
        "passage_text": (
            "1. UN 안전보장이사회 상임이사국의 특징으로 옳은 것은?\n"
            "① 모든 회원국이 돌아가며 맡는다 ② 거부권을 행사할 수 있다 ③ 임기가 2년이다 ④ 투표권이 없다\n\n"
            "2. 안전보장이사회의 거부권 제도가 국제 사회의 의사 결정에 미치는 영향을 서술하시오."
        ),
        "standards": ["국제 사회의 행위 주체와 국제기구"],
    },
    {
        "id": "str_020",
        "passage_text": (
            "1. 세대 내 이동에 해당하는 사례로 옳은 것은?\n"
            "① 농부의 자녀가 의사가 된 경우 ② 평사원이 임원으로 승진한 경우 ③ 귀족 자녀가 귀족 지위를 유지한 경우 ④ 노비의 자녀가 노비가 된 경우\n\n"
            "2. 개방적 계층 구조가 사회 이동에 미치는 영향을 서술하시오."
        ),
        "standards": ["사회 이동과 계층 구조"],
    },
    {
        "id": "str_021",
        "passage_text": (
            "1. 독과점 시장에서 나타나는 문제점으로 옳은 것은?\n"
            "① 가격 경쟁 심화 ② 자원 배분의 비효율성 ③ 소비자 후생 증대 ④ 신규 기업 진입 용이\n\n"
            "2. 공정거래법이 독과점 규제를 위해 사용하는 수단으로 옳은 것은?\n"
            "① 기업 결합 무조건 승인 ② 시장 지배적 지위 남용 행위 금지 ③ 담합 장려 ④ 중소기업 진입 제한"
        ),
        "standards": ["시장 구조와 정부 규제"],
    },
    {
        "id": "str_022",
        "passage_text": (
            "1. 헌법 제37조 2항에 따라 국민의 기본권을 법률로 제한할 때 지켜야 할 원칙을 서술하시오."
        ),
        "standards": ["기본권의 제한과 한계"],
    },
    {
        "id": "str_023",
        "passage_text": (
            "1. 다문화 정책 중 동화주의 모델의 특징으로 옳은 것은?\n"
            "① 이주민 고유문화를 그대로 유지시킨다 ② 이주민이 주류 문화에 흡수되도록 유도한다 "
            "③ 다양한 문화가 대등하게 공존하도록 한다 ④ 문화 간 우열을 인정하지 않는다\n\n"
            "2. 다문화 사회에서 문화적 다양성과 사회 통합을 동시에 추구하기 위한 방안을 서술하시오."
        ),
        "standards": ["다문화 사회와 문화 다양성"],
    },
]


def generate_one(sample: dict, budget: int) -> dict:
    spec: ExamSpec = {
        "passage_text": sample["passage_text"],
        "standards": sample.get("standards", []),
    }
    init_session()
    graph = get_exam_graph()
    state = graph.invoke(
        {
            "spec": spec,
            "budget": budget,
            "draft_items": [],
            "agent_messages": [],
            "validation_passed": False,
            "similarity_judge_result": {},
            "error": "",
        }
    )
    items = state.get("draft_items", [])
    generated_items = [
        {
            "question": it.get("question", ""),
            "options": it.get("options", []),
            "answer": it.get("answer", ""),
            "item_type": it.get("item_type", ""),
            "difficulty": it.get("difficulty", ""),
        }
        for it in items
    ]
    return {
        "id": sample["id"],
        "passage_text": sample["passage_text"],
        "generated_items": generated_items,
        "model_similarity_judge_self_report": state.get("similarity_judge_result", {}),
        "human_label": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--only", type=str, default="")
    parser.add_argument("--drop", type=str, default="", help="기존 출력에서 완전히 제거할 id (예: 문항 0개로 골든셋 부적합 판정된 것)")
    args = parser.parse_args()

    only_ids = {s.strip() for s in args.only.split(",") if s.strip()} or None
    samples = [s for s in PASSAGE_SAMPLES if only_ids is None or s["id"] in only_ids]
    drop_ids = {s.strip() for s in args.drop.split(",") if s.strip()}

    existing_by_id = {}
    if os.path.exists(_OUT_PATH):
        with open(_OUT_PATH, encoding="utf-8") as f:
            for e in json.load(f).get("entries", []):
                if e["id"] not in drop_ids:
                    existing_by_id[e["id"]] = e

    print(f"=== STRUCTURE_GOLDEN 후보 생성 ({os.environ['OLLAMA_MODEL']}, budget={args.budget}) ===")
    print(f"대상 passage 수: {len(samples)}\n")

    for i, sample in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {sample['id']} 생성 중...")
        try:
            entry = generate_one(sample, args.budget)
        except Exception as e:
            print(f"  실패: {e}")
            entry = {
                "id": sample["id"],
                "passage_text": sample["passage_text"],
                "generated_items": [],
                "error": str(e),
                "human_label": None,
            }
        n_items = len(entry.get("generated_items", []))
        print(f"  생성된 문항: {n_items}개")
        existing_by_id[sample["id"]] = entry

    # PASSAGE_SAMPLES 순서대로 정렬해 저장(병합된 다른 id가 있으면 뒤에 붙임)
    order = [s["id"] for s in PASSAGE_SAMPLES]
    entries = [existing_by_id[i] for i in order if i in existing_by_id]
    entries += [e for eid, e in existing_by_id.items() if eid not in order]

    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "_note": (
                    "gen_structure_golden.py가 생성한 라벨링 대기 항목. "
                    "human_label을 사람이 채운 뒤 structure_golden.json의 entries에 수동 병합할 것. "
                    "이 파일 자체는 eval_exam.py가 읽지 않음(human_label 없어 eval_structure_judge가 KeyError 남)."
                ),
                "entries": entries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n완료 — {_OUT_PATH} 에 {len(entries)}개 저장 (이번 실행 {len(samples)}개 갱신)")


if __name__ == "__main__":
    main()
