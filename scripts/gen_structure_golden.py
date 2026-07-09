#!/usr/bin/env python
"""STRUCTURE_GOLDEN 생성 스크립트 (num_items 아키텍처 반영, 2026-07-09 전면 재구성).

실제 출제 그래프(agent_node, qwen2.5:7b)를 그대로 호출해 passage_text·num_items
조합별로 "진짜 모델이 생성한" generated_items를 뽑는다. 라벨링(human_label)은
하지 않음 — 이 스크립트는 모델 출력 생성까지만 담당하고, 사람이 이후
data/golden/structure_golden.json을 열어 human_label을 직접 채운다.

count_match(생성 개수가 예시 문제 개수와 일치하는가)라는 옛 전제는 폐기됐다.
생성 개수는 passage_text와 무관하게 num_items가 결정하므로, 여기서는
passage_text(스타일/난이도 참고용)와 num_items(목표 개수)를 각 샘플에 함께 지정한다.

--budget N: graph budget (기본 3)
--only id1,id2,...: 지정한 id만 재생성 (기본: 전체). 기존 파일에서 같은 id만
                     교체하고, 라벨링된(human_label 있는) 다른 항목은 그대로 보존한다.
--drop id1,id2,...: 기존 파일에서 완전히 제거할 id (예: 문항 0개로 골든셋 부적합 판정된 것)

출력: data/golden/structure_golden.json (entries 배열에 직접 기록, human_label: null)
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

_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "structure_golden.json")

_SCHEMA = {
    "description": (
        "STRUCTURE_GOLDEN — num_items 기반 아키텍처(passage_text는 스타일/난이도 참고용, "
        "개수는 spec.num_items가 결정) 하의 구조 유사도 Judge 신뢰도 검증용 골든셋. "
        "개수 일치 여부는 코드가 결정론적으로 검증하므로 사람 라벨링·kappa 대상에서 제외."
    ),
    "entry_fields": {
        "id": "고유 id",
        "passage_text": "예시 문제 원문 (스타일/난이도 참고용, 개수와 무관)",
        "num_items": "이 생성에 지정된 목표 개수",
        "generated_items": "실제 생성된 문항 세트",
        "human_label": {
            "type_ratio_score": "사람이 판단한 유형 구성 비율 유사도 (0.0~1.0)",
            "difficulty_match": "사람이 판단한 난이도 구성 부합 여부 (bool)",
            "overall_score": "사람이 매긴 종합 유사도 점수 (0~5 정수). human_label 전체가 null이면 라벨링 전(eval_structure_judge가 자동 스킵).",
        },
    },
    "provenance": (
        "2026-07-09 count_match 폐기·num_items 도입 이후 실제 qwen2.5:7b 출력 기반으로 "
        "전면 재생성. 이전 부트스트랩(Claude 합성 str_001~003)과 count_match 기반 pending "
        "항목은 아키텍처 변경으로 전부 폐기하고 새로 생성함."
    ),
    "how_to_label": (
        "passage_text·num_items와 generated_items를 비교해 사람이 human_label을 채운다 "
        "(개수 일치 여부는 라벨링 대상이 아님 — len(generated_items)==num_items로 코드가 이미 검증). "
        "eval_structure_judge()는 이 고정된 쌍을 LLM에게 다시 보여주고 판단시킨 뒤, "
        "그 결과를 human_label과 비교한다."
    ),
}

# num_items를 passage_text 자체의 예시 문항 수와 다르게 줘서, 생성 개수가 예시가 아니라
# num_items를 따르는지(디커플링) 검증한다. 단일 지문형(예시 1문항)과 다중 지문형을 섞는다.
PASSAGE_SAMPLES = [
    {
        "id": "str_001",
        "passage_text": (
            "1. 헌법이 보장하는 기본권 중 자유권에 해당하는 것은?\n"
            "① 교육받을 권리 ② 신체의 자유 ③ 근로의 권리 ④ 환경권"
        ),
        "standards": ["기본권의 종류와 보장"],
        "num_items": 3,
    },
    {
        "id": "str_002",
        "passage_text": (
            "1. 수요와 공급의 법칙에 따라 균형 가격이 형성되는 원리를 서술하시오."
        ),
        "standards": ["시장 가격의 결정 원리"],
        "num_items": 5,
    },
    {
        "id": "str_003",
        "passage_text": (
            "1. 삼권 분립 원칙에 따라 입법부가 담당하는 기능으로 옳은 것은?\n"
            "① 법률 집행 ② 법률 제정 ③ 재판 ④ 행정 감독\n\n"
            "2. 사회 계층화 현상이 나타나는 원인을 서술하시오."
        ),
        "standards": ["권력 분립의 원리", "사회 계층과 불평등"],
        "num_items": 5,
    },
    {
        "id": "str_004",
        "passage_text": (
            "1. (하) 공정 무역이 추구하는 목표로 가장 적절한 것은?\n"
            "① 다국적 기업의 이윤 극대화 ② 생산자에게 정당한 대가 지급 ③ 관세 철폐 ④ 저가 경쟁 촉진\n\n"
            "2. (중) 다국적 기업의 공간적 분업이 개발도상국 경제에 미치는 영향을 서술하시오.\n\n"
            "3. (상) 다음 사례에서 나타난 국제 무역 갈등의 원인을 분석하여 서술하시오."
        ),
        "standards": ["국제 무역과 다국적 기업"],
        "num_items": 3,
    },
    {
        "id": "str_005",
        "passage_text": (
            "1. 근로자의 단결권·단체교섭권·단체행동권을 통틀어 이르는 말은?\n"
            "① 노동 3권 ② 사회권 ③ 자유권 ④ 청구권\n\n"
            "2. 부당 해고를 당한 근로자가 구제받을 수 있는 기관으로 옳은 것은?\n"
            "① 헌법재판소 ② 노동위원회 ③ 국가인권위원회 ④ 감사원\n\n"
            "3. 최저임금제도가 노동 시장에 미치는 영향을 서술하시오.\n\n"
            "4. 비정규직 근로자 보호를 위한 제도적 방안을 두 가지 이상 서술하시오."
        ),
        "standards": ["노동자의 권리와 노동법"],
        "num_items": 7,
    },
    {
        "id": "str_006",
        "passage_text": (
            "1. 국제 사회에서 비정부기구(NGO)의 역할로 가장 적절한 것은?\n"
            "① 조약 체결의 당사자가 된다 ② 인도적 지원과 감시 활동을 한다 ③ 군사력을 행사한다 ④ 관세를 부과한다"
        ),
        "standards": ["국제 사회의 행위 주체"],
        "num_items": 5,
    },
    {
        "id": "str_007",
        "passage_text": (
            "1. 물가 상승률과 실업률 사이의 상충 관계를 나타내는 곡선은?\n"
            "① 로렌츠 곡선 ② 필립스 곡선 ③ 로지스틱 곡선 ④ 래퍼 곡선\n\n"
            "2. 경기 침체기에 정부가 취할 수 있는 재정 정책을 서술하시오."
        ),
        "standards": ["경기 변동과 안정화 정책"],
        "num_items": 7,
    },
    {
        "id": "str_008",
        "passage_text": (
            "1. 대중매체가 여론 형성에 미치는 긍정적 영향으로 옳은 것은?\n"
            "① 정보 왜곡 ② 다양한 의견의 공론화 ③ 여론 조작 ④ 정보 독점\n\n"
            "2. 미디어 리터러시가 필요한 이유를 서술하시오.\n\n"
            "3. 가짜 뉴스가 민주주의에 미치는 부정적 영향으로 옳은 것은?\n"
            "① 시민의 합리적 판단 저해 ② 언론의 자유 신장 ③ 정보 접근성 향상 ④ 여론의 다양성 확대\n\n"
            "4. SNS를 통한 정치 참여 확대의 순기능을 서술하시오.\n\n"
            "5. 필터 버블 현상이 여론 양극화에 미치는 영향을 서술하시오."
        ),
        "standards": ["미디어와 사회 참여"],
        "num_items": 5,
    },
    {
        "id": "str_009",
        "passage_text": (
            "1. 다음 사례에서 나타난 문화 접변의 유형을 쓰고, 그렇게 판단한 근거를 서술하시오."
        ),
        "standards": ["문화 접변의 유형과 사례"],
        "num_items": 3,
    },
    {
        "id": "str_010",
        "passage_text": (
            "1. 지방분권이 필요한 이유로 가장 적절한 것은?\n"
            "① 중앙정부 권한 강화 ② 지역 실정에 맞는 행정 실현 ③ 행정 절차 단축 ④ 예산 절감"
        ),
        "standards": ["지방분권과 중앙집권"],
        "num_items": 5,
    },
    {
        "id": "str_011",
        "passage_text": (
            "1. 환율이 상승(원화 가치 하락)할 때 나타나는 현상으로 옳은 것은?\n"
            "① 수출 감소 ② 수입 물가 상승 ③ 해외여행 비용 감소 ④ 외채 부담 감소\n\n"
            "2. 경상수지 흑자가 지속될 때 국내 경제에 미치는 영향을 서술하시오."
        ),
        "standards": ["환율과 국제수지"],
        "num_items": 5,
    },
    {
        "id": "str_012",
        "passage_text": (
            "1. 사회적 소수자를 보호하기 위한 제도적 방안으로 가장 적절한 것은?\n"
            "① 차별금지법 제정 ② 다수결 원칙 강화 ③ 동화 정책 강제 ④ 정보 제한"
        ),
        "standards": ["사회적 소수자와 인권 보호"],
        "num_items": 3,
    },
    {
        "id": "str_013",
        "passage_text": (
            "1. 다음 중 정치 참여 유형이 다른 것은?\n"
            "① 선거 투표 ② 정당 가입 ③ 이익집단 활동 ④ 헌법재판소 위헌 결정\n\n"
            "2. 이익집단과 정당의 차이점을 서술하시오.\n\n"
            "3. 시민의 정치 참여가 저조할 때 나타나는 문제점을 서술하시오."
        ),
        "standards": ["정치 참여의 유형과 방법"],
        "num_items": 5,
    },
    {
        "id": "str_014",
        "passage_text": (
            "1. 저출산·고령화 현상이 사회보장 재정에 미치는 영향을 서술하시오."
        ),
        "standards": ["인구 구조 변화와 사회 문제"],
        "num_items": 3,
    },
    {
        "id": "str_015",
        "passage_text": (
            "1. 세계인권선언과 국제인권규약의 차이점으로 옳은 것은?\n"
            "① 세계인권선언은 법적 구속력이 있다 ② 국제인권규약은 조약으로서 비준국에 구속력을 가진다 "
            "③ 국제인권규약은 선언적 의미만 갖는다 ④ 둘 다 구속력이 없다\n\n"
            "2. 국제인권규약이 비준국의 국내법에 미치는 영향을 서술하시오."
        ),
        "standards": ["국제 인권 규범"],
        "num_items": 7,
    },
    {
        "id": "str_016",
        "passage_text": (
            "1. 소비자 잉여에 대한 설명으로 옳은 것은?\n"
            "① 생산자가 얻는 이윤 ② 소비자가 지불할 용의보다 실제로 적게 지불해 얻는 이득 "
            "③ 정부가 거두는 세금 ④ 시장에서 발생하는 손실"
        ),
        "standards": ["시장의 효율성과 잉여"],
        "num_items": 5,
    },
    {
        "id": "str_017",
        "passage_text": (
            "1. 지역 이기주의 중 님비(NIMBY) 현상의 사례로 옳은 것은?\n"
            "① 지역에 종합병원 유치를 요구하는 것 ② 지역에 쓰레기 소각장 건설을 반대하는 것 "
            "③ 지역에 지하철역 신설을 요구하는 것 ④ 지역에 대학 유치를 요구하는 것\n\n"
            "2. 님비 현상과 핌피(PIMFY) 현상의 공통된 발생 원인을 서술하시오."
        ),
        "standards": ["지역 이기주의와 공동체 갈등"],
        "num_items": 3,
    },
    {
        "id": "str_018",
        "passage_text": (
            "1. 세대 간 갈등이 심화되는 원인으로 가장 적절한 것은?\n"
            "① 세대별 가치관과 이해관계의 차이 ② 세대 간 완전한 가치관 일치 ③ 인구 감소 ④ 지역 격차 해소"
        ),
        "standards": ["세대 갈등과 사회 통합"],
        "num_items": 5,
    },
    {
        "id": "str_019",
        "passage_text": (
            "1. 사법부의 독립이 필요한 이유로 옳은 것은?\n"
            "① 행정부의 효율적 통제를 위해 ② 공정한 재판을 통한 국민의 권리 보장을 위해 "
            "③ 입법부의 권한 강화를 위해 ④ 신속한 재판 진행만을 위해\n\n"
            "2. 법관의 신분 보장이 사법권 독립에 기여하는 방식을 서술하시오."
        ),
        "standards": ["사법부의 독립과 기능"],
        "num_items": 7,
    },
    {
        "id": "str_020",
        "passage_text": "1. 국민참여재판제도에서 배심원의 역할로 옳은 것은?\n① 판결의 최종 확정 ② 유·무죄에 대한 의견 제시 ③ 형량의 강제 결정 ④ 검사 역할 대행",
        "standards": ["국민참여재판제도"],
        "num_items": 3,
    },
    {
        "id": "str_021",
        "passage_text": "1. 환경영향평가 제도의 목적으로 가장 적절한 것은?\n① 개발 사업의 신속한 승인 ② 개발이 환경에 미치는 영향을 사전에 예측·평가 ③ 기업의 이윤 극대화 ④ 인허가 절차 폐지",
        "standards": ["환경 정책과 지속가능발전"],
        "num_items": 5,
    },
    {
        "id": "str_022",
        "passage_text": "1. 소득 불평등 정도를 나타내는 지표로 옳은 것은?\n① 지니계수 ② 물가상승률 ③ 실업률 ④ 경제성장률",
        "standards": ["소득 분배와 불평등 지표"],
        "num_items": 3,
    },
    {
        "id": "str_023",
        "passage_text": "1. 언론의 자유가 민주주의에서 중요한 이유를 서술하시오.",
        "standards": ["언론의 자유와 알 권리"],
        "num_items": 5,
    },
    {
        "id": "str_024",
        "passage_text": "1. 다수결 원칙의 한계로 옳은 것은?\n① 소수 의견이 항상 반영됨 ② 소수자의 권리가 침해될 수 있음 ③ 결정이 항상 정확함 ④ 시간이 오래 걸리지 않음",
        "standards": ["다수결 원칙과 소수자 보호"],
        "num_items": 3,
    },
    {
        "id": "str_025",
        "passage_text": "1. 지적재산권 보호가 필요한 이유로 가장 적절한 것은?\n① 창작 의욕 저하 ② 창작자의 권리 보호와 기술 혁신 촉진 ③ 정보 독점 강화 ④ 가격 상승 유도",
        "standards": ["지적재산권의 의의와 보호"],
        "num_items": 5,
    },
    {
        "id": "str_026",
        "passage_text": "1. 조세 저항이 발생하는 주된 원인을 서술하시오.",
        "standards": ["조세의 종류와 조세 저항"],
        "num_items": 3,
    },
    {
        "id": "str_027",
        "passage_text": "1. 사회 갈등이 순기능을 하는 경우로 옳은 것은?\n① 갈등이 폭력으로 번질 때 ② 갈등을 통해 문제를 공론화하고 해결책을 모색할 때 ③ 갈등이 장기화될 때 ④ 갈등 당사자가 소통을 거부할 때",
        "standards": ["사회 갈등의 기능"],
        "num_items": 5,
    },
    {
        "id": "str_028",
        "passage_text": "1. 디지털 디바이드(정보 격차)가 심화될 때 나타날 수 있는 문제를 서술하시오.",
        "standards": ["정보사회와 디지털 격차"],
        "num_items": 3,
    },
    {
        "id": "str_029",
        "passage_text": "1. 선거공영제를 시행하는 목적으로 옳은 것은?\n① 후보자 간 재력 차이에 따른 불공정 완화 ② 특정 정당 지원 ③ 선거 비용 증대 ④ 투표율 저하",
        "standards": ["선거공영제와 선거의 공정성"],
        "num_items": 5,
    },
    {
        "id": "str_030",
        "passage_text": "1. 헌법 개정 절차 중 국민투표가 필요한 이유를 서술하시오.",
        "standards": ["헌법 개정 절차"],
        "num_items": 3,
    },
    {
        "id": "str_031",
        "passage_text": "1. 노동조합의 주요 기능으로 옳은 것은?\n① 사용자의 이윤 극대화 지원 ② 근로자의 근로 조건 개선을 위한 단체교섭 ③ 근로자 해고 결정 ④ 정부 정책 집행",
        "standards": ["노동조합의 기능과 역할"],
        "num_items": 5,
    },
    {
        "id": "str_032",
        "passage_text": "1. 공공부조와 사회보험의 차이점으로 옳은 것은?\n① 공공부조는 보험료 사전 납부가 필수이다 ② 공공부조는 조세를 재원으로 하고 소득 조사를 거친다 ③ 사회보험은 소득 조사를 거친다 ④ 둘은 재원이 동일하다",
        "standards": ["사회보장제도의 유형"],
        "num_items": 3,
    },
    {
        "id": "str_033",
        "passage_text": "1. 국제기구 유엔환경계획(UNEP)의 주요 활동으로 옳은 것은?\n① 군사 개입 ② 국제 환경 문제에 대한 조사·연구·정책 조정 ③ 관세 부과 ④ 통화 발행",
        "standards": ["국제환경기구의 역할"],
        "num_items": 5,
    },
    {
        "id": "str_034",
        "passage_text": "1. 다음 사례에서 나타난 세대 간 소득 재분배 방식의 특징을 서술하시오.",
        "standards": ["세대 간 소득 재분배"],
        "num_items": 3,
    },
]


def generate_one(sample: dict, budget: int) -> dict:
    spec: ExamSpec = {
        "passage_text": sample["passage_text"],
        "standards": sample.get("standards", []),
        "num_items": sample["num_items"],
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
        "num_items": sample["num_items"],
        "generated_items": generated_items,
        "human_label": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--only", type=str, default="")
    parser.add_argument("--drop", type=str, default="", help="기존 파일에서 완전히 제거할 id")
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

    print(f"=== STRUCTURE_GOLDEN 생성 ({os.environ['OLLAMA_MODEL']}, budget={args.budget}) ===")
    print(f"대상 passage 수: {len(samples)}\n")

    for i, sample in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {sample['id']} (num_items={sample['num_items']}) 생성 중...")
        try:
            entry = generate_one(sample, args.budget)
        except Exception as e:
            print(f"  실패: {e}")
            entry = {
                "id": sample["id"],
                "passage_text": sample["passage_text"],
                "num_items": sample["num_items"],
                "generated_items": [],
                "error": str(e),
                "human_label": None,
            }
        n_items = len(entry.get("generated_items", []))
        print(f"  생성된 문항: {n_items}개 (목표 {sample['num_items']}개)")
        existing_by_id[sample["id"]] = entry

    order = [s["id"] for s in PASSAGE_SAMPLES]
    entries = [existing_by_id[i] for i in order if i in existing_by_id]
    entries += [e for eid, e in existing_by_id.items() if eid not in order]

    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"_schema": _SCHEMA, "entries": entries}, f, ensure_ascii=False, indent=2)
    print(f"\n완료 — {_OUT_PATH} 에 {len(entries)}개 저장 (이번 실행 {len(samples)}개 갱신)")


if __name__ == "__main__":
    main()
