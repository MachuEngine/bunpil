#!/usr/bin/env python
"""STRUCTURE_GOLDEN 생성 스크립트 (num_items 아키텍처 반영, 2026-07-09 전면 재구성).

실제 출제 그래프(agent_node, qwen2.5:7b)를 그대로 호출해 passage_text·num_items
조합별로 "진짜 모델이 생성한" generated_items를 뽑는다. 라벨링(human_label)은
하지 않음 — 이 스크립트는 모델 출력 생성까지만 담당하고, 사람이 이후
data/golden/structure_golden.json을 열어 human_label을 직접 채운다.

2026-07-23: 런타임 구조 유사도 판단은 더 이상 생성 에이전트의 자기채점(self-judge)이
아니다 — get_judge_backend()를 호출하는 별도 judge_node로 분리됐고, 이는 오프라인
eval(judge_structure_one())과 완전히 같은 코드(app/modules/exam/judge.py)를 공유한다.
따라서 이 스크립트가 뽑는 generated_items를 오프라인 eval_structure_judge()로 채점한
결과가 곧 "런타임에 실제로 쓰이는 judge"의 신뢰도이며, 별도의 self-judge 캡처 필드는
더 이상 필요 없다(2026-07-22에 잠시 도입했던 self_judge_result/self_judge_passed
필드는 self-judge 자체가 폐기되며 함께 제거됨).

count_match(생성 개수가 예시 문제 개수와 일치하는가)라는 옛 전제는 폐기됐다.
생성 개수는 passage_text와 무관하게 num_items가 결정하므로, 여기서는
passage_text(스타일/난이도 참고용)와 num_items(목표 개수)를 각 샘플에 함께 지정한다.

--budget N: graph budget (기본 3)
--only id1,id2,...: 지정한 id만 재생성 (기본: 전체). 기존 파일에서 같은 id만
                     교체하고, 라벨링된(human_label 있는) 다른 항목은 그대로 보존한다.
--drop id1,id2,...: 기존 파일에서 완전히 제거할 id (예: 문항 0개로 골든셋 부적합 판정된 것)

주의(2026-07-23): judge_node가 이제 실제로 매 생성마다 get_judge_backend()를 호출한다.
JUDGE_BACKEND 기본값은 openai(gpt-5.6-luna)라 OPENAI_API_KEY가 없으면 이 스크립트도
그대로 fail-fast로 실패한다. 골든셋 생성만 목적이면 `JUDGE_BACKEND=local` 환경변수로
로컬 Judge를 쓰는 것을 권장(비용 없음, 대량 생성에 적합).

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

from app.common.llm.tracing import init_langsmith_project
init_langsmith_project()

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
        "그 결과를 human_label과 비교한다.\n"
        "difficulty_match 판단 기준: (1) passage_text에 상/중/하가 명시된 경우(예: '1. (하) ...') "
        "그 라벨을 기준으로 generated_items의 difficulty와 비교. (2) 명시적 난이도 라벨이 없는 경우"
        "(예: str_001), 라벨러가 문항 내용(다루는 개념의 난이도, 보기 구성의 복잡도 등)을 보고 실제 "
        "교사라면 이 예시 문제를 어떤 난이도로 인식할지 암묵적으로 추론한 뒤, 그 추론된 기준선을 "
        "generated_items의 explicit difficulty 라벨과 비교해 difficulty_match를 채운다."
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
    # str_035~040: 2026-07-11, 재시도 구조 개선(부분 진행 보존) + strong 프롬프트 +
    # num_ctx=16384 + temperature=0.7(유지) 전부 반영된 코드로 생성. 기존 str_001~032는
    # 이 개선 이전(2026-07-10 01:20) 코드로 생성됐음 — human_label 시 구분 참고.
    {
        "id": "str_035",
        "passage_text": "1. 헌법재판소의 위헌법률심판 결정 유형 중 '헌법불합치'의 의미로 옳은 것은?\n① 법률이 즉시 효력을 상실한다 ② 법률의 위헌성은 인정하되 일정 기간 효력을 유지시킨다 ③ 법률이 처음부터 무효였다고 본다 ④ 법률의 합헌성을 확정한다",
        "standards": ["헌법재판소의 위헌법률심판"],
        "num_items": 5,
    },
    {
        "id": "str_036",
        "passage_text": "1. 완전경쟁시장의 특징으로 옳은 것은?\n① 소수의 공급자가 시장을 지배한다 ② 다수의 수요자와 공급자가 존재하고 진입·퇴출이 자유롭다 ③ 상품 차별화가 이루어진다 ④ 개별 기업이 가격을 결정한다\n\n2. 독점시장에서 자원 배분이 비효율적인 이유를 서술하시오.",
        "standards": ["시장 구조의 유형"],
        "num_items": 5,
    },
    {
        "id": "str_037",
        "passage_text": "1. 다음 사례에서 나타난 사회 문제 해결 방식의 한계를 서술하시오.",
        "standards": ["사회 문제와 해결 방안"],
        "num_items": 3,
    },
    {
        "id": "str_038",
        "passage_text": "1. 국제 사회에서 국가 주권 평등의 원칙이 지니는 의의로 옳은 것은?\n① 강대국의 우월한 지위를 인정 ② 모든 국가가 국제법상 동등한 지위를 가짐 ③ 약소국의 발언권을 제한 ④ 국제기구의 결정을 무시할 수 있음\n\n2. 국가 주권 평등 원칙이 현실에서 제약받는 사례를 서술하시오.",
        "standards": ["국가 주권과 국제 질서"],
        "num_items": 7,
    },
    {
        "id": "str_039",
        "passage_text": "1. 근로기준법상 부당해고 구제 절차로 옳은 것은?\n① 곧바로 민사소송을 제기해야 한다 ② 노동위원회에 구제를 신청할 수 있다 ③ 국회에 청원해야 한다 ④ 구제 절차가 없다",
        "standards": ["근로자 권리 구제 절차"],
        "num_items": 3,
    },
    {
        "id": "str_040",
        "passage_text": "1. 대의 민주주의에서 정당의 역할로 가장 적절한 것은?\n① 사법부 견제 ② 국민의 다양한 의사를 결집해 정책으로 제시 ③ 행정부 대체 ④ 언론 통제\n\n2. 대의 민주주의의 한계를 보완하는 직접 민주주의적 요소를 두 가지 이상 서술하시오.\n\n3. 정당 명부식 비례대표제가 지역구 국회의원 선거와 다른 점을 서술하시오.",
        "standards": ["정당 정치와 대의 민주주의"],
        "num_items": 5,
    },
    # str_041~046: 2026-07-11, 언어 오염(중국어)으로 분리된 6개(str_003/007/017/032/038/040)의
    # 대체분. 현재 코드(부분 진행 보존 + strong 프롬프트 + num_ctx=16384) 기준, budget=1.
    {
        "id": "str_041",
        "passage_text": "1. 문화 지체 현상의 사례로 가장 적절한 것은?\n① 기술 발전 속도를 제도와 의식이 따라가지 못하는 것 ② 전통 문화의 완전한 소멸 ③ 외래 문화의 전면 수용 ④ 문화 간 접촉의 단절",
        "standards": ["문화 변동과 문화 지체"],
        "num_items": 5,
    },
    {
        "id": "str_042",
        "passage_text": "1. 기회비용의 개념으로 옳은 것은?\n① 실제 지출한 금액만 포함한다 ② 어떤 선택으로 포기한 대안 중 가장 가치 있는 것 ③ 항상 화폐 단위로만 측정된다 ④ 선택과 무관하게 발생한다\n\n2. 매몰 비용을 의사 결정에서 고려하지 말아야 하는 이유를 서술하시오.",
        "standards": ["합리적 선택과 기회비용"],
        "num_items": 7,
    },
    {
        "id": "str_043",
        "passage_text": "1. 다음 사례에서 나타난 일탈 이론의 관점을 쓰고, 그 근거를 서술하시오.",
        "standards": ["일탈 행동의 이론적 관점"],
        "num_items": 3,
    },
    {
        "id": "str_044",
        "passage_text": "1. 국제 하천을 둘러싼 국가 간 갈등의 해결 방식으로 가장 적절한 것은?\n① 상류 국가의 일방적 개발 ② 국제 협약을 통한 공동 관리 ③ 군사적 봉쇄 ④ 하류 국가의 보상 포기\n\n2. 국제 환경 협약이 실효성을 갖기 어려운 이유를 서술하시오.",
        "standards": ["국제 환경 갈등과 협력"],
        "num_items": 5,
    },
    {
        "id": "str_045",
        "passage_text": "1. (하) 민법상 미성년자의 법률행위에 대한 설명으로 옳은 것은?\n① 모든 법률행위가 무효이다 ② 법정대리인의 동의 없이 한 행위는 취소할 수 있다 ③ 성년자와 동일하게 취급된다 ④ 취소권이 인정되지 않는다\n\n2. (중) 미성년자 보호 제도가 거래 상대방 보호와 충돌하는 지점을 서술하시오.\n\n3. (상) 다음 사례에서 미성년자의 계약 취소 가능 여부를 판단하고 근거를 서술하시오.",
        "standards": ["미성년자의 법률행위와 보호"],
        "num_items": 3,
    },
    {
        "id": "str_046",
        "passage_text": "1. 실업의 유형 중 경기적 실업에 해당하는 것은?\n① 더 나은 직장을 찾는 과정에서 생기는 실업 ② 산업 구조 변화로 기존 기술이 쓸모없어져 생기는 실업 ③ 경기 침체로 노동 수요가 줄어 생기는 실업 ④ 계절 변화에 따라 생기는 실업",
        "standards": ["실업의 유형과 대책"],
        "num_items": 5,
    },
    # str_047~049: 2026-07-11, save_item 언어 게이트(_check_korean) 적용 후 생성.
    # str_044~046이 언어 오염으로 분리되어 그 부족분(목표 20개 복원).
    {
        "id": "str_047",
        "passage_text": "1. 사회 계약설의 관점에서 국가의 성립 목적으로 옳은 것은?\n① 지배 계급의 이익 보호 ② 시민의 생명·자유·재산 보호 ③ 종교적 권위의 유지 ④ 영토 확장",
        "standards": ["사회 계약설과 국가의 역할"],
        "num_items": 5,
    },
    {
        "id": "str_048",
        "passage_text": "1. 보통 선거 원칙에 위배되는 사례로 옳은 것은?\n① 일정 연령 이상 모든 국민에게 투표권 부여 ② 납세액에 따라 투표권 차등 부여 ③ 비밀 투표 보장 ④ 직접 투표 실시\n\n2. 평등 선거 원칙과 보통 선거 원칙의 차이를 서술하시오.",
        "standards": ["민주 선거의 기본 원칙"],
        "num_items": 3,
    },
    {
        "id": "str_049",
        "passage_text": "1. 다음 사례에서 나타난 규모의 경제 효과를 설명하고, 이것이 시장 구조에 미치는 영향을 서술하시오.",
        "standards": ["규모의 경제와 시장 구조"],
        "num_items": 7,
    },
    # str_050~051: 2026-07-11, str_025/str_049가 v3(게이트+재시도 로직 적용)에서도
    # 0문항으로 끝나 골든셋 목표 20개를 채우기 위한 대체분.
    {
        "id": "str_050",
        "passage_text": "1. 지방재정자립도가 낮은 지방자치단체가 겪는 문제로 가장 적절한 것은?\n① 중앙정부에 대한 재정 의존도 심화 ② 지방세 수입 증가 ③ 자치 행정의 독립성 강화 ④ 지역 개발 사업의 자율적 확대",
        "standards": ["지방재정과 지방자치"],
        "num_items": 5,
    },
    {
        "id": "str_051",
        "passage_text": "1. 다음 사례에서 나타난 공유지의 비극 현상이 발생하는 근본 원인을 설명하고, 이를 해결하기 위한 방안을 서술하시오.",
        "standards": ["공유자원의 특성과 공유지의 비극"],
        "num_items": 7,
    },
    # str_052~076: 2026-07-12, STRUCTURE_GOLDEN 20개 → 40~50개 확대(야간 자율 세션).
    # 기존에 없던 주제로 채우고, 단일 지문형/다중 지문형·num_items(3/5/7)를 고르게 섞음.
    # human_label은 비워둠(사람 라벨링은 별도 진행).
    {
        "id": "str_052",
        "passage_text": "1. 최저임금제도의 시행 목적으로 가장 적절한 것은?\n① 기업의 인건비 부담 완화 ② 저임금 근로자의 생활 안정 ③ 물가 상승 억제 ④ 노동 공급 확대",
        "standards": ["최저임금제도와 근로자 보호"],
        "num_items": 5,
    },
    {
        "id": "str_053",
        "passage_text": "1. 소비자 물가 지수(CPI)가 상승할 때 나타나는 현상으로 옳은 것은?\n① 화폐의 실질 구매력 상승 ② 화폐의 실질 구매력 하락 ③ 명목 소득과 실질 소득이 항상 같아짐 ④ 저축의 실질 가치 상승\n\n2. 인플레이션이 채권자와 채무자에게 미치는 영향을 서술하시오.",
        "standards": ["물가와 인플레이션"],
        "num_items": 5,
    },
    {
        "id": "str_054",
        "passage_text": "1. 중앙은행이 기준금리를 인상할 때 나타나는 효과로 옳은 것은?\n① 시중 통화량 증가 ② 시중 통화량 감소 ③ 투자와 소비 촉진 ④ 물가 상승 촉진",
        "standards": ["통화정책과 중앙은행의 역할"],
        "num_items": 3,
    },
    {
        "id": "str_055",
        "passage_text": "1. 비교우위에 따른 국제 분업이 무역 당사국 모두에게 이익이 되는 이유를 서술하시오.",
        "standards": ["비교우위론과 국제 분업"],
        "num_items": 3,
    },
    {
        "id": "str_056",
        "passage_text": "1. 헌법상 평등권의 의미로 가장 적절한 것은?\n① 모든 국민을 예외 없이 동일하게 대우 ② 합리적 이유 없는 차별을 금지 ③ 결과의 평등을 절대적으로 보장 ④ 기회의 평등을 전면 배제\n\n2. 적극적 평등 실현 조치(예: 여성 할당제)가 평등권 논쟁에서 쟁점이 되는 이유를 서술하시오.",
        "standards": ["평등권의 의미와 실현"],
        "num_items": 7,
    },
    {
        "id": "str_057",
        # 2026-07-12: 원래 "표현의 자유"→"환경세" 문항 둘 다 budget=3(재시도 포함)에서도
        # 0문항으로 실패해 다시 교체(str_050/051 대체 사례와 동일한 처리 방식).
        "passage_text": "1. 청소년 아르바이트 근로계약에서 보호자 동의가 필요한 이유를 서술하시오.",
        "standards": ["미성년 근로자 보호"],
        "num_items": 3,
    },
    {
        "id": "str_058",
        "passage_text": "1. 지방자치단체가 조례를 제정할 때 지켜야 할 원칙으로 옳은 것은?\n① 법률의 범위를 벗어나도 무방하다 ② 법령의 범위 안에서 제정해야 한다 ③ 국회의 사전 승인이 필요하다 ④ 조례는 상위법을 개정할 수 있다\n\n2. 지방자치단체의 조례 제정권이 주민 자치 실현에 기여하는 방식을 서술하시오.",
        "standards": ["지방자치단체의 자치입법권"],
        "num_items": 5,
    },
    {
        "id": "str_059",
        # 2026-07-12: 원래 "국제법의 법원" 문항이 budget=3(재시도 포함)에서도 0문항으로
        # 실패해 다른 주제로 교체(str_050/051 대체 사례와 동일한 처리 방식).
        "passage_text": "1. 근로시간 단축 제도가 근로자와 기업에 미치는 영향을 서술하시오.",
        "standards": ["근로시간과 노동 조건"],
        "num_items": 3,
    },
    {
        "id": "str_060",
        "passage_text": "1. 다음 사례에서 나타난 사회화 기관의 유형을 쓰고, 그 특징을 서술하시오.",
        "standards": ["사회화 기관의 유형과 기능"],
        "num_items": 3,
    },
    {
        "id": "str_061",
        "passage_text": "1. 사회 계층 이동 중 세대 간 이동에 해당하는 사례로 옳은 것은?\n① 개인이 평생 동안 계층이 상승하는 경우 ② 부모 세대와 자녀 세대의 계층 지위가 달라지는 경우 ③ 한 개인이 짧은 기간 안에 계층이 변동하는 경우 ④ 계층 구조 자체가 고정된 경우\n\n2. 계층 이동이 활발한 사회와 폐쇄적인 사회의 차이를 서술하시오.",
        "standards": ["사회 계층 이동의 유형"],
        "num_items": 5,
    },
    {
        "id": "str_062",
        "passage_text": "1. 문화 상대주의의 관점으로 가장 적절한 것은?\n① 특정 문화의 우월성을 전제한다 ② 각 문화를 그 사회의 맥락에서 이해하려 한다 ③ 보편적 인권을 부정한다 ④ 자기 문화의 기준으로 타 문화를 평가한다",
        "standards": ["문화 상대주의와 자문화중심주의"],
        "num_items": 5,
    },
    {
        "id": "str_063",
        "passage_text": "1. 사회 실재론의 관점에 가까운 주장으로 옳은 것은?\n① 사회는 개인의 합에 불과하다 ② 사회는 개인으로 환원할 수 없는 독자적 실체이다 ③ 사회 문제의 원인은 항상 개인에게 있다 ④ 사회는 개인의 의지와 무관하게 존재하지 않는다\n\n2. 사회 명목론이 개인의 자율성을 강조하는 이유를 서술하시오.",
        "standards": ["사회를 바라보는 관점(실재론·명목론)"],
        "num_items": 5,
    },
    {
        "id": "str_064",
        "passage_text": "1. 근대 민주주의가 발전하는 과정에서 시민혁명이 기여한 바를 서술하시오.",
        "standards": ["근대 민주주의의 발전 과정"],
        "num_items": 3,
    },
    {
        "id": "str_065",
        "passage_text": "1. 대통령제의 특징으로 옳은 것은?\n① 행정부 수반과 국가 원수가 분리된다 ② 의회 다수당이 내각을 구성한다 ③ 대통령과 의회 의원의 임기가 서로 독립적으로 보장된다 ④ 내각 불신임권이 존재한다\n\n2. 의원내각제에서 내각 불신임권과 의회 해산권이 상호 견제 수단으로 기능하는 방식을 서술하시오.",
        "standards": ["대통령제와 의원내각제 비교"],
        "num_items": 7,
    },
    {
        "id": "str_066",
        "passage_text": "1. 소선거구제의 특징으로 옳은 것은?\n① 한 선거구에서 여러 명을 선출한다 ② 한 선거구에서 최다 득표자 1인을 선출한다 ③ 사표가 발생하지 않는다 ④ 군소 정당에 절대적으로 유리하다",
        "standards": ["선거구제와 대표 결정 방식"],
        "num_items": 3,
    },
    {
        "id": "str_067",
        "passage_text": "1. UN 안전보장이사회의 특징으로 옳은 것은?\n① 모든 결정에 만장일치가 필요하다 ② 상임이사국은 거부권을 행사할 수 있다 ③ 총회의 하위 기관이다 ④ 군사적 제재 권한이 전혀 없다\n\n2. UN 총회와 안전보장이사회의 권한 차이를 서술하시오.",
        "standards": ["UN의 주요 기관과 역할"],
        "num_items": 5,
    },
    {
        "id": "str_068",
        "passage_text": "1. 지속가능발전목표(SDGs)가 추구하는 방향으로 가장 적절한 것은?\n① 경제 성장만을 최우선 목표로 삼는다 ② 환경·사회·경제의 균형 있는 발전을 추구한다 ③ 선진국 중심의 발전만을 다룬다 ④ 단기적 이익 극대화를 목표로 한다",
        "standards": ["지속가능발전목표와 환경 정의"],
        "num_items": 5,
    },
    {
        "id": "str_069",
        "passage_text": "1. 정보 사회의 특징으로 옳은 것은?\n① 정보의 생산과 유통이 감소한다 ② 지식과 정보가 부가가치 창출의 핵심 자원이 된다 ③ 시공간의 제약이 강화된다 ④ 쌍방향 소통이 어려워진다\n\n2. 정보 사회에서 발생할 수 있는 사생활 침해 문제를 서술하시오.\n\n3. 정보 격차(디지털 디바이드)를 해소하기 위한 방안을 두 가지 이상 서술하시오.",
        "standards": ["정보 사회의 특징과 문제점"],
        "num_items": 7,
    },
    {
        "id": "str_070",
        "passage_text": "1. 사회보장제도가 사회 통합에 기여하는 방식을 서술하시오.",
        "standards": ["사회보장제도의 기능"],
        "num_items": 3,
    },
    {
        "id": "str_071",
        "passage_text": "1. 공정거래위원회의 주요 역할로 옳은 것은?\n① 기업의 독과점을 조장 ② 불공정 거래 행위를 규제하고 시장 경쟁을 촉진 ③ 기업의 세금 납부를 대행 ④ 노동조합 설립을 인가",
        "standards": ["공정거래위원회와 독과점 규제"],
        "num_items": 5,
    },
    {
        "id": "str_072",
        "passage_text": "1. 사유재산 제도가 자원 배분의 효율성에 기여하는 이유를 서술하시오.",
        "standards": ["재산권과 사유재산 제도"],
        "num_items": 3,
    },
    {
        "id": "str_073",
        "passage_text": "1. 국제 사회가 인권 보장을 위해 마련한 제도적 장치로 옳은 것은?\n① 국제형사재판소(ICC) ② 세계무역기구(WTO) ③ 국제통화기금(IMF) ④ 경제협력개발기구(OECD)\n\n2. 국제 인권 규범이 개별 국가의 주권과 충돌할 수 있는 지점을 서술하시오.",
        "standards": ["국제 인권 보장을 위한 노력"],
        "num_items": 5,
    },
    {
        "id": "str_074",
        "passage_text": "1. 다문화 사회로의 전환이 가져오는 긍정적 효과로 옳은 것은?\n① 문화적 다양성 축소 ② 다양한 문화 간 교류를 통한 사회적 창의성 증대 ③ 노동력 유입의 전면 차단 ④ 단일 문화의 강화\n\n2. 다문화 사회에서 발생할 수 있는 갈등 요인과 그 해결 방안을 서술하시오.",
        "standards": ["다문화 사회와 문화 다양성"],
        "num_items": 5,
    },
    {
        "id": "str_075",
        "passage_text": "1. 시민 불복종이 정당화되기 위한 조건으로 옳은 것은?\n① 폭력적 수단 사용 ② 공익을 목적으로 하며 최후의 수단으로 사용 ③ 개인의 사적 이익 추구 ④ 처벌을 회피하려는 목적",
        "standards": ["시민 불복종의 정당화 조건"],
        "num_items": 3,
    },
    {
        "id": "str_076",
        "passage_text": "1. 응능부담의 원칙에 따른 조세 부과 방식으로 가장 적절한 것은?\n① 모든 납세자에게 동일한 세액을 부과 ② 개인의 소득·재산 등 부담 능력에 따라 세액을 차등 부과 ③ 이용한 만큼만 부담 ④ 소비량과 무관하게 정액 부과\n\n2. 응익부담 원칙이 적용되는 대표적 사례를 서술하시오.",
        "standards": ["조세 원칙(응능부담·응익부담)"],
        "num_items": 5,
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
