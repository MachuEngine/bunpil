#!/usr/bin/env python
"""VLM(이미지 → 텍스트 추출) 정확도 골든셋 생성기.

data/golden/vlm_golden_images/ 아래에 합성 시험 문제 이미지 40장(text_only 20 +
figure 15 + adversarial 5)을 PIL로 렌더링하고, data/golden/vlm_extraction_golden.json에
정답(ground_truth_text)·메타데이터를 기록한다.

실제 스크린샷·학생 데이터는 전혀 쓰지 않는다 — 전부 이 스크립트가 합성한다
(하드룰 1과 동일한 이유). 차트(막대/원/꺾은선/표)도 matplotlib 없이 PIL 기본
도형(rectangle/pieslice/line)만으로 그린다 — 의존성을 늘릴 이유가 없어서다
(BM25를 rank_bm25 대신 직접 구현한 것과 같은 판단, MODEL_SELECTION.md 5절 참고).

figure 카테고리의 `figure_summary`는 Claude가 초안만 작성했고 `reviewed: false` —
evals/eval_vlm.py를 정식으로 정기 평가에 편입하기 전에 사람 검수가 필요하다.

재실행하면 이미지·JSON을 덮어쓴다(idempotent, 랜덤 요소 없음).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "data", "golden", "vlm_golden_images")
JSON_PATH = os.path.join(ROOT, "data", "golden", "vlm_extraction_golden.json")

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:\\Windows\\Fonts\\malgun.ttf",
]


def _font_path() -> str:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    raise RuntimeError("한글 폰트를 찾지 못했습니다 — _FONT_CANDIDATES에 경로를 추가하세요.")


_FP = _font_path()


def F(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_FP, size)


# ── 공통 렌더링 유틸 ─────────────────────────────────────────────────────
# 2026-08-20 수정: 처음엔 문자열을 그대로 d.text()에 넘겼는데, 실제 렌더링해보니
# (1) 개행이 포함된 문자열(<보기> 블록)은 d.text()가 내부적으로 여러 줄을 그리는데
#     y 증가량은 한 줄분만 세서 다음 내용이 그 위에 겹쳐 찍히고,
# (2) 캔버스 폭을 넘는 긴 줄은 잘려서 안 보이지 않고 그냥 화면 밖으로 사라졌다
#     (즉 골든셋 ground_truth_text에는 있는데 이미지엔 없는 텍스트가 생김 — 골든셋
#     자체가 틀린 상태였다). 실제 폭 측정(textlength) 기반 줄바꿈으로 교체.

def _wrap_block(d: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    """text 안의 기존 개행은 문단 구분으로 유지하고, 각 문단을 max_width 안에 들어오게
    공백 기준으로 다시 줄바꿈한다."""
    out: list[str] = []
    for para in text.split("\n"):
        if para == "":
            out.append("")
            continue
        words = para.split(" ")
        cur = ""
        for w in words:
            trial = (cur + " " + w).strip()
            if not cur or d.textlength(trial, font=font) <= max_width:
                cur = trial
            else:
                out.append(cur)
                cur = w
        out.append(cur)
    return out


def _draw_block(d: ImageDraw.ImageDraw, text: str, x: int, y: int, font, max_width: int, line_h: int = 28) -> int:
    for line in _wrap_block(d, text, font, max_width):
        d.text((x, y), line, fill="black", font=font)
        y += line_h
    return y


MAX_W = 680  # 캔버스 폭 720 기준 좌우 여백(20px씩) 뺀 실제 텍스트 폭


def render_text_only(path: str, question: str, options: list[str]) -> None:
    W, H = 720, 420
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    y = _draw_block(d, question, 20, 20, F(20), MAX_W, line_h=30)
    y += 20
    _draw_block(d, "  ".join(options), 20, y, F(18), MAX_W, line_h=28)
    img.save(path)


def render_adversarial_leak(path: str, question: str, options: list[str], leak_line: str | None) -> None:
    """정답/해설이 실제로 이미지에 있는 경우(leak_line 지정)와, 없는 일반 문항(leak_line=None)
    둘 다 만들 수 있게 함 — 후자는 "이미지에 없는 정답을 VLM이 스스로 지어내지 않는가"를 검증."""
    W, H = 720, 340
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    y = _draw_block(d, question, 20, 20, F(20), MAX_W, line_h=30)
    y += 20
    y = _draw_block(d, "  ".join(options), 20, y, F(18), MAX_W, line_h=28)
    if leak_line:
        y += 20
        _draw_block(d, leak_line, 20, y, F(16), MAX_W, line_h=24)
    img.save(path)


def render_pii_variant(path: str, header_lines: list[str], question: str, options: list[str]) -> None:
    W, H = 720, 340
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    y = 15
    for line in header_lines:
        y = _draw_block(d, line, 20, y, F(18), MAX_W, line_h=28)
    y += 10
    y = _draw_block(d, question, 20, y, F(20), MAX_W, line_h=30)
    y += 20
    _draw_block(d, "  ".join(options), 20, y, F(18), MAX_W, line_h=28)
    img.save(path)


def _chart_frame(question: str) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
    W, H = 720, 620  # 문항이 2~3줄로 줄바꿈돼도 차트·선지가 밀리지 않도록 여유 확보
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    y = _draw_block(d, question, 20, 15, F(19), MAX_W, line_h=28)
    return img, d, y + 10


def render_bar(path: str, question: str, options: list[str], categories: list[str], values: list[float], unit: str) -> None:
    img, d, y0 = _chart_frame(question)
    box = (80, y0, 640, y0 + 260)
    x0, y0b, x1, y1b = box
    max_v = max(values) * 1.25
    n = len(categories)
    bar_w = (x1 - x0) / (n * 2)
    baseline = y1b - 20
    top = y0b + 20
    d.line([(x0, baseline), (x1, baseline)], fill="black", width=2)
    fs = F(15)
    for i, (cat, v) in enumerate(zip(categories, values)):
        bx0 = x0 + (2 * i + 0.5) * bar_w
        bx1 = bx0 + bar_w
        bh = (v / max_v) * (baseline - top)
        by0 = baseline - bh
        d.rectangle([bx0, by0, bx1, baseline], outline="black", fill="#A9C6E8")
        d.text(((bx0 + bx1) / 2, by0 - 12), str(v), font=fs, anchor="mm", fill="black")
        d.text(((bx0 + bx1) / 2, baseline + 16), cat, font=fs, anchor="mm", fill="black")
    d.text((x0, y0b - 18), f"(단위: {unit})", font=F(13), fill="black")
    y = baseline + 45
    _draw_block(d, "  ".join(options), 20, y, F(18), MAX_W, line_h=28)
    img.save(path)


def render_line(path: str, question: str, options: list[str], years: list[str], values: list[float], unit: str) -> None:
    img, d, y0 = _chart_frame(question)
    x0, y0b, x1, y1b = 80, y0, 640, y0 + 260
    max_v = max(values) * 1.2
    min_v = min(0, min(values))
    baseline = y1b - 20
    top = y0b + 20
    d.line([(x0, baseline), (x1, baseline)], fill="black", width=2)
    d.line([(x0, top), (x0, baseline)], fill="black", width=2)
    n = len(years)
    step = (x1 - x0 - 40) / max(1, n - 1)
    pts = []
    for i, v in enumerate(values):
        px = x0 + 30 + i * step
        py = baseline - ((v - min_v) / (max_v - min_v)) * (baseline - top)
        pts.append((px, py))
    d.line(pts, fill="#1D4ED8", width=3)
    fs = F(14)
    for (px, py), yy, v in zip(pts, years, values):
        d.ellipse([px - 4, py - 4, px + 4, py + 4], fill="#1D4ED8")
        d.text((px, py - 16), str(v), font=fs, anchor="mm", fill="black")
        d.text((px, baseline + 16), yy, font=fs, anchor="mm", fill="black")
    d.text((x0, y0b - 18), f"(단위: {unit})", font=F(13), fill="black")
    y = baseline + 45
    _draw_block(d, "  ".join(options), 20, y, F(18), MAX_W, line_h=28)
    img.save(path)


def render_pie(path: str, question: str, options: list[str], labels: list[str], pcts: list[float]) -> None:
    img, d, y0 = _chart_frame(question)
    cx, cy, r = 240, y0 + 130, 110
    colors = ["#A9C6E8", "#F3B9C4", "#E8C468", "#B7E0C4", "#D8C4E8", "#F0A98A"]
    start = -90.0
    for i, pct in enumerate(pcts):
        end = start + pct * 3.6
        d.pieslice([cx - r, cy - r, cx + r, cy + r], start, end, fill=colors[i % len(colors)], outline="white")
        start = end
    lx, ly = 420, y0 + 40
    fs = F(15)
    for i, (lab, pct) in enumerate(zip(labels, pcts)):
        d.rectangle([lx, ly + i * 30, lx + 16, ly + i * 30 + 16], fill=colors[i % len(colors)])
        d.text((lx + 24, ly + i * 30 + 8), f"{lab} {pct}%", font=fs, anchor="lm", fill="black")
    y = y0 + 280
    _draw_block(d, "  ".join(options), 20, y, F(18), MAX_W, line_h=28)
    img.save(path)


def render_table(path: str, question: str, options: list[str], headers: list[str], rows: list[list[str]]) -> None:
    img, d, y0 = _chart_frame(question)
    x0, y0b = 60, y0 + 10
    col_w = 560 // len(headers)
    row_h = 34
    fs = F(15)
    all_rows = [headers] + rows
    for r, row in enumerate(all_rows):
        for c, cell in enumerate(row):
            x = x0 + c * col_w
            y = y0b + r * row_h
            d.rectangle([x, y, x + col_w, y + row_h], outline="black", fill="#EFEFE8" if r == 0 else "white")
            d.text((x + col_w / 2, y + row_h / 2), str(cell), font=fs, anchor="mm", fill="black")
    y = y0b + len(all_rows) * row_h + 30
    _draw_block(d, "  ".join(options), 20, y, F(18), MAX_W, line_h=28)
    img.save(path)


# ── 골든셋 콘텐츠 정의 ────────────────────────────────────────────────────

TEXT_ONLY = [
    dict(id="t01", topic="정치와 법 — 입법 절차", question="1. 다음 중 법률안 제정 절차에 해당하지 않는 것은?",
         options=["① 법률안 제출", "② 위원회 심사", "③ 본회의 의결", "④ 대통령 공포", "⑤ 헌법재판소 위헌 심사"]),
    dict(id="t02", topic="정치와 법 — 헌법재판소", question="2. 헌법재판소의 권한으로 옳지 않은 것은?",
         options=["① 위헌 법률 심판", "② 탄핵 심판", "③ 정당 해산 심판", "④ 권한쟁의 심판", "⑤ 대통령 임명"]),
    dict(id="t03", topic="정치와 법 — 선거 원칙", question="3. 다음 <보기>가 설명하는 선거의 4대 원칙은?\n\n<보기>\n선거권자가 직접 투표소에 나가 투표해야 하며, 대리 투표는 인정되지 않는다.",
         options=["① 보통 선거", "② 평등 선거", "③ 직접 선거", "④ 비밀 선거", "⑤ 자유 선거"]),
    dict(id="t04", topic="경제 — 수요공급", question="4. 공급이 일정할 때 수요가 증가하면 나타나는 변화로 옳은 것은?",
         options=["① 가격 하락, 거래량 감소", "② 가격 하락, 거래량 증가", "③ 가격 상승, 거래량 증가",
                  "④ 가격 상승, 거래량 감소", "⑤ 가격·거래량 모두 불변"]),
    dict(id="t05", topic="경제 — 기회비용", question="5. 다음 <보기>의 밑줄 친 개념에 해당하는 것은?\n\n<보기>\n민수는 저녁에 아르바이트 대신 스터디를 선택하며 포기한 시급 만원을 아까워했다.",
         options=["① 매몰 비용", "② 기회비용", "③ 명시적 비용", "④ 한계 비용", "⑤ 고정 비용"]),
    dict(id="t06", topic="경제 — 통화정책", question="6. 중앙은행이 기준 금리를 인상했을 때 나타나는 효과로 가장 적절한 것은?",
         options=["① 시중 통화량 증가", "② 물가 상승 압력 증가", "③ 저축 유인 감소",
                  "④ 대출 수요 감소", "⑤ 환율 하락 압력 감소"]),
    dict(id="t07", topic="사회문화 — 사회화 기관", question="7. 다음 중 2차적 사회화 기관에 해당하는 것은?",
         options=["① 가족", "② 또래 집단", "③ 학교", "④ 친족", "⑤ 이웃"]),
    dict(id="t08", topic="사회문화 — 사회 불평등", question="8. 다음 <보기>에서 설명하는 사회 불평등 이론은?\n\n<보기>\n사회 불평등은 사회 유지를 위해 기능적으로 필요하며, 중요한 직위일수록 더 많은 보상이 따른다.",
         options=["① 기능론", "② 갈등론", "③ 상징적 상호작용론", "④ 교환 이론", "⑤ 낙인 이론"]),
    dict(id="t09", topic="사회문화 — 문화 상대주의", question="9. 문화를 이해하는 태도 중 문화 상대주의에 해당하는 서술은?",
         options=["① 우리 문화 기준으로 다른 문화를 평가한다", "② 모든 문화는 그 사회의 맥락에서 이해해야 한다",
                  "③ 선진국 문화가 후진국 문화보다 우월하다", "④ 특정 문화만이 보편적으로 옳다",
                  "⑤ 문화 간 우열을 명확히 가릴 수 있다"]),
    dict(id="t10", topic="한국지리 — 인구 분포", question="10. 우리나라 인구의 수도권 집중 현상의 원인으로 옳지 않은 것은?",
         options=["① 일자리 집중", "② 교육 기반 시설 집중", "③ 각종 편의 시설 집중",
                  "④ 신생아 출산 장려금 지역 편중", "⑤ 행정·경제 기능 집중"]),
    dict(id="t11", topic="한국지리 — 도시 재개발", question="11. 다음 중 도시 재개발 방식과 그 설명이 잘못 짝지어진 것은?",
         options=["① 철거 재개발 — 기존 건물을 완전히 철거", "② 보존 재개발 — 역사·문화적 가치 보존",
                  "③ 수복 재개발 — 기존 골격 유지하며 부분 개량", "④ 개량 재개발 — 노후 시설만 교체",
                  "⑤ 철거 재개발 — 기존 건물을 그대로 보존"]),
    dict(id="t12", topic="세계지리 — 기후대", question="12. 다음 <보기>가 설명하는 기후는?\n\n<보기>\n연중 고온다습하며 스콜성 강우가 잦고, 열대 우림이 발달한다.",
         options=["① 열대 우림 기후", "② 사바나 기후", "③ 지중해성 기후", "④ 온대 계절풍 기후", "⑤ 툰드라 기후"]),
    dict(id="t13", topic="세계지리 — 지역 갈등", question="13. 종교 차이가 주요 원인으로 작용한 지역 갈등에 해당하지 않는 것은?",
         options=["① 카슈미르 분쟁", "② 팔레스타인 분쟁", "③ 벨기에 언어 갈등", "④ 북아일랜드 분쟁", "⑤ 스리랑카 내전"]),
    dict(id="t14", topic="동아시아사 — 유교 사상", question="14. 다음 <보기>의 사상가가 강조한 개념으로 가장 적절한 것은?\n\n<보기>\n그는 인(仁)을 최고의 덕목으로 삼고, 자기 수양을 통한 도덕적 완성을 강조했다.",
         options=["① 공자", "② 노자", "③ 한비자", "④ 묵자", "⑤ 순자"]),
    dict(id="t15", topic="세계사 — 시민혁명", question="15. 다음 중 프랑스 혁명의 배경으로 옳지 않은 것은?",
         options=["① 절대 왕정의 재정 위기", "② 계몽사상의 확산", "③ 제3신분의 불만 고조",
                  "④ 산업 혁명 이후 노동 계급 형성", "⑤ 신분제 사회의 모순"]),
    dict(id="t16", topic="윤리 — 공리주의", question="16. 다음 <보기>의 입장에 해당하는 윤리 사상은?\n\n<보기>\n행위의 옳고 그름은 그 행위가 가져오는 결과, 즉 최대 다수의 최대 행복 여부로 판단해야 한다.",
         options=["① 의무론", "② 공리주의", "③ 덕 윤리", "④ 배려 윤리", "⑤ 담론 윤리"]),
    dict(id="t17", topic="윤리 — 칸트 의무론", question="17. 칸트의 윤리 사상에 대한 설명으로 옳지 않은 것은?",
         options=["① 행위의 결과보다 동기를 중시한다", "② 정언 명령을 도덕 법칙으로 제시한다",
                  "③ 인간을 목적으로 대우해야 한다고 본다", "④ 도덕적 행위의 기준을 유용성에 둔다",
                  "⑤ 보편화 가능성을 도덕 판단의 기준으로 삼는다"]),
    dict(id="t18", topic="정치와 법 — 지방자치제도", question="18. 지방자치제도의 의의로 가장 적절하지 않은 것은?",
         options=["① 주민의 정치 참여 기회 확대", "② 지역 실정에 맞는 행정 실현", "③ 권력 분립을 통한 견제와 균형",
                  "④ 중앙 정부의 재정 부담 전면 해소", "⑤ 지방 자치 단체 간 자율성 보장"]),
    dict(id="t19", topic="경제 — 시장실패", question="19. 다음 <보기>의 사례가 나타내는 시장 실패의 원인은?\n\n<보기>\n공장이 오염 물질을 배출하지만, 이로 인한 주변 주민의 피해는 시장 가격에 반영되지 않는다.",
         options=["① 공공재", "② 외부 효과", "③ 정보 비대칭", "④ 독과점", "⑤ 진입 장벽"]),
    dict(id="t20", topic="사회문화 — 사회 변동", question="20. 사회 변동을 설명하는 이론 중 <보기>에 해당하는 것은?\n\n<보기>\n사회는 단순한 형태에서 복잡한 형태로, 미분화 상태에서 분화된 상태로 진보한다고 본다.",
         options=["① 진화론", "② 순환론", "③ 갈등론", "④ 기능론", "⑤ 균형론"]),
]

FIGURES = [
    dict(id="f01", kind="bar", question="1. 다음 자료를 보고 물음에 답하시오.\n\n다음은 A~D 지역의 인구를 나타낸 것이다. 인구가 가장 많은 지역은?",
         options=["① A", "② B", "③ C", "④ D", "⑤ 네 지역 모두 같다"],
         categories=["A", "B", "C", "D"], values=[820, 450, 610, 300], unit="만 명",
         figure_summary="A~D 4개 지역의 인구를 막대그래프로 비교. A지역이 820만 명으로 가장 많고, B(450)·C(610)·D(300) 순. 값이 가장 큰 막대는 A."),
    dict(id="f02", kind="bar", question="2. 다음은 갑국의 산업별 취업자 비율을 나타낸 것이다. 2차 산업 취업자 비율이 가장 높은 시기는?",
         options=["① 1970년", "② 1990년", "③ 2010년", "④ 2020년", "⑤ 자료로 알 수 없다"],
         categories=["1970", "1990", "2010", "2020"], values=[15, 35, 28, 20], unit="%",
         figure_summary="1970/1990/2010/2020년 4개 시점의 2차 산업 취업자 비율 막대그래프. 1990년이 35%로 가장 높고, 1970년(15)→1990년(35)→2010년(28)→2020년(20) 순으로 1990년 정점 후 감소."),
    dict(id="f03", kind="line", question="3. 다음 자료는 갑국의 연도별 실업률 추이이다. 실업률이 가장 낮았던 해는?",
         options=["① 2018년", "② 2019년", "③ 2020년", "④ 2021년", "⑤ 2022년"],
         years=["2018", "2019", "2020", "2021", "2022"], values=[3.8, 3.5, 4.9, 3.7, 3.0], unit="%",
         figure_summary="2018~2022년 5개년 실업률 꺾은선 그래프. 2020년에 4.9%로 급등했다가(코로나 시기 연상) 이후 하락, 2022년이 3.0%로 가장 낮음."),
    dict(id="f04", kind="line", question="4. 다음은 갑국의 연도별 합계 출산율 추이이다. 이 자료에 대한 설명으로 옳은 것은?",
         options=["① 출산율은 매년 증가했다", "② 2021년이 가장 높다", "③ 전반적으로 감소 추세이다",
                  "④ 2019년이 가장 낮다", "⑤ 변화가 전혀 없다"],
         years=["2018", "2019", "2020", "2021", "2022"], values=[0.98, 0.92, 0.84, 0.81, 0.78], unit="명",
         figure_summary="2018~2022년 합계 출산율 꺾은선 그래프. 0.98→0.92→0.84→0.81→0.78로 5개년 연속 하락하는 감소 추세."),
    dict(id="f05", kind="pie", question="5. 다음은 갑 지역 주민 300명을 대상으로 한 여가 활동 설문 조사 결과이다. 가장 높은 비율을 차지하는 활동은?",
         options=["① 운동", "② 독서", "③ 여행", "④ 게임", "⑤ 기타"],
         labels=["운동", "독서", "여행", "게임", "기타"], pcts=[35, 15, 25, 18, 7],
         figure_summary="여가 활동 설문 조사 원그래프(5개 항목). 운동이 35%로 가장 큰 비중, 여행 25%, 게임 18%, 독서 15%, 기타 7% 순."),
    dict(id="f06", kind="pie", question="6. 다음은 갑국 유권자의 정치 성향 조사 결과이다. 이 자료에 대한 해석으로 옳은 것은?",
         options=["① 중도 성향이 가장 많다", "② 보수 성향이 가장 적다", "③ 진보 성향이 과반이다",
                  "④ 응답 불명은 존재하지 않는다", "⑤ 보수와 진보 비율이 같다"],
         labels=["진보", "중도", "보수", "응답 불명"], pcts=[32, 40, 24, 4],
         figure_summary="정치 성향 설문 원그래프(4개 항목). 중도 40%로 가장 많고, 진보 32%, 보수 24%, 응답 불명 4%."),
    dict(id="f07", kind="table", question="7. 다음은 갑~병국의 무역 수지를 나타낸 표이다. 무역 적자를 기록한 나라는?",
         options=["① 갑국", "② 을국", "③ 병국", "④ 갑국과 병국", "⑤ 세 나라 모두"],
         headers=["국가", "수출(억 달러)", "수입(억 달러)"],
         rows=[["갑국", "520", "480"], ["을국", "310", "310"], ["병국", "270", "340"]],
         figure_summary="갑·을·병 3개국 수출입 표. 갑국은 수출(520)>수입(480)로 흑자, 을국은 수출=수입(310)으로 균형, 병국은 수출(270)<수입(340)로 적자."),
    dict(id="f08", kind="table", question="8. 다음은 갑국의 시대별 인구 구성비를 나타낸 표이다. 고령화가 가장 심화된 시기는?",
         options=["① 1990년대", "② 2000년대", "③ 2010년대", "④ 2020년대", "⑤ 변화 없음"],
         headers=["시기", "유소년(%)", "노년(%)"],
         rows=[["1990년대", "28", "5"], ["2000년대", "20", "8"], ["2010년대", "15", "13"], ["2020년대", "11", "18"]],
         figure_summary="1990~2020년대 4개 시기 유소년/노년 인구 비율 표. 노년 비율이 5→8→13→18%로 계속 증가, 2020년대가 고령화 가장 심함."),
    dict(id="f09", kind="bar", question="9. 다음은 갑국의 연령대별 스마트폰 이용률을 나타낸 것이다. 이용률이 가장 낮은 연령대는?",
         options=["① 10대", "② 20대", "③ 40대", "④ 60대", "⑤ 70대 이상"],
         categories=["10대", "20대", "40대", "60대", "70대+"], values=[98, 99, 92, 70, 42], unit="%",
         figure_summary="연령대별 스마트폰 이용률 막대그래프(5개 구간). 10대·20대는 98~99%로 최상위, 연령이 높아질수록 감소해 70대 이상이 42%로 가장 낮음."),
    dict(id="f10", kind="line", question="10. 다음은 갑국의 연도별 이산화탄소 배출량 추이이다. 이 자료에 대한 설명으로 옳은 것은?",
         options=["① 매년 증가했다", "② 2020년 이후 뚜렷이 감소했다", "③ 변화가 없다",
                  "④ 2019년이 가장 낮다", "⑤ 자료로 알 수 없다"],
         years=["2017", "2018", "2019", "2020", "2021"], values=[620, 640, 655, 590, 560], unit="백만 톤",
         figure_summary="2017~2021년 이산화탄소 배출량 꺾은선. 2019년(655)까지 증가하다가 2020년(590)부터 뚜렷이 감소해 2021년 560까지 하락."),
    dict(id="f11", kind="pie", question="11. 다음은 갑국의 종교 인구 비율을 나타낸 것이다. 가장 높은 비율을 차지하는 종교는?",
         options=["① 개신교", "② 불교", "③ 천주교", "④ 무교", "⑤ 기타"],
         labels=["개신교", "불교", "천주교", "무교", "기타"], pcts=[20, 16, 8, 53, 3],
         figure_summary="종교 인구 비율 원그래프(5개 항목). 무교가 53%로 과반을 차지, 개신교 20%, 불교 16%, 천주교 8%, 기타 3%."),
    dict(id="f12", kind="bar", question="12. 다음은 갑국 대통령 선거의 지역별 A후보 득표율이다. A후보 득표율이 가장 높은 지역은?",
         options=["① 수도권", "② 충청권", "③ 영남권", "④ 호남권", "⑤ 강원권"],
         categories=["수도권", "충청권", "영남권", "호남권", "강원권"], values=[48, 42, 55, 30, 46], unit="%",
         figure_summary="5개 권역별 A후보 득표율 막대그래프. 영남권이 55%로 가장 높고, 호남권이 30%로 가장 낮음."),
    dict(id="f13", kind="table", question="13. 다음은 갑~정 지역의 1월 평균 기온과 연 강수량을 나타낸 표이다. 가장 한랭 건조한 지역은?",
         options=["① 갑", "② 을", "③ 병", "④ 정", "⑤ 자료로 알 수 없다"],
         headers=["지역", "1월 평균기온(℃)", "연 강수량(mm)"],
         rows=[["갑", "-8", "600"], ["을", "5", "1400"], ["병", "-12", "450"], ["정", "12", "1800"]],
         figure_summary="갑·을·병·정 4개 지역의 1월 평균기온·연 강수량 표. 병 지역이 기온 -12℃로 가장 낮고 강수량도 450mm로 가장 적어 가장 한랭 건조."),
    dict(id="f14", kind="line", question="14. 다음은 갑국의 연도별 1인당 온실가스 배출량 추이이다. 이 자료에 대한 해석으로 옳은 것은?",
         options=["① 지속적으로 증가했다", "② 2021년까지 증가하다 이후 감소했다", "③ 매년 동일하다",
                  "④ 2018년이 가장 높다", "⑤ 자료로 알 수 없다"],
         years=["2018", "2019", "2020", "2021", "2022"], values=[11.2, 11.8, 12.3, 12.9, 11.5], unit="톤",
         figure_summary="2018~2022년 1인당 온실가스 배출량 꺾은선. 2018년(11.2)부터 2021년(12.9)까지 증가하다 2022년(11.5)에 감소로 전환."),
    dict(id="f15", kind="pie", question="15. 다음은 갑 가구의 월평균 소비 지출 항목별 비중이다. 가장 큰 비중을 차지하는 항목은?",
         options=["① 주거비", "② 식료품비", "③ 교육비", "④ 의료비", "⑤ 기타"],
         labels=["주거비", "식료품비", "교육비", "의료비", "기타"], pcts=[28, 22, 19, 11, 20],
         figure_summary="가구 월평균 소비 지출 원그래프(5개 항목). 주거비가 28%로 가장 크고, 식료품비 22%, 교육비 19%, 기타 20%, 의료비 11%."),
]

ADVERSARIAL = [
    dict(id="a01", kind="no_leak", question="1. 다음 중 삼권분립의 목적으로 가장 적절한 것은?",
         options=["① 행정 효율성 극대화", "② 권력 남용 방지", "③ 입법 속도 향상", "④ 사법부 권한 확대", "⑤ 국가 통합 강화"],
         leak_line=None,
         notes="정답·해설이 이미지에 전혀 없음 — VLM이 스스로 '정답은 ②입니다' 같은 문구를 지어내지 않는지 확인(요약·해설·정답 추론 금지 규칙 검증)."),
    dict(id="a02", kind="no_leak", question="2. 다음 중 시장 가격의 기능으로 옳지 않은 것은?",
         options=["① 자원 배분 기능", "② 정보 전달 기능", "③ 소득 재분배 기능", "④ 경제 활동 유인 기능", "⑤ 조절 기능"],
         leak_line=None,
         notes="a01과 동일 취지의 두 번째 케이스(주제만 경제로 변경) — 반복 측정으로 우연 여부 배제."),
    dict(id="a03", kind="leak_present", question="3. 다음 중 국회의 권한에 해당하지 않는 것은?",
         options=["① 법률 제정권", "② 예산 심의·확정권", "③ 조약 체결·비준 동의권", "④ 국정 감사권", "⑤ 대법원장 임명권"],
         leak_line="[채점 메모] 정답: ⑤ (대법원장 임명권은 대통령의 권한)",
         notes="이미지 자체에 정답 메모가 실제로 찍혀 있는 케이스 — 이번엔 반대로, 이미지에 있는 내용을 빠뜨리지 않고 그대로(추론이 아니라 전사로서) 옮기는지 확인. '원문 그대로 옮긴다'는 지시와 '정답 추론 금지'가 충돌하지 않는 경계 사례."),
    dict(id="a04", kind="pii_variant", header=["분필고등학교 2학년 3반 사회 수행평가지", "성명 : 김민준     연락처 : 010-9876-5432"],
         question="4. 다음 중 문화의 속성으로 옳지 않은 것은?",
         options=["① 학습성", "② 공유성", "③ 전체성", "④ 고정성", "⑤ 변동성"],
         pii_labels_expected=["이름", "전화번호"], pii_labels_known_limitation=[],
         notes="라벨·값 사이 공백이 불규칙(캡처마다 서식이 다를 수 있음)하지만 이름·연락처 자체는 정상 표기 — mask_pii()가 이 정도 변형은 잡아내는지 확인(공백 실측: `mask_pii()`가 실제로 ['이름','전화번호']를 반환함을 golden_gen 실행 시 확인)."),
    dict(id="a05", kind="pii_variant", header=["분필고등학교 1학년 5반", "이름: 이서연  /  연락처(공일공-일이삼사-오육칠팔)"],
         question="5. 다음 중 근로자의 권리에 해당하지 않는 것은?",
         options=["① 단결권", "② 단체 교섭권", "③ 단체 행동권", "④ 무제한 파업권", "⑤ 근로 기준법상 최저 임금 보장"],
         pii_labels_expected=["이름"], pii_labels_known_limitation=["전화번호"],
         notes="연락처를 한글 숫자('공일공-...')로 표기 — mask_pii()의 숫자 기반 정규식(\\d)이 구조적으로 못 잡는 케이스. 이름은 '이름:' 레이블이 있어 정상적으로 잡히므로 필수 항목, 전화번호만 알려진 한계로 분리(실패로 채점하지 않되 리포트에는 남김)."),
]


def _wrap_ground_truth_options(options: list[str]) -> str:
    return " ".join(options)


def build_entries() -> list[dict]:
    entries = []

    for q in TEXT_ONLY:
        img_name = f"{q['id']}_{q['topic'].split(' — ')[0]}.png".replace(" ", "")
        img_path = os.path.join(IMG_DIR, img_name)
        render_text_only(img_path, q["question"], q["options"])
        entries.append({
            "id": q["id"],
            "category": "text_only",
            "topic": q["topic"],
            "image": f"vlm_golden_images/{img_name}",
            "ground_truth_text": q["question"] + "\n" + _wrap_ground_truth_options(q["options"]),
        })

    for q in FIGURES:
        img_name = f"{q['id']}_{q['kind']}.png"
        img_path = os.path.join(IMG_DIR, img_name)
        if q["kind"] == "bar":
            render_bar(img_path, q["question"], q["options"], q["categories"], q["values"], q["unit"])
        elif q["kind"] == "line":
            render_line(img_path, q["question"], q["options"], q["years"], q["values"], q["unit"])
        elif q["kind"] == "pie":
            render_pie(img_path, q["question"], q["options"], q["labels"], q["pcts"])
        elif q["kind"] == "table":
            render_table(img_path, q["question"], q["options"], q["headers"], q["rows"])
        entries.append({
            "id": q["id"],
            "category": "figure",
            "figure_kind": q["kind"],
            "image": f"vlm_golden_images/{img_name}",
            "ground_truth_text": q["question"] + "\n" + _wrap_ground_truth_options(q["options"]),
            "figure_summary": q["figure_summary"],
            "reviewed": False,
        })

    for q in ADVERSARIAL:
        img_name = f"{q['id']}_{q['kind']}.png"
        img_path = os.path.join(IMG_DIR, img_name)
        entry = {
            "id": q["id"],
            "category": "adversarial",
            "adversarial_kind": q["kind"],
            "image": f"vlm_golden_images/{img_name}",
            "notes": q["notes"],
        }
        if q["kind"] in ("no_leak", "leak_present"):
            render_adversarial_leak(img_path, q["question"], q["options"], q.get("leak_line"))
            gt = q["question"] + "\n" + _wrap_ground_truth_options(q["options"])
            if q.get("leak_line"):
                gt += "\n" + q["leak_line"]
            entry["ground_truth_text"] = gt
            entry["forbidden_answer_leak"] = q["kind"] == "no_leak"
        elif q["kind"] == "pii_variant":
            render_pii_variant(img_path, q["header"], q["question"], q["options"])
            entry["ground_truth_text"] = "\n".join(q["header"]) + "\n\n" + q["question"] + "\n" + _wrap_ground_truth_options(q["options"])
            entry["pii_labels_expected"] = q["pii_labels_expected"]
            entry["pii_labels_known_limitation"] = q["pii_labels_known_limitation"]
        entries.append(entry)

    return entries


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    entries = build_entries()

    doc = {
        "_schema": {
            "description": (
                "VLM(이미지→텍스트 추출, POST /exam/extract) 정확도 골든셋. "
                f"text_only {len(TEXT_ONLY)}건 + figure {len(FIGURES)}건 + adversarial {len(ADVERSARIAL)}건 "
                f"= {len(entries)}건. 이미지는 golden_gen/gen_vlm_golden.py가 PIL로 합성(재실행 시 idempotent)."
            ),
            "entry_fields": {
                "id": "고유 ID",
                "category": "text_only | figure | adversarial",
                "image": "data/golden/ 기준 상대경로",
                "ground_truth_text": "이미지에 실제로 적힌 발문·<보기>·선지(및 있다면 메모) 원문 — CER/WER 채점 기준",
                "figure_summary": "(figure만) 자료가 담아야 할 핵심 사실. Claude 초안, reviewed=false면 사람 검수 전",
                "reviewed": "(figure만) figure_summary를 사람이 검수했는지",
                "forbidden_answer_leak": "(adversarial, no_leak/leak_present만) true면 이미지에 정답 표시가 없으므로 VLM 출력에 '정답은/정답:/따라서 답은' 등이 나오면 안 됨",
                "pii_labels_expected": "(adversarial, pii_variant만) mask_pii()가 반드시 잡아야 하는 PII 라벨 — 누락되면 실패로 채점",
                "pii_labels_known_limitation": "(adversarial, pii_variant만) mask_pii() 정규식이 구조적으로 못 잡는다고 알려진 라벨 — 누락돼도 실패로 채점하지 않고 리포트에만 남김",
                "notes": "이 항목을 넣은 이유",
            },
            "provenance": (
                "전부 Claude가 golden_gen/gen_vlm_golden.py로 합성(PIL, matplotlib 미사용 — 의존성 최소화). "
                "실제 스크린샷·학생 데이터 미사용(CLAUDE.md 하드룰 1). "
                "figure_summary는 Claude 초안 — 사람 검수 전까지 reviewed=false."
            ),
        },
        "entries": entries,
    }

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(f"[완료] 이미지 {len(entries)}장 → {IMG_DIR}")
    print(f"[완료] 골든셋 JSON → {JSON_PATH}")
    print(f"  text_only={len(TEXT_ONLY)} figure={len(FIGURES)} adversarial={len(ADVERSARIAL)} 총 {len(entries)}건")


if __name__ == "__main__":
    main()
