"""분필 Gradio UI — 출제 모드 + 생기부 윤문 모드."""
import html
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:1.5b")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db")

import gradio as gr

from app.common.rag import chunk_document, parse_pdf
from app.modules.exam import ExamSpec, get_exam_graph
from app.modules.record import get_record_chain


# ── HTML 컴포넌트 헬퍼 ───────────────────────────────────────────────

def _status(text: str, state: str = "idle") -> str:
    """상태 배지 HTML. state: idle | running | done | error"""
    cfg = {
        "idle":    ("#adb5bd", "#f8f9fa", "#e9ecef", "대기"),
        "running": ("#0ca678", "#e6fcf5", "#b2f2cc", "진행 중"),
        "done":    ("#1c7ed6", "#e7f5ff", "#bac8ff", "완료"),
        "error":   ("#e03131", "#fff5f5", "#ffc9c9", "오류"),
    }
    dot_color, bg, border, label = cfg.get(state, cfg["idle"])
    anim = "animation:pulse 1.4s ease-in-out infinite;" if state == "running" else ""
    return (
        f'<div style="display:flex;align-items:center;gap:10px;padding:12px 16px;'
        f'background:{bg};border:1px solid {border};border-radius:10px;margin-bottom:16px;">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:{dot_color};'
        f'flex-shrink:0;{anim}"></span>'
        f'<span style="font-size:0.875rem;color:#1a1a2e;font-weight:500;flex:1;">'
        f'{html.escape(text)}</span>'
        f'<span style="font-size:0.75rem;color:{dot_color};font-weight:700;'
        f'letter-spacing:0.04em;text-transform:uppercase;">{label}</span>'
        f'</div>'
    )


def _alert(content: str, kind: str = "info") -> str:
    """Alert 컴포넌트 HTML. kind: info | success | warning | danger"""
    cfg = {
        "info":    ("#1c7ed6", "#e7f5ff", "#74c0fc", "ℹ"),
        "success": ("#2b8a3e", "#ebfbee", "#8ce99a", "✓"),
        "warning": ("#b45309", "#fffbeb", "#fcd34d", "⚠"),
        "danger":  ("#c92a2a", "#fff5f5", "#ffa8a8", "✕"),
    }
    text_color, bg, border, icon = cfg.get(kind, cfg["info"])
    return (
        f'<div style="display:flex;gap:10px;padding:14px 16px;background:{bg};'
        f'border:1px solid {border};border-radius:10px;margin-top:8px;">'
        f'<span style="color:{text_color};font-weight:700;flex-shrink:0;">{icon}</span>'
        f'<div style="color:#1a1a2e;font-size:0.875rem;line-height:1.6;">{content}</div>'
        f'</div>'
    )


def _result_card(title: str, content: str, muted: bool = False) -> str:
    """결과 카드 HTML."""
    title_color = "#6c757d" if muted else "#1a1a2e"
    content_color = "#495057" if muted else "#212529"
    border = "#e9ecef"
    return (
        f'<div style="background:#ffffff;border:1px solid {border};border-radius:12px;'
        f'padding:20px;margin-top:12px;">'
        f'<p style="margin:0 0 8px;font-size:0.75rem;font-weight:700;color:{title_color};'
        f'letter-spacing:0.08em;text-transform:uppercase;">{html.escape(title)}</p>'
        f'<div style="font-size:0.9375rem;color:{content_color};line-height:1.7;'
        f'white-space:pre-wrap;word-break:break-word;">{content}</div>'
        f'</div>'
    )


def _empty_result_card(title: str) -> str:
    return (
        f'<div style="background:#f8f9fa;border:1px dashed #dee2e6;border-radius:12px;'
        f'padding:20px;margin-top:12px;">'
        f'<p style="margin:0;font-size:0.75rem;font-weight:700;color:#adb5bd;'
        f'letter-spacing:0.08em;text-transform:uppercase;">{html.escape(title)}</p>'
        f'<p style="margin:8px 0 0;font-size:0.875rem;color:#ced4da;">결과가 여기에 표시됩니다.</p>'
        f'</div>'
    )


_IDLE_EXAM = (
    _status("대기 중", "idle"),
    "",
)

_IDLE_RECORD = (
    _status("대기 중", "idle"),
    _empty_result_card("개인정보 처리 결과"),
    _empty_result_card("생기부 작성 결과"),
    _empty_result_card("규정 확인 및 교사 안내"),
)


# ── 출제 모드 핸들러 ──────────────────────────────────────────────────

def run_exam(pdf_file, unit, item_type, difficulty, standards_text):
    if pdf_file is None:
        yield _status("PDF 파일을 업로드해주세요.", "error"), ""
        return
    if not unit.strip():
        yield _status("단원명을 입력해주세요.", "error"), ""
        return

    standards = [s.strip() for s in standards_text.split(",") if s.strip()]
    if not standards:
        standards = [f"{unit.strip()} 핵심 개념 이해"]

    spec: ExamSpec = {
        "unit": unit.strip(),
        "num_items": 1,
        "type_dist": {item_type: 1},
        "difficulty_dist": {difficulty: 1},
        "target": "내신",
        "standards": standards,
    }

    try:
        yield _status("PDF를 분석하고 있습니다...", "running"), ""
        pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name
        doc = parse_pdf(pdf_path)
        chunks = chunk_document(doc)
        if not chunks:
            yield _status("PDF에서 텍스트를 추출할 수 없습니다.", "error"), ""
            return

        spec["passage_text"] = "\n\n".join(c["text"] for c in chunks)[:4000]

        yield _status("AI 에이전트가 문항을 생성하고 있습니다. 수 분 소요됩니다...", "running"), ""
        graph = get_exam_graph()
        state = graph.invoke({
            "spec": spec,
            "source_collection": "",
            "budget": 2,
            "draft_items": [],
            "agent_messages": [],
            "coverage_map": {},
            "validation_passed": False,
            "error": "",
        })

        all_items = state.get("draft_items", [])
        approved = [it for it in all_items if it.get("status") == "approved"]
        passed = state.get("validation_passed", False)

        display_items = approved or sorted(
            all_items, key=lambda x: x.get("judge_score", 0), reverse=True
        )[:1]

        if not all_items:
            error_msg = state.get("error", "")
            backend = os.getenv("LLM_BACKEND", "local")
            hint = f"LLM_BACKEND={backend}" + (f" | {error_msg}" if error_msg else "")
            yield _status("문항이 생성되지 않았습니다.", "error"), _alert(
                f"문항 생성에 실패했습니다. ({html.escape(hint)})", "danger"
            )
            return

        # 결과 HTML 구성
        val_badge = (
            '<span style="display:inline-block;padding:2px 8px;border-radius:6px;'
            'background:#ebfbee;color:#2b8a3e;font-size:0.75rem;font-weight:700;">검증 통과</span>'
            if passed else
            '<span style="display:inline-block;padding:2px 8px;border-radius:6px;'
            'background:#fff5f5;color:#c92a2a;font-size:0.75rem;font-weight:700;">검증 미통과</span>'
        )
        meta_line = (
            f'<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:16px;">'
            f'{val_badge}'
            f'<span style="font-size:0.8125rem;color:#6c757d;">단원 <strong style="color:#212529;">'
            f'{html.escape(unit)}</strong></span>'
            f'<span style="font-size:0.8125rem;color:#6c757d;">유형 <strong style="color:#212529;">'
            f'{html.escape(item_type)}</strong></span>'
            f'<span style="font-size:0.8125rem;color:#6c757d;">난이도 <strong style="color:#212529;">'
            f'{html.escape(difficulty)}</strong></span>'
            f'</div>'
        )

        items_html = ""
        for item in display_items:
            score = item.get("judge_score", 0)
            q = html.escape(item.get("question", "(없음)"))
            opts = item.get("options", [])
            ans = html.escape(item.get("answer", "?"))

            opts_html = ""
            if opts:
                opts_html = "<ol style='margin:10px 0 10px 20px;padding:0;color:#212529;line-height:1.8;list-style:none;'>"
                for o in opts:
                    opts_html += f"<li style='margin-bottom:4px;'>{html.escape(str(o))}</li>"
                opts_html += "</ol>"
                opts_html += (
                    f'<div style="padding:8px 12px;background:#e7f5ff;border-radius:8px;'
                    f'font-size:0.875rem;color:#1c7ed6;font-weight:600;">정답: {ans}</div>'
                )

            score_bar = "".join(
                f'<span style="display:inline-block;width:14px;height:14px;border-radius:3px;'
                f'background:{"#12b886" if i < score else "#e9ecef"};margin-right:2px;"></span>'
                for i in range(5)
            )

            items_html += (
                f'<div style="background:#ffffff;border:1px solid #e9ecef;border-radius:12px;'
                f'padding:20px;margin-bottom:12px;">'
                f'<p style="margin:0 0 12px;font-size:1rem;color:#1a1a2e;font-weight:600;'
                f'line-height:1.6;">{q}</p>'
                f'{opts_html}'
                f'<div style="display:flex;align-items:center;gap:8px;margin-top:12px;'
                f'padding-top:12px;border-top:1px solid #f1f3f5;">'
                f'<span style="font-size:0.75rem;color:#6c757d;">품질 점수</span>'
                f'{score_bar}'
                f'<span style="font-size:0.75rem;color:#495057;font-weight:600;">{int(score)}/5</span>'
                f'</div>'
                f'</div>'
            )

        result_html = (
            f'<div style="padding:4px 0;">'
            f'{meta_line}'
            f'{items_html}'
            f'</div>'
        )
        yield _status("문항 생성 완료", "done"), result_html

    except Exception:
        yield _status("오류가 발생했습니다.", "error"), _alert(
            f"<pre style='margin:0;font-size:0.8rem;white-space:pre-wrap;'>"
            f"{html.escape(traceback.format_exc())}</pre>", "danger"
        )


# ── 생기부 모드 핸들러 ────────────────────────────────────────────────

async def run_record(memo: str):
    if not memo.strip():
        yield _status("관찰 메모를 입력해주세요.", "error"), *_IDLE_RECORD[1:]
        return

    try:
        yield (
            _status("개인정보를 확인하고 있습니다...", "running"),
            _status("처리 중...", "running"),
            _status("처리 중...", "running"),
            _status("처리 중...", "running"),
        )

        chain = get_record_chain()

        yield (
            _status("학교생활기록부 문체로 다듬고 있습니다...", "running"),
            _status("처리 중...", "running"),
            _status("처리 중...", "running"),
            _status("처리 중...", "running"),
        )
        out = await chain.run(memo.strip())

        # 개인정보 카드
        pii_list = out.get("pii_found") or []
        if pii_list:
            pii_items = "".join(
                f'<li style="margin-bottom:4px;">{html.escape(str(p))}</li>'
                for p in pii_list
            )
            pii_body = (
                f'<p style="margin:0 0 8px;color:#b45309;font-size:0.875rem;font-weight:600;">'
                f'감지된 개인정보 {len(pii_list)}건 — 마스킹 처리됨</p>'
                f'<ul style="margin:0;padding-left:20px;color:#495057;font-size:0.875rem;">'
                f'{pii_items}</ul>'
            )
            pii_card = _result_card("개인정보 처리 결과", pii_body)
        else:
            pii_card = _result_card(
                "개인정보 처리 결과",
                '<span style="color:#2b8a3e;font-weight:600;">✓ 감지된 개인정보 없음</span>'
            )

        masked = html.escape(out.get("masked_memo", ""))
        if masked:
            pii_card += _result_card("마스킹된 메모", masked, muted=True)

        # 생기부 작성 결과 카드
        polished = html.escape(out.get("polished", ""))
        polished_card = _result_card("생기부 작성 결과", polished) if polished else _empty_result_card("생기부 작성 결과")

        # 규정 확인 + 교사 안내 카드
        violations = out.get("violations") or []
        warning = (out.get("warning") or "").strip()

        check_html = ""
        if violations:
            viol_items = "".join(
                f'<li style="margin-bottom:6px;">{html.escape(str(v))}</li>'
                for v in violations
            )
            check_html += _alert(
                f'<strong>규정 위반 {len(violations)}건 감지</strong>'
                f'<ul style="margin:8px 0 0;padding-left:20px;line-height:1.7;">'
                f'{viol_items}</ul>',
                "danger"
            )
        else:
            check_html += _alert("규정 위반 사항이 감지되지 않았습니다.", "success")

        if warning:
            # 경고 텍스트에서 [교사 확인 사항] 제목 제거 후 표시
            body = warning.replace("[교사 확인 사항]", "").strip()
            check_html += _alert(html.escape(body), "info")

        check_card = (
            f'<div style="background:#ffffff;border:1px solid #e9ecef;border-radius:12px;'
            f'padding:20px;margin-top:12px;">'
            f'<p style="margin:0 0 4px;font-size:0.75rem;font-weight:700;color:#6c757d;'
            f'letter-spacing:0.08em;text-transform:uppercase;">규정 확인 및 교사 안내</p>'
            f'{check_html}'
            f'</div>'
        )

        yield _status("다듬기 완료", "done"), pii_card, polished_card, check_card

    except Exception as e:
        yield (
            _status(f"오류: {html.escape(str(e))}", "error"),
            _empty_result_card("개인정보 처리 결과"),
            _empty_result_card("생기부 작성 결과"),
            _alert(f"<pre style='margin:0;font-size:0.8rem;white-space:pre-wrap;'>"
                   f"{html.escape(traceback.format_exc())}</pre>", "danger"),
        )


# ── CSS 디자인 시스템 ─────────────────────────────────────────────────

_CSS = """
/* ── Design Tokens ── */
:root {
    --bg:             #f0f2f5;
    --surface:        #ffffff;
    --surface-muted:  #f8f9fa;
    --primary:        #12b886;
    --primary-hover:  #0ca678;
    --primary-light:  #e6fcf5;
    --primary-ring:   rgba(18,184,134,0.18);
    --text:           #111827;
    --text-secondary: #4b5563;
    --text-muted:     #9ca3af;
    --border:         #e5e7eb;
    --border-focus:   #12b886;
    --radius-sm:      8px;
    --radius-md:      12px;
    --radius-lg:      20px;
    --shadow-sm:      0 1px 2px rgba(0,0,0,0.05);
    --shadow-md:      0 4px 16px rgba(0,0,0,0.08);
    --shadow-lg:      0 8px 32px rgba(0,0,0,0.10);
}

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
.dark, [data-theme="dark"] { color-scheme: light !important; }

body, .gradio-container, .gradio-container > .main, .gradio-container > .main > .wrap {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Pretendard', 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Gradio 내부 블록 배경 전부 투명 — 우리 카드만 배경 갖는다 */
.gradio-container .block,
.gradio-container .form,
.gradio-container .gap,
.gradio-container .panel {
    background: transparent !important;
    box-shadow: none !important;
}

/* ── Layout ── */
.gradio-container > .main {
    max-width: 1160px !important;
    margin: 0 auto !important;
    padding: 0 20px 60px !important;
}

/* ── Header ── */
.bp-header {
    padding: 28px 0 20px;
    margin-bottom: 8px;
    border-bottom: 1px solid var(--border);
}
.bp-header h1 {
    margin: 0 0 3px;
    font-size: 1.375rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.025em;
    display: flex;
    align-items: center;
    gap: 10px;
}
.bp-header h1::before {
    content: '';
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--primary);
    flex-shrink: 0;
}
.bp-header p {
    margin: 0;
    font-size: 0.8125rem;
    color: var(--text-muted);
    padding-left: 20px;
}

/* ── Tabs ── */
.bp-tabs > .tab-nav {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 !important;
    gap: 0 !important;
    margin-bottom: 24px !important;
}
.bp-tabs > .tab-nav button {
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    background: transparent !important;
    color: var(--text-muted) !important;
    font-size: 0.9375rem !important;
    font-weight: 600 !important;
    padding: 14px 20px !important;
    margin-bottom: -1px !important;
    transition: color 0.15s, border-color 0.15s !important;
}
.bp-tabs > .tab-nav button:hover:not(.selected) {
    color: var(--text-secondary) !important;
}
.bp-tabs > .tab-nav button.selected {
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
}

/* ── Info Banner ── */
.bp-info {
    background: var(--primary-light) !important;
    border: 1px solid #b2f2cc !important;
    border-radius: var(--radius-sm) !important;
    padding: 11px 16px !important;
    margin-bottom: 20px !important;
    font-size: 0.8125rem !important;
    color: #0b6e4f !important;
    line-height: 1.55 !important;
}

/* ── Input Panel (left column card) ── */
.bp-panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 0 !important;
    box-shadow: var(--shadow-md) !important;
    overflow: hidden !important;
}

/* 패널 안 각 블록 구분선 */
.bp-panel .block,
.bp-panel .form {
    border: none !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 16px 20px !important;
    margin: 0 !important;
}
.bp-panel .block:last-child,
.bp-panel .form:last-child {
    border-bottom: none !important;
}

/* 블록 라벨 */
.bp-panel .block-label,
.bp-panel .label-wrap {
    background: transparent !important;
    border: none !important;
    padding: 0 0 6px !important;
}
.bp-panel label > span,
.bp-panel .label-wrap > span {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ── File Upload ── */
/* 파일 업로드 블록 자체도 동일한 패딩 구조 */
#bp-pdf-upload {
    border-bottom: 1px solid var(--border) !important;
    padding: 16px 20px !important;
    background: transparent !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}
#bp-pdf-upload .wrap {
    background: var(--primary-light) !important;
    border: 2px dashed var(--primary) !important;
    border-radius: var(--radius-sm) !important;
    min-height: 110px !important;
    transition: background 0.15s !important;
    cursor: pointer !important;
}
#bp-pdf-upload .wrap:hover { background: #ccf5e7 !important; }
#bp-pdf-upload .wrap * { background: transparent !important; color: var(--text-secondary) !important; }
#bp-pdf-upload .wrap svg { stroke: var(--primary) !important; fill: none !important; }
#bp-pdf-upload .block-label, #bp-pdf-upload .label-wrap {
    background: transparent !important; border: none !important; padding: 0 0 8px !important;
}
#bp-pdf-upload .label-wrap span {
    font-size: 0.75rem !important; font-weight: 700 !important;
    color: var(--text-muted) !important; text-transform: uppercase !important; letter-spacing: 0.06em !important;
}

/* ── Inputs & Textarea ── */
.gradio-container input[type="text"],
.gradio-container input[type="search"],
.gradio-container textarea {
    background: var(--surface-muted) !important;
    color: var(--text) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 13px !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    transition: border-color 0.15s, background 0.15s, box-shadow 0.15s !important;
}
.gradio-container input[type="text"]:focus,
.gradio-container textarea:focus {
    background: var(--surface) !important;
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-ring) !important;
    outline: none !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder { color: var(--text-muted) !important; }

/* ── Radio — Pill Buttons ── */
.bp-radio input[type="radio"] {
    appearance: none !important; -webkit-appearance: none !important;
    width: 0 !important; height: 0 !important; margin: 0 !important; padding: 0 !important;
    position: absolute !important; opacity: 0 !important; pointer-events: none !important;
}
.bp-radio > .wrap, .bp-radio fieldset, .bp-radio > div {
    display: flex !important; flex-wrap: wrap !important; gap: 6px !important;
    border: none !important; padding: 0 !important; margin: 0 !important; background: transparent !important;
}
.bp-radio label {
    display: inline-flex !important; align-items: center !important;
    padding: 5px 16px !important; border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-lg) !important; background: var(--surface-muted) !important;
    cursor: pointer !important; font-size: 0.8125rem !important; font-weight: 500 !important;
    color: var(--text-secondary) !important; transition: all 0.15s !important; user-select: none !important;
}
.bp-radio label:hover {
    border-color: var(--primary) !important; color: var(--primary) !important; background: var(--primary-light) !important;
}
.bp-radio label:has(input:checked) {
    border-color: var(--primary) !important; background: var(--primary) !important;
    color: #fff !important; font-weight: 700 !important; box-shadow: 0 2px 6px rgba(18,184,134,0.3) !important;
}
.bp-radio label:has(input:checked) span { color: #fff !important; }

/* ── Primary Button ── */
.gradio-container button.primary {
    background: var(--primary) !important;
    border: none !important; border-radius: var(--radius-sm) !important;
    color: #fff !important; font-size: 0.9375rem !important; font-weight: 700 !important;
    padding: 13px 24px !important; width: 100% !important; letter-spacing: 0.01em !important;
    box-shadow: 0 2px 8px rgba(18,184,134,0.3) !important;
    transition: background 0.15s, box-shadow 0.15s, transform 0.1s !important;
    cursor: pointer !important;
}
.gradio-container button.primary:hover {
    background: var(--primary-hover) !important;
    box-shadow: 0 4px 16px rgba(18,184,134,0.4) !important;
    transform: translateY(-1px) !important;
}
.gradio-container button.primary:active { transform: none !important; }
.gradio-container button.primary:disabled { opacity: 0.45 !important; transform: none !important; cursor: not-allowed !important; }

/* ── Result Panel (right column) ── */
.bp-result-panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 20px !important;
    box-shadow: var(--shadow-md) !important;
    min-height: 200px !important;
}

/* HTML 결과 컴포넌트 배경 투명 */
.bp-result-panel .block { padding: 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Animations ── */
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
}
:focus-visible { outline: 2px solid var(--primary) !important; outline-offset: 2px !important; }
"""

_HEADER_HTML = """
<div class="bp-header">
    <h1>분필</h1>
    <p>고등학교 사회 교사용 AI 어시스턴트 &nbsp;·&nbsp; AI-powered exam &amp; school record tool</p>
</div>
"""

_EXAM_INFO_HTML = """
<div class="bp-info">
    지문 PDF를 업로드하고 유형·난이도를 지정한 뒤 <strong>문항 생성</strong>을 누르세요. 생성에 수 분 소요됩니다.
</div>
"""

_RECORD_INFO_HTML = """
<div class="bp-info">
    교사 관찰 메모를 입력하면 학교생활기록부 문체로 다듬어 드립니다.
    입력 내용은 <strong>저장되지 않으며</strong>, 개인정보는 모델 호출 전에 자동으로 가려집니다.
</div>
"""

_EXAM_EMPTY_HTML = """
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
            min-height:240px;text-align:center;padding:32px 20px;">
    <div style="width:52px;height:52px;border-radius:14px;background:#e6fcf5;
                display:flex;align-items:center;justify-content:center;margin-bottom:16px;">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
             stroke="#12b886" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="16" y1="13" x2="8" y2="13"/>
            <line x1="16" y1="17" x2="8" y2="17"/>
            <polyline points="10 9 9 9 8 9"/>
        </svg>
    </div>
    <p style="margin:0 0 6px;font-size:0.9375rem;font-weight:600;color:#374151;">생성된 문항이 여기에 표시됩니다</p>
    <p style="margin:0;font-size:0.8125rem;color:#9ca3af;line-height:1.6;">
        PDF를 업로드하고 설정을 지정한 뒤<br>문항 생성 버튼을 누르세요
    </p>
</div>
"""

_RECORD_EMPTY_HTML = """
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
            min-height:180px;text-align:center;padding:32px 20px;">
    <div style="width:52px;height:52px;border-radius:14px;background:#e6fcf5;
                display:flex;align-items:center;justify-content:center;margin-bottom:16px;">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
             stroke="#12b886" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
        </svg>
    </div>
    <p style="margin:0 0 6px;font-size:0.9375rem;font-weight:600;color:#374151;">다듬기 결과가 여기에 표시됩니다</p>
    <p style="margin:0;font-size:0.8125rem;color:#9ca3af;">관찰 메모를 입력하고 버튼을 누르세요</p>
</div>
"""


def build_ui() -> gr.Blocks:
    _theme = gr.themes.Base(
        primary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.slate,
        font=["Pretendard", "Noto Sans KR", "system-ui", "sans-serif"],
    ).set(
        # 라이트 모드 강제 — dark 변형까지 동일한 밝은 색으로 덮어씀
        body_background_fill="#f7f8f9",
        body_background_fill_dark="#f7f8f9",
        body_text_color="#1a1a2e",
        body_text_color_dark="#1a1a2e",
        body_text_color_subdued="#6c757d",
        body_text_color_subdued_dark="#6c757d",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#ffffff",
        background_fill_secondary="#f1f3f5",
        background_fill_secondary_dark="#f1f3f5",
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        block_border_color="#dee2e6",
        block_border_color_dark="#dee2e6",
        block_label_background_fill="#ffffff",
        block_label_background_fill_dark="#ffffff",
        block_label_text_color="#1a1a2e",
        block_label_text_color_dark="#1a1a2e",
        block_title_text_color="#1a1a2e",
        block_title_text_color_dark="#1a1a2e",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        input_border_color="#dee2e6",
        input_border_color_dark="#dee2e6",
        input_placeholder_color="#adb5bd",
        input_placeholder_color_dark="#adb5bd",
        border_color_primary="#dee2e6",
        border_color_primary_dark="#dee2e6",
        border_color_accent="#12b886",
        border_color_accent_dark="#12b886",
        color_accent_soft="rgba(18,184,134,0.1)",
        color_accent_soft_dark="rgba(18,184,134,0.1)",
        checkbox_background_color="#ffffff",
        checkbox_background_color_dark="#ffffff",
        table_even_background_fill="#ffffff",
        table_even_background_fill_dark="#ffffff",
        table_odd_background_fill="#f7f8f9",
        table_odd_background_fill_dark="#f7f8f9",
        panel_background_fill="#ffffff",
        panel_background_fill_dark="#ffffff",
        panel_border_color="#dee2e6",
        panel_border_color_dark="#dee2e6",
    )

    with gr.Blocks(
        title="분필",
        css=_CSS,
        theme=_theme,
    ) as demo:

        gr.HTML(_HEADER_HTML)

        with gr.Tabs(elem_classes=["bp-tabs"]):

            # ── 탭 1: 문항 출제 ────────────────────────────────────────
            with gr.TabItem("문항 출제"):
                gr.HTML(_EXAM_INFO_HTML)
                with gr.Row(equal_height=False):

                    # 좌측: 입력 패널
                    with gr.Column(scale=1, min_width=300):
                        with gr.Group(elem_classes=["bp-panel"]):
                            pdf_input = gr.File(
                                label="지문 PDF",
                                file_types=[".pdf"],
                                type="filepath",
                                elem_id="bp-pdf-upload",
                            )
                            unit_input = gr.Textbox(
                                label="단원명",
                                placeholder="예: 민주주의와 헌법",
                            )
                            standards_input = gr.Textbox(
                                label="성취기준 (쉼표 구분 · 비워두면 자동)",
                                placeholder="예: 민주주의 원리 이해, 기본권 보장의 의미",
                            )
                            type_radio = gr.Radio(
                                choices=["객관식", "서술형"],
                                value="객관식",
                                label="문항 유형",
                                elem_classes=["bp-radio"],
                            )
                            diff_radio = gr.Radio(
                                choices=["상", "중", "하"],
                                value="중",
                                label="난이도",
                                elem_classes=["bp-radio"],
                            )
                        exam_btn = gr.Button("문항 생성", variant="primary", size="lg")

                    # 우측: 결과 패널
                    with gr.Column(scale=2, min_width=420):
                        with gr.Group(elem_classes=["bp-result-panel"]):
                            exam_status = gr.HTML(value=_status("대기 중", "idle"))
                            exam_output = gr.HTML(value=_EXAM_EMPTY_HTML)

                exam_btn.click(
                    fn=run_exam,
                    inputs=[pdf_input, unit_input, type_radio, diff_radio, standards_input],
                    outputs=[exam_status, exam_output],
                )

            # ── 탭 2: 생기부 다듬기 ────────────────────────────────────
            with gr.TabItem("생기부 다듬기"):
                gr.HTML(_RECORD_INFO_HTML)
                with gr.Row(equal_height=False):

                    # 좌측: 입력 패널
                    with gr.Column(scale=1, min_width=300):
                        with gr.Group(elem_classes=["bp-panel"]):
                            memo_input = gr.Textbox(
                                label="관찰 메모",
                                placeholder=(
                                    "예: 수업 시간에 발표를 잘 함. "
                                    "모둠 토론에서 논리적으로 의견을 제시하고 "
                                    "친구들이 이해하지 못할 때 적극적으로 도움."
                                ),
                                lines=8,
                            )
                        record_btn = gr.Button("생기부 다듬기", variant="primary", size="lg")

                    # 우측: 결과 패널
                    with gr.Column(scale=2, min_width=420):
                        with gr.Group(elem_classes=["bp-result-panel"]):
                            record_status = gr.HTML(value=_status("대기 중", "idle"))
                            pii_output     = gr.HTML(value=_RECORD_EMPTY_HTML)
                            polished_output = gr.HTML(value="")
                            check_output   = gr.HTML(value="")

                record_btn.click(
                    fn=run_record,
                    inputs=[memo_input],
                    outputs=[record_status, pii_output, polished_output, check_output],
                )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
