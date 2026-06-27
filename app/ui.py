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
    --bg:             #f7f8f9;
    --surface:        #ffffff;
    --surface-muted:  #f1f3f5;
    --primary:        #12b886;
    --primary-hover:  #0ca678;
    --primary-light:  #e6fcf5;
    --primary-ring:   rgba(18,184,134,0.18);
    --text:           #1a1a2e;
    --text-secondary: #495057;
    --text-muted:     #868e96;
    --border:         #dee2e6;
    --border-focus:   #12b886;
    --radius-sm:      8px;
    --radius-md:      12px;
    --radius-lg:      16px;
    --shadow-sm:      0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md:      0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
}

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

body,
.gradio-container,
.gradio-container > .main,
.gradio-container > .main > .wrap {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Pretendard', 'Noto Sans KR', -apple-system, BlinkMacSystemFont,
                 'Segoe UI', sans-serif !important;
}

/* Gradio 기본 다크 배경 제거 */
.dark, [data-theme="dark"] { color-scheme: light !important; }

/* ── Page Layout ── */
.gradio-container > .main {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0 24px 48px !important;
}

/* ── Header ── */
.bp-header {
    background: var(--surface);
    border-bottom: 2px solid var(--primary);
    padding: 22px 32px;
    margin: 0 -24px 28px;
}
.bp-header h1 {
    margin: 0 0 4px;
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.02em;
}
.bp-header p {
    margin: 0;
    font-size: 0.875rem;
    color: var(--text-muted);
}

/* ── Tabs ── */
.bp-tabs > .tab-nav {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 4px !important;
    gap: 0 !important;
}
.bp-tabs > .tab-nav button {
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    background: transparent !important;
    color: var(--text-muted) !important;
    font-size: 0.9375rem !important;
    font-weight: 600 !important;
    padding: 12px 20px !important;
    margin-bottom: -1px !important;
    transition: color 0.15s, border-color 0.15s !important;
}
.bp-tabs > .tab-nav button:hover:not(.selected) {
    color: var(--text) !important;
    background: var(--surface-muted) !important;
}
.bp-tabs > .tab-nav button.selected {
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
    background: transparent !important;
}

/* ── Card ── */
.bp-card {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 24px !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Info Banner ── */
.bp-info {
    background: var(--primary-light) !important;
    border-left: 3px solid var(--primary) !important;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0 !important;
    padding: 10px 14px !important;
    margin-bottom: 20px !important;
    font-size: 0.875rem !important;
    color: #0b6e4f !important;
    line-height: 1.5 !important;
}

/* ── Labels ── */
.gradio-container label > span,
.gradio-container .label-wrap > span {
    color: var(--text) !important;
    font-size: 0.8125rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
}

/* ── Inputs & Textarea ── */
.gradio-container input[type="text"],
.gradio-container input[type="search"],
.gradio-container textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 14px !important;
    font-size: 0.9375rem !important;
    line-height: 1.6 !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
.gradio-container input[type="text"]:focus,
.gradio-container textarea:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px var(--primary-ring) !important;
    outline: none !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ── Readonly / Disabled Textbox ── */
.gradio-container textarea[readonly],
.gradio-container textarea:disabled,
.gradio-container input[readonly],
.gradio-container input:disabled {
    background: var(--surface-muted) !important;
    color: var(--text-secondary) !important;
    border-color: var(--border) !important;
}

/* ── Radio (Segmented style) ── */
.bp-card .gr-radio-group,
.bp-card fieldset {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
.bp-card .gr-radio-group label,
.bp-card fieldset label {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    padding: 7px 16px !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    background: var(--surface) !important;
    cursor: pointer !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    transition: all 0.15s !important;
    user-select: none !important;
}
.bp-card .gr-radio-group label:hover,
.bp-card fieldset label:hover {
    border-color: var(--primary) !important;
    color: var(--primary) !important;
    background: var(--primary-light) !important;
}
.bp-card input[type="radio"]:checked + span {
    color: var(--primary) !important;
    font-weight: 700 !important;
}

/* ── File Upload ── */
.gradio-container .upload-container,
.gradio-container .file-preview,
.gradio-container [data-testid="file"] {
    background: var(--primary-light) !important;
    border: 2px dashed var(--primary) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text) !important;
    transition: background 0.15s !important;
}
.gradio-container .upload-container:hover,
.gradio-container [data-testid="file"]:hover {
    background: #d3f9e8 !important;
}

/* ── Primary Button ── */
.gradio-container button.primary,
.gradio-container .gr-button.primary,
button[variant="primary"] {
    background: var(--primary) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    color: #ffffff !important;
    font-size: 0.9375rem !important;
    font-weight: 700 !important;
    padding: 12px 24px !important;
    width: 100% !important;
    box-shadow: 0 2px 8px rgba(18,184,134,0.25) !important;
    transition: background 0.15s, box-shadow 0.15s, transform 0.1s !important;
    cursor: pointer !important;
}
.gradio-container button.primary:hover {
    background: var(--primary-hover) !important;
    box-shadow: 0 4px 14px rgba(18,184,134,0.35) !important;
    transform: translateY(-1px) !important;
}
.gradio-container button.primary:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 6px rgba(18,184,134,0.25) !important;
}
.gradio-container button.primary:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
    transform: none !important;
}

/* ── Markdown Result ── */
.bp-result .prose,
.bp-result p,
.bp-result li,
.bp-result h1,
.bp-result h2,
.bp-result h3 {
    color: var(--text) !important;
}

/* ── Section Divider ── */
.bp-divider {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 20px 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Animations ── */
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
}

/* ── Focus visible ── */
:focus-visible {
    outline: 2px solid var(--primary) !important;
    outline-offset: 2px !important;
}
"""

_HEADER_HTML = """
<div class="bp-header">
    <h1>분필</h1>
    <p>고등학교 사회 교사용 AI 어시스턴트 &nbsp;·&nbsp; AI-powered exam &amp; school record tool</p>
</div>
"""

_EXAM_INFO_HTML = """
<div class="bp-info">
    지문 PDF를 업로드하고 유형·난이도를 지정한 뒤 <strong>문항 생성</strong>을 누르세요.
    생성에 수 분 소요됩니다.
</div>
"""

_RECORD_INFO_HTML = """
<div class="bp-info">
    교사 관찰 메모를 입력하면 학교생활기록부 문체로 다듬어 드립니다.
    입력 내용은 <strong>저장되지 않으며</strong>, 개인정보는 모델 호출 전에 자동으로 가려집니다.
</div>
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="분필",
        css=_CSS,
        theme=gr.themes.Base(
            font=["Pretendard", "Noto Sans KR", "system-ui", "sans-serif"],
            primary_hue=gr.themes.colors.green,
            neutral_hue=gr.themes.colors.slate,
        ),
    ) as demo:

        gr.HTML(_HEADER_HTML)

        with gr.Tabs(elem_classes=["bp-tabs"]):

            # ── 탭 1: 문항 출제 ────────────────────────────────────────
            with gr.TabItem("문항 출제"):
                gr.HTML(_EXAM_INFO_HTML)
                with gr.Row(equal_height=False):

                    # 좌측: 입력 패널
                    with gr.Column(scale=1, min_width=280):
                        with gr.Group(elem_classes=["bp-card"]):
                            pdf_input = gr.File(
                                label="지문 PDF",
                                file_types=[".pdf"],
                                type="filepath",
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
                            )
                            diff_radio = gr.Radio(
                                choices=["상", "중", "하"],
                                value="중",
                                label="난이도",
                            )
                        exam_btn = gr.Button("문항 생성", variant="primary", size="lg")

                    # 우측: 결과 패널
                    with gr.Column(scale=2, min_width=400):
                        exam_status = gr.HTML(value=_status("대기 중", "idle"))
                        exam_output = gr.HTML(value="", elem_classes=["bp-result"])

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
                    with gr.Column(scale=1, min_width=280):
                        with gr.Group(elem_classes=["bp-card"]):
                            memo_input = gr.Textbox(
                                label="관찰 메모",
                                placeholder=(
                                    "예: 수업 시간에 발표를 잘 함. "
                                    "모둠 토론에서 논리적으로 의견을 제시하고 "
                                    "친구들이 이해하지 못할 때 적극적으로 도움."
                                ),
                                lines=7,
                            )
                        record_btn = gr.Button("생기부 다듬기", variant="primary", size="lg")

                    # 우측: 결과 패널
                    with gr.Column(scale=2, min_width=400):
                        record_status = gr.HTML(value=_status("대기 중", "idle"))
                        pii_output     = gr.HTML(value=_empty_result_card("개인정보 처리 결과"))
                        polished_output = gr.HTML(value=_empty_result_card("생기부 작성 결과"))
                        check_output   = gr.HTML(value=_empty_result_card("규정 확인 및 교사 안내"))

                record_btn.click(
                    fn=run_record,
                    inputs=[memo_input],
                    outputs=[record_status, pii_output, polished_output, check_output],
                )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
