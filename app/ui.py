"""분필 Gradio UI — 출제 모드 + 생기부 윤문 모드."""
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

# ── 상태 표시 헬퍼 ───────────────────────────────────────────────────

def _status(text: str, state: str = "running") -> str:
    """상태 칩 HTML 반환. state: running | done | error | idle"""
    cfg = {
        "running": ("#00C471", "진행 중",  "animation:status-pulse 1.2s ease-in-out infinite;"),
        "done":    ("#00C471", "완료",     ""),
        "error":   ("#E03131", "오류",     ""),
        "idle":    ("#ADB5BD", "대기",     ""),
    }
    color, label, anim = cfg.get(state, cfg["idle"])
    return (
        f'<div style="display:flex;align-items:center;gap:10px;padding:12px 16px;'
        f'background:#F8F9FA;border-radius:8px;border:1px solid #E9ECEF;margin-bottom:12px;">'
        f'<span style="width:9px;height:9px;border-radius:50%;background:{color};'
        f'display:inline-block;flex-shrink:0;{anim}"></span>'
        f'<span style="font-size:0.9rem;color:#212529;font-weight:500;">{text}</span>'
        f'<span style="margin-left:auto;font-size:0.75rem;color:{color};font-weight:700;'
        f'letter-spacing:0.06em;">{label}</span>'
        f'</div>'
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
        yield _status("PDF를 분석하고 있습니다..."), ""
        pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name
        doc = parse_pdf(pdf_path)
        chunks = chunk_document(doc)
        if not chunks:
            yield _status("PDF에서 텍스트를 추출할 수 없습니다.", "error"), ""
            return

        # 지문 텍스트를 에이전트 프롬프트에 직접 전달 (임베딩 단계 불필요)
        spec["passage_text"] = "\n\n".join(c["text"] for c in chunks)[:4000]

        yield _status("AI 에이전트가 문항을 생성하고 있습니다. 수 분 소요됩니다..."), ""
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

        # 승인된 문항만 표시. 없으면 점수 최고 1개로 fallback
        display_items = approved or sorted(
            all_items, key=lambda x: x.get("judge_score", 0), reverse=True
        )[:1]

        status_label = "통과" if passed else "미통과"
        lines = [
            "## 출제 결과",
            f"**검증**: {status_label}  |  **단원**: {unit}  |  **유형**: {item_type}  |  **난이도**: {difficulty}",
            "",
        ]
        for item in display_items:
            score = item.get("judge_score", 0)
            lines.append(f"**질문**: {item.get('question','(없음)')}")
            opts = item.get("options", [])
            if opts:
                for o in opts:
                    lines.append(f"- {o}")
                lines.append(f"\n**정답**: {item.get('answer','?')}")
            lines.append(f"\n품질 점수: {score:.0f}/5")
            lines.append("")

        if not all_items:
            error_msg = state.get("error", "")
            backend = os.getenv("LLM_BACKEND", "local")
            hint = f"LLM_BACKEND={backend}" + (f" | {error_msg}" if error_msg else "")
            lines.append(f"문항이 생성되지 않았습니다. ({hint})")

        result = "\n".join(lines)
        yield _status("문항 생성 완료", "done"), result

    except Exception:
        yield _status("오류가 발생했습니다.", "error"), f"```\n{traceback.format_exc()}\n```"


# ── 생기부 모드 핸들러 ────────────────────────────────────────────────

async def run_record(memo: str):
    if not memo.strip():
        yield _status("관찰 메모를 입력해주세요.", "error"), "", "", ""
        return

    try:
        yield _status("개인정보를 확인하고 있습니다..."), "", "", ""

        chain = get_record_chain()

        yield _status("학교생활기록부 문체로 다듬고 있습니다..."), "", "", ""
        out = await chain.run(memo.strip())

        pii_info = f"감지된 개인정보: {', '.join(out['pii_found'])}" if out["pii_found"] else "감지된 개인정보 없음"
        masked_display = f"{out['masked_memo']}\n\n({pii_info})"
        viol_md = (
            "**규정 위반 감지:**\n" + "\n".join(f"- {v}" for v in out["violations"])
            if out["violations"] else "규정 위반 없음"
        )
        if out.get("warning"):
            viol_md += f"\n\n---\n{out['warning']}"

        yield _status("다듬기 완료", "done"), masked_display, out["polished"], viol_md

    except Exception as e:
        yield _status(f"오류: {e}", "error"), "", "", ""


_CSS = """
/* ── Velog-inspired design ── */
:root {
    --velog-green: #00C471;
    --velog-green-dark: #00A862;
    --velog-green-light: #E8FAF2;
    --velog-text: #212529;
    --velog-text-secondary: #868E96;
    --velog-bg: #F8F9FA;
    --velog-white: #FFFFFF;
    --velog-border: #E9ECEF;
    --velog-radius: 8px;
}

/* 전체 배경 */
body, .gradio-container {
    background-color: var(--velog-bg) !important;
    font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: var(--velog-text) !important;
}

/* 헤더 영역 */
.bunpil-header {
    background: var(--velog-white);
    border-bottom: 2px solid var(--velog-green);
    padding: 20px 32px;
    margin-bottom: 24px;
    border-radius: 0;
}

.bunpil-header h1 {
    color: var(--velog-text) !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    margin: 0 0 4px 0 !important;
}

.bunpil-header p {
    color: var(--velog-text-secondary) !important;
    font-size: 0.9rem !important;
    margin: 0 !important;
}

/* 탭 */
.tab-nav button {
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    color: var(--velog-text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 20px !important;
    background: transparent !important;
    transition: all 0.2s !important;
}

.tab-nav button.selected {
    border-bottom: 2px solid var(--velog-green) !important;
    color: var(--velog-green) !important;
    background: transparent !important;
}

.tab-nav button:hover:not(.selected) {
    color: var(--velog-text) !important;
    background: var(--velog-green-light) !important;
}

/* 카드 패널 */
.card-panel {
    background: var(--velog-white) !important;
    border: 1px solid var(--velog-border) !important;
    border-radius: var(--velog-radius) !important;
    padding: 24px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}

/* 입력 필드 */
input[type="text"], textarea, .gr-input, .gr-textarea {
    border: 1px solid var(--velog-border) !important;
    border-radius: var(--velog-radius) !important;
    padding: 10px 14px !important;
    font-size: 0.95rem !important;
    background: var(--velog-white) !important;
    color: var(--velog-text) !important;
    transition: border-color 0.2s !important;
}

input[type="text"]:focus, textarea:focus {
    border-color: var(--velog-green) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(0, 196, 113, 0.12) !important;
}

/* 라벨 */
label span, .gr-form label {
    color: var(--velog-text) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em !important;
}

/* 라디오 버튼 */
.gr-radio label {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    cursor: pointer !important;
}

input[type="radio"]:checked + span {
    color: var(--velog-green) !important;
    font-weight: 700 !important;
}

/* Primary 버튼 (문항 생성) */
button.primary, .gr-button-primary {
    background: var(--velog-green) !important;
    border: none !important;
    border-radius: var(--velog-radius) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 24px !important;
    cursor: pointer !important;
    transition: background 0.2s, transform 0.1s !important;
    width: 100% !important;
    box-shadow: 0 2px 6px rgba(0, 196, 113, 0.3) !important;
}

button.primary:hover, .gr-button-primary:hover {
    background: var(--velog-green-dark) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0, 196, 113, 0.4) !important;
}

button.primary:active, .gr-button-primary:active {
    transform: translateY(0) !important;
}

/* 파일 업로드 영역 */
.gr-file-upload, .upload-container {
    border: 2px dashed var(--velog-green) !important;
    border-radius: var(--velog-radius) !important;
    background: var(--velog-green-light) !important;
    transition: background 0.2s !important;
}

.gr-file-upload:hover {
    background: #d4f5e7 !important;
}

/* 결과 출력 Markdown 카드 */
.result-box {
    background: var(--velog-white) !important;
    border: 1px solid var(--velog-border) !important;
    border-radius: var(--velog-radius) !important;
    padding: 20px !important;
    min-height: 300px !important;
}

/* 구분선 */
hr {
    border: none !important;
    border-top: 1px solid var(--velog-border) !important;
    margin: 16px 0 !important;
}

/* 안내 박스 */
.info-box {
    background: var(--velog-green-light) !important;
    border-left: 3px solid var(--velog-green) !important;
    border-radius: 0 var(--velog-radius) var(--velog-radius) 0 !important;
    padding: 10px 14px !important;
    font-size: 0.88rem !important;
    color: #1a7a4a !important;
    margin-bottom: 16px !important;
}

/* Textbox 비활성 */
.gr-textbox[readonly], .gr-textbox:disabled {
    background: var(--velog-bg) !important;
    color: var(--velog-text) !important;
}

/* 상태 표시 점 애니메이션 */
@keyframes status-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.35; transform: scale(0.75); }
}
"""

_HEADER_HTML = """
<div class="bunpil-header">
    <h1>분필</h1>
    <p>고등학교 사회 교사용 AI 어시스턴트 &nbsp;·&nbsp; AI-powered exam & school record tool</p>
</div>
"""

_EXAM_INFO_HTML = """
<div class="info-box">
    지문 PDF를 업로드하고 유형·난이도를 선택한 뒤 <strong>문항 생성</strong>을 누르세요.
    생성에 수 분 소요됩니다.
</div>
"""

_RECORD_INFO_HTML = """
<div class="info-box">
    교사 관찰 메모를 입력하면 학교생활기록부 문체로 다듬어 드립니다.
    입력 내용은 저장되지 않으며, 개인정보는 자동으로 가려집니다.
</div>
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="분필", css=_CSS) as demo:

        gr.HTML(_HEADER_HTML)

        with gr.Tabs(elem_classes=["tab-nav"]):

            # ── 탭 1: 출제 모드 ────────────────────────────────────────
            with gr.TabItem("문항 출제"):
                gr.HTML(_EXAM_INFO_HTML)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=300):
                        with gr.Group(elem_classes=["card-panel"]):
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
                        exam_btn = gr.Button(
                            "문항 생성",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=2, min_width=400):
                        exam_status = gr.HTML(value=_status("대기 중", "idle"))
                        exam_output = gr.Markdown(
                            value="",
                            elem_classes=["result-box"],
                        )

                exam_btn.click(
                    fn=run_exam,
                    inputs=[pdf_input, unit_input, type_radio, diff_radio, standards_input],
                    outputs=[exam_status, exam_output],
                )

            # ── 탭 2: 생기부 다듬기 모드 ─────────────────────────────────
            with gr.TabItem("생기부 다듬기"):
                gr.HTML(_RECORD_INFO_HTML)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=300):
                        with gr.Group(elem_classes=["card-panel"]):
                            memo_input = gr.Textbox(
                                label="관찰 메모",
                                placeholder="예: 수업 시간에 발표를 잘 함. 친구들이 이해 못할 때 도와줌.",
                                lines=6,
                            )
                        record_btn = gr.Button(
                            "생기부 다듬기",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=2, min_width=400):
                        record_status = gr.HTML(value=_status("대기 중", "idle"))
                        with gr.Group(elem_classes=["card-panel"]):
                            masked_output = gr.Textbox(
                                label="개인정보 처리 결과",
                                lines=3,
                                interactive=False,
                            )
                            polished_output = gr.Textbox(
                                label="생기부 작성 결과",
                                lines=5,
                                interactive=False,
                            )
                            violation_output = gr.Markdown(label="규정 확인")

                record_btn.click(
                    fn=run_record,
                    inputs=[memo_input],
                    outputs=[record_status, masked_output, polished_output, violation_output],
                )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
