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

from app.common.rag import BGEEmbedder, RAGStore, chunk_document, parse_pdf
from app.modules.exam import ExamSpec, get_exam_graph
from app.modules.record import get_record_chain

_store = RAGStore()
_embedder = BGEEmbedder()

# ── 출제 모드 핸들러 ──────────────────────────────────────────────────

def run_exam(pdf_file, unit, mc_count, essay_count, high_count, mid_count, low_count, standards_text):
    if pdf_file is None:
        return "⚠️ PDF 파일을 업로드해주세요."
    if not unit.strip():
        return "⚠️ 단원명을 입력해주세요."

    type_dist = {}
    if int(mc_count) > 0:
        type_dist["객관식"] = int(mc_count)
    if int(essay_count) > 0:
        type_dist["서술형"] = int(essay_count)
    if not type_dist:
        return "⚠️ 문항 유형(객관식/서술형)을 하나 이상 설정해주세요."

    diff_dist = {}
    if int(high_count) > 0:
        diff_dist["상"] = int(high_count)
    if int(mid_count) > 0:
        diff_dist["중"] = int(mid_count)
    if int(low_count) > 0:
        diff_dist["하"] = int(low_count)
    if not diff_dist:
        return "⚠️ 난이도를 하나 이상 설정해주세요."

    standards = [s.strip() for s in standards_text.split(",") if s.strip()]
    if not standards:
        standards = [f"{unit.strip()} 핵심 개념 이해"]

    num_items = sum(type_dist.values())
    spec: ExamSpec = {
        "unit": unit.strip(),
        "num_items": num_items,
        "type_dist": type_dist,
        "difficulty_dist": diff_dist,
        "target": "내신",
        "standards": standards,
    }

    collection_name = None

    try:
        pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name
        doc = parse_pdf(pdf_path)
        chunks = chunk_document(doc)
        if not chunks:
            return "⚠️ PDF에서 텍스트를 추출할 수 없습니다."

        vecs = _embedder.embed([c["text"] for c in chunks])
        collection_name = _store.create_temp_collection()
        _store.add_chunks(collection_name, chunks, vecs)

        graph = get_exam_graph()
        state = graph.invoke({
            "spec": spec,
            "source_collection": collection_name,
            "budget": 2,
            "draft_items": [],
            "agent_messages": [],
            "coverage_map": {},
            "validation_passed": False,
            "error": "",
        })

        items = state.get("draft_items", [])
        approved = [it for it in items if it.get("status") == "approved"]
        passed = state.get("validation_passed", False)

        lines = [
            "## 📝 출제 결과",
            f"**검증 통과**: {'✓' if passed else '✗'}  |  **생성**: {len(items)}문항  |  **승인**: {len(approved)}문항",
            f"**단원**: {unit}  |  **유형**: {type_dist}  |  **난이도**: {diff_dist}",
            "",
        ]

        for i, item in enumerate(items, 1):
            ok = item.get("status") == "approved"
            icon = "✅" if ok else "❌"
            lines.append(
                f"### {icon} 문항 {i} — {item.get('item_type','?')} / "
                f"난이도:{item.get('difficulty','?')} / Judge:{item.get('judge_score',0):.0f}/5"
            )
            lines.append(f"**질문**: {item.get('question','(없음)')}")
            opts = item.get("options", [])
            if opts:
                lines.append("  " + "  ".join(str(o) for o in opts))
                lines.append(f"**정답**: {item.get('answer','?')}")
            lines.append("")

        if not items:
            error_msg = state.get("error", "")
            backend = os.getenv("LLM_BACKEND", "local")
            hint = f"LLM_BACKEND={backend}" + (f" | {error_msg}" if error_msg else "")
            lines.append(f"⚠️ 문항이 생성되지 않았습니다. ({hint})")

        return "\n".join(lines)

    except Exception as e:
        return f"⚠️ 오류 발생:\n```\n{traceback.format_exc()}\n```"

    finally:
        if collection_name:
            try:
                _store.delete_collection(collection_name)
            except Exception:
                pass


# ── 생기부 모드 핸들러 ────────────────────────────────────────────────

def run_record(memo: str):
    if not memo.strip():
        return "", "", "⚠️ 관찰 메모를 입력해주세요.", ""

    try:
        chain = get_record_chain()
        out = chain.run(memo.strip())

        pii_info = f"감지된 PII: {', '.join(out['pii_found'])}" if out["pii_found"] else "PII 없음"
        masked_display = f"{out['masked_memo']}\n\n({pii_info})"

        if out["violations"]:
            viol_md = "⚠️ **규정 위반 감지:**\n" + "\n".join(f"- {v}" for v in out["violations"])
        else:
            viol_md = "✅ 규정 위반 없음"

        return masked_display, out["polished"], viol_md, out["warning"]

    except Exception as e:
        return "", "", f"⚠️ 오류: {e}", ""


# ── UI 구성 ───────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="분필", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🏫 분필 — 고등학교 사회 교사용 AI 어시스턴트\n"
            "> **출제 모드**: 지문 업로드 → 문항 세트 자동 출제  |  "
            "**생기부 모드**: 관찰 메모 → 생기부 문체 윤문"
        )

        with gr.Tabs():

            # ── 탭 1: 출제 모드 ────────────────────────────────────────
            with gr.TabItem("📝 출제 모드"):
                gr.Markdown(
                    "지문 PDF를 업로드하고 출제 조건을 설정한 뒤 **문항 생성** 버튼을 누르세요.\n\n"
                    "> ⚠️ 생성에 수 분 소요됩니다. (CPU 추론)"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        pdf_input = gr.File(
                            label="📄 지문 PDF 업로드",
                            file_types=[".pdf"],
                            type="filepath",
                        )
                        unit_input = gr.Textbox(
                            label="단원명",
                            placeholder="예: 민주주의와 헌법",
                        )
                        standards_input = gr.Textbox(
                            label="성취기준 (쉼표 구분, 비워두면 자동 생성)",
                            placeholder="예: 민주주의 원리 이해, 기본권 보장의 의미",
                        )
                        gr.Markdown("**문항 유형 수**")
                        with gr.Row():
                            mc_slider    = gr.Slider(0, 8, value=3, step=1, label="객관식")
                            essay_slider = gr.Slider(0, 4, value=1, step=1, label="서술형")
                        gr.Markdown("**난이도별 수**")
                        with gr.Row():
                            high_slider = gr.Slider(0, 4, value=1, step=1, label="상")
                            mid_slider  = gr.Slider(0, 4, value=2, step=1, label="중")
                            low_slider  = gr.Slider(0, 4, value=1, step=1, label="하")
                        exam_btn = gr.Button("🎯 문항 생성", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        exam_output = gr.Markdown(label="출제 결과", value="결과가 여기에 표시됩니다.")

                exam_btn.click(
                    fn=run_exam,
                    inputs=[pdf_input, unit_input, mc_slider, essay_slider,
                            high_slider, mid_slider, low_slider, standards_input],
                    outputs=exam_output,
                )

            # ── 탭 2: 생기부 윤문 모드 ─────────────────────────────────
            with gr.TabItem("📋 생기부 윤문"):
                gr.Markdown(
                    "교사 관찰 메모를 입력하면 학교생활기록부 문체로 윤문합니다.\n\n"
                    "> **보안**: 입력 내용은 저장되지 않습니다. 개인정보(전화번호·주민번호 등)는 자동 마스킹됩니다."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        memo_input = gr.Textbox(
                            label="교사 관찰 메모",
                            placeholder="예: 수학 시간에 발표를 잘 함. 친구들이 이해못할 때 도와줌.",
                            lines=5,
                        )
                        record_btn = gr.Button("✍️ 윤문 생성", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        masked_output   = gr.Textbox(label="🔒 마스킹 결과", lines=3, interactive=False)
                        polished_output = gr.Textbox(label="📄 윤문 결과", lines=4, interactive=False)
                        violation_output = gr.Markdown(label="📋 규정 검증")
                        warning_output   = gr.Markdown(label="⚠️ 교사 확인 사항")

                record_btn.click(
                    fn=run_record,
                    inputs=[memo_input],
                    outputs=[masked_output, polished_output, violation_output, warning_output],
                )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
