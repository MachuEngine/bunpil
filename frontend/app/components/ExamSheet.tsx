import { forwardRef } from "react";
import type { ExamItem, SheetMode } from "@/lib/exam";

// 다운로드(PDF·PNG)용 문항지. 화면 밖에 그려 두고 html-to-image로 캡처한다.
// PDF는 [data-sheet-block] 단위로 이미지를 만들어 페이지를 넘기므로 문항이 페이지 사이에서 잘리지 않는다.
// 색은 hex 인라인 스타일로 고정 — 캡처 결과가 화면 테마·Tailwind 색 함수에 좌우되지 않게 한다.
const ExamSheet = forwardRef<
  HTMLDivElement,
  { items: ExamItem[]; mode: SheetMode; explanations: Record<string, string> }
>(function ExamSheet({ items, mode, explanations }, ref) {
  return (
    <div
      ref={ref}
      style={{ width: 794, padding: 48, background: "#FFFFFF", color: "#111111", fontSize: 15, lineHeight: 1.6 }}
    >
      <div data-sheet-block style={{ paddingBottom: 16, marginBottom: 16, borderBottom: "2px solid #111111" }}>
        <div style={{ fontSize: 20, fontWeight: 700 }}>사회 문항지 ({mode === "teacher" ? "교사용" : "학생용"})</div>
      </div>
      {items.map((item, i) => (
        <div key={item.item_id} data-sheet-block style={{ paddingBottom: 24 }}>
          <div style={{ fontWeight: 600 }}>
            {i + 1}. {item.question}
          </div>
          {item.stimulus && (
            <div style={{ border: "1px solid #111111", padding: "8px 12px", margin: "8px 0", whiteSpace: "pre-wrap" }}>
              {item.stimulus}
            </div>
          )}
          {item.options.map((opt, j) => (
            <div key={j} style={{ paddingLeft: 12 }}>
              {opt}
            </div>
          ))}
          {mode === "teacher" && (
            <div style={{ marginTop: 8, padding: "8px 12px", background: "#F3F4EE", fontSize: 14 }}>
              {item.answer && (
                <div style={{ fontWeight: 600 }}>
                  {item.options.length > 0 ? "정답" : "예시 답안"}: {item.answer}
                </div>
              )}
              {explanations[item.item_id] && (
                <div style={{ whiteSpace: "pre-wrap" }}>{explanations[item.item_id]}</div>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
});

export default ExamSheet;
