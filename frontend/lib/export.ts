// 문항 다운로드(TXT·PNG·PDF) — 전부 브라우저에서 만든다. 서버는 파일 생성에 관여하지 않아
// 생성 문항이 서버를 다시 거치지 않는다(하드룰 3). 한글은 브라우저가 렌더링한 화면을 캡처하므로
// PDF에 폰트를 임베딩할 필요가 없다. 대신 PDF는 이미지 기반이라 텍스트 선택이 되지 않는다.
import type { ExamItem, SheetMode } from "@/lib/exam";

export function buildText(items: ExamItem[], mode: SheetMode, explanations: Record<string, string>): string {
  const lines = [`사회 문항지 (${mode === "teacher" ? "교사용" : "학생용"})`, ""];
  items.forEach((item, i) => {
    lines.push(`${i + 1}. ${item.question}`);
    if (item.stimulus) lines.push(item.stimulus);
    lines.push(...item.options);
    if (mode === "teacher") {
      if (item.answer) lines.push(`${item.options.length > 0 ? "정답" : "예시 답안"}: ${item.answer}`);
      if (explanations[item.item_id]) lines.push(`해설: ${explanations[item.item_id]}`);
    }
    lines.push("");
  });
  return lines.join("\n");
}

function downloadUrl(url: string, filename: string) {
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
}

export function downloadText(text: string, filename: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
  downloadUrl(url, filename);
  URL.revokeObjectURL(url);
}

const CAPTURE_OPTIONS = { pixelRatio: 2, backgroundColor: "#FFFFFF" };

export async function downloadPng(node: HTMLElement, filename: string) {
  const { toPng } = await import("html-to-image");
  downloadUrl(await toPng(node, CAPTURE_OPTIONS), filename);
}

export async function downloadPdf(node: HTMLElement, filename: string) {
  const [{ toJpeg }, { jsPDF }] = await Promise.all([import("html-to-image"), import("jspdf")]);
  const pdf = new jsPDF({ unit: "mm", format: "a4" });
  const pageWidth = pdf.internal.pageSize.getWidth();
  const pageHeight = pdf.internal.pageSize.getHeight();
  const margin = 12;
  const contentWidth = pageWidth - margin * 2;
  let y = margin;

  // 문항 블록 단위로 붙인다 — 남은 공간보다 크면 새 페이지에서 시작해 문항이 잘리지 않게 한다
  for (const block of Array.from(node.querySelectorAll<HTMLElement>("[data-sheet-block]"))) {
    // PNG를 그대로 넣으면 문항 2개에 3~4MB가 된다 — PDF 안에서는 JPEG로 압축한다
    const dataUrl = await toJpeg(block, { ...CAPTURE_OPTIONS, quality: 0.9 });
    const height = (block.offsetHeight / block.offsetWidth) * contentWidth;
    if (y + height > pageHeight - margin && y > margin) {
      pdf.addPage();
      y = margin;
    }
    pdf.addImage(dataUrl, "JPEG", margin, y, contentWidth, height);
    y += height;
  }
  pdf.save(filename);
}
