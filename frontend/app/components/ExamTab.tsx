"use client";

import { useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { ExamItem, SheetMode } from "@/lib/exam";
import { buildText, downloadPdf, downloadPng, downloadText } from "@/lib/export";
import ExamSheet from "./ExamSheet";

const MAX_PASSAGE_LENGTH = 8000;
// 백엔드 /exam/revise 한도와 같게 맞춘다(app/main.py _MAX_HISTORY_TURNS·_MAX_CHAT_LENGTH)
const MAX_HISTORY_TURNS = 6;
const MAX_CHAT_LENGTH = 1000;

type ChatMessage = { role: "user" | "assistant"; content: string; failed?: boolean };

// 해설 상태 — 교사용 다운로드·수정 반영 시에도 쓰므로 ExamTab이 item_id별로 들고 있다(2026-09)
type ExplanationState = { status: "loading" } | { status: "done"; text: string } | { status: "error" };

function ItemCard({
  item,
  explanation,
  onExplain,
}: {
  item: ExamItem;
  explanation?: ExplanationState;
  onExplain: (item: ExamItem) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [showAnswer, setShowAnswer] = useState(false);

  return (
    <div
      className="border border-[#DBDCD2] rounded-xl p-4 bg-white cursor-pointer hover:border-[#2F4A3D] transition-colors"
      onClick={() => setExpanded((v) => !v)}
    >
      <div className="flex items-center gap-2 mb-2 flex-wrap">
        <Badge variant={item.item_type === "객관식" ? "mc" : "sa"}>
          {item.item_type}
        </Badge>
        <Badge variant={item.difficulty === "상" ? "hard" : item.difficulty === "중" ? "med" : "easy"}>
          난이도 {item.difficulty}
        </Badge>
      </div>

      <p className="text-[14px] text-[#1C2620] line-clamp-2">
        {item.question || "—"}
      </p>

      {expanded && (
        <div className="mt-3 pt-3 border-t border-[#DBDCD2]">
          {item.stimulus && (
            <div className="mb-3 rounded-md border border-[#1C2620] px-3 py-2 text-[13px] text-[#1C2620] whitespace-pre-wrap">
              {item.stimulus}
            </div>
          )}
          {item.options.length > 0 && (
            <ol className="space-y-1 mb-2">
              {item.options.map((opt, i) => (
                <li
                  key={i}
                  className={`text-[13px] pl-2 ${showAnswer && item.answer && opt.startsWith(item.answer) ? "text-[#2F4A3D] font-medium" : "text-[#6E7469]"}`}
                >
                  {opt}
                </li>
              ))}
            </ol>
          )}
          {showAnswer && item.options.length === 0 && item.answer && (
            <p className="text-[13px] text-[#6E7469]">
              <span className="font-medium text-[#1C2620]">예시 답안: </span>
              {item.answer}
            </p>
          )}
          {item.standard && (
            <p className="text-[13px] text-[#6E7469] mt-1">
              성취기준: {item.standard}
            </p>
          )}

          {/* 해설 보기 — 누르기 전에는 정답도 숨긴다(카드를 펼친 것만으로 정답이 보이지 않게) */}
          <div className="mt-3" onClick={(e) => e.stopPropagation()}>
            {!showAnswer ? (
              <Button
                type="button"
                variant="secondary"
                size="sm"
                onClick={() => {
                  setShowAnswer(true);
                  if (!explanation || explanation.status === "error") onExplain(item);
                }}
              >
                해설 보기
              </Button>
            ) : (
              <div className="rounded-lg bg-[#F3F4EE] px-3 py-2 text-[13px] text-[#1C2620]">
                {item.options.length > 0 && (
                  <p className="font-medium mb-1">정답: {item.answer}</p>
                )}
                {explanation?.status === "loading" && (
                  <p className="text-[#6E7469]">해설을 작성하고 있습니다...</p>
                )}
                {explanation?.status === "done" && (
                  <p className="whitespace-pre-wrap">{explanation.text}</p>
                )}
                {explanation?.status === "error" && (
                  <p className="text-[#A63B2E]">
                    해설을 만들지 못했습니다.{" "}
                    <button type="button" className="underline" onClick={() => onExplain(item)}>
                      다시 시도
                    </button>
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function ExamTab() {
  const [passageText, setPassageText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [stepMsg, setStepMsg] = useState("");
  const [items, setItems] = useState<ExamItem[]>([]);
  const [error, setError] = useState("");
  const [truncated, setTruncated] = useState(false);
  const [piiFound, setPiiFound] = useState<string[]>([]);
  const [explanations, setExplanations] = useState<Record<string, ExplanationState>>({});

  // 챗봇 수정 — 2026-09. 대화는 이 state에만 있고 서버에 저장되지 않는다(하드룰 3).
  // 매 요청에 생성 당시 예시 문제·현재 문항·최근 대화를 함께 보낸다.
  const [generatedPassage, setGeneratedPassage] = useState("");
  const [chat, setChat] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isRevising, setIsRevising] = useState(false);

  const handleRevise = async () => {
    const instruction = chatInput.trim().slice(0, MAX_CHAT_LENGTH);
    if (!instruction || isRevising) return;
    const history = chat
      .filter((m) => !m.failed)
      .slice(-MAX_HISTORY_TURNS)
      .map(({ role, content }) => ({ role, content: content.slice(0, MAX_CHAT_LENGTH) }));
    setChat((prev) => [...prev, { role: "user", content: instruction }]);
    setChatInput("");
    setIsRevising(true);
    try {
      const fd = new FormData();
      fd.append("passage_text", generatedPassage);
      fd.append("items", JSON.stringify(items));
      fd.append("history", JSON.stringify(history));
      fd.append("instruction", instruction);
      const res = await fetch("/api/exam/revise", { method: "POST", body: fd });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const data: any = await res.json().catch(() => null);
      if (!res.ok || typeof data?.message !== "string") {
        const msg =
          res.status === 413
            ? "문항과 대화가 너무 길어 요청을 보낼 수 없습니다. 새로 생성해 주세요."
            : res.status === 429
              ? "다른 요청을 처리 중입니다. 잠시 후 다시 시도해 주세요."
              : "수정 요청을 처리하지 못했습니다.";
        setChat((prev) => [...prev, { role: "assistant", content: msg, failed: true }]);
        return;
      }
      // 바뀐 문항은 새 item_id로 교체한다 — 카드가 다시 그려지면서 이전 해설·정답 표시가 초기화된다
      const changes = new Map<number, ExamItem>(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (data.changes ?? []).map((c: any) => [c.number, c.item]),
      );
      setItems((prev) =>
        prev.map((it, i) => {
          const changed = changes.get(i + 1);
          return changed ? { ...changed, item_id: crypto.randomUUID().slice(0, 8) } : it;
        }),
      );
      setChat((prev) => [...prev, { role: "assistant", content: data.message }]);
    } catch {
      setChat((prev) => [...prev, { role: "assistant", content: "서버 연결 오류가 발생했습니다.", failed: true }]);
    } finally {
      setIsRevising(false);
    }
  };

  // 해설은 서버에 저장되지 않는다 — 문항을 다시 보내 생성하고 이 state에만 캐시한다(하드룰 3)
  const fetchExplanation = async (item: ExamItem): Promise<string | null> => {
    setExplanations((prev) => ({ ...prev, [item.item_id]: { status: "loading" } }));
    try {
      const fd = new FormData();
      fd.append("item", JSON.stringify(item));
      const res = await fetch("/api/exam/explain", { method: "POST", body: fd });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const data: any = await res.json().catch(() => null);
      if (!res.ok || typeof data?.explanation !== "string") throw new Error();
      setExplanations((prev) => ({ ...prev, [item.item_id]: { status: "done", text: data.explanation } }));
      return data.explanation;
    } catch {
      setExplanations((prev) => ({ ...prev, [item.item_id]: { status: "error" } }));
      return null;
    }
  };

  // 다운로드 — 2026-09. 교사용은 아직 안 받은 해설을 먼저 순차로 받는다(동시 요청 슬롯 2개)
  const sheetRef = useRef<HTMLDivElement>(null);
  const [sheetMode, setSheetMode] = useState<SheetMode>("student");
  const [exportMsg, setExportMsg] = useState("");

  const doneExplanations = (): Record<string, string> =>
    Object.fromEntries(
      Object.entries(explanations).flatMap(([id, e]) => (e.status === "done" ? [[id, e.text]] : [])),
    );

  const handleDownload = async (mode: SheetMode, format: "pdf" | "txt" | "png") => {
    const texts = doneExplanations();
    try {
      if (mode === "teacher") {
        const missing = items.filter((it) => !texts[it.item_id]);
        for (const [i, it] of missing.entries()) {
          setExportMsg(`해설 준비 중 (${i + 1}/${missing.length})...`);
          const text = await fetchExplanation(it);
          if (text) texts[it.item_id] = text;
        }
      }
      setExportMsg("파일을 만드는 중...");
      const date = new Date().toISOString().slice(0, 10).replaceAll("-", "");
      const filename = `분필_문항_${mode === "teacher" ? "교사용" : "학생용"}_${date}.${format}`;
      if (format === "txt") {
        downloadText(buildText(items, mode, texts), filename);
        return;
      }
      setSheetMode(mode);
      // 시트가 새 모드·해설로 다시 그려진 뒤 캡처한다
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      if (!sheetRef.current) return;
      if (format === "png") await downloadPng(sheetRef.current, filename);
      else await downloadPdf(sheetRef.current, filename);
    } catch {
      setError("파일을 만들지 못했습니다. 다시 시도해 주세요.");
    } finally {
      setExportMsg("");
    }
  };

  // 이미지(캡처) 입력 — 2026-08-19
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [extractError, setExtractError] = useState("");
  const [extractPiiFound, setExtractPiiFound] = useState<string[]>([]);

  const handleImageUpload = async (file: File) => {
    setExtractError("");
    setExtractPiiFound([]);
    setIsExtracting(true);
    try {
      const fd = new FormData();
      fd.append("image", file);
      const res = await fetch("/api/exam/extract", { method: "POST", body: fd });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const data: any = await res.json().catch(() => null);
      if (!res.ok || typeof data?.text !== "string") {
        setExtractError("이미지에서 문제를 읽지 못했습니다. 직접 입력해 주세요.");
        return;
      }
      setPassageText((prev) => (prev.trim() ? `${prev.trim()}\n\n${data.text}` : data.text));
      setExtractPiiFound(data.pii_found ?? []);
    } catch {
      setExtractError("이미지에서 문제를 읽지 못했습니다. 직접 입력해 주세요.");
    } finally {
      setIsExtracting(false);
    }
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const item = Array.from(e.clipboardData.items).find((it) => it.type.startsWith("image/"));
    if (!item) return; // 텍스트 붙여넣기는 기본 동작 그대로 둔다
    e.preventDefault();
    const file = item.getAsFile();
    if (file) handleImageUpload(file);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleImageUpload(file);
    e.target.value = ""; // 같은 파일을 다시 선택해도 onChange가 발생하도록 초기화
  };

  const handleGenerate = async () => {
    if (!passageText.trim()) { setError("예시 문제를 붙여넣어 주세요."); return; }
    setError("");
    setItems([]);
    setExplanations({});
    setChat([]);
    setTruncated(false);
    setPiiFound([]);
    setIsLoading(true);
    setStepMsg("준비 중...");

    try {
      const fd = new FormData();
      fd.append("passage_text", passageText.trim());
      setGeneratedPassage(passageText.trim());

      const res = await fetch("/api/exam/stream", { method: "POST", body: fd });
      if (!res.ok || !res.body) {
        let msg = "문항 생성에 실패했습니다.";
        try {
          const errBody = await res.clone().json();
          if (typeof errBody?.detail === "string") msg = errBody.detail;
          else if (typeof errBody?.error === "string") msg = errBody.error;
        } catch {
          // 본문이 JSON이 아니면(예: 스트림이 이미 일부 소비됨) 기본 메시지 유지
        }
        setError(msg);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let sepIndex;
        while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
          const frame = buffer.slice(0, sepIndex);
          buffer = buffer.slice(sepIndex + 2);

          const line = frame.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          let data: any;
          try {
            data = JSON.parse(line.slice("data: ".length));
          } catch {
            continue; // 프레임 하나가 깨져도 이미 표시된 진행 상황은 유지
          }

          if (data.status === "progress") {
            setStepMsg(data.msg ?? "");
          } else if (data.status === "truncated") {
            setTruncated(true);
          } else if (data.status === "pii_masked") {
            setPiiFound(data.pii_found ?? []);
          } else if (data.status === "done") {
            setItems(data.items ?? []);
            setTruncated(Boolean(data.truncated));
            setPiiFound(data.pii_found ?? []);
          } else if (data.status === "error") {
            setError(data.msg ?? "문항 생성에 실패했습니다.");
          }
        }
      }
    } catch {
      setError("서버 연결 오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  };

  const overLimit = passageText.length > MAX_PASSAGE_LENGTH;

  return (
    <div className="flex flex-col lg:flex-row gap-6 h-full">
      {/* 좌측: 컨트롤 */}
      <div className="lg:w-80 xl:w-96 shrink-0 space-y-5">
        {/* 예시 문제 붙여넣기 */}
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label className="block text-[13px] font-medium text-[#6E7469]">
              예시 문제
            </label>
            <span className={`text-[12px] ${overLimit ? "text-[#A63B2E]" : "text-[#6E7469]"}`}>
              {passageText.length.toLocaleString()} / {MAX_PASSAGE_LENGTH.toLocaleString()}자
            </span>
          </div>
          <textarea
            rows={12}
            placeholder="참고할 예시 문제를 그대로 붙여넣어 주세요. 문항 수, 유형(객관식/서술형), 난이도 구성을 그대로 파악해 새 문항 세트를 만듭니다. 이미지를 붙여넣거나(Ctrl+V) 첨부해도 됩니다."
            value={passageText}
            onChange={(e) => setPassageText(e.target.value)}
            onPaste={handlePaste}
            className="w-full rounded-lg border border-[#DBDCD2] bg-white px-3 py-2 text-[13px] text-[#1C2620] placeholder:text-[#6E7469] focus:outline-none focus:border-[#2F4A3D] transition-colors resize-none"
          />
          {overLimit && (
            <p className="text-[12px] text-[#A63B2E] mt-1">
              8,000자를 초과하면 앞부분만 반영됩니다.
            </p>
          )}
          <p className="text-[12px] text-[#6E7469] mt-1">
            실제 학생 정보는 입력하지 마세요. 감지된 개인정보는 모델 호출 전에 마스킹됩니다.
          </p>

          <div className="flex items-center gap-2 mt-2">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp"
              className="hidden"
              onChange={handleFileSelect}
            />
            <Button
              type="button"
              variant="secondary"
              size="sm"
              onClick={() => fileInputRef.current?.click()}
              disabled={isExtracting}
            >
              이미지 첨부
            </Button>
            {isExtracting && (
              <span className="text-[12px] text-[#6E7469] flex items-center gap-1.5">
                <span className="w-3 h-3 border-2 border-[#2F4A3D] border-t-transparent rounded-full animate-spin" />
                문제를 읽는 중...
              </span>
            )}
          </div>
          <p className="text-[12px] text-[#6E7469] mt-1">
            이미지는 텍스트 추출을 위해 외부 모델로 전달된 뒤 마스킹됩니다(텍스트와 달리 마스킹이 추출 이후에 적용됨). 학생 개인정보가 찍힌 캡처는 넣지 마세요.
          </p>

          {extractError && (
            <p className="text-[12px] text-[#A63B2E] mt-1">{extractError}</p>
          )}
          {extractPiiFound.length > 0 && (
            <p className="text-[12px] text-[#93601F] bg-[#F5EBD8] rounded-lg px-3 py-2 mt-2">
              추출된 텍스트에서 마스킹된 개인정보: {extractPiiFound.join(", ")}
            </p>
          )}
        </div>


        {error && (
          <p className="text-[13px] text-[#A63B2E] bg-[#F7E9E4] rounded-lg px-3 py-2">
            {error}
          </p>
        )}

        <Button
          onClick={handleGenerate}
          disabled={isLoading || isExtracting}
          className="w-full"
        >
          {isLoading ? "생성 중..." : "문항 생성"}
        </Button>
      </div>

      {/* 우측: 결과 */}
      <div className="flex-1 min-w-0">
        {piiFound.length > 0 && (
          <p className="text-[13px] text-[#93601F] bg-[#F5EBD8] rounded-lg px-3 py-2 mb-3">
            모델 호출 전에 마스킹된 개인정보: {piiFound.join(", ")}
          </p>
        )}
        {isLoading && (
          <div className="flex flex-col items-center justify-center h-48 gap-3">
            <div className="w-8 h-8 border-2 border-[#2F4A3D] border-t-transparent rounded-full animate-spin" />
            <p className="text-[14px] text-[#6E7469]">{stepMsg}</p>
          </div>
        )}

        {!isLoading && items.length === 0 && (
          <div className="flex items-center justify-center h-48">
            <p className="text-[14px] text-[#6E7469]">
              좌측에 예시 문제를 붙여넣고 문항 생성 버튼을 눌러주세요.
            </p>
          </div>
        )}

        {!isLoading && items.length > 0 && (
          <div>
            {truncated && (
              <p className="text-[13px] text-[#93601F] bg-[#F5EBD8] rounded-lg px-3 py-2 mb-3">
                입력이 길어 앞부분만 반영되었습니다.
              </p>
            )}
            <div className="flex items-center justify-between mb-4 gap-3 flex-wrap">
              <h2 className="text-[14px] font-semibold text-[#1C2620]">
                생성된 문항 ({items.length}개)
              </h2>
              <div className="flex flex-col gap-1 items-end">
                {(["student", "teacher"] as const).map((mode) => (
                  <div key={mode} className="flex items-center gap-1.5">
                    <span className="text-[12px] text-[#6E7469]">
                      {mode === "teacher" ? "교사용(정답·해설)" : "학생용"}
                    </span>
                    {(["pdf", "txt", "png"] as const).map((format) => (
                      <Button
                        key={format}
                        type="button"
                        variant="secondary"
                        size="sm"
                        disabled={Boolean(exportMsg)}
                        onClick={() => handleDownload(mode, format)}
                      >
                        {format.toUpperCase()}
                      </Button>
                    ))}
                  </div>
                ))}
                {exportMsg && <span className="text-[12px] text-[#6E7469]">{exportMsg}</span>}
              </div>
            </div>
            <div className="space-y-3">
              {items.map((item) => (
                <ItemCard
                  key={item.item_id}
                  item={item}
                  explanation={explanations[item.item_id]}
                  onExplain={fetchExplanation}
                />
              ))}
            </div>
            {/* 챗봇 수정 */}
            <div className="mt-6 border border-[#DBDCD2] rounded-xl bg-white p-4">
              <h3 className="text-[13px] font-semibold text-[#1C2620] mb-2">문항 수정 요청</h3>
              {chat.length > 0 && (
                <div className="space-y-2 mb-3 max-h-72 overflow-y-auto">
                  {chat.map((m, i) => (
                    <p
                      key={i}
                      className={`text-[13px] rounded-lg px-3 py-2 whitespace-pre-wrap ${
                        m.role === "user"
                          ? "bg-[#2F4A3D] text-white ml-8"
                          : m.failed
                            ? "bg-[#F7E9E4] text-[#A63B2E] mr-8"
                            : "bg-[#F3F4EE] text-[#1C2620] mr-8"
                      }`}
                    >
                      {m.content}
                    </p>
                  ))}
                  {isRevising && <p className="text-[12px] text-[#6E7469]">문항을 고치고 있습니다...</p>}
                </div>
              )}
              <div className="flex gap-2">
                <textarea
                  rows={2}
                  maxLength={MAX_CHAT_LENGTH}
                  placeholder="예: 2번 선지를 더 헷갈리게 바꿔줘 / 1번 난이도를 상으로 올려줘"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                      e.preventDefault();
                      handleRevise();
                    }
                  }}
                  className="flex-1 rounded-lg border border-[#DBDCD2] bg-white px-3 py-2 text-[13px] text-[#1C2620] placeholder:text-[#6E7469] focus:outline-none focus:border-[#2F4A3D] resize-none"
                />
                <Button type="button" onClick={handleRevise} disabled={isRevising || !chatInput.trim() || Boolean(exportMsg)}>
                  보내기
                </Button>
              </div>
              <p className="text-[12px] text-[#6E7469] mt-1">
                대화는 저장되지 않으며, 새로 생성하면 초기화됩니다. 개인정보는 모델 호출 전에 마스킹됩니다.
              </p>
            </div>

            {/* 다운로드 캡처용 시트 — 화면 밖에 그린다 */}
            <div aria-hidden style={{ position: "fixed", left: -10000, top: 0 }}>
              <ExamSheet ref={sheetRef} items={items} mode={sheetMode} explanations={doneExplanations()} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
