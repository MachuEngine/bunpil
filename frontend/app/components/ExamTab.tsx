"use client";

import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";

const MAX_PASSAGE_LENGTH = 8000;

interface ExamItem {
  item_id: string;
  question: string;
  options: string[];
  answer: string;
  item_type: "객관식" | "서술형";
  difficulty: "상" | "중" | "하";
  standard: string;
  judge_score: number;
  status: "approved" | "rejected";
}

function ItemCard({ item }: { item: ExamItem }) {
  const [expanded, setExpanded] = useState(false);
  const scorePercent = (item.judge_score / 5) * 100;

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
        <Badge variant={item.status === "approved" ? "approved" : "rejected"}>
          {item.status === "approved" ? "✓ 승인" : "✗ 반려"}
        </Badge>
      </div>

      <p className="text-[14px] text-[#1C2620] line-clamp-2 mb-3">
        {item.question || "—"}
      </p>

      <div className="flex items-center gap-2">
        <span className="text-[13px] text-[#6E7469] w-16 shrink-0">
          품질 {item.judge_score.toFixed(1)}/5
        </span>
        <Progress value={scorePercent} className="flex-1" />
      </div>

      {expanded && (
        <div className="mt-3 pt-3 border-t border-[#DBDCD2]">
          {item.options.length > 0 && (
            <ol className="space-y-1 mb-2">
              {item.options.map((opt, i) => (
                <li
                  key={i}
                  className={`text-[13px] pl-2 ${opt.startsWith(item.answer) ? "text-[#2F4A3D] font-medium" : "text-[#6E7469]"}`}
                >
                  {opt}
                </li>
              ))}
            </ol>
          )}
          {item.options.length === 0 && item.answer && (
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

  const handleGenerate = async () => {
    if (!passageText.trim()) { setError("예시 문제를 붙여넣어 주세요."); return; }
    setError("");
    setItems([]);
    setTruncated(false);
    setPiiFound([]);
    setIsLoading(true);
    setStepMsg("준비 중...");

    try {
      const fd = new FormData();
      fd.append("passage_text", passageText.trim());

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

  const approved = items.filter((i) => i.status === "approved").length;
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
            placeholder="참고할 예시 문제를 그대로 붙여넣어 주세요. 문항 수, 유형(객관식/서술형), 난이도 구성을 그대로 파악해 새 문항 세트를 만듭니다."
            value={passageText}
            onChange={(e) => setPassageText(e.target.value)}
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
        </div>


        {error && (
          <p className="text-[13px] text-[#A63B2E] bg-[#F7E9E4] rounded-lg px-3 py-2">
            {error}
          </p>
        )}

        <Button
          onClick={handleGenerate}
          disabled={isLoading}
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
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-[14px] font-semibold text-[#1C2620]">
                생성된 문항 ({items.length}개)
              </h2>
              <span className="text-[13px] text-[#6E7469]">
                승인 {approved} / 반려 {items.length - approved}
              </span>
            </div>
            <div className="space-y-3">
              {items.map((item) => (
                <ItemCard key={item.item_id} item={item} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
