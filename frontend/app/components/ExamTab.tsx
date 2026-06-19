"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";

interface ExamItem {
  item_id: string;
  question: string;
  options: string[];
  answer: string;
  item_type: "객관식" | "서술형";
  difficulty: "상" | "중" | "하";
  standard: string;
  judge_score: number;
  is_duplicate: boolean;
  status: "approved" | "rejected";
}

const LOADING_STEPS = [
  "PDF 파싱 중...",
  "임베딩 생성 중...",
  "문항 생성 중...",
  "품질 검사 중...",
  "중복 확인 중...",
];

function ItemCard({ item }: { item: ExamItem }) {
  const [expanded, setExpanded] = useState(false);
  const scorePercent = (item.judge_score / 5) * 100;

  return (
    <div
      className="border border-[#E5E3DE] rounded-xl p-4 bg-white cursor-pointer hover:border-[#D97706] transition-colors"
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
        {item.is_duplicate && (
          <Badge className="bg-yellow-100 text-yellow-700">중복 의심</Badge>
        )}
      </div>

      <p className="text-[14px] text-[#1A1A1A] line-clamp-2 mb-3">
        {item.question || "—"}
      </p>

      <div className="flex items-center gap-2">
        <span className="text-[13px] text-[#6B6B6B] w-16 shrink-0">
          품질 {item.judge_score.toFixed(1)}/5
        </span>
        <Progress value={scorePercent} className="flex-1" />
      </div>

      {expanded && (
        <div className="mt-3 pt-3 border-t border-[#E5E3DE]">
          {item.options.length > 0 && (
            <ol className="space-y-1 mb-2">
              {item.options.map((opt, i) => (
                <li
                  key={i}
                  className={`text-[13px] pl-2 ${opt.startsWith(item.answer) ? "text-[#D97706] font-medium" : "text-[#6B6B6B]"}`}
                >
                  {opt}
                </li>
              ))}
            </ol>
          )}
          {item.options.length === 0 && item.answer && (
            <p className="text-[13px] text-[#6B6B6B]">
              <span className="font-medium text-[#1A1A1A]">예시 답안: </span>
              {item.answer}
            </p>
          )}
          {item.standard && (
            <p className="text-[13px] text-[#6B6B6B] mt-1">
              성취기준: {item.standard}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function SliderRow({
  label,
  value,
  onChange,
  min = 0,
  max = 10,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-[13px] text-[#6B6B6B] w-28 shrink-0">{label}</span>
      <Slider
        min={min}
        max={max}
        step={1}
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        className="flex-1"
      />
      <span className="text-[13px] font-medium text-[#1A1A1A] w-5 text-right">{value}</span>
    </div>
  );
}

export default function ExamTab() {
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [unit, setUnit] = useState("");
  const [numMc, setNumMc] = useState(5);
  const [numSa, setNumSa] = useState(2);
  const [numHard, setNumHard] = useState(2);
  const [numMed, setNumMed] = useState(3);
  const [numEasy, setNumEasy] = useState(2);
  const [standards, setStandards] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [stepMsg, setStepMsg] = useState("");
  const [items, setItems] = useState<ExamItem[]>([]);
  const [error, setError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const stepRef = useRef(0);

  useEffect(() => {
    if (!isLoading) return;
    stepRef.current = 0;
    setStepMsg(LOADING_STEPS[0]);
    const id = setInterval(() => {
      stepRef.current = (stepRef.current + 1) % LOADING_STEPS.length;
      setStepMsg(LOADING_STEPS[stepRef.current]);
    }, 2500);
    return () => clearInterval(id);
  }, [isLoading]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file?.type === "application/pdf") setPdfFile(file);
  }, []);

  const handleGenerate = async () => {
    if (!pdfFile) { setError("PDF 파일을 업로드하세요."); return; }
    if (!unit.trim()) { setError("단원명을 입력하세요."); return; }
    setError("");
    setItems([]);
    setIsLoading(true);

    try {
      const fd = new FormData();
      fd.append("pdf", pdfFile);
      fd.append("unit", unit.trim());
      fd.append("num_mc", String(numMc));
      fd.append("num_sa", String(numSa));
      fd.append("num_hard", String(numHard));
      fd.append("num_med", String(numMed));
      fd.append("num_easy", String(numEasy));
      fd.append("standards", standards);

      const res = await fetch("/api/exam", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error ?? "문항 생성에 실패했습니다.");
      } else {
        setItems(data.items ?? []);
      }
    } catch {
      setError("서버 연결 오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  };

  const approved = items.filter((i) => i.status === "approved").length;

  return (
    <div className="flex flex-col lg:flex-row gap-6 h-full">
      {/* 좌측: 컨트롤 */}
      <div className="lg:w-80 xl:w-96 shrink-0 space-y-5">
        {/* PDF 업로드 */}
        <div>
          <label className="block text-[13px] font-medium text-[#6B6B6B] mb-1.5">
            수업 지문 PDF
          </label>
          <div
            className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${
              isDragging
                ? "border-[#D97706] bg-[#FEF3C7]"
                : "border-[#E5E3DE] hover:border-[#D97706] hover:bg-[#F0EEE9]"
            }`}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) setPdfFile(f);
              }}
            />
            {pdfFile ? (
              <div>
                <p className="text-[14px] font-medium text-[#D97706]">
                  {pdfFile.name}
                </p>
                <p className="text-[13px] text-[#6B6B6B] mt-0.5">
                  {(pdfFile.size / 1024).toFixed(0)} KB
                </p>
              </div>
            ) : (
              <div>
                <p className="text-[14px] text-[#6B6B6B]">
                  PDF를 드래그하거나 클릭해서 업로드
                </p>
              </div>
            )}
          </div>
        </div>

        {/* 단원명 */}
        <div>
          <label className="block text-[13px] font-medium text-[#6B6B6B] mb-1.5">
            단원명
          </label>
          <input
            type="text"
            placeholder="예: 민주주의와 헌법"
            value={unit}
            onChange={(e) => setUnit(e.target.value)}
            className="w-full rounded-lg border border-[#E5E3DE] bg-white px-3 py-2 text-[14px] text-[#1A1A1A] placeholder:text-[#6B6B6B] focus:outline-none focus:border-[#D97706] transition-colors"
          />
        </div>

        {/* 문항 유형 */}
        <div className="space-y-3">
          <p className="text-[13px] font-medium text-[#6B6B6B]">문항 유형</p>
          <SliderRow label="객관식" value={numMc} onChange={setNumMc} max={10} />
          <SliderRow label="서술형" value={numSa} onChange={setNumSa} max={5} />
        </div>

        {/* 난이도 배분 */}
        <div className="space-y-3">
          <p className="text-[13px] font-medium text-[#6B6B6B]">난이도 배분</p>
          <SliderRow label="상" value={numHard} onChange={setNumHard} max={10} />
          <SliderRow label="중" value={numMed} onChange={setNumMed} max={10} />
          <SliderRow label="하" value={numEasy} onChange={setNumEasy} max={10} />
        </div>

        {/* 성취기준 */}
        <div>
          <label className="block text-[13px] font-medium text-[#6B6B6B] mb-1.5">
            성취기준 (선택, 줄바꿈으로 구분)
          </label>
          <textarea
            rows={3}
            placeholder={"[사문9101-1] 민주주의의 의미와 원리\n[사문9101-2] 헌법의 기본 원리"}
            value={standards}
            onChange={(e) => setStandards(e.target.value)}
            className="w-full rounded-lg border border-[#E5E3DE] bg-white px-3 py-2 text-[13px] text-[#1A1A1A] placeholder:text-[#6B6B6B] focus:outline-none focus:border-[#D97706] transition-colors resize-none"
          />
        </div>

        {error && (
          <p className="text-[13px] text-red-600 bg-red-50 rounded-lg px-3 py-2">
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
        {isLoading && (
          <div className="flex flex-col items-center justify-center h-48 gap-3">
            <div className="w-8 h-8 border-2 border-[#D97706] border-t-transparent rounded-full animate-spin" />
            <p className="text-[14px] text-[#6B6B6B]">{stepMsg}</p>
          </div>
        )}

        {!isLoading && items.length === 0 && (
          <div className="flex items-center justify-center h-48">
            <p className="text-[14px] text-[#6B6B6B]">
              좌측에서 설정 후 문항 생성 버튼을 눌러주세요.
            </p>
          </div>
        )}

        {!isLoading && items.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-[14px] font-semibold text-[#1A1A1A]">
                생성된 문항 ({items.length}개)
              </h2>
              <span className="text-[13px] text-[#6B6B6B]">
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
