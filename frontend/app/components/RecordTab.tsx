"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";

interface RecordOutput {
  masked_memo: string;
  pii_found: string[];
  polished: string;
  violations: string[];
  /** 차단하지 않는 주의 사항 — 결과는 정상 반환되고 교사가 보고 판단한다(2026-08-03) */
  warnings?: string[];
  validation_status: "passed" | "violations_found" | "unavailable";
  /** 교사 책임 고지 문구 — 위 warnings와 다름 */
  warning: string;
}

function ResultSection({
  title,
  content,
  accent = false,
}: {
  title: string;
  content: string;
  accent?: boolean;
}) {
  return (
    <div className={`rounded-xl border p-4 ${accent ? "border-transparent bg-[#E7EDE8]" : "border-[#DBDCD2] bg-white"}`}>
      <p className="text-[13px] font-medium text-[#6E7469] mb-2">{title}</p>
      <p className="text-[14px] text-[#1C2620] whitespace-pre-wrap leading-relaxed">{content}</p>
    </div>
  );
}

export default function RecordTab() {
  const [memo, setMemo] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<RecordOutput | null>(null);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);

  const handlePolish = async () => {
    if (!memo.trim()) { setError("관찰 메모를 입력하세요."); return; }
    setError("");
    setResult(null);
    setIsLoading(true);
    try {
      const res = await fetch("/api/record", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ memo: memo.trim() }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error ?? "윤문 생성에 실패했습니다.");
      } else {
        setResult(data as RecordOutput);
      }
    } catch {
      setError("서버 연결 오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = async () => {
    if (!result?.polished) return;
    await navigator.clipboard.writeText(result.polished);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col lg:flex-row gap-6 h-full">
      {/* 좌측: 입력 */}
      <div className="lg:w-80 xl:w-96 shrink-0 space-y-4">
        <div>
          <label className="block text-[13px] font-medium text-[#6E7469] mb-1.5">
            교사 관찰 메모
          </label>
          <textarea
            rows={12}
            placeholder={
              "예: 수학 시간에 발표 잘 함. 친구들과 협력도 잘 하고\n모둠 활동에서 리더 역할 했음.\n좀 더 구체적인 근거 필요."
            }
            value={memo}
            onChange={(e) => setMemo(e.target.value)}
            className="w-full rounded-xl border border-[#DBDCD2] bg-white px-3 py-2.5 text-[14px] text-[#1C2620] placeholder:text-[#6E7469] focus:outline-none focus:border-[#2F4A3D] transition-colors resize-none"
          />
          <p className="text-[12px] text-[#6E7469] mt-1">
            실제 학생 정보는 입력하지 마세요. 감지된 개인정보는 모델 호출 전에 마스킹됩니다.
          </p>
        </div>

        {error && (
          <p className="text-[13px] text-[#A63B2E] bg-[#F7E9E4] rounded-lg px-3 py-2">
            {error}
          </p>
        )}

        <Button onClick={handlePolish} disabled={isLoading} className="w-full">
          {isLoading ? (
            <span className="flex items-center gap-2">
              <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              윤문 생성 중...
            </span>
          ) : (
            "윤문 생성"
          )}
        </Button>

        {/* 교사 고지 — 항상 표시 */}
        <div className="rounded-xl border border-[#DBDCD2] border-l-[3px] border-l-[#A63B2E] bg-white px-4 py-3">
          <p className="text-[13px] font-semibold text-[#A63B2E] mb-1">
            교사 확인 필수
          </p>
          <p className="text-[13px] text-[#6E7469] leading-relaxed">
            이 문장은 AI 보조 도구로 생성된 초안입니다. 최종 기재 여부와 내용의
            정확성은 담당 교사가 반드시 확인·책임져야 합니다.
          </p>
        </div>
      </div>

      {/* 우측: 결과 */}
      <div className="flex-1 min-w-0 space-y-4">
        {!isLoading && !result && (
          <div className="flex items-center justify-center h-48">
            <p className="text-[14px] text-[#6E7469]">
              좌측에서 메모를 입력 후 윤문 생성 버튼을 눌러주세요.
            </p>
          </div>
        )}

        {isLoading && (
          <div className="flex flex-col items-center justify-center h-48 gap-3">
            <div className="w-8 h-8 border-2 border-[#2F4A3D] border-t-transparent rounded-full animate-spin" />
            <p className="text-[14px] text-[#6E7469]">윤문 생성 중...</p>
          </div>
        )}

        {result && (
          <>
            {result.pii_found.length > 0 && (
              <div className="rounded-xl bg-[#F5EBD8] px-4 py-3">
                <p className="text-[13px] font-medium text-[#93601F]">
                  마스킹된 개인정보: {result.pii_found.join(", ")}
                </p>
              </div>
            )}

            <ResultSection title="마스킹 결과" content={result.masked_memo} />

            <div className="rounded-xl border border-[#DBDCD2] bg-white p-4">
              <div className="flex items-center justify-between mb-2">
                <p className="text-[13px] font-medium text-[#6E7469]">윤문 결과</p>
                {result.polished && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleCopy}
                  >
                    {copied ? "복사됨 ✓" : "복사"}
                  </Button>
                )}
              </div>
              <p className="text-[14px] text-[#1C2620] whitespace-pre-wrap leading-relaxed">
                {result.polished || "안전 검증을 통과한 윤문 결과가 없습니다."}
              </p>
            </div>

            {result.validation_status === "unavailable" ? (
              <div className="rounded-xl bg-[#F5EBD8] p-4 space-y-1">
                <p className="text-[13px] font-medium text-[#93601F] mb-1">
                  안전 검증을 완료할 수 없어 결과를 숨겼습니다
                </p>
                {result.violations.map((v, i) => (
                  <p key={i} className="text-[13px] text-[#93601F]">
                    · {v}
                  </p>
                ))}
              </div>
            ) : result.violations.length > 0 ? (
              <div className="rounded-xl bg-[#F7E9E4] p-4 space-y-1">
                <p className="text-[13px] font-medium text-[#A63B2E] mb-1">
                  안전 규칙 위반 — 결과를 숨겼습니다
                </p>
                {result.violations.map((v, i) => (
                  <p key={i} className="text-[13px] text-[#A63B2E]">
                    · {v}
                  </p>
                ))}
              </div>
            ) : result.warnings && result.warnings.length > 0 ? (
              <div className="rounded-xl bg-[#F5EBD8] p-4 space-y-1">
                <p className="text-[13px] font-medium text-[#93601F] mb-1">
                  확인이 필요한 표현이 있습니다 — 기재 여부는 선생님이 판단하세요
                </p>
                {result.warnings.map((w, i) => (
                  <p key={i} className="text-[13px] text-[#93601F]">
                    · {w}
                  </p>
                ))}
                <p className="text-[12px] text-[#93601F] opacity-80 pt-1">
                  교과 주제로서 언급한 경우(예: 사회 수업에서 다룬 정치·종교 개념)라면
                  그대로 두셔도 됩니다.
                </p>
              </div>
            ) : (
              <div className="rounded-xl bg-[#E5EEE4] px-4 py-3">
                <p className="text-[13px] font-medium text-[#3D6B4A]">
                  ✓ 규정 검증 통과
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
