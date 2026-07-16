"use client";

import { useState } from "react";
import ExamTab from "@/app/components/ExamTab";
import RecordTab from "@/app/components/RecordTab";

type Tab = "exam" | "record";

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>("exam");

  return (
    <div className="min-h-screen flex flex-col bg-[#F5F6F1]">
      {/* TopBar */}
      <header className="sticky top-0 z-10 border-b border-[#DBDCD2] bg-[#F5F6F1]/95 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-[18px] font-semibold text-[#1C2620] tracking-tight">
              분필
            </span>
            <span className="text-[13px] text-[#6E7469]">
              사회 교사 AI 어시스턴트
            </span>
          </div>

          <nav className="flex gap-1 bg-[#E7EDE8] p-0.5 rounded-lg">
            <button
              onClick={() => setActiveTab("exam")}
              className={`px-4 py-1.5 rounded-md text-[13px] font-medium transition-colors ${
                activeTab === "exam"
                  ? "bg-[#2F4A3D] text-[#EFEAD9]"
                  : "text-[#6E7469] hover:text-[#1C2620]"
              }`}
            >
              출제 모드
            </button>
            <button
              onClick={() => setActiveTab("record")}
              className={`px-4 py-1.5 rounded-md text-[13px] font-medium transition-colors ${
                activeTab === "record"
                  ? "bg-[#2F4A3D] text-[#EFEAD9]"
                  : "text-[#6E7469] hover:text-[#1C2620]"
              }`}
            >
              생기부 윤문
            </button>
          </nav>
        </div>
      </header>

      {/* Content */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-4 sm:px-6 py-8">
        {activeTab === "exam" ? <ExamTab /> : <RecordTab />}
      </main>
    </div>
  );
}
