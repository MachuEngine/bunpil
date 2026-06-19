"use client";

import { useState } from "react";
import ExamTab from "@/app/components/ExamTab";
import RecordTab from "@/app/components/RecordTab";

type Tab = "exam" | "record";

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>("exam");

  return (
    <div className="min-h-screen flex flex-col bg-[#FAF9F6]">
      {/* TopBar */}
      <header className="sticky top-0 z-10 border-b border-[#E5E3DE] bg-[#FAF9F6]/95 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-[18px] font-semibold text-[#1A1A1A] tracking-tight">
              분필
            </span>
            <span className="text-[13px] text-[#6B6B6B]">
              사회 교사 AI 어시스턴트
            </span>
          </div>

          <nav className="flex gap-1">
            <button
              onClick={() => setActiveTab("exam")}
              className={`px-4 py-1.5 rounded-lg text-[13px] font-medium transition-colors ${
                activeTab === "exam"
                  ? "bg-[#D97706] text-white"
                  : "text-[#6B6B6B] hover:bg-[#F0EEE9] hover:text-[#1A1A1A]"
              }`}
            >
              출제 모드
            </button>
            <button
              onClick={() => setActiveTab("record")}
              className={`px-4 py-1.5 rounded-lg text-[13px] font-medium transition-colors ${
                activeTab === "record"
                  ? "bg-[#D97706] text-white"
                  : "text-[#6B6B6B] hover:bg-[#F0EEE9] hover:text-[#1A1A1A]"
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
