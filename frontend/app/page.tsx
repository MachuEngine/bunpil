"use client";

import ExamTab from "@/app/components/ExamTab";

export default function Home() {
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
              사회 교사 문항 출제 어시스턴트
            </span>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-4 sm:px-6 py-8">
        <ExamTab />
      </main>
    </div>
  );
}
