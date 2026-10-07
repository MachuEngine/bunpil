// 생성 문항 타입 — 화면(ExamTab), 다운로드용 시트(ExamSheet), 내보내기(export.ts)가 함께 쓴다.
export interface ExamItem {
  item_id: string;
  question: string;
  stimulus?: string; // <보기>·자료 등 제시문. 없으면 "" (2026-09 형식 인식)
  options: string[];
  answer: string;
  item_type: "객관식" | "서술형";
  difficulty: "상" | "중" | "하";
  // 2026-10-07: `standard` 제거 — 생성 경로에서 성취기준 검색(search_standards)을 뺀
  // 결정과 함께. 이 필드가 채우던 화면 표시(성취기준: ...)를 사용자가 쓰지 않았다.
  // 2026-08-06: `judge_score`·`status` 제거 — AI가 자기 문항에 스스로 매기던 점수라
  // 검증된 적이 없었고, 교사 화면에 "품질"로 보이는 것이 오해를 유발했다(EVAL.md 17절).
}

// 학생용: 문항만 / 교사용: 정답·해설 포함
export type SheetMode = "student" | "teacher";
