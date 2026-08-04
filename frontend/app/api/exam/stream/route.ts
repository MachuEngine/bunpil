import { NextRequest } from "next/server";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";
const API_KEY = process.env.BUNPIL_API_KEY ?? "";

export async function POST(request: NextRequest) {
  const formData = await request.formData();
  if (!API_KEY) {
    // 아직 SSE 스트림을 열지 않은 단계의 실패라 순수 JSON으로 응답한다 —
    // 예전엔 이것도 `data: ` SSE 프레임으로 감쌌는데, ExamTab.tsx는 이 응답을
    // JSON으로 파싱해 에러 메시지를 꺼내므로 형식이 맞아야 실제로 노출된다.
    return Response.json({ error: "서버 인증이 설정되지 않았습니다." }, { status: 503 });
  }

  let response: Response;
  try {
    response = await fetch(`${BACKEND}/exam/stream`, {
      method: "POST",
      headers: { "X-Bunpil-Api-Key": API_KEY },
      body: formData,
    });
  } catch {
    return Response.json({ error: "FastAPI 서버에 연결할 수 없습니다." }, { status: 503 });
  }

  // 백엔드가 실제 SSE 스트림을 열기 전에 실패하면(인증 401, 동시요청 초과 429 등)
  // 일반 JSON 에러 응답이 온다 — 이 경우까지 무조건 text/event-stream으로 씌우면
  // 프론트가 SSE 프레임 파서로 순수 JSON을 못 읽어 에러가 조용히 사라진다.
  const isStream = response.headers.get("content-type")?.includes("text/event-stream");
  if (!isStream) {
    return new Response(response.body, {
      status: response.status,
      headers: { "Content-Type": response.headers.get("content-type") ?? "application/json" },
    });
  }

  return new Response(response.body, {
    status: response.status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "X-Accel-Buffering": "no",
    },
  });
}
