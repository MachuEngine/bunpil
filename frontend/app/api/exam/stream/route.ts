import { NextRequest } from "next/server";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";
const API_KEY = process.env.BUNPIL_API_KEY ?? "";

export async function POST(request: NextRequest) {
  const formData = await request.formData();
  if (!API_KEY) {
    const msg = JSON.stringify({ status: "error", msg: "서버 인증이 설정되지 않았습니다." });
    return new Response(`data: ${msg}\n\n`, {
      status: 503,
      headers: { "Content-Type": "text/event-stream" },
    });
  }

  let response: Response;
  try {
    response = await fetch(`${BACKEND}/exam/stream`, {
      method: "POST",
      headers: { "X-Bunpil-Api-Key": API_KEY },
      body: formData,
    });
  } catch {
    const msg = JSON.stringify({ status: "error", msg: "FastAPI 서버에 연결할 수 없습니다." });
    return new Response(`data: ${msg}\n\n`, {
      status: 503,
      headers: { "Content-Type": "text/event-stream" },
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
