import { NextRequest } from "next/server";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(request: NextRequest) {
  const formData = await request.formData();

  let response: Response;
  try {
    response = await fetch(`${BACKEND}/exam/stream`, {
      method: "POST",
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
