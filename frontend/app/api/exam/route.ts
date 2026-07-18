import { NextRequest, NextResponse } from "next/server";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";
const API_KEY = process.env.BUNPIL_API_KEY ?? "";

export async function POST(request: NextRequest) {
  const formData = await request.formData();
  if (!API_KEY) {
    return NextResponse.json({ error: "서버 인증이 설정되지 않았습니다." }, { status: 503 });
  }

  let response: Response;
  try {
    response = await fetch(`${BACKEND}/exam`, {
      method: "POST",
      headers: { "X-Bunpil-Api-Key": API_KEY },
      body: formData,
    });
  } catch {
    return NextResponse.json(
      { error: "FastAPI 서버에 연결할 수 없습니다." },
      { status: 503 }
    );
  }

  const data = response.headers.get("content-type")?.includes("application/json")
    ? await response.json()
    : { error: "백엔드 요청 처리에 실패했습니다." };
  return NextResponse.json(data, { status: response.status });
}
