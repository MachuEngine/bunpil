import { NextRequest, NextResponse } from "next/server";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(request: NextRequest) {
  const formData = await request.formData();

  let response: Response;
  try {
    response = await fetch(`${BACKEND}/exam`, {
      method: "POST",
      body: formData,
    });
  } catch {
    return NextResponse.json(
      { error: "FastAPI 서버에 연결할 수 없습니다." },
      { status: 503 }
    );
  }

  const data = await response.json();
  return NextResponse.json(data, { status: response.status });
}
