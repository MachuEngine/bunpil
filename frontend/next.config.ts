import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Docker 배포 시 node_modules 전체 없이 최소 런타임만 복사(.next/standalone)하기 위함.
  // frontend/Dockerfile이 이 산출물을 그대로 실행한다.
  output: "standalone",
};

export default nextConfig;
