FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# next.config.ts의 output: "standalone" 덕분에 node_modules 전체 없이
# .next/standalone만으로 실행 가능 (이미지 크기 축소)
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

# localhost로 하면 Alpine 리졸버가 IPv6(::1)를 먼저 시도하는데 Next.js는 IPv4(0.0.0.0)에만
# 바인딩돼 있어 헬스체크가 항상 실패함(서비스는 정상) — 127.0.0.1로 명시해 회피
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD wget -qO- http://127.0.0.1:3000/ || exit 1

CMD ["node", "server.js"]
