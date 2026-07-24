#!/bin/bash
# ============================================================
# 목적: Bash 도구 실행 전 위험 명령(chroma_db/data 강제 삭제, .env 직접 수정)을 차단
# 트리거: PreToolUse (matcher: Bash)
# 로그: .claude/logs/hooks.log (PASS/BLOCK 이벤트, 매 Bash 실행마다 1줄)
# ============================================================

LOG_FILE="$(dirname "$0")/../../logs/hooks.log"
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null

log_event() {
    echo "$(date +%Y-%m-%dT%H:%M:%S%z) [guard] $1" >> "$LOG_FILE" 2>/dev/null
}

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_input', {}).get('command', ''))
except Exception:
    print('')
" 2>/dev/null)

# rm -rf chroma_db 또는 rm -rf data 차단
if echo "$COMMAND" | grep -qE 'rm\s+-rf\s+(\./)?(chroma_db|data)(/|$)'; then
    echo "[guard] 🚫 위험 명령 차단: chroma_db 또는 data 디렉토리 강제 삭제는 금지되어 있습니다." >&2
    echo "[guard] 명령: $COMMAND" >&2
    log_event "BLOCK rm-rf-chroma_db-or-data"
    exit 1
fi

# chroma_db/ 경로 직접 삭제 패턴 차단 (rm 명령 + chroma_db 포함)
if echo "$COMMAND" | grep -qE 'rm\s+.*chroma_db'; then
    echo "[guard] 🚫 위험 명령 차단: chroma_db 경로 삭제는 금지되어 있습니다." >&2
    echo "[guard] 명령: $COMMAND" >&2
    log_event "BLOCK rm-chroma_db"
    exit 1
fi

# .env 파일 직접 수정 차단 (리다이렉션으로 덮어쓰기)
if echo "$COMMAND" | grep -qE '(>|tee)\s*\.env\b'; then
    echo "[guard] 🚫 위험 명령 차단: .env 파일 직접 수정은 금지되어 있습니다." >&2
    echo "[guard] 명령: $COMMAND" >&2
    log_event "BLOCK env-overwrite"
    exit 1
fi

log_event "PASS"
exit 0
