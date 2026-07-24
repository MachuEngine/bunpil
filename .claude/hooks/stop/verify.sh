#!/bin/bash
# ============================================================
# 목적: 완료 선언(Stop) 전 scripts/test_*.py 실행 여부를 상기시키는 리마인더
# 트리거: Stop
# 로그: .claude/logs/hooks.log (REMINDER 이벤트, 발견된 테스트 스크립트 개수 기록)
# ============================================================

LOG_FILE="$(dirname "$0")/../../logs/hooks.log"
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null

log_event() {
    echo "$(date +%Y-%m-%dT%H:%M:%S%z) [verify] $1" >> "$LOG_FILE" 2>/dev/null
}

SCRIPTS_DIR="$(dirname "$0")/../../../scripts"
SCRIPTS_DIR="$(cd "$SCRIPTS_DIR" 2>/dev/null && pwd)"

if [ -z "$SCRIPTS_DIR" ]; then
    # fallback: 절대 경로 사용
    SCRIPTS_DIR="/Users/anjongmin/bunpil/scripts"
fi

TEST_FILES=$(ls "$SCRIPTS_DIR"/test_*.py 2>/dev/null)
TEST_COUNT=$(echo "$TEST_FILES" | grep -c . 2>/dev/null || echo 0)

if [ -n "$TEST_FILES" ]; then
    echo "[verify] ✅ 검증 스크립트를 실행했는지 확인하세요:" >&2
    echo "$TEST_FILES" | while read -r f; do
        echo "  - $(basename "$f")" >&2
    done
    echo "[verify] 실행 방법: python scripts/test_<모듈>.py" >&2
fi

log_event "REMINDER test_files=$TEST_COUNT"
exit 0
