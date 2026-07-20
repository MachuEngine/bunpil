#!/bin/bash
# 완료 선언 전 기본 검증 확인

SCRIPTS_DIR="$(dirname "$0")/../../../scripts"
SCRIPTS_DIR="$(cd "$SCRIPTS_DIR" 2>/dev/null && pwd)"

if [ -z "$SCRIPTS_DIR" ]; then
    # fallback: 절대 경로 사용
    SCRIPTS_DIR="/Users/anjongmin/bunpil/scripts"
fi

TEST_FILES=$(ls "$SCRIPTS_DIR"/test_*.py 2>/dev/null)

if [ -n "$TEST_FILES" ]; then
    echo "[verify] ✅ 검증 스크립트를 실행했는지 확인하세요:" >&2
    echo "$TEST_FILES" | while read -r f; do
        echo "  - $(basename "$f")" >&2
    done
    echo "[verify] 실행 방법: python scripts/test_<모듈>.py" >&2
fi

exit 0
