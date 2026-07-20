#!/bin/bash
# Python 파일 수정 후 PII 패턴 노출 여부 스캔

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_input', {}).get('file_path', ''))
except Exception:
    print('')
" 2>/dev/null)

# Python 파일이 아니면 스킵
if [[ "$FILE_PATH" != *.py ]]; then
    exit 0
fi

# 파일이 없으면 스킵
if [ ! -f "$FILE_PATH" ]; then
    exit 0
fi

FOUND=0

# 전화번호 (010-xxxx-xxxx 형태)
if grep -nqE '010-[0-9]{4}-[0-9]{4}' "$FILE_PATH"; then
    echo "[pii_check] ⚠️  전화번호 패턴 발견: $FILE_PATH" >&2
    grep -nE '010-[0-9]{4}-[0-9]{4}' "$FILE_PATH" | head -3 >&2
    FOUND=1
fi

# 이메일 주소
if grep -nqE '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}' "$FILE_PATH"; then
    echo "[pii_check] ⚠️  이메일 주소 패턴 발견: $FILE_PATH" >&2
    grep -nE '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}' "$FILE_PATH" | head -3 >&2
    FOUND=1
fi

# 주민등록번호 패턴 (6자리-7자리)
if grep -nqE '[0-9]{6}-[0-9]{7}' "$FILE_PATH"; then
    echo "[pii_check] ⚠️  주민등록번호 패턴 발견: $FILE_PATH" >&2
    grep -nE '[0-9]{6}-[0-9]{7}' "$FILE_PATH" | head -3 >&2
    FOUND=1
fi

if [ "$FOUND" -eq 1 ]; then
    echo "[pii_check] 위 패턴이 실제 PII인지 확인하세요. 실제 개인정보라면 즉시 제거하십시오." >&2
fi

# 경고만 출력, 진행은 허용
exit 0
