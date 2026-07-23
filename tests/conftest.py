"""pytest 세션 전체에 적용되는 공용 설정.

2026-07-24 출제 모듈 프로덕션 트레이싱 허용(CLAUDE.md 하드룰 3 예외) 이후, app.main이
LANGCHAIN_TRACING_V2를 더 이상 강제로 끄지 않는다. 그대로 두면 테스트가 개발자 로컬 .env의
값에 영향받아 순수 로직 테스트에서도 실제 LangSmith 네트워크 호출을 시도할 수 있어(테스트
비결정성·속도 저하·API 사용량 소모), 테스트 세션 동안은 항상 꺼둔다.
"""
import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
