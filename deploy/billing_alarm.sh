#!/bin/bash
# AWS 월간 요금 알람 설정
# 실행 위치: 로컬 터미널 (aws configure 완료 후)
# 주의: 빌링 알람은 반드시 us-east-1 리전에서 설정해야 함

set -euo pipefail

ALARM_EMAIL="ann10266@gmail.com"  # 알람 수신 이메일
THRESHOLD_USD=10                   # 임계값 달러

echo "=== [1/3] SNS 토픽 생성 (알람 발송 채널) ==="
# 왜 SNS? CloudWatch가 알람을 SNS 토픽으로 전송 → 이메일로 전달.
TOPIC_ARN=$(aws sns create-topic \
    --name BunpilBillingAlert \
    --region us-east-1 \
    --query 'TopicArn' --output text)

echo "  토픽 ARN: $TOPIC_ARN"

aws sns subscribe \
    --topic-arn "$TOPIC_ARN" \
    --protocol email \
    --notification-endpoint "$ALARM_EMAIL" \
    --region us-east-1

echo "  ✉️  $ALARM_EMAIL 로 구독 확인 이메일이 발송됩니다. 반드시 Confirm을 클릭하세요!"

echo ""
echo "=== [2/3] CloudWatch 빌링 알람 생성 ==="
# 왜 us-east-1? AWS 빌링 메트릭은 글로벌이지만 us-east-1에서만 조회 가능.
aws cloudwatch put-metric-alarm \
    --alarm-name "BunpilMonthlyCost" \
    --alarm-description "분필 월 요금 \$${THRESHOLD_USD} 초과" \
    --namespace "AWS/Billing" \
    --metric-name "EstimatedCharges" \
    --dimensions Name=Currency,Value=USD \
    --statistic Maximum \
    --period 86400 \
    --evaluation-periods 1 \
    --threshold "$THRESHOLD_USD" \
    --comparison-operator GreaterThanThreshold \
    --alarm-actions "$TOPIC_ARN" \
    --region us-east-1

echo "  알람 설정 완료: 월 누적 요금이 \$$THRESHOLD_USD 초과 시 이메일 발송"

echo ""
echo "=== [3/3] 빌링 메트릭 활성화 확인 ==="
echo "  AWS 콘솔 → Billing → Billing Preferences → Receive Billing Alerts 체크 확인"
echo "  (CLI로는 설정 불가 — 콘솔에서 직접 켜야 합니다)"
echo ""
echo "=== 완료 ==="
echo "  알람 확인: aws cloudwatch describe-alarms --alarm-names BunpilMonthlyCost --region us-east-1"
