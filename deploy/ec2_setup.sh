#!/bin/bash
# EC2 t3.small 프로비저닝 스크립트
# 실행 위치: 로컬 터미널 (AWS CLI 설정 완료 후)
# 전제: aws configure 완료, 키페어 이름과 도메인 확인

set -euo pipefail

# ── 설정값 (직접 수정) ──────────────────────────────────────────────
KEY_NAME="bunpil-key"          # aws ec2 create-key-pair로 생성한 키페어 이름
REGION="ap-northeast-2"        # 서울 리전
AMI_ID="ami-0f3a440bbcff3d043" # Amazon Linux 2023 (서울, 2024-01 기준)
INSTANCE_TYPE="t3.small"
EBS_SIZE=20                     # GB (ChromaDB + 모델 캐시)
REPO_URL="https://github.com/MachuEngine/bunpil.git"
# ────────────────────────────────────────────────────────────────────

echo "=== [1/5] 보안 그룹 생성 ==="
# 왜? EC2에 들어오는 트래픽을 규칙으로 제어. 22(SSH), 7860(Gradio)만 허용.
# Caddy를 올리면 80/443도 추가로 허용.
SG_ID=$(aws ec2 create-security-group \
    --group-name bunpil-sg \
    --description "쌤조 앱 보안 그룹" \
    --region "$REGION" \
    --query 'GroupId' --output text)

aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --region "$REGION" \
    --ip-permissions \
    'IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=0.0.0.0/0,Description=SSH}]' \
    'IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges=[{CidrIp=0.0.0.0/0,Description=HTTP}]' \
    'IpProtocol=tcp,FromPort=443,ToPort=443,IpRanges=[{CidrIp=0.0.0.0/0,Description=HTTPS}]'

echo "  보안 그룹: $SG_ID"

echo "=== [2/5] EC2 인스턴스 시작 ==="
# 왜 t3.small? vCPU 2, 메모리 2GB — BGE 임베딩(CPU)·Gradio·FastAPI 구동 최소 사양.
# EBS 20GB: 모델 캐시 ~8GB + ChromaDB ~2GB + 여유 공간.
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":$EBS_SIZE,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=bunpil-app}]' \
    --region "$REGION" \
    --query 'Instances[0].InstanceId' --output text)

echo "  인스턴스 ID: $INSTANCE_ID"
echo "  인스턴스 시작 대기 중..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "  퍼블릭 IP: $PUBLIC_IP"

echo "=== [3/5] 인스턴스 초기화 (SSH) ==="
echo "  잠시 후 아래 명령으로 SSH 접속하세요:"
echo ""
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP}"
echo ""
echo "  접속 후 실행할 명령어:"
cat <<'SETUP_CMDS'
# Docker 설치
sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
newgrp docker

# 프로젝트 클론
git clone https://github.com/MachuEngine/bunpil.git && cd bunpil

# .env 파일 생성 (RunPod 키 등 실값 입력)
cp .env.example .env
nano .env   # LLM_BACKEND=runpod, RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID 입력

# 앱 빌드 & 실행 (첫 빌드: BGE 모델 다운로드로 10-20분 소요)
docker compose up -d --build

# 로그 확인
docker compose logs -f
SETUP_CMDS

echo ""
echo "=== [4/5] Elastic IP 연결 권장 ==="
echo "  왜? EC2를 재시작하면 IP가 바뀝니다. Elastic IP를 고정 IP로 연결하면"
echo "  도메인 DNS 설정이 유지됩니다."
echo ""
echo "  aws ec2 allocate-address --domain vpc --region $REGION"
echo "  # 반환된 AllocationId를 아래에 입력:"
echo "  aws ec2 associate-address --instance-id $INSTANCE_ID --allocation-id <alloc-id> --region $REGION"

echo ""
echo "=== [5/5] 완료 ==="
echo "  다음 단계: Caddyfile 설정 → deploy/caddy_setup.sh 참고"
echo "  인스턴스 ID: $INSTANCE_ID | IP: $PUBLIC_IP | SG: $SG_ID"
