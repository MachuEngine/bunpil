#!/bin/bash
# EC2 인스턴스 내부에서 실행 — Caddy 설치 + 서비스 등록
# 전제: your-domain.com DNS A 레코드가 이 서버 IP를 가리키고 있어야 함

set -euo pipefail

DOMAIN="your-domain.com"   # 실제 도메인으로 교체

echo "=== Caddy 설치 ==="
# 왜 Caddy? Let's Encrypt 인증서 자동 발급/갱신. 설정 한 줄로 HTTPS 완성.
sudo dnf install -y 'dnf-command(copr)'
sudo dnf copr enable -y @caddy/caddy
sudo dnf install -y caddy

echo "=== Caddyfile 배포 ==="
# 프로젝트 Caddyfile에서 도메인을 실제 값으로 치환 후 복사
sed "s/your-domain.com/$DOMAIN/g" ~/bunpil/Caddyfile | sudo tee /etc/caddy/Caddyfile

echo "=== 로그 디렉토리 생성 ==="
sudo mkdir -p /var/log/caddy
sudo chown caddy:caddy /var/log/caddy

echo "=== Caddy 서비스 시작 ==="
sudo systemctl enable --now caddy
sudo systemctl reload caddy

echo ""
echo "=== 완료 ==="
echo "  https://$DOMAIN 은 Next.js(3000, docker-compose frontend 서비스)로 연결됩니다."
echo "  (docker compose up -d --build 로 app+frontend 컨테이너가 함께 떠 있어야 함)"
echo "  인증서 상태: sudo caddy certificates"
echo "  로그 확인:   sudo journalctl -u caddy -f"
