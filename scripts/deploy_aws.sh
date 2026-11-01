#!/usr/bin/env bash
# ==============================================================================
# Automated 1-Click AWS EC2 Free Tier Deployment Script
# Targets: Ubuntu 22.04 / 24.04 LTS on t2.micro or t3.micro
# ==============================================================================
set -e

echo "===================================================================="
echo " Starting AWS EC2 Free Tier Setup for Codebase Debugging Agent"
echo "===================================================================="

# 1. Configure 4GB Swap Space (Essential for 1GB RAM t2.micro instances)
SWAP_EXISTS=$(swapon --show | wc -l)
if [ "$SWAP_EXISTS" -le 1 ]; then
    echo "--> [1/5] Configuring 4GB swapfile to prevent out-of-memory errors..."
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "    [OK] 4GB swap configured."
else
    echo "--> [1/5] Swap space already present. Skipping swap creation."
fi
free -h

# 2. Install Docker and Docker Compose
echo "--> [2/5] Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "    Installing Docker and Docker Compose plugin..."
    sudo apt-get update -y
    sudo apt-get install -y docker.io docker-compose-v2 curl git
    sudo usermod -aG docker "$USER"
    sudo systemctl enable docker
    sudo systemctl start docker
    echo "    [OK] Docker installed and enabled."
else
    echo "    [OK] Docker is already installed: $(docker --version)"
fi

# 3. Setup .env file
echo "--> [3/5] Checking environment configuration (.env)..."
if [ ! -f .env ]; then
    if [ -z "$GROQ_API_KEY" ]; then
        echo "===================================================================="
        echo " Enter your GROQ_API_KEY (starts with gsk_): "
        read -r INPUT_GROQ_KEY
    else
        INPUT_GROQ_KEY="$GROQ_API_KEY"
    fi

    cat <<EOF > .env
MODEL_PROVIDER=groq
GROQ_API_KEY=${INPUT_GROQ_KEY}
GROQ_MODEL=qwen/qwen3.8-27b
FALLBACK_MODEL=openai/gpt-oss-20b
PORT=8000
VECTOR_STORE_DIR=vector_store
SESSION_DB_PATH=data/sessions.db
EOF
    echo "    [OK] .env file created."
else
    echo "    [OK] Existing .env file found."
fi

# 4. Build and Start Container Services
echo "--> [4/5] Building and launching container services..."
sudo docker compose down --remove-orphans 2>/dev/null || true
sudo docker compose up --build -d

# 5. Verify Health Probe
echo "--> [5/5] Waiting for services to become healthy..."
MAX_ATTEMPTS=30
ATTEMPT=0
HEALTHY=false

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        HEALTHY=true
        break
    fi
    echo "    Waiting for API health probe (attempt $ATTEMPT/$MAX_ATTEMPTS)..."
    sleep 3
done

PUBLIC_IP=$(curl -s https://checkip.amazonaws.com || curl -s ifconfig.me || echo "<YOUR_EC2_PUBLIC_IP>")

echo ""
echo "===================================================================="
if [ "$HEALTHY" = true ]; then
    echo " DEPLOYMENT SUCCESSFUL! 🎉"
    echo "===================================================================="
    echo " Live Swagger Docs:  http://${PUBLIC_IP}:8000/docs"
    echo " Health Endpoint:    http://${PUBLIC_IP}:8000/health"
    echo " Ready Endpoint:     http://${PUBLIC_IP}:8000/ready"
else
    echo " Service started, but healthcheck is still warming up."
    echo " Check container logs with: sudo docker compose logs -f agent"
fi
echo "===================================================================="
