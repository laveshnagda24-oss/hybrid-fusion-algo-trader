#!/bin/bash
set -e

echo "=================================================================="
echo "DEPLOYING HYBRID FUSION CHAMPION TO GOOGLE CLOUD COMPUTE ENGINE VM"
echo "=================================================================="

# 1. Update and install dependencies
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git curl

# 2. Setup project directory
APP_DIR="/opt/algo-trader"
sudo mkdir -p $APP_DIR
sudo chown -R $USER:$USER $APP_DIR

# 3. Clone / Pull repository
if [ -d "$APP_DIR/.git" ]; then
    echo "Updating existing repository..."
    cd $APP_DIR
    git pull origin main
else
    echo "Cloning repository..."
    git clone https://github.com/laveshnagda24-oss/hybrid-fusion-algo-trader.git $APP_DIR
fi

cd $APP_DIR

# 4. Install Python dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt
pip3 install requests pandas numpy pytz fastapi uvicorn

# 5. Create Systemd Service for 24/7/365 Always-On Execution
SERVICE_FILE="/etc/systemd/system/algo-trader.service"
echo "Configuring systemd service at $SERVICE_FILE..."

sudo bash -c "cat > $SERVICE_FILE" <<EOL
[Unit]
Description=Hybrid Fusion Champion Live Cloud Trader
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR
ExecStart=/usr/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 80
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOL

# 6. Allow port 80 for non-root or give python bind permissions
sudo setcap 'cap_net_bind_service=+ep' $(readlink -f $(which python3)) || true

# 7. Reload systemd and start service
sudo systemctl daemon-reload
sudo systemctl enable algo-trader
sudo systemctl restart algo-trader

echo "=================================================================="
echo "DEPLOYMENT COMPLETE! ALGO TRADER IS RUNNING 24/7 ON GOOGLE CLOUD!"
echo "Status check command: sudo systemctl status algo-trader"
echo "Live logs command   : sudo journalctl -u algo-trader -f"
echo "=================================================================="
