#!/bin/bash

# Script'in bulunduğu dizinden backend kök dizinine git
cd "$(dirname "$0")"

# 8000 portunu kullanan eski bir süreç varsa onu bul ve sonlandır (Port Çakışması Çözümü)
echo "[*] Checking for processes on port 8000..."
PIDS=$(lsof -ti:8000)
if [ ! -z "$PIDS" ]; then
    echo "[!] Found stale process on port 8000. Killing process: $PIDS"
    kill -9 $PIDS
fi

# .venv'i bul ve aktifleştir (varsayılan sanal ortam yolu)
source ../.venv/bin/activate

# Backend'i modül olarak çalıştır (böylece core modülü sorunsuz bulunur)
echo "[*] Starting BirdBase Backend..."
python -m app.main
