#!/bin/bash
# Robust tunnel with auto-restart

TUNNEL_LOG="/tmp/tunnel.log"

while true; do
    echo "$(date): Starting tunnel..." | tee -a $TUNNEL_LOG

    # Run SSH tunnel with keepalive settings
    ssh -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -R 80:localhost:8000 \
        nokey@localhost.run 2>&1 | tee -a $TUNNEL_LOG

    EXIT_CODE=$?
    echo "$(date): Tunnel died with exit code $EXIT_CODE. Restarting in 3s..." | tee -a $TUNNEL_LOG
    sleep 3
done
