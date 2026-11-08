#!/bin/bash

for port in 8001 8501; do
  pids=$(/usr/sbin/lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null)
  if [ -n "$pids" ]; then
    echo "停止端口 ${port}：${pids}"
    kill $pids 2>/dev/null
  fi
done
