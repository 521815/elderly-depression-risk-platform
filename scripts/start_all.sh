#!/bin/bash

set -u

PROJECT="$HOME/Desktop/大创/老年抑郁风险预测大模型"
BACKEND_PORT="${BACKEND_PORT:-8001}"
WEBAPP_PORT="${WEBAPP_PORT:-8501}"

cd "$PROJECT" || exit 1
mkdir -p "$PROJECT/output"

is_up() {
  local port="$1"
  local path="$2"
  /usr/bin/curl -sf --connect-timeout 1 --max-time 2 "http://localhost:${port}${path}" >/dev/null 2>&1
}

if ! is_up "$BACKEND_PORT" "/health"; then
  echo "正在启动后端 API (${BACKEND_PORT})..."
  /usr/bin/nohup /bin/bash "$PROJECT/scripts/start_backend.sh" >> "$PROJECT/output/backend.log" 2>&1 &
  disown
  for _i in {1..60}; do
    if is_up "$BACKEND_PORT" "/health"; then
      break
    fi
    sleep 0.5
  done
fi

if ! is_up "$WEBAPP_PORT" "/healthz"; then
  echo "正在启动前端工作台 (${WEBAPP_PORT})..."
  /usr/bin/nohup /opt/anaconda3/bin/streamlit run webapp/app.py \
    --server.headless true \
    --server.port "$WEBAPP_PORT" \
    --server.fileWatcherType none >> "$PROJECT/output/webapp.log" 2>&1 &
  disown
  for _i in {1..60}; do
    if is_up "$WEBAPP_PORT" "/healthz"; then
      break
    fi
    sleep 0.5
  done
fi

if is_up "$BACKEND_PORT" "/health" && is_up "$WEBAPP_PORT" "/healthz"; then
  echo "后端 API：http://localhost:${BACKEND_PORT}"
  echo "前端工作台：http://localhost:${WEBAPP_PORT}"
else
  echo "启动失败，请查看 output/backend.log 和 output/webapp.log"
  exit 1
fi
