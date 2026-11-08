#!/bin/bash

set -u

PROJECT="$HOME/Desktop/大创/老年抑郁风险预测大模型"
WEBAPP_PORT="${WEBAPP_PORT:-8501}"

cd "$PROJECT" || exit 1
mkdir -p "$PROJECT/output"

if ! curl -sf --connect-timeout 1 http://localhost:8001/health >/dev/null 2>&1; then
  echo "启动后端（waitress）..."
  nohup /bin/bash "$PROJECT/scripts/start_backend_prod.sh" >> "$PROJECT/output/backend.log" 2>&1 &
  disown
fi

if ! curl -sf --connect-timeout 1 "http://localhost:${WEBAPP_PORT}/healthz" >/dev/null 2>&1; then
  echo "启动前端（Streamlit）..."
  nohup /opt/anaconda3/bin/streamlit run "$PROJECT/webapp/app.py" \
    --server.headless true \
    --server.port "$WEBAPP_PORT" \
    --server.fileWatcherType none >> "$PROJECT/output/webapp.log" 2>&1 &
  disown
fi

if [ -z "$WEBAPP_NO_OPEN" ]; then
  sleep 2
  /usr/bin/open "http://localhost:${WEBAPP_PORT}"
fi

echo "后端：http://localhost:8001"
echo "前端：http://localhost:${WEBAPP_PORT}"
