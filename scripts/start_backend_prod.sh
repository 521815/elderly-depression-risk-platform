#!/bin/bash

set -u

PROJECT="$HOME/Desktop/大创/老年抑郁风险预测大模型"
cd "$PROJECT" || exit 1

if [ -f "$PROJECT/.env" ]; then
  set -a
  . "$PROJECT/.env"
  set +a
fi

if /opt/anaconda3/bin/python -c "import waitress" >/dev/null 2>&1; then
  exec /opt/anaconda3/bin/python -m waitress --listen=127.0.0.1:8001 --threads=8 backend_api.wsgi:application
else
  echo "waitress 未安装，回退到 Flask 开发服务器"
  exec /opt/anaconda3/bin/python backend_api/app.py
fi
