#!/bin/zsh

PROJECT="$HOME/Desktop/大创/老年抑郁风险预测大模型"
/bin/bash "$PROJECT/scripts/start_all.sh"

if [ -z "$WEBAPP_NO_OPEN" ]; then
  /usr/bin/open "http://localhost:${WEBAPP_PORT:-8501}"
fi
