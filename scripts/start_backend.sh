#!/bin/bash

export PATH="/opt/anaconda3/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export LANG="zh_CN.UTF-8"

cd "$HOME/Desktop/大创/老年抑郁风险预测大模型" || exit 1

if [ -f "$HOME/Desktop/大创/老年抑郁风险预测大模型/.env" ]; then
  set -a
  . "$HOME/Desktop/大创/老年抑郁风险预测大模型/.env"
  set +a
fi

exec /opt/anaconda3/bin/python backend_api/app.py
