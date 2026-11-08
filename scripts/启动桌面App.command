#!/bin/zsh

PROJECT="$HOME/Desktop/大创/老年抑郁风险预测大模型"
cd "$PROJECT" || exit 1
export PYTHONPATH="$PROJECT/src:$PROJECT/backend_api:$PROJECT/webapp"
exec /opt/anaconda3/bin/python "$PROJECT/scripts/desktop_app.py"
