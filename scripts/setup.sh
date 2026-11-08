#!/bin/bash

set -e

PROJECT="$HOME/Desktop/大创/老年抑郁风险预测大模型"
cd "$PROJECT" || exit 1

echo "==> 安装 Python 依赖"
/opt/anaconda3/bin/pip install -r requirements.txt

echo "==> 准备数据与输出目录"
mkdir -p "$PROJECT/data/synthetic" "$PROJECT/output"

echo "==> 检查后台服务器"
/opt/anaconda3/bin/python -c "import waitress" || /opt/anaconda3/bin/pip install waitress

echo "完成。启动方式："
echo "  scripts/start_all.sh          开发模式（前后端 + 自动打开浏览器）"
echo "  scripts/start_all_prod.sh     生产模式（waitress + Streamlit）"
