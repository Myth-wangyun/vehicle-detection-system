#!/usr/bin/env bash
set -euo pipefail

# 基于脚本位置定位项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# 显式声明关键环境变量
export PORT=5000

# 清理 5000 端口残留进程
fuser -k 5000/tcp 2>/dev/null || true
sleep 1

echo "启动预览服务..."
echo "项目目录: $PROJECT_DIR"
echo "监听端口: 5000"
echo "预览页面: http://localhost:5000/preview.html"

# 使用 Python 内置 HTTP 服务器提供预览
exec python3 -m http.server 5000 --bind 0.0.0.0
