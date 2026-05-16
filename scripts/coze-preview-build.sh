#!/usr/bin/env bash
set -euo pipefail

# 基于脚本位置定位项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "预览准备完成 - 静态文件已就绪"
echo "项目目录: $PROJECT_DIR"
echo "预览文件: preview.html"
