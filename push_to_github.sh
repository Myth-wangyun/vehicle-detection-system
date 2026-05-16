#!/bin/bash

# GitHub Push Script
# 用于快速将项目推送到 GitHub

echo "=========================================="
echo "Vehicle Detection System - GitHub Push"
echo "=========================================="

# 检查是否在正确目录
if [ ! -f "README.md" ]; then
    echo "错误: 请在 vehicle_detection_system 目录下运行"
    exit 1
fi

# 检查 git 是否已初始化
if [ ! -d ".git" ]; then
    echo "初始化 Git 仓库..."
    git init
    git config user.name "Myth-wangyun"
    git config user.email "myth-wangyun@github.com"
fi

# 设置远程仓库
REPO_URL="https://github.com/Myth-wangyun/vehicle-detection-system.git"

# 检查远程是否已设置
if ! git remote get-url origin &>/dev/null; then
    echo "添加远程仓库..."
    git remote add origin "$REPO_URL"
else
    echo "更新远程仓库..."
    git remote set-url origin "$REPO_URL"
fi

# 显示状态
echo ""
echo "Git 状态:"
git status -s

# 添加所有更改
echo ""
echo "添加文件..."
git add .

# 提交
echo ""
read -p "输入提交信息 (直接回车使用默认): " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Update: Vehicle Detection System $(date +%Y-%m-%d)"
fi

git commit -m "$commit_msg"

# 推送到 GitHub
echo ""
echo "推送到 GitHub..."
echo "仓库地址: $REPO_URL"
echo ""

git push -u origin main --force

echo ""
echo "=========================================="
echo "推送完成!"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 访问 https://github.com/Myth-wangyun/vehicle-detection-system"
echo "2. 进入 Settings > Pages"
echo "3. Source 选择 'Deploy from a branch'"
echo "4. Branch 选择 'gh-pages' / '(root)'"
echo "5. 等待 1-2 分钟访问:"
echo "   https://myth-wangyun.github.io/vehicle-detection-system/"
echo ""
