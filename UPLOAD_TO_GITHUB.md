# GitHub 上传指南

## 🚀 一键上传到 GitHub

### 步骤 1: 在 GitHub 上创建仓库

1. 访问 https://github.com/new
2. 仓库名称: `vehicle-detection-system`
3. 选择 Private 或 Public
4. **不要**勾选 "Initialize this repository with a README"
5. 点击 "Create repository"

### 步骤 2: 本地推送

在项目目录下执行:

```bash
cd vehicle-detection-system

# 添加远程仓库
git remote add origin https://github.com/Myth-wangyun/vehicle-detection-system.git

# 提交代码
git add .
git commit -m "Initial commit: Vehicle Detection System with YOLOv8"

# 推送（首次推送设置上游分支）
git push -u origin main
```

### 步骤 3: 启用 GitHub Pages

1. 进入仓库 Settings
2. 左侧菜单选择 "Pages"
3. Source 选择 "Deploy from a branch"
4. Branch 选择 `gh-pages` / `(root)` 
5. 点击 Save

等待 1-2 分钟，页面将发布到:
```
https://myth-wangyun.github.io/vehicle-detection-system/
```

## 📁 已配置的文件

```
vehicle-detection-system/
├── .github/
│   └── workflows/
│       └── deploy.yml          # 自动部署配置
├── docs/
│   ├── index.html              # GitHub Pages 首页
│   └── images/                 # 示例图片
├── README.md                   # 项目文档
├── .gitignore                  # Git 忽略配置
└── ...（其他项目文件）
```

## 🔄 自动部署

每次推送到 main 分支后，GitHub Actions 会自动:
1. 构建页面
2. 部署到 GitHub Pages
3. 更新访问地址

## 🌐 访问地址

| 环境 | 地址 |
|------|------|
| GitHub Pages | https://myth-wangyun.github.io/vehicle-detection-system/ |
| GitHub 仓库 | https://github.com/Myth-wangyun/vehicle-detection-system |

## ❓ 常见问题

### Q: 推送失败?

```bash
# 检查远程仓库
git remote -v

# 如果 origin 已存在，先移除
git remote remove origin

# 重新添加
git remote add origin https://github.com/Myth-wangyun/vehicle-detection-system.git
```

### Q: GitHub Pages 没更新?

1. 检查 Actions 是否有错误: https://github.com/Myth-wangyun/vehicle-detection-system/actions
2. 确认 Settings > Pages 中已启用
3. 等待 2-5 分钟

### Q: 需要帮助?

创建 Issue 或联系 [@Myth-wangyun](https://github.com/Myth-wangyun)
