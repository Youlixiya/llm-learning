# 🚀 GitHub Pages 自动部署指南

本项目已配置 GitHub Actions 自动部署工作流，当你推送代码到 `main` 或 `master` 分支时，会自动部署到 GitHub Pages。

## 📋 启用步骤

### 1. 在 GitHub 仓库中启用 Pages

1. 访问你的仓库：`https://github.com/Youlixiya/llm-learning`
2. 进入 **Settings** → **Pages**
3. 在 **Source** 部分，选择 **GitHub Actions**
4. 点击 **Save**

### 2. 推送代码触发部署

```bash
git add .
git commit -m "Update project"
git push origin main
```

### 3. 查看部署状态

- 访问 **Actions** 标签页查看部署进度
- 部署完成后，访问：`https://youlixiya.github.io/llm-learning/`

## 🔧 Workflow 配置说明

部署工作流文件位于：`.github/workflows/deploy.yml`

**触发条件：**
- 推送到 `main` 或 `master` 分支
- 手动触发（workflow_dispatch）

**部署内容：**
- 自动部署 `web/` 目录下的所有文件
- 包括 HTML、CSS、JavaScript 和文档

## 🐛 故障排查

### 部署失败

1. **检查 Actions 日志**
   - 进入仓库的 **Actions** 标签页
   - 查看失败的 workflow 运行详情

2. **检查权限设置**
   - 确保仓库 Settings → Actions → General → Workflow permissions 设置为 "Read and write permissions"

3. **检查 Pages 设置**
   - 确保 Pages 源设置为 "GitHub Actions"

### 页面无法访问

1. **等待部署完成**
   - 首次部署可能需要几分钟
   - 检查 Actions 中是否有正在运行的 workflow

2. **检查 URL**
   - 确保 URL 格式正确：`https://<username>.github.io/<repository-name>/`
   - 注意仓库名称大小写

3. **清除浏览器缓存**
   - 使用无痕模式访问
   - 或强制刷新（Ctrl+F5 / Cmd+Shift+R）

## 📝 自定义部署

如果需要修改部署配置，编辑 `.github/workflows/deploy.yml`：

```yaml
- name: Upload artifact
  uses: actions/upload-pages-artifact@v3
  with:
    path: './web'  # 修改为你的部署目录
```

## 🔗 相关链接

- [GitHub Pages 文档](https://docs.github.com/en/pages)
- [GitHub Actions 文档](https://docs.github.com/en/actions)
- [部署 Actions 文档](https://github.com/actions/deploy-pages)
