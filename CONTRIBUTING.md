# 贡献指南

欢迎为 tts-skill 贡献代码！

## 如何贡献

### 报告问题

- 使用 [GitHub Issues](https://github.com/boommanpro/tts-skill/issues) 报告 bug 或提出功能建议
- 请包含：操作系统、Python 版本、GPU 类型、完整的错误信息
- 可先运行 `python -m tts_skill doctor` 获取环境信息

### 提交代码

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m 'feat: add your feature'`
4. 推送分支：`git push origin feature/your-feature`
5. 提交 Pull Request

### 开发规范

- 代码风格遵循 PEP 8
- 新功能必须包含测试用例
- 提交前运行测试：`python -m pytest tests/ -v`
- 提交信息使用约定式提交（Conventional Commits）：
  - `feat:` 新功能
  - `fix:` 修复 bug
  - `docs:` 文档更新
  - `refactor:` 重构
  - `test:` 测试相关
  - `chore:` 构建/工具相关

### 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_utils.py -v

# 生成覆盖率报告
python -m pytest tests/ --cov=tts_skill --cov-report=html
```

### 项目结构

详见 [README.md](README.md) 的「仓库结构」部分。核心代码在 `tts_skill/` 目录，测试在 `tests/` 目录。

### 添加新功能

- **新生成模式**：在 `tts_skill/infer.py` 和 `tts_skill/webui.py` 中添加
- **新平台支持**：在 `tts_skill/utils.py` 和 `tts_skill/setup_env.py` 中添加
- **新参数**：在 `tts_skill/infer.py` 的 `build_infer_parser()` 中添加，并在 `tts_skill/config.py` 的 `GenerationRecord` 中扩展

## 行为准则

- 保持友善和尊重
- 欢迎新手贡献
- 关注问题本身，不针对个人

## 许可证

提交的代码将遵循 [MIT 许可证](LICENSE)。
