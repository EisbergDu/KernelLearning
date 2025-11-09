# Repository Guidelines

## Coding Style & Naming Conventions
- 代码全写 Python：使用 4 空格缩进，函数/变量采用 snake_case，类名保持 PascalCase，遵循 PEP 8。
- 导入分组为标准库、第三方、项目内模块，组内按字母排序，避免无用导入。
- 代码中的注释使用中文简体，聚焦解释非显而易见的设计；保持英文命名以保持一致。
- 若引入格式化/静态检查工具（如 `black`、`ruff`），请记录版本和命令，并在每次提交前运行。

## Testing Guidelines
- 当前尚无正式测试框架，实验通过脚本或 Notebook 记录。新增测试建议新建 `tests/` 目录或与被测模块共存。
- 测试文件命名 `test_*.py`，示例 `tests/test_kernel_loss.py`，保持描述性且紧靠实现文件。
- 运行命令：`python tests/test_kernel_loss.py`，若基于 Notebook，可在文档中注明执行顺序与关键单元。

## Commit & Pull Request Guidelines
- 遵循 `type: short summary`（如 `fix: align kernel matrix`）格式，主题不超过 50 个字符、小写且以冒号分隔。
- PR 描述需包含：简要摘要、关联 issue（若有）、执行验证步骤（脚本、Notebook 运行），若涉及可视化输出，请附前后对比说明或截图。
- 变更备注、PR 讨论、文档均使用中文简体；提交代码时仍保持英文命名。

## 文档与协作要求
- 所有新文档、注释和 PR 描述都需使用中文简体，便于团队统一沟通。
- 工作时若遇已有改动（dirty tree），只处理任务相关文件，未经指示不要重置或撤销他人的变动。

## Language Requirements
- **Primary Language**: 所有文档、注释、PR 描述使用**中文（简体）**
- **Code Language**: 项目代码统一使用 **Python**
- **Variable/Function Names**: 使用英文命名（遵循 PEP 8）
- **Comments**: 中文注释，解释复杂逻辑