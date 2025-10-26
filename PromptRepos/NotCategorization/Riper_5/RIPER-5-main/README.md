# AI 提示词工程与工作流集合

[![GitHub stars](https://img.shields.io/github/stars/NeekChaw/RIPER-5.svg)](https://github.com/NeekChaw/RIPER-5/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/NeekChaw/RIPER-5.svg)](https://github.com/NeekChaw/RIPER-5/network)
[![GitHub issues](https://img.shields.io/github/issues/NeekChaw/RIPER-5.svg)](https://github.com/NeekChaw/RIPER-5/issues)

<div align="center">
  <img src="image.png" width="250" alt="公众号二维码">
  <p><strong>扫码关注我的公众号，获取更多 AI 前沿资讯与实践</strong></p>
</div>

欢迎来到 AI 提示词工程与工作流集合仓库！本项目旨在收录和展示一系列高质量、结构化的 AI 提示词（Prompts）和工作流（Workflows），专注于提升 AI 在软件开发、代码重构、深度思考和项目管理等领域的协作效率和可靠性。

## 更多场景优秀prompt请移步repo：[awesome-prompt](https://github.com/NeekChaw/awesome-prompt)

## 目录

- [仓库核心内容](#仓库核心内容)
- [RIPER-5：AI 编码行为协议](#1-riper-5ai-编码行为协议)
- [Claude Code：高效编程提示词集合](#2-claude-code高效编程提示词集合)
- [快速开始](#快速开始)
- [如何使用](#如何使用)
- [贡献](#贡献)
- [许可](#许可)

## 仓库核心内容

本仓库主要包含两大部分：`RIPER-5 行为协议` 和 `Claude Code 专用提示词`。

### 1. [RIPER-5：AI 编码行为协议](./RIPER-5/)

`RIPER-5` 不仅仅是一套提示词，它是一套为高级 AI 助手（尤其是在 IDE 中集成的 AI）设计的严格**行为协议和工作流框架**。其核心目标是通过一个强制性的、分阶段的流程来约束 AI 的行为，确保其在执行复杂编码任务时的每一步操作都**安全、可控且符合预期**。

👉 **[点击此处阅读 RIPER-5 的详细说明](./RIPER-5/README.md)**

- 📘 中文全文：[`RIPER-5/RIPER-5-CN.md`](./RIPER-5/RIPER-5-CN.md)
- 📗 English: [`RIPER-5/RIPER-5-EN.md`](./RIPER-5/RIPER-5-EN.md)
- 🧪 使用示例：[`RIPER-5/使用示例.md`](./RIPER-5/使用示例.md)

### 2. [Claude Code：高效编程提示词集合](./Claude%20Code/)

`Claude Code` 目录包含一系列为 [Claude](https://claude.ai/) 系列模型优化的提示词和工作流，旨在帮助开发者更高效地利用 AI 进行编程。每个子目录都针对一个特定的开发场景。

| 模块                                                                                     | 描述                                                               | 来源                                                                                                         |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| [**AI协同开发规范 (AICP)**](./Claude%20Code/AI%E5%8D%8F%E5%90%8C%E5%BC%80%E5%8F%91%E8%A7%84%E8%8C%83/) | 规范驱动开发流程，四阶段协作产出（含 [prompt.md](./Claude%20Code/AI%E5%8D%8F%E5%90%8C%E5%BC%80%E5%8F%91%E8%A7%84%E8%8C%83/prompt.md) 可即用）。 | 互联网                                                                                                       |
| [**Kiro需求收集与规划**](./Claude%20Code/kiro%E9%9C%80%E6%B1%82%E6%94%B6%E9%9B%86/)               | 一个严谨的三阶段工作流，将模糊想法转化为包含需求、设计和实施计划的完整开发文档。 | [kingkongshot/prompts](https://github.com/kingkongshot/prompts)                                              |
| [**Linux之父帮你重构代码**](./Claude%20Code/Linux%E4%B9%8B%E7%88%B6%E5%B8%AE%E4%BD%A0%E9%87%8D%E6%9E%84%E4%BB%A3%E7%A0%81/) | 角色扮演提示词，模拟 Linus Torvalds 的思维模式，以犀利、深刻的视角审查和重构代码。 | [kingkongshot/prompts](https://github.com/kingkongshot/prompts)                                              |
| [**一个让Claude更靠谱的Workflow**](./Claude%20Code/%E4%B8%80%E4%B8%AA%E8%AE%A9cc%E6%9B%B4%E9%9D%A0%E8%B0%B1%E7%9A%84workflow/) | 一套强制性的五步工作流，旨在约束 AI 行为，解决其随意创建文件、忽视现有架构等问题。 | [Reddit 讨论](https://www.reddit.com/r/ClaudeAI/comments/1m3pol4/my_best_workflow_for_working_with_claude_code/) |
| [**专业高效提交Git**](./Claude%20Code/%E4%B8%93%E4%B8%9A%E9%AB%98%E6%95%88%E6%8F%90%E4%BA%A4Git/)         | 引入 "Commit-as-Prompt" 理念，将 Git 提交信息结构化，使其能作为高质量上下文供 AI 使用。 | [kingkongshot/prompts](https://github.com/kingkongshot/prompts)                                              |
| [**超深度思考**](./Claude%20Code/%E8%B6%85%E6%B7%B1%E5%BA%A6%E6%80%9D%E8%80%83/)                       | 一个多代理协作工作流，通过模拟专家团队（架构师、研究员、程序员、测试员）来解决复杂问题。| [Reddit 讨论](https://www.reddit.com/r/ClaudeAI/comments/1lpvj7z/ultrathink_task_command/)                       |

## 快速开始

1. 选择你的目标：
   - 全局规范与工作流约束 → 进入 `RIPER-5`
   - 场景化高效提示词 → 进入 `Claude Code`
2. 打开对应目录的 `README.md` 了解理念和结构。
3. 复制 `prompt.md` 到你的 AI 助手（系统提示/自定义指令/对话中）。
4. 按照工作流执行与迭代。

## 如何使用

1.  **确定您的需求:** 您是想为 AI 设定一套全局的行为准则（参考 `RIPER-5`），还是想在特定任务中应用某个高效的工作流（参考 `Claude Code`）？
2.  **浏览相应目录:** 进入您感兴趣的目录，阅读其 `README.md` 文件，以快速了解该模块的核心理念和使用方法。
3.  **阅读 `prompt.md`:** 每个具体的工作流目录下都有一个 `prompt.md` 文件，其中包含了可以直接使用的提示词内容。
4.  **应用于您的 AI 助手:** 将提示词内容设置为 AI 的系统提示、自定义指令或在对话中直接使用，然后观察并引导 AI 按照预设的流程工作。

## 贡献

我们欢迎任何形式的贡献！如果您有创新的 AI 工作流、高效的提示词实践，或者对现有内容有改进建议，请随时提交 Pull Request 或创建 Issue。

## 许可

本项目采用 [MIT 许可证](LICENSE)。