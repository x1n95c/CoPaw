---
name: acp_delegation
description: Use this skill when a task is a good fit for one-shot delegation to an ACP-compatible external agent such as opencode, qwen, or gemini. It explains when to delegate, how to verify that spawn_agent is enabled and a runner is available, how to choose the right runner, and how to write a complete self-contained delegation prompt. | 当任务适合一次性委派给 opencode、qwen、gemini 等兼容 ACP 的外部 agent 时使用；说明何时该委派、如何确认 spawn_agent 已启用且 runner 可用、如何选择合适 runner，以及如何编写完整自包含的 delegation prompt
metadata: { "builtin_skill_version": "1.0", "copaw": { "emoji": "🛰️" } }
---

# ACP Delegation

把 ACP delegation 当作“一次性外包执行”能力，而不是长期对话能力。
只有当任务边界清晰、输入能一次写全、结果能直接回收进当前会话时，才使用它。

## 何时使用

- 需要外部 agent 做一次性代码分析、review、实现建议或定向调研
- 用户明确指定要用某个 runner
- 任务可以写成完整 prompt，不依赖多轮追问或长期上下文
- 你已经确认当前环境存在可用 runner

## 何时不要使用

- 任务非常简单，自己直接完成更快
- 任务需要状态延续、多轮追问或长期 session
- 输出强依赖当前会话里的隐式上下文
- 当前还没有确认 runner 可用
- 用户只是点名 `opencode`、`qwen` 或 `gemini`，但你还没核实可用性

## 调用前检查

在调用 `spawn_agent` 前，先确认两件事：

1. `spawn_agent` 工具已启用
2. 当前环境里存在可用 runner。优先顺序是：

- 代码内置 runner 预设
- 当前 agent 的 `agent.json` 里的覆盖配置

如果你需要检查 workspace 覆盖配置，可关注：

```json
{
  "spawn_agent": {
    "runners": {
      "<agent_type>": {
        "enabled": true
      }
    }
  }
}
```

如果你无法确认 runner 可用：

- 先查看当前 agent 配置或可用 runner 信息
- 不要盲目调用 `spawn_agent`
- 先告诉用户当前 runner 尚未就绪，或暂时无法确认可用性

这些 runner 名字是常见示例。即使没有 workspace 级覆盖配置，代码也可能已经提供了默认 runner 预设。

## 选择 runner

先看用户是否明确指定，再看任务目标，最后看当前可用的 `agent_type`。

默认决策顺序：

1. 优先选择用户明确指定的 runner
2. 否则选择当前环境里最匹配任务的 runner
3. 若没有明显匹配，则不要强行委派
4. 若用户指定的 runner 不可用，先指出 runner 尚未就绪，不要伪造调用

## 编写 delegation prompt

每次委派都要把 prompt 写成完整、自包含的一次性任务说明。至少写清楚：

- 目标：希望外部 agent 完成什么
- 范围：涉及哪些文件、目录、模块、接口或限制范围
- 约束：哪些东西不能改、不能假设、不能忽略
- 输出：希望它返回什么，格式是什么

推荐顺序：

1. 先说明任务目标
2. 再限定范围和约束
3. 最后说明输出格式

不要把当前会话里的隐式背景留给对方猜。后续如果还要再次委派，也要重新提供完整上下文。

## 推荐输出格式

按任务类型要求外部 agent 返回明确结果：

- 代码分析：问题列表 + 风险判断 + 涉及文件
- review：按严重程度列 findings
- 实现建议：修改思路 + 受影响文件 + patch 建议
- 调研：结论 + 依据 + 未确认项

## 最小工作流

1. 判断任务是否适合一次性委派
2. 确认 `spawn_agent` 已启用
3. 确认目标 runner 可用
4. 选择合适 runner
5. 写完整 delegation prompt
6. 调用 `spawn_agent`
7. 把结果回收进当前会话并继续推进

## 失败处理

如果 `spawn_agent` 返回失败、未认证、环境冲突或其他 runner 级错误：

- 先把失败原因原样解释给用户
- 明确指出是外部 runner 失败，不要伪装成委派已经成功
- 如果错误提示要求登录、认证或修复环境配置，就直接告诉用户先处理这些前置条件
- 不要把失败说成“我来帮你继续完成”并偷偷切换成自己本地执行
- 不要在未经用户同意的情况下，自动改用另一个 runner
- 不要在未经用户同意的情况下，自动退化成 `execute_shell_command`、`read_file` 或其他本地工具来冒充 delegation 结果

只有在用户明确同意后，才能改成：

- 使用另一个已配置 runner
- 改为你自己直接完成任务
- 改为使用本地工具继续分析

如果 `spawn_agent` 已经明确返回可操作提示，例如：

- `Please run qwen auth`
- `runner is not authenticated`
- `environment overrides are not compatible`

就优先转述这些提示，并暂停 delegation 流程，等待用户决定下一步。

## 重要限制

当前 MVP 只有 one-shot 语义：

- 不保留持久 session
- 不支持 `send_agent`
- 不支持 `wait_agent`
- 不支持 `cancel_agent`

这意味着你不能假设外部 runner 记得之前的对话。后续如果还要继续委派，必须重新提供完整上下文。

## 当前实现方式

当前 CoPaw 内部通过 `spawn_agent` 调用可用的 ACP-compatible runner。
把 `spawn_agent` 视为底层执行入口即可。重点先放在：

- 任务是否值得委派
- runner 是否可用
- prompt 是否完整

只有这三点都成立时，再真正执行 delegation。
