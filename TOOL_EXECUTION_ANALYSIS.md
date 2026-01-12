# 工具执行机制分析 / Tool Execution Mechanism Analysis

## 中文版本

### 摘要

本文档分析了 LoopTool 项目中工具调用的执行机制。**结论：LoopTool 项目中的工具调用是通过大语言模型（LLM）模拟实现的，而非在真实的沙盒环境中执行。**

### 详细分析

#### 1. 工具执行流程

在 LoopTool 项目中，整个对话生成流程采用多智能体模拟框架，包括以下角色：

- **Planner（规划者）**：生成对话计划
- **User（用户）**：模拟用户查询
- **Assistant（助手）**：生成工具调用决策
- **Tool（工具）**：**模拟工具执行结果**

关键发现：所有这些角色都是由 LLM 扮演的，包括"工具"角色。

#### 2. 核心证据

##### 2.1 工具角色的系统提示词

在 `dialog_generation/prompts/tool_prompts.py` 文件中：

```python
tool_role_system_prompt = "你的任务是扮演工具，给定工具的定义和相应的工具调用语句，你需要根据你自身知识模拟工具返回结果，然后以特定格式返回给用户。"

tool_role_en_system_prompt = "Your task is to act as the tool. Given the tool definitions and the corresponding tool invocation statements, you need to simulate the tool's return results based on your own knowledge and then return them to the user in a specific format."
```

**关键词**：
- "扮演工具" (act as the tool)
- "根据你自身知识模拟" (simulate based on your own knowledge)
- "模拟工具返回结果" (simulate the tool's return results)

这明确表明工具不是实际执行的，而是由 LLM 基于其训练知识来模拟结果。

##### 2.2 工具模拟提示词

```python
tool_mock_prompt = '''以下是工具定义：
[tool_defs]

你接受到的工具调用语句如下：
[tool_calls]

请用以下Json格式返回你模拟工具生成的返回结果，其中每个列表元素为一次调用的返回。注意不要总是返回预期的正向结果，可根据情况模拟返回非预期结果、异常或错误信息等。确保可以用json.loads读取整个结果，除此之外不要任何额外的分析和解释：
[
    {
        "name": ...,
        "results": ...
    },
    ...
]'''
```

这个提示词明确要求 LLM：
1. 模拟工具生成返回结果
2. 可以模拟返回错误、异常等情况
3. 以 JSON 格式返回模拟结果

##### 2.3 工具调用实现

在 `dialog_generation/tool_call.py` 中：

```python
def moc_tool_call(tool_defs, parameters, moc_prompt=None, sys_prompt=None, curr_time=None, used_model=None, try_num=5):
    """
    tool_defs：工具定义，每一行是一个工具的定义的json.dumps，每个工具定义默认为符合json schema的dict
    parameters：入参，list of dict，每个list元素为一次工具调用
    curr_time：数据的当前时间，如无则为None
    used_model: 使用什么gpt模型，默认为gpt-4-turbo-2024-04-09
    try_num：尝试爬取最多次数，如超过仍无法爬取成功返回None
    """
    messages = convert_messages_for_tool_role(tool_defs, parameters, moc_prompt, sys_prompt, curr_time)
    if used_model is None:
        used_model = "gpt-4-turbo-2024-04-09"
    success, valid, response = call_gpt(used_model, messages, tools=None, try_num=try_num)
    if success and valid:
        response_message = response['choices'][0]['message']
        if response_message is None or response_message["content"] is None:
            return None
        return convert_gpt_tool_role_output_to_toolace(response_message)
    else:
        return None
```

函数名 `moc_tool_call` 中的 "moc" 很可能是 "mock"（模拟）的缩写，进一步证实了工具调用是模拟的。

##### 2.4 数据爬取流程

在 `dialog_generation/data_crawl.py` 的 `InteractiveCrawlThread` 类中：

```python
elif messages[-1]["role"] == "assistant" and "tool_usage" in messages[-1]:  # 应是工具轮次
    current_role = "tool"
    if row.is_en:
        new_messages = row.convert_messages_for_tool_role(
            tool_prompts.tool_role_en_system_prompt
        )
    else:
        new_messages = row.convert_messages_for_tool_role(
            tool_prompts.tool_role_system_prompt
        )
    gpt_tools = None
```

当助手生成工具调用后，系统将当前角色设置为 "tool"，并通过调用 GPT API 来模拟工具的返回结果。

#### 3. 为什么采用 LLM 模拟？

这种设计有几个原因：

1. **数据生成灵活性**：LoopTool 的核心目标是生成高质量的工具调用训练数据。使用 LLM 模拟可以快速生成大量多样化的对话样本。

2. **避免真实 API 依赖**：真实执行工具需要：
   - 实际的 API 密钥和凭证
   - 处理 API 限流和成本
   - 管理复杂的错误情况
   - 可能涉及真实的数据修改（如发送邮件、创建订单等）

3. **可控性**：LLM 可以模拟各种场景，包括成功、失败、异常等情况，这对于生成鲁棒的训练数据很重要。

4. **可扩展性**：支持 1.2 万个工具的定义（来自 ToolBench），无需为每个工具实现真实的执行逻辑。

#### 4. 与真实工具执行的对比

| 特性 | LoopTool (LLM 模拟) | 真实沙盒执行 |
|------|---------------------|-------------|
| 执行方式 | LLM 基于知识生成结果 | 实际调用 API 或执行代码 |
| 结果真实性 | 模拟的、推测的 | 真实的 API 返回 |
| 成本 | 仅 LLM 推理成本 | LLM + API 调用成本 |
| 依赖 | 无需真实 API | 需要 API 密钥、网络访问 |
| 副作用 | 无 | 可能有（如数据修改） |
| 可控性 | 高（可模拟任意场景） | 受 API 限制 |

#### 5. 适用场景

LoopTool 的 LLM 模拟方法适合：
- **训练数据生成**：快速生成大量工具调用对话样本
- **模型微调**：为强化学习训练提供监督信号
- **评估基准测试**：在 BFCL-v3 和 ACEBench 上评估模型的工具调用能力

不适合：
- **生产环境部署**：需要真实工具执行结果的应用
- **真实任务完成**：需要与外部系统实际交互的场景

### 结论

LoopTool 项目采用 **LLM 模拟工具执行** 的方式，而非在真实沙盒环境中执行工具。这是一个刻意的设计选择，目的是高效地生成高质量的工具调用训练数据，用于训练和优化 LLM 的工具使用能力。该方法在数据生成阶段非常有效，但在实际部署时，训练出的模型仍需连接真实的工具执行环境。

---

## English Version

### Summary

This document analyzes the tool execution mechanism in the LoopTool project. **Conclusion: Tool calls in the LoopTool project are simulated by Large Language Models (LLMs), not executed in a real sandbox environment.**

### Detailed Analysis

#### 1. Tool Execution Workflow

In the LoopTool project, the entire dialogue generation process uses a multi-agent simulation framework, including the following roles:

- **Planner**: Generates conversation plans
- **User**: Simulates user queries
- **Assistant**: Generates tool call decisions
- **Tool**: **Simulates tool execution results**

Key finding: All these roles are played by LLMs, including the "Tool" role.

#### 2. Core Evidence

##### 2.1 Tool Role System Prompt

In the `dialog_generation/prompts/tool_prompts.py` file:

```python
tool_role_system_prompt = "你的任务是扮演工具，给定工具的定义和相应的工具调用语句，你需要根据你自身知识模拟工具返回结果，然后以特定格式返回给用户。"

tool_role_en_system_prompt = "Your task is to act as the tool. Given the tool definitions and the corresponding tool invocation statements, you need to simulate the tool's return results based on your own knowledge and then return them to the user in a specific format."
```

**Keywords**:
- "act as the tool"
- "simulate based on your own knowledge"
- "simulate the tool's return results"

This clearly indicates that tools are not actually executed, but rather simulated by the LLM based on its training knowledge.

##### 2.2 Tool Mock Prompt

```python
tool_mock_en_prompt = '''Here are the tool definitions:  
[tool_defs]  

The tool invocation statements you received are as follows:  
[tool_calls]  

Please return the simulated tool results in the following JSON format, where each list element represents the result of one invocation. Be careful not to always return the expected positive results; depending on the situation, you can simulate returning unexpected results, exceptions, or error messages, etc. Ensure that the entire result can be read using `json.loads`, and provide no additional analysis or explanation:  
[
    {
        "name": ...,
        "results": ...
    },
    ...
]'''
```

This prompt explicitly asks the LLM to:
1. Simulate tool-generated return results
2. Simulate errors, exceptions, and other scenarios
3. Return simulated results in JSON format

##### 2.3 Tool Call Implementation

In `dialog_generation/tool_call.py`:

```python
def moc_tool_call(tool_defs, parameters, moc_prompt=None, sys_prompt=None, curr_time=None, used_model=None, try_num=5):
    """
    tool_defs: Tool definitions, each line is a json.dumps of a tool definition
    parameters: Input parameters, list of dict, each element is one tool call
    curr_time: Current time for the data, None if not available
    used_model: Which GPT model to use, default is gpt-4-turbo-2024-04-09
    try_num: Maximum number of attempts, return None if still unsuccessful
    """
    messages = convert_messages_for_tool_role(tool_defs, parameters, moc_prompt, sys_prompt, curr_time)
    if used_model is None:
        used_model = "gpt-4-turbo-2024-04-09"
    success, valid, response = call_gpt(used_model, messages, tools=None, try_num=try_num)
    if success and valid:
        response_message = response['choices'][0]['message']
        if response_message is None or response_message["content"] is None:
            return None
        return convert_gpt_tool_role_output_to_toolace(response_message)
    else:
        return None
```

The function name `moc_tool_call` where "moc" likely stands for "mock" (simulate), further confirming that tool calls are simulated.

##### 2.4 Data Crawl Workflow

In the `InteractiveCrawlThread` class in `dialog_generation/data_crawl.py`:

```python
elif messages[-1]["role"] == "assistant" and "tool_usage" in messages[-1]:  # Tool turn
    current_role = "tool"
    if row.is_en:
        new_messages = row.convert_messages_for_tool_role(
            tool_prompts.tool_role_en_system_prompt
        )
    else:
        new_messages = row.convert_messages_for_tool_role(
            tool_prompts.tool_role_system_prompt
        )
    gpt_tools = None
```

When the assistant generates a tool call, the system sets the current role to "tool" and calls the GPT API to simulate the tool's return results.

#### 3. Why Use LLM Simulation?

This design has several reasons:

1. **Data Generation Flexibility**: LoopTool's core goal is to generate high-quality tool call training data. Using LLM simulation allows rapid generation of large amounts of diverse conversation samples.

2. **Avoid Real API Dependencies**: Real tool execution requires:
   - Actual API keys and credentials
   - Handling API rate limits and costs
   - Managing complex error conditions
   - Potential real data modifications (e.g., sending emails, creating orders)

3. **Controllability**: LLMs can simulate various scenarios, including success, failure, and exceptions, which is important for generating robust training data.

4. **Scalability**: Supports 12,000 tool definitions (from ToolBench) without needing to implement real execution logic for each tool.

#### 4. Comparison with Real Tool Execution

| Feature | LoopTool (LLM Simulation) | Real Sandbox Execution |
|---------|---------------------------|------------------------|
| Execution Method | LLM generates results based on knowledge | Actually calls APIs or executes code |
| Result Authenticity | Simulated, speculative | Real API returns |
| Cost | Only LLM inference cost | LLM + API call costs |
| Dependencies | No real APIs needed | Requires API keys, network access |
| Side Effects | None | Possible (e.g., data modifications) |
| Controllability | High (can simulate any scenario) | Limited by API constraints |

#### 5. Use Cases

LoopTool's LLM simulation approach is suitable for:
- **Training Data Generation**: Quickly generate large amounts of tool call dialogue samples
- **Model Fine-tuning**: Provide supervision signals for reinforcement learning training
- **Benchmark Evaluation**: Evaluate model tool call capabilities on BFCL-v3 and ACEBench

Not suitable for:
- **Production Deployment**: Applications requiring real tool execution results
- **Real Task Completion**: Scenarios requiring actual interaction with external systems

### Conclusion

The LoopTool project uses **LLM simulation of tool execution** rather than executing tools in a real sandbox environment. This is a deliberate design choice aimed at efficiently generating high-quality tool call training data for training and optimizing LLM tool use capabilities. This approach is very effective during the data generation phase, but when deployed in practice, the trained model still needs to be connected to a real tool execution environment.

---

## References

### Key Files Analyzed

1. **dialog_generation/prompts/tool_prompts.py** - Contains system prompts for tool role
2. **dialog_generation/tool_call.py** - Implements tool call simulation logic
3. **dialog_generation/api_call.py** - Handles GPT API calls
4. **dialog_generation/data_crawl.py** - Orchestrates multi-agent dialogue generation
5. **dialog_generation/utils.py** - Utility functions for format conversion

### Project Architecture

```
LoopTool Data Generation Pipeline:
┌─────────────────────────────────────────────────────┐
│  1. Planner (LLM) → Generates conversation plan     │
│  2. User (LLM) → Generates user query               │
│  3. Assistant (LLM) → Generates tool call decision  │
│  4. Tool (LLM) → Simulates tool execution result    │ ← Key Finding
│  5. Assistant (LLM) → Summarizes final response     │
└─────────────────────────────────────────────────────┘
         ↓
   Training Data for GRPO
         ↓
   Fine-tune LLM for Tool Use
         ↓
   Deploy Model (needs real tool execution)
```

### Implications for Users

If you're using LoopTool:
- **For Training**: The simulated approach is perfect for generating diverse training data
- **For Deployment**: You'll need to integrate real tool execution environments
- **For Evaluation**: The models trained with LoopTool can be evaluated on standard benchmarks (BFCL, ACEBench)

The project demonstrates that high-quality tool call training data can be generated through careful LLM simulation, which is validated by the strong performance of LoopTool-8B and LoopTool-32B on various benchmarks.
