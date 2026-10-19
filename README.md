# GPT-4 on GRPO

GPT-4's training data ends before DeepSeekMath introduced Group Relative Policy
Optimization (GRPO). I described GRPO to GPT-4 using the paper's own method
text, with the name removed, and asked it two questions:

1. **Predict the result.** Given a 7B math model after instruction tuning and its
   baseline scores, predict its scores after this RL stage on eight benchmarks.
2. **Give its take.** How does this algorithm compare with actor-critic methods
   that learn a value function, such as PPO with GAE?

The two user prompts and GPT-4's two replies are published here verbatim.

| | Prompt | Reply |
|---|---|---|
| 1. Predictions | [prompt.txt](1-predictions/prompt.txt) | [reply.txt](1-predictions/reply.txt) |
| 2. Take vs PPO | [prompt.txt](2-take-vs-ppo/prompt.txt) | [reply.txt](2-take-vs-ppo/reply.txt) |

[TRANSCRIPT.md](TRANSCRIPT.md) shows all four in one page.

## Setup

- **Model:** `openai/gpt-4` through OpenRouter, temperature 0.4, web search off.
- **Date:** 19 October 2026. One sample per question.
- **Name removed:** neither prompt contains "GRPO" or "Group Relative".
- **Independent calls:** the second call is a fresh conversation. It saw only
  the method text, with no setup numbers, no results, and nothing from the first call.
- **System prompt:** the first call also had a fixed system prompt that set the
  answer format, a closing JSON block of predictions. It is not included here.

## Source

The method text in both prompts is quoted from:

> Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei
> Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, Daya Guo. *DeepSeekMath: Pushing the
> Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300, 2024.
> https://arxiv.org/abs/2402.03300

The quoted passages remain the authors' work.
