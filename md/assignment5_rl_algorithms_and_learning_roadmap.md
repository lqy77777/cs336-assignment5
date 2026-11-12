# 从 Assignment 5 到 PPO、GRPO 与 Agentic RL：概念、实现与学习路线

整理日期：2026-11-10。

范围：先核对当前作业实际实现，再解释 PPO、GRPO 及相关方法的关系，最后给出可以逐步完成的项目和论文阅读路线。本文是知识梳理与学习计划，不把未来建议写成已经实现的功能。

对应代码：[`tool.py`](../cs336_alignment/tool.py)、[`training_step.py`](../cs336_alignment/training_step.py)、[`tuning_grpo_varients.py`](../cs336_alignment/tuning_grpo_varients.py)、[`tests/test_grpo.py`](../tests/test_grpo.py)。

本地依据：[主作业讲义](../cs336_spring2026_assignment5_alignment.pdf)、[可选 SFT/DPO 补充作业](../cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)、[2026-11-06 实验总结](../summary/grpo_variants_results_2026-11-06.md)。外部论文和项目链接在本次整理时核对；框架接口和默认配置以后可能变化。

本次只读取代码、运行现有测试并编写这份文档，没有修改代码，也没有启动模型训练。

## 1. 先校准：这份作业究竟在训练什么

你说“使用 GRPO，让模型解数学题的能力增强”，这个训练目标基本正确。更准确的描述是：

> 我正在完成 CS336 Assignment 5 的数学推理强化学习主线：以 OLMo-2-0425-1B 为基础模型，在 GSM8K 上使用可验证奖励，通过 GRPO 及相关策略优化方法，提高模型生成正确答案的概率。

这里有三个需要区分的概念。

### 1.1 GRPO 是更新算法，Agentic RL 是交互场景

| 名称 | 它回答什么问题 | 在当前作业中的对应关系 |
|---|---|---|
| 强化学习 RL | 如何通过奖励改善决策行为？ | 根据回答奖励调整生成策略 |
| PPO、GRPO、GSPO | 如何估计优势、构造目标、更新策略？ | 训练算法与目标函数的选择 |
| RLVR | 奖励是否来自可以验证的规则或结果？ | 根据数学答案等规则判分 |
| RLHF | 是否利用人类反馈或由此学出的偏好信号？ | 当前数学主线没有要求训练人类偏好奖励模型 |
| Reasoning RL | 要改善什么能力？ | 数学解题、推理任务表现 |
| Agentic RL | 模型是否在行动、观察、工具反馈的交互闭环中学习？ | 当前单轮回答主线还没有这样的工具交互循环 |

这些概念可以组合。例如：“用 GRPO 训练一个调用计算器的 agent，并用最终答案正确性给奖励”，可以同时属于 GRPO、工具交互式 Agentic RL 和 RLVR。

广义 RL 中，生成 token 的模型当然可以被看作 agent；但在当下 LLM 语境里，Agentic RL 通常特别强调多轮行动与环境反馈。你当前的“问题 → 生成完整回答 → 判分”更适合称为 reasoning RL / RLVR。官方 Agentic RL 实例可以参考 [verl 的多轮工具训练文档](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)。

### 1.2 文件名里的 alignment，比这次必做主线更宽

主讲义重点是 reasoning RL。仓库的可选 supplement 才进一步涉及指令微调、偏好数据、对话质量和安全评估，要求实现 SFT 和 DPO。

因此，完成数学 RL 主线，说明你掌握了一条重要的后训练路线；它不等同于完整覆盖通用助手对齐、偏好建模、安全行为和工具使用训练。

### 1.3 “训练目标是提升能力”与“已经证明提升”要分开

单元测试验证的是实现是否符合预期。数学能力是否提高，需要比较训练前后的独立评估表现，并保持解码方式、提示模板和评估题目一致。训练 reward 上升不能单独证明泛化推理能力变强。

你的历史总结记录了五种方法、每种三个种子的数学实验。这是已有实验记录；本次没有重新读取远端 W&B 原始记录，也没有重跑训练。评价某种方法优劣时，应保留模型规模、数据、训练预算和种子数这些适用条件。

## 2. 结合当前代码：已经完成什么，还差什么

你当前实现的训练主线是：

```text
抽取 B 个数学问题
    → 每题生成 G 个回答，共 N = B × G 条
    → 判分
    → 根据同题回答的奖励计算优势
    → 计算当前模型对已生成 token 的 log 概率
    → 构造逐 token 或序列级策略目标
    → mask 和归一化
    → backward 累积梯度
    → optimizer.step 更新参数
```

### 2.1 函数和数据怎样对应

| 当前函数或变量 | 输入/输出与职责 |
|---|---|
| `tokenize_prompt_and_output` | 把文本变成对齐的 `input_ids`、`labels`、`response_mask`，形状为 `(N, S)` |
| `compute_rollout_rewards` | 每条回答得到一个奖励，返回长度为 N 的奖励张量与日志数据 |
| `compute_group_normalized_rewards_grpo` | 按每题 G 个回答组织奖励，计算每条回答的优势 `(N,)` |
| `get_response_log_probs` | 用当前模型打分，取实际 token 的 log 概率，形状为 `(N, S)` |
| `compute_policy_gradient_loss_on_policy` | 名字仍带 `on_policy`，但当前已有 `none`、`noclip`、`grpo`、GSPO 分支，返回 loss 与 metadata |
| `aggregate_loss_across_microbatch_sequence` | 排除 prompt/padding，再按选择的方式聚合 loss |
| `grpo_train_step` | 组织 microbatch、反向传播、梯度裁剪和一次参数更新 |

这里 S 是对齐后的序列宽度；每条回答实际有效 token 数为 `response_mask.sum(-1)`，不一定等于 S。当前 GSPO 先计算 `(N, 1)` 的序列 loss，再通过 `repeat` 扩展成 `(N, S)`，适配现有接口。

这些实现已经把几个可以独立研究的设计拆开：优势 baseline、优势归一化、重要性权重、clipping、loss 聚合方式。以后阅读论文时，可以先问“它修改了这里的哪个部分”。

### 2.2 本次实际验证结果

本次执行：

```bash
.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_grpo.py -q --tb=short
```

结果是 **16 项通过，3 项失败**。

| 检查范围 | 本次结果 |
|---|---|
| tokenization、模型 log 概率、reward、优势计算、loss 聚合 | 相关测试通过 |
| `none`、`noclip`、`grpo`、`gspo` 的独立 loss 计算 | 相关测试通过 |
| 标准 on-policy 与现有 on-policy 变体训练步骤 | 相关测试通过 |
| `test_grpo_train_step_off_policy[noclip]` | 失败 |
| `test_grpo_train_step_off_policy[grpo]` | 失败 |
| `test_grpo_train_step_off_policy[gspo]` | 失败 |

三个失败有相同原因：`training_step.py` 中当前 log 概率已经按照 microbatch 切成 2 条回答，但传给 loss 的 `old_log_probs` 仍包含整批 4 条回答，减法时发生第一维不匹配。

这里还需要让旧概率和当前输入执行相同的 microbatch 切片、`keep` 过滤与设备/位置对齐。旧概率也需要保持为固定数据。本文只指出问题，没有应用修复。

### 2.3 “基本掌握核心 loss”与“完成整份作业”之间还有哪些验收

从当前 `training_step.py` 的入口看，主循环仍然每批 rollout 调用一次训练步骤，调用时没有传入 off-policy 配置和缓存的旧概率。这是静态阅读结论，不代表其他未检查位置绝对不存在实验。

完成主作业还需要逐项核对：

1. Section 6.4：接通旧策略 log 概率、同一批 rollout 的多次更新、microbatch 对齐，并通过 off-policy train-step 测试。
2. Section 6.4：比较 naive、noclip、GRPO clipping、GSPO；讲义要求每种四个随机种子，记录 clip fraction，并提交曲线与解释。
3. Section 7：提出一个相对已有方法只改变一项设计的估计器，说明来源、理论或直觉，进行多种子对照。
4. 核对全部书面题、实验设置、种子数、图表与最终提交要求。

没有在本次检查中验证的实验，不能写成已经完成。当前准确的进度描述是：“核心 loss 与 on-policy 训练测试已通过，正在完成 off-policy 训练集成及后续实验验收。”

## 3. 从你熟悉的 GRPO 回到 RL 的共同基础

### 3.1 模型在 RL 中对应什么

| RL 概念 | 普通控制任务 | 当前语言模型任务 |
|---|---|---|
| 状态 $s$ | 当前环境观测 | 问题和已生成前缀 |
| 动作 $a$ | 左移、右移等 | 下一个 token |
| 策略 $\pi$ | 各动作的概率 | 模型的 next-token 分布 |
| 轨迹 $\tau$ | 一串状态与动作 | 完整回答的生成过程 |
| 奖励 $r$ | 环境反馈 | 答案/格式规则产生的分数 |
| 回报 | 后续奖励的累积 | 终局奖励或其他奖励的累积 |

你的作业主要使用回答结束后的奖励；因此需要把整条回答的结果转换成各 token 的训练信号。

### 3.2 为什么推导梯度后，还要调用 backward

目标是当前策略下的期望奖励：

$$
J(\theta)=\mathbb E_{y\sim\pi_\theta}[r(y)].
$$

离散回答采样和判题通常没有直接可微的路径。对一批已经生成的奖励取平均，得到的是目标值的估计，不自动得到正确梯度。

利用概率的 log 导数恒等式，可以把梯度写成：

$$
\nabla_\theta J
=\mathbb E_{y\sim\pi_\theta}
[r(y)\nabla_\theta\log\pi_\theta(y)].
$$

这告诉我们，在固定采样结果后，可以构造 `-reward * log_prob` 这样的可微 loss，让自动求导产生需要的估计器。推导决定应该对什么求导；`backward()` 执行具体的链式法则。换成 off-policy 权重后，要重新检查构造出的 loss 对应哪个目标。

### 3.3 四个常见的“模型角色”

| 角色 | 做什么 | 容易混淆的地方 |
|---|---|---|
| Actor / policy | 生成动作或 token | 就是你当前训练的语言模型策略 |
| Critic / value function | 估计某个状态下的未来回报 | 不等于给完整回答判分的奖励模型 |
| Reward model / verifier | 产生奖励 | 可以是学出来的偏好评分器，也可以是本作业的规则判题器 |
| Reference policy | 作为 KL 正则的参照 | 不等于每批 rollout 的采样旧策略 |

旧策略 $\pi_{\mathrm{old}}$ 记录“这批数据从哪里来”；参考策略 $\pi_{\mathrm{ref}}$ 记录“希望相对于哪个模型约束偏移”。二者可能采用不同更新节奏。

## 4. PPO 是什么：你已经认识 clipping，还需要补 critic 和 GAE

PPO 是 Proximal Policy Optimization。下面主要讨论常用的 PPO-Clip。它通过一个保守的替代目标，减少单批数据推动策略过度变化的激励。原论文也讨论 KL penalty 版本；PPO-Clip 不要求必须在目标里额外加入 KL 项。[PPO 原论文](https://arxiv.org/abs/1707.06347)

### 4.1 先认识 PPO 中新增的变量

| 符号 | 含义 |
|---|---|
| $\pi_\theta(a_t \mid s_t)$ | actor 在状态 $s_t$ 下选择动作 $a_t$ 的概率 |
| $V_\phi(s_t)$ | critic 预测从 $s_t$ 开始能获得的期望未来回报 |
| $r_t$ | 当前动作对应的奖励 |
| $\gamma$ | 折扣因子，控制未来奖励的权重 |
| $\lambda$ | GAE 参数，控制不同时间跨度的估计如何混合 |
| $\delta_t$ | 一步 TD 残差 |
| $\widehat A_t$ | 估计出来的优势，作为 actor 更新系数 |
| $w_t$ | 当前策略与 rollout 旧策略对实际动作的概率比 |

典型 PPO 使用 actor 和 critic，但 critic 不一定是另一个完整同规模模型：可以是独立网络，也可以与 actor 共享部分网络、使用 value head。是否共享会影响工程成本和梯度处理。

### 4.2 Critic 帮我们判断“比预期好多少”

一个直观的优势近似是：

$$
\widehat A_t=\widehat R_t-V_\phi(s_t),
$$

其中 $\widehat R_t$ 是实际回报或构造的回报目标。

例如终局奖励为 1、折扣为 1。在一个困难前缀下，critic 原本只预期 0.2 的回报，成功带来的改进约为 0.8；在另一个几乎已经解决问题的前缀下，critic 预期 0.9，改进约为 0.1。这里仅用回报减 baseline 说明直觉，实际 GAE 的数值还取决于后续 TD 残差。

这提供了逐状态的 baseline。对于 LLM，状态就是不同的文本前缀，因此同一条回答的各 token 可以有不同优势。

### 4.3 GAE 怎样形成优势

GAE 是 Generalized Advantage Estimation，是一种优势估计方法，不是单独的完整训练算法。用 $m_t$ 表示该步是否允许从下一状态 bootstrap，在最简单的真实终止处理下：

$$
\delta_t=r_t+\gamma m_tV_\phi(s_{t+1})-V_\phi(s_t),
$$

$$
\widehat A_t=\delta_t+\gamma\lambda m_t\widehat A_{t+1}.
$$

沿轨迹反向递推，即可把多步信息结合起来。较短时间跨度更依赖价值预测，较长跨度更多使用实际后续回报，这形成偏差与方差的取舍。实际环境还要区分真正终止与时间截断，不能一律把下一状态价值置零。[GAE 原论文](https://arxiv.org/abs/1506.02438)

时间截断时，TD 残差可以使用截断前最终观测的价值进行 bootstrap；但优势递推不能跨越 reset，接到下一条新 episode 的优势。实际实现可能需要分别处理 bootstrap mask 与轨迹连续性 mask，以上共用 $m_t$ 的公式仅用于说明简单的真实终止情形。环境返回值含义可对照 [Gymnasium API](https://gymnasium.farama.org/introduction/basic_usage/)。

### 4.4 Actor 更新用的 clipping，你已经学过

$$
w_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\rm old}(a_t\mid s_t)},
$$

$$
J_{\rm clip}(\theta)
=\mathbb E_t\left[
\min\left(w_t\widehat A_t,
\operatorname{clip}(w_t,1-\epsilon,1+\epsilon)\widehat A_t\right)
\right].
$$

与作业 Section 6.2 相同：正优势在上侧进入平台区，负优势在下侧进入平台区。Clipping 改变优化激励，不保证真实概率比或 KL 严格被限制在某个范围内。

典型实现同时训练 critic，让它回归固定的回报目标；还可以加入熵项鼓励探索。actor loss 中的优势一般作为固定估计量使用，不通过优势反传去优化 critic。采样、优势估计、actor 更新、value 回归之间的分工，可对照 [Spinning Up 的 PPO 伪代码](https://spinningup.openai.com/en/latest/algorithms/ppo.html)。

### 4.5 一个完整 PPO 循环

```text
当前策略收集一批轨迹，保存旧 log 概率和 value
    → 根据 reward、value、终止信息计算 return/GAE
    → 固定这批数据中的优势与训练目标
    → 分 minibatch 更新 actor 的 clipped objective
    → 同时训练 critic 的 value loss
    → 重复若干轮，但监控策略偏移
    → 重新收集轨迹
```

你在 GRPO 中已经掌握了策略 log 概率、概率比、clipping 和 microbatch。学习 PPO 最值得补的部分是：价值函数、回报目标、bootstrap、GAE，以及 actor 与 critic 的训练关系。

## 5. PPO 与 GRPO 的关系

原始 GRPO 在 DeepSeekMath 中提出。其 outcome-supervision 版本用同一道题的一组回答奖励估计相对优势，从而省去 learned critic；保留了 PPO 风格的 token 级裁剪目标。原论文的完整目标还包含相对于参考策略的 KL 正则。[DeepSeekMath §4](https://arxiv.org/html/2402.03300v3#S4)

| 比较项 | 典型 PPO | 原始 outcome-supervision GRPO |
|---|---|---|
| baseline 从哪里来 | 训练出来的 value function | 同题多条回答的组内奖励均值 |
| 优势怎样计算 | 常用 GAE，也可使用其他估计器 | 通常 `(r − 组均值) / 组标准差` |
| 是否需要 learned critic | 常见实现需要 | 不需要 |
| 同一回答的 token 优势 | 可以随状态/前缀变化 | 通常共享一个回答优势 |
| 是否需要同题多回答 | 不是必要的结构要求 | 用同题多回答形成组内比较 |
| 策略更新 | token/action 概率比与 clipping | 类似的 token 概率比与 clipping |
| 主要额外成本 | value 预测与学习 | 同题多回答的生成成本 |
| 对参考模型的 KL | LLM RLHF 常加入，非 PPO-Clip 必备 | 原始论文有，本作业主线省略 |

因此，“GRPO 没有 critic”不等于“GRPO 不需要奖励”，也不保证每个设置下都比 PPO 更便宜或效果更好。

### 5.1 对照你自己的优势计算

你的标准配置是：

$$
A_i=\frac{r_i-\bar r}{\operatorname{std}(r)+\varepsilon_A}.
$$

这里的 $\varepsilon_A$ 对应 `advantage_eps`，用于数值稳定；它与 clipping 的 `cliprange` 是不同参数。

例如一题有四个回答，奖励是 `[1, 0, 1, 0]`。沿用你当前 `torch.std()` 的默认样本标准差口径，均值是 0.5，标准差约为 0.577；忽略很小的稳定项，优势约为 `[0.866, -0.866, 0.866, -0.866]`。每个回答内部的 token 共享该回答的优势。

组内全对或全错时，减均值后优势全为零。这解释了你的实现为什么能跳过零优势回答，也解释了研究动态采样等方法的动机。优化必须保持原本的归一化口径，否则“跳过计算”也会改变目标。

### 5.2 补充前面对 on-policy/off-policy 的说法

经典 RL 文献通常把 PPO 归为 **on-policy** 算法：它围绕最新采集的轨迹训练，然后重新采样；同一批数据内部仍可做多轮 minibatch 更新。

本作业 Section 6 则强调：第一次更新后，当前参数已经变化，剩余样本来自旧策略，因此从严格的分布一致性角度称其为 off-policy 更新。

两种说法的关注层次不同。不要因此把 PPO 理解成可以像 DQN/SAC 那样，直接使用长期 replay buffer 中任意久远的数据。读文章时要先确认作者对“off-policy 程度”的定义。[PPO 文档的分类与训练说明](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

## 6. RL 算法的地图：先分路线，再看变体

不用把所有算法名字背成一条进化链。先看它在学什么：策略、价值函数，还是环境模型。算法之间可以组合，下面只是定位工具。[Spinning Up 算法分类](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)

| 路线 | 代表方法 | 核心思路 | 与你当前项目的距离 |
|---|---|---|---|
| 直接策略梯度 | REINFORCE、带 baseline 的 REINFORCE | 用回报或优势给 log 概率梯度加权 | 最近，是你推导过的起点 |
| Actor-critic / 保守策略更新 | A2C/A3C、TRPO、PPO | 用价值估计帮助策略更新，或限制更新幅度 | PPO 是下一步重点 |
| 基于动作价值的学习 | Q-learning、DQN | 学 $Q(s,a)$，再依据 $Q$ 选择动作 | 帮助理解 Bellman backup 与 replay |
| Off-policy actor-critic | DDPG、TD3、SAC | 联合学习策略和 Q，复用历史经验；SAC 加入熵目标 | 原版主要面向连续控制，先了解即可 |
| Model-based RL | 使用已知或学习的转移模型进行规划的方法 | 预测环境后果，再规划或改进策略 | 与当前 LLM policy loss 不是一项小差异 |
| 偏好优化 | DPO 等 | 从偏好对构造直接训练目标 | 适合连接本地 supplement |

TRPO 用信赖域思路约束策略变化，是理解 PPO 动机的一条背景线；可以先理解目的，再决定是否实现其较复杂的优化过程。[TRPO](https://arxiv.org/abs/1502.05477)

DQN 和 SAC 都值得认识，但不是理解 GRPO 必须先复现的算法。前者侧重动作价值学习，后者展示带熵目标、replay 与 Q 学习的另一种 actor-critic 路线。[DQN](https://arxiv.org/abs/1312.5602)、[SAC](https://arxiv.org/abs/1801.01290)

### 6.1 和当前 LLM RL 最相关的方法

| 方法 | 主要改变什么 | 相对你学过的 GRPO，应注意什么 |
|---|---|---|
| Dr. GRPO | 去掉组内标准差归一化，使用固定常数代替按实际回答长度归一化 | 研究题目难度和长度带来的权重偏置；不是删除 reward baseline |
| RLOO | 每条回答以同组其他回答的平均奖励为 baseline | baseline 排除自身，不等于完整算法只是“GRPO 去 std” |
| GSPO | 序列概率比取几何平均，在序列层面 clipping | 几何平均与裁剪改变目标，不是完整序列重要性采样的无偏修正 |
| DAPO | 非对称 clipping、动态采样、token 级聚合、过长回答奖励处理 | 是一组训练设计，单独换一个 loss 分母不等于复现全部 DAPO |
| CISPO | 对梯度估计器中的重要性系数设置上界 | 超出上界后仍可有非零梯度；与 PPO 目标平台区不同 |
| RFT，按本作业配置 | 用原始奖励加权，奖励为零的回答不贡献梯度 | 是当前作业的比较基线；一般的 rejection fine-tuning 还涉及筛选与监督训练流程 |
| MaxRL，按本作业配置 | 用组均值归一化中心化奖励 | 当前系数为 $(r-\mu)/(\mu+\varepsilon_A)$，需要结合固定分母理解其题目权重 |
| DPO | 用 chosen/rejected 偏好对直接优化策略 | 标准 DPO 不要求训练时在线 rollout，不是 GRPO 的 clipping 变体 |

Dr. GRPO、RLOO、GSPO 和 DAPO 的定义分别见 [Dr. GRPO 论文](https://arxiv.org/abs/2503.20783)、[RLOO 论文](https://arxiv.org/abs/2402.14740)、[GSPO 论文](https://arxiv.org/abs/2507.18071)、[DAPO 官方项目](https://dapo-sia.github.io/)。CISPO、RFT、MaxRL 这里按本地主讲义介绍的版本解释。

两点值得单独记住：

1. RLOO 的优势为 $r_i-\frac{1}{G-1}\sum_{j\ne i}r_j$。固定组大小时，它等于 $\frac{G}{G-1}(r_i-\bar r)$，其中 $\bar r$ 是组均值。这是优势系数层面的关系，完整训练方法还受 clipping、KL、归一化等设计影响。
2. Dr. GRPO 讨论去掉特定归一化引入的偏置；如果同时保留 token 级 off-policy 近似或 clipping，不能因此宣称整个估计器在所有条件下都无偏。

### 6.2 DPO 应放在哪里

SFT 学习“给定问题，示范回答是什么”；DPO 学习“给定同一问题，哪个回答更受偏好”。标准 DPO 从偏好数据构造分类式目标，不需要在训练过程中一边生成回答、一边执行策略梯度 rollout，也不需要显式训练奖励模型。

它与 RLHF 的 KL 正则化偏好目标存在理论联系，但工程流程明显不同。你本地 supplement 正好可以用来理解这条分支。[DPO 原论文](https://arxiv.org/abs/2305.18290)

## 7. 怎样学到能独立实现：按项目验收推进

下面的算力等级是按任务结构作出的相对判断，不是本次测量的 GPU 小时。先跑小规模正确性与资源预检，再决定训练预算。已有项目有实验积累，优先复用它。

### 项目 A：完成当前 Assignment 5 的训练闭环

**要做什么：** 先补 Section 6.4 的 old-log-prob 缓存和切片、每批 rollout 多次更新、日志，再完成 Section 7 的估计器对照。

**验收内容：**

- `tests/test_grpo.py` 全部通过；单测通过后仍需真实小规模训练冒烟验证。
- `old_log_probs` 确实来自生成当前 rollout 的策略，在这一批数据的多个更新中不刷新。
- microbatch、零优势过滤和 token 对齐不会把不同回答的数据混在一起。
- 记录准确率、回答长度、熵、clip fraction 与实际耗时；说明 clip fraction 的分母和口径。
- 保持问题分组的优势计算正确，再进行训练 minibatch 的切分。
- 区分同生成 token 预算、同优化步骤数和同实际时间的比较；不同预算回答不同问题。
- 按讲义完成四种 off-policy 配置、每种四个种子；不要只挑最高的一条曲线。

**你会学到：** loss 正确、训练步骤正确、训练调度正确和实验结论可靠，是四个层次。

### 项目 B：在小环境里，从 REINFORCE 写到 PPO

**推荐作为理解 PPO 的下一站。** 用 CartPole 和小 MLP，通常可在本机 CPU 学习，不必先增加一个大语言模型 critic。

按下面顺序实现，每次只增加一个机制：

```text
REINFORCE
    → learned value baseline
    → TD / GAE
    → 保存 old log probability
    → PPO clipping 与 minibatch 多轮更新
```

**验收内容：** 能手算短轨迹的 return 与优势；区分 terminated/truncated；确认 old log probability 固定、优势不向 actor loss 之外错误传导；比较多个种子的回报曲线，并记录 value loss、entropy、approx KL 和 clip fraction。

先写自己的简版，再逐段对照 [CleanRL 的 PPO 实现说明](https://docs.cleanrl.dev/rl-algorithms/ppo/) 和 [官方仓库](https://github.com/vwxyzjn/cleanrl)。环境接口查 [Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) 和 [基本 API](https://gymnasium.farama.org/introduction/basic_usage/)。

**你会学到：** critic 到底预测什么、GAE 如何产生 token/step 级优势，以及为什么 PPO 除 actor loss 外还要训练 value。

### 项目 C：完成本地 SFT/DPO supplement

如果更关心通用助手对齐，可以把它放在项目 B 前面。

**要做什么：** 按 [补充讲义](../cs336_spring2026_assignment5_supplement_safety_rlhf.pdf) 完成示范数据处理、SFT loss、偏好对处理、DPO loss 和相关评估。它要求的是 SFT/DPO，不是 PPO 实现。

**验收内容：** 能解释 chosen/rejected 如何进入目标；正确屏蔽 prompt/padding；比较 SFT 与 DPO 前后回答行为；把任务能力、偏好胜率与安全评估分开报告。

讲义使用 8B 模型并含更大模型的评判环节，不能把整个 supplement 当成普通 CPU 小实验。可以先做小模型机制验证，但若用于正式作业提交，要遵守讲义规定的模型与评估设置。

**你会学到：** 为什么“数学 RLVR”和“偏好对齐”不是同一个训练问题。

### 项目 D：在现有框架中复现一个机制，必要时再做 LLM-PPO

优先选择一个清晰问题：

- 去掉优势标准差归一化，会怎样改变不同题目的训练权重？
- 只改变长度归一化，准确率与回答长度如何变化？
- 相同 rollout 复用程度下，token clipping 与序列 clipping 有什么差异？
- 使用 value head / GAE 后，与组内优势相比有什么收益和成本？

前三项可直接沿用你的代码；第四项是可选的 LLM-PPO 项目，适合在小环境 PPO 学会后再做。它需要额外的 value 预测、回报目标与训练，工程和计算开销都会增加。

机制复现要控制其他变量。可以先消融单项设计，再比较完整命名方法；完整 GRPO 与完整 Dr. GRPO 同时改变两个归一化选择，不能把全部差异都归因于其中一项。

参考 [Dr. GRPO 作者仓库](https://github.com/sail-sg/understand-r1-zero) 核对公式和实现。其完整实验环境与模型设置可能比当前作业大，使用现有 OLMo/GSM8K 做机制对照更容易定位原因。小模型复现能验证机制，不等于复现了原论文的全部指标与结论。

### 项目 E：进入真正的工具交互式 Agentic RL

**推荐起点：** 给数学模型一个确定性的计算器工具，先实现完整交互轨迹，再训练策略。

```text
模型生成工具调用
    → 执行计算器
    → 工具结果加入上下文
    → 模型继续推理或调用工具
    → 输出最终答案
    → 按最终结果等规则给奖励
```

先比较三种基线：相同模型无工具、相同模型有工具但不做 RL、有工具并做 RL。这样才能区分“工具本身有用”和“训练改善了工具使用”。

**验收内容：** 保存多轮轨迹；区分模型动作与工具 observation；策略 loss 只作用于模型实际生成的 token；统计任务成功率、工具调用格式成功率、调用次数、输出长度和实际成本。

参考 [verl Agentic RL 文档](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)。阅读 [ReAct](https://arxiv.org/abs/2210.03629) 可以先理解 reasoning/action/observation 的交互结构，但使用 ReAct 提示本身不等于进行了 RL 训练。

到这一步，你才会面对当前作业较少涉及的问题：多轮信用分配、工具失败、环境延迟、观测 token mask、长轨迹和成本约束。

## 8. 论文阅读顺序与每篇要回答的问题

不建议先追最新算法名称。对你目前的基础，优先顺序是 **GAE/PPO → 原始 GRPO 对照 → 一个归一化或权重变体 → 偏好或工具交互方向**。

| 顺序 | 论文 | 阅读后应能回答 |
|---|---|---|
| 1 | [GAE](https://arxiv.org/abs/1506.02438) | return、value、TD residual、advantage 的关系是什么？为什么需要 λ？ |
| 2 | [PPO](https://arxiv.org/abs/1707.06347) | 为什么要保留旧概率？clipping 究竟截断什么？actor 与 critic 如何训练？ |
| 3 | [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) | 用组内奖励替代 learned critic，省掉什么、依赖什么？原版与作业差在哪？ |
| 4，二选一 | [Dr. GRPO](https://arxiv.org/abs/2503.20783) 或 [GSPO](https://arxiv.org/abs/2507.18071) | 哪个权重被改变？引入或去掉了什么偏置？怎样做单变量实验？ |
| 5，偏好路线 | [DPO](https://arxiv.org/abs/2305.18290) | 为什么可以从偏好对直接优化策略？与在线采样的策略梯度有什么区别？ |
| 6，可选拓展 | [RLOO](https://arxiv.org/abs/2402.14740)、[DAPO](https://arxiv.org/abs/2503.14476) | baseline 或训练配方具体改了哪些环节？能否拆成小实验？ |

对每篇论文，建议只先写清六件事：优化目标、数据从哪里来、优势怎么估计、如何复用旧数据、如何归一化/裁剪、相对哪一个基线改变了什么。之后再读完整实验结果，判断其结论是否能迁移到你的模型与数据。

如果以后要扩展到整个 RL 领域，再系统学习 Bellman 方程、Q-learning/DQN、SAC/TD3、model-based planning。它们很重要，但无需作为当前 GRPO 项目的前置条件全部实现。

## 9. 框架怎么选：先理解自己的代码，再借助框架

| 资源 | 推荐用途 | 使用时注意 |
|---|---|---|
| 本项目 | 推导与实现一一对应、做小规模消融 | 当前最熟悉，先完成训练闭环 |
| [Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html) | PPO 数学、伪代码与核心概念 | 教程价值高，历史代码依赖不必直接迁入当前环境 |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | 小环境的单文件 PPO 对照 | 逐段核对 rollout、GAE、loss、日志，避免只运行脚本 |
| [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) | 对照 LLM loss/config 的实现语义 | Trainer 名字不能代替算法配置说明 |
| [verl](https://github.com/verl-project/verl) | 大模型策略训练、系统扩展与工具交互 | 项目与依赖更复杂，进入相应阶段再选择 |

截至本次核对，TRL 的 `GRPOConfig` 文档显示默认 `loss_type="dapo"`。因此“用了 GRPOTrainer”并不表示使用了你讲义中的原始 GRPO 配置。对照时要检查优势标准化、loss 分母、KL、裁剪上下界以及 token/sequence 权重。[配置说明](https://huggingface.co/docs/trl/grpo_trainer)

verl 当前官方仓库位于 `verl-project/verl`，仓库记录了 2026 年组织迁移及 recipe 拆分。跟随教程时锁定版本或 commit，并核对文档对应版本，不要混用不同年代的目录和配置。[官方仓库](https://github.com/verl-project/verl)

不必同时掌握多个大框架。先用自己的小实现回答算法问题，再用一个框架学习规模化训练。

## 10. 下一次开始学习时，可以按这个顺序做

1. 自己修正 off-policy train-step 的旧概率切片与过滤对齐，重新运行全部 GRPO 测试。
2. 做一次短的真实 off-policy 训练，检查概率比、梯度、mask 和日志，再扩展实验预算。
3. 完成 Section 6.4 的多种子对照与 Section 7 的估计器实验，记录证据与限制。
4. 在 CartPole 中实现 value baseline、GAE 和 PPO，补齐当前最缺的 critic 知识。
5. 根据兴趣选择 SFT/DPO supplement，或在当前框架复现 Dr. GRPO/GSPO 的某个机制。
6. 最后加入计算器工具，建立多轮轨迹与只覆盖模型动作的 loss mask，进入 Agentic RL。

用于自查的问题：我能否不用论文缩写，解释每种方法的数据、优势、权重和更新过程？我能否解释 critic、reward model、旧策略和参考策略各做什么？我能否根据一条短轨迹手算优势，并判断某个 token 是否被裁剪？我能否拿出独立评估与多种子结果，支持“这个设计有效”的结论？
