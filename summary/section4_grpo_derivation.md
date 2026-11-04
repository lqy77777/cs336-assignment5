# Section 4：On-policy GRPO 完整推导与当前实现

整理日期：2026-11-04。

对应讲义：[Assignment 5 主讲义](../cs336_spring2026_assignment5_alignment.pdf)，Section 4，重点是 4.1 的推导、Algorithm 1，以及 4.2 中把梯度估计写成 loss 的过程。

对应实际代码：

- [tool.py](../cs336_alignment/tool.py)：tokenize、逐 token log-prob、奖励、组内 advantage、策略梯度 loss、序列聚合。
- [training_step.py](../cs336_alignment/training_step.py)：一次 GRPO 更新、vLLM 采样、权重同步和训练循环。
- [tests/adapters.py](../tests/adapters.py)：把上述实现连接到作业测试。

范围说明：本文总结本作业 Section 4 的 **standard on-policy GRPO**。重要性重加权和 PPO/GRPO 式概率比裁剪在 Section 6 展开，不属于这里的推导。文中“当前实现”来自本次对实际源码的静态阅读；本次执行了小规模数学核对，没有重新运行项目单元测试或 GPU 训练，也没有判断训练实验是否全部完成。

## 1. 整体思路：从答对概率到可执行的参数更新

Section 3 固定模型参数，研究提示词如何影响解题表现。Section 4 开始更新模型参数，目标是让模型更经常生成评分函数认可的正确回答。

完整推导路线是：

```text
最大化模型自己生成回答的期望奖励 J(θ)
    ↓ log-derivative trick
得到 REINFORCE：奖励 × 回答的 log-prob 梯度
    ↓ 对一批题目和回答取样本平均
得到可计算的策略梯度估计
    ↓ 减去同一道题的组均值
把绝对奖励改为组内相对优势
    ↓ 除以组内标准差
调整不同组的更新尺度
    ↓ 每条回答除以自身长度
先在回答内部平均，再对所有回答平均
    ↓ 取负号，冻结奖励和 advantage
构造可由 PyTorch backward 的 loss
    ↓ 梯度累积、梯度范数裁剪、一次 optimizer.step()
更新模型，再同步给 vLLM，重新采样
```

GRPO 的名字可以结合这条流程记忆：**Group** 是同一道题的多个回答；**Relative** 是相对同组平均奖励的优势；**Policy Optimization** 是用这些优势来更新语言模型的生成策略。

这里没有单独训练一个价值模型来给出 baseline，当前实现直接使用同题多个回答的奖励统计量。

## 2. 数据表示、符号和接口

### 2.1 先认识推导中的变量

| 符号 | 含义 | 在代码中的对应 |
|---|---|---|
| $\theta$ | 语言模型参数 | `model.parameters()` |
| $\pi_\theta$ | 由模型参数定义的生成策略 | Hugging Face 模型；同步权重后的 vLLM 副本 |
| $\rho$ | 题目／提示词分布 | 题目数据集及采样过程 |
| $x^{(i)}$ | 第 $i$ 个完整 prompt，包含题目及模板 | `prompt`、`repeated_prompts` |
| $y^{(i,j)}$ | 第 $i$ 道题的第 $j$ 个模型回答 | `rollout_responses` 中的一项 |
| $y_t$、$y_{<t}$ | 当前生成的 token、此前生成的回答前缀 | `labels` 中的目标 token，以及输入前缀 |
| $r_{ij}$ | 回答的原始奖励，即 $r(y^{(i,j)}\mid x^{(i)})$ | `raw_rewards` |
| $B$ | 一批采样的题目数 | `B` |
| $G$ | 每道题生成的回答数 | `group_size`、`G` |
| $N=BG$ | 一批回答的总数 | `N`、`len(repeated_prompts)` |
| $\mu_i$ | 第 $i$ 组奖励的均值 | `mu` 中对应组的值 |
| $s_i$ | 实际实现使用的组内样本标准差 | `reward.std()` |
| $\epsilon$ | 防止零除的小常数 | `advantage_eps` |
| $A_{ij}$ | 回答的组内标准化优势 | `normed_advantages` |
| $L_{ij}$ | 一条回答的有效 token 数 | `response_mask.sum(dim=-1)` |
| $M_{n,t}$ | 是否把某个目标 token 计入 loss 的 mask | `response_mask` |
| $\alpha$ | 学习率 | `learning_rate` |

这里的 $A_{ij}$ 是 advantage。后面的二元动作方差小题使用大写 $A\in\{0,1\}$ 表示动作，二者含义不同。

### 2.2 数据为什么要按组排列

当前实现把二维的“题目 × 回答”展开成一维列表：

```text
repeated_prompts     = [x1, x1, ..., x1, x2, x2, ..., x2, ...]
rollout_responses    = [y11,y12,...,y1G,y21,y22,...,y2G,...]
repeated_ground_truths = [a1,a1,...,a1,a2,a2,...,a2,...]
```

三个列表长度都是 $N=BG$，同一道题的 $G$ 条回答必须连续。`compute_group_normalized_rewards_grpo` 依靠连续切片恢复每组；如果排列被打乱，计算出来的就不再是同题之间的相对奖励。

参考最终答案只用于评分。参与 log-prob 计算的 `labels` 来自模型自己生成的回答，并非数据集提供的参考推理过程。

### 2.3 当前函数的输入、返回值和职责

设 $S$ 是一批拼接后的 prompt＋response 经右侧 padding 后的最大长度，减去因果预测的一个位置。

| 实际函数 | 主要输入 | 返回值与职责 |
|---|---|---|
| `tokenize_prompt_and_output` | prompt 列表、回答列表、tokenizer | 字典，包含形状均为 $(N,S)$ 的 `input_ids`、`labels`、`response_mask` |
| `get_response_log_probs` | 模型、输入 token、目标 token、是否计算熵 | 字典，`log_probs` 形状 $(N,S)$；可选 `token_entropy` 同形状 |
| `compute_rollout_rewards` | 评分函数、模型回答、参考最终答案 | `(raw_rewards, metadata)`，奖励形状 $(N,)$ |
| `compute_group_normalized_rewards_grpo` | 奖励、组大小、baseline 与归一化设置 | `(advantages, metadata)`，优势形状 $(N,)$ |
| `compute_policy_gradient_loss_on_policy` | 每回答一个 advantage、逐 token log-prob | `(per_token_loss, metadata)`，loss 形状 $(N,S)$ |
| `aggregate_loss_across_microbatch_sequence` | 逐 token loss、mask | 标量 Tensor：先每回答平均，再对回答平均 |
| `grpo_train_step` | 模型、优化器、一批采样回答及配置 | `(batch_loss, metadata)`，完成一次参数更新；返回的 loss 用于日志 |

`compute_group_normalized_rewards_grpo` 当前内部把返回张量命名为 `rewards`，但它已经是标准化后的 **advantages**，不是原始的 0/1 奖励。

## 3. 算法执行过程：把数学推导与代码连接起来

### 3.1 把语言模型看成策略（4.1.1—4.1.2）

使用从 1 开始的 token 下标，在生成第 $t$ 个回答 token 之前：

$$
s_t=(x,y_{<t}),\qquad a_t=y_t.
$$

状态包含题目和提示词 $x$，也包含已经生成的回答。语言模型给出的下一个 token 分布就是策略：

$$
\pi_\theta(a_t\mid s_t)
=\pi_\theta(y_t\mid x,y_{<t})
=\left[\operatorname{softmax}(f_\theta(s_t))\right]_{y_t}.
$$

采样 token 后，状态转移是把这个 token 拼接到前缀后面：

$$
s_{t+1}=(x,y_{\le t}).
$$

一次从 prompt 开始、一直到结束标记或长度上限的生成过程，就是 trajectory／episode／rollout。在当前任务中，给定 $x$ 后，整段回答 $y$ 就足以描述这条轨迹。

这里需要两种不同操作：

1. **采样**：从模型生成分布抽取一个回答，当前由 vLLM 完成。
2. **计算概率**：对已经采样出的回答，计算每个 token 在当前策略下的 log-prob，当前由 Hugging Face 模型完成并保留梯度。

采样得到的是离散 token。训练时会重新前向计算这些 token 的概率，而不会通过 token 抽样操作反向传播。

### 3.2 完整回答的概率如何拆成 token 概率

根据概率链式法则：

$$
\pi_\theta(y\mid x)
=\prod_{t=1}^{L}\pi_\theta(y_t\mid x,y_{<t}).
$$

取对数，乘积变成和：

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{L}\log\pi_\theta(y_t\mid x,y_{<t}).
$$

因此：

$$
\nabla_\theta\log\pi_\theta(y\mid x)
=\sum_{t=1}^{L}\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t}).
$$

这是后面能够先计算逐 token loss、再聚合成整条回答 loss 的依据。到这一步只有求和，**还没有除以回答长度**。

### 3.3 奖励、return 和最终目标（4.1.3）

本作业在生成完整回答后，使用评分函数给出终止奖励。中间 token 没有独立的正确性奖励，整条轨迹的 return 就是最后的评分：

$$
R(\tau)=r(y\mid x)\in\{0,1\}.
$$

更准确地说，奖励为 1 表示评分程序成功接受该回答为正确；格式或解析失败也可能造成奖励为 0。

我们希望最大化：

$$
J(\theta)
=\mathbb E_{x\sim\rho}\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[r(y\mid x)].
$$

当奖励是 0/1 时，期望奖励就是在这个题目分布和生成分布下，被评分函数判对的概率。

关键困难是：**回答 $y$ 是模型自己采样出来的，它的分布也依赖 $\theta$。** 改变参数会改变哪些回答更容易被生成。

所以，不能只看某次抽到的奖励数字，再尝试直接对那个 0 或 1 求导。我们需要计算“回答概率如何随参数变化”。

### 3.4 从目标函数推导 REINFORCE（4.1.4，讲义式 9—14）

先固定一道题 $x$，记：

$$
J_x(\theta)=\sum_y\pi_\theta(y\mid x)r(y\mid x).
$$

这里假设评分规则没有对 $\theta$ 的直接依赖；题目分布也不依赖 $\theta$。在允许交换求导与求和的条件下：

$$
\begin{aligned}
\nabla_\theta J_x(\theta)
&=\sum_y r(y\mid x)\nabla_\theta\pi_\theta(y\mid x)\\
&=\sum_y r(y\mid x)\pi_\theta(y\mid x)
\frac{\nabla_\theta\pi_\theta(y\mid x)}{\pi_\theta(y\mid x)}\\
&=\sum_y\pi_\theta(y\mid x)r(y\mid x)
\nabla_\theta\log\pi_\theta(y\mid x)\\
&=\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}
\left[r(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x)\right].
\end{aligned}
$$

中间使用的 log-derivative trick 是：

$$
\nabla_\theta\pi_\theta(y\mid x)
=\pi_\theta(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x).
$$

再把对题目的期望加回来：

$$
\boxed{
\nabla_\theta J(\theta)
=\mathbb E_{x\sim\rho,y\sim\pi_\theta}
\left[r(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x)\right]
}
$$

这就是 REINFORCE 策略梯度。它把原来的求导问题，改写成可以通过模型采样来估计的期望。

采样 $B$ 道题，每题采样 $G$ 个回答后：

$$
\hat g_{\mathrm{RF}}
=\frac1{BG}\sum_{i=1}^{B}\sum_{j=1}^{G}
r_{ij}\nabla_\theta\log\pi_\theta(y^{(i,j)}\mid x^{(i)}).
$$

在相应的 on-policy 采样假设下，这是原始期望奖励梯度的无偏估计。

如果直接使用 0/1 奖励，正确回答的该项系数为 1，错误回答的该项系数为 0。因此，原始形式会利用正确样本，而错误样本没有自身直接的梯度贡献。

### 3.5 为什么能够减去 baseline（4.1.5，讲义式 15—17）

先定义 score function：

$$
q(y)=\nabla_\theta\log\pi_\theta(y\mid x).
$$

它的期望为 0：

$$
\begin{aligned}
\mathbb E_y[q(y)]
&=\sum_y\pi_\theta(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x)\\
&=\sum_y\nabla_\theta\pi_\theta(y\mid x)\\
&=\nabla_\theta\sum_y\pi_\theta(y\mid x)\\
&=\nabla_\theta 1=0.
\end{aligned}
$$

如果 baseline $b(x)$ 在给定题目时不依赖当前抽到的回答，则：

$$
\begin{aligned}
\mathbb E_y[(r(y\mid x)-b(x))q(y)]
&=\mathbb E_y[r(y\mid x)q(y)]-b(x)\mathbb E_y[q(y)]\\
&=\mathbb E_y[r(y\mid x)q(y)].
\end{aligned}
$$

所以，减去这样的 baseline 不改变梯度期望。它可以改变方差，但**不保证任何 baseline 都会降低方差**。

例如取 $b=0.5$：正确回答的权重变成 $+0.5$，错误回答的权重变成 $-0.5$，从而得到正向和负向两种样本信号。

实现时，baseline／advantage 应当作为本次更新的固定权重。如果 baseline 的计算依赖可求导参数，也要阻断它在策略 loss 中的梯度；上述等式讨论的是 $bq$，不包含额外的 $\nabla b$ 项。

### 3.6 `baseline_calcs`：完整计算 baseline 的方差

这个小题帮助理解“保持梯度期望”与“降低方差”是两件事。

先解释变量：动作 $A\in\{0,1\}$；动作 1 的概率为 $p=\sigma(\theta)$；奖励 $r(A)=A$；$n$ 是独立采样次数；$b$ 是给定 $\theta$ 下用于梯度估计的固定 baseline。

Bernoulli 策略的 log-prob 为：

$$
\log\pi_\theta(A)=A\log p+(1-A)\log(1-p).
$$

由于 $\frac{\partial p}{\partial\theta}=p(1-p)$：

$$
\frac{\partial}{\partial\theta}\log\pi_\theta(A)=A-p.
$$

单样本 baseline-adjusted 梯度估计为：

$$
Z_b=(A-b)(A-p).
$$

它只有两种取值：

| 动作 | 概率 | $Z_b$ |
|---|---|---|
| $A=1$ | $p$ | $(1-b)(1-p)$ |
| $A=0$ | $1-p$ | $bp$ |

因此：

$$
\mathbb E[Z_b]
=p(1-b)(1-p)+(1-p)bp
=p(1-p).
$$

这恰好等于 $\partial_\theta J=\partial_\theta p$，与 $b$ 无关。

对取值为 $z_1,z_0$ 的两点随机变量，方差可以写成 $p(1-p)(z_1-z_0)^2$。这里：

$$
z_1-z_0=(1-b)(1-p)-bp=1-p-b.
$$

于是：

$$
\operatorname{Var}(Z_b)=p(1-p)(1-p-b)^2.
$$

独立样本均值的方差再除以 $n$：

$$
\boxed{
\operatorname{Var}(\hat g_b)
=\frac{p(1-p)(1-p-b)^2}{n}
}
$$

题目三问的结果分别是：

$$
\boxed{\operatorname{Var}(\hat g_0)=\frac{p(1-p)^3}{n}}
$$

$$
\boxed{\operatorname{Var}(\hat g_b)=\frac{p(1-p)(1-p-b)^2}{n}}
$$

$$
\boxed{\operatorname{Var}(\hat g_{b=p})=\frac{p(1-p)(1-2p)^2}{n}}
$$

比较 $b=p$ 与不使用 baseline：

$$
\operatorname{Var}(\hat g_{b=p})-\operatorname{Var}(\hat g_0)
=\frac{p^2(1-p)(3p-2)}{n}.
$$

所以对 $0<p<1$：

- $p<2/3$：总体均值 baseline $b=p$ 的方差更低。
- $p=2/3$：方差相同。
- $p>2/3$：方差反而更高。

在边界 $p=0,1$，上述方差都为 0。对本小题，最小方差的 baseline 是 $b^*=1-p$，可使 $Z_b$ 两种取值完全相同。它说明“奖励均值”也未必是最小方差的 baseline。

这里 $b=p$ 表示在当前 $\theta$ 处取该数值作为权重，并不对这个权重继续求导。

### 3.7 使用组均值 baseline：GRPO 的核心（讲义式 20—24）

同一道题 $x^{(i)}$ 的 $G$ 个奖励均值是：

$$
\mu_i=\frac1G\sum_{j=1}^{G}r_{ij}.
$$

先只减去均值，得到：

$$
\hat g_{\mathrm{group}}
=\frac1{BG}\sum_{i,j}(r_{ij}-\mu_i)
\nabla_\theta\log\pi_\theta(y^{(i,j)}\mid x^{(i)}).
$$

这个均值包含当前回答自身的奖励，因此不能直接套用“baseline 与当前回答无关”的证明。需要单独计算。

固定一道题，省略 $i$，令 $q_j=\nabla_\theta\log\pi_\theta(y^{(j)}\mid x)$，并记 $g_x=\mathbb E[r_jq_j]$。假设组内回答是给定题目后的独立采样。

展开：

$$
\mathbb E[(r_j-\mu)q_j]
=\mathbb E[r_jq_j]-\frac1G\sum_{k=1}^{G}\mathbb E[r_kq_j].
$$

当 $k\ne j$，根据条件独立性和 $\mathbb E[q_j]=0$：

$$
\mathbb E[r_kq_j]=\mathbb E[r_k]\mathbb E[q_j]=0.
$$

当 $k=j$，该项是 $\mathbb E[r_jq_j]=g_x$。所以：

$$
\mathbb E[(r_j-\mu)q_j]
=g_x-\frac1Gg_x=\frac{G-1}{G}g_x.
$$

再对 $j$ 平均并对题目取期望，得到：

$$
\boxed{
\mathbb E[\hat g_{\mathrm{group}}]
=\frac{G-1}{G}\nabla_\theta J(\theta)
}
$$

这表示：**只使用组均值中心化时，梯度期望与原始梯度相差固定比例 $(G-1)/G$**。本作业当前实现没有额外乘回 $G/(G-1)$。

这个结论还没有包含接下来的标准差归一化与长度归一化，不能把同一个比例直接套到最终 GRPO 梯度上。

### 3.8 除以组内标准差：advantage normalization（4.1.6）

讲义展示的总体形式是：

$$
\sigma_i=\sqrt{\frac1G\sum_{j=1}^{G}(r_{ij}-\mu_i)^2}.
$$

但讲义明确要求实现时使用默认 `torch.std()`。你当前代码确实使用 `reward.std()`，对应样本标准差：

$$
s_i=\sqrt{\frac1{G-1}\sum_{j=1}^{G}(r_{ij}-\mu_i)^2},\qquad G>1.
$$

当前实际 advantage 为：

$$
\boxed{A_{ij}=\frac{r_{ij}-\mu_i}{s_i+\epsilon}}
$$

这一步重新调整每组的更新尺度。标准差较小时，给定的奖励偏差会被赋予更大的相对权重。

讲义把它作为稳定性设计来讨论，但它并不严格保证不同组的梯度范数相同；梯度还受到模型、回答内容和方向等因素影响。

尤其要记住：**标准差由本组回答及其奖励计算而来，所以除以它通常不再保持原始奖励梯度的期望。** 这一步是算法设计上的改变，不能表述为一个保持目标完全不变的代数恒等变形。

当前实现的具体做法是：

1. 每次切出连续 $G$ 个奖励。
2. 把 `reward.mean().item()` 重复 $G$ 次，加入 `mu`。
3. 把 `reward.std().item() + advantage_eps` 重复 $G$ 次，加入 `norm`。
4. 转为 Tensor，计算 `(raw_rewards - mu) / norm`。
5. 返回优势张量和奖励统计。

因此，每个回答都与它所在组的均值、标准差配对。`.item()` 以及重新创建 Tensor 的过程也使这些统计值不参与反向传播；当前原始奖励本来就是外部评分得到的不带梯度的数值。

### 3.9 再除以每条回答的长度（4.1.7）

如果只把回答 log-prob 展开成 token 求和，得到：

$$
\hat g=\frac1{BG}\sum_{i,j}A_{ij}
\sum_{t=1}^{L_{ij}}\nabla_\theta
\log\pi_\theta(y_t^{(i,j)}\mid x^{(i)},y_{<t}^{(i,j)}).
$$

标准 GRPO 进一步加入每条回答的 $1/L_{ij}$：

$$
\boxed{
\hat g_{\mathrm{GRPO}}
=\frac1{BG}\sum_{i,j}\frac{A_{ij}}{L_{ij}}
\sum_{t=1}^{L_{ij}}\nabla_\theta
\log\pi_\theta(y_t^{(i,j)}\mid x^{(i)},y_{<t}^{(i,j)})
}
$$

这相当于先计算每条回答的平均 token 项，再对所有回答平均。同样的 advantage 下，较长回答中每个 token 的显式权重更小。

注意三种归约方式的区别：

| 方式 | 计算形式 | 含义 |
|---|---|---|
| 每回答求和，再对回答平均 | $\frac1N\sum_n\sum_t\ell_{n,t}$ | 原始序列 log-prob 梯度的结构 |
| 每回答平均，再对回答平均 | $\frac1N\sum_n\frac1{L_n}\sum_t\ell_{n,t}$ | 本节标准 GRPO 的结构 |
| 所有有效 token 一起平均 | $\frac{\sum_n\sum_t\ell_{n,t}}{\sum_nL_n}$ | 按 token 数决定各条回答在均值中的权重 |

第二种与第三种在回答长度不同时通常不同。长度由生成内容决定，因此这一步同样会改变相对于原始策略梯度的权重。

“每条回答都先平均”不等于保证每条回答的梯度范数相同；那还取决于各 token 梯度的大小和方向。

### 3.10 把梯度估计写成 PyTorch loss（4.2.3，讲义式 32）

先固定这次采样得到的数据。令 $\operatorname{sg}(\cdot)$ 表示停止梯度，即把该量作为本次更新的常数。

构造一个采样后的代理目标：

$$
\widetilde J(\theta)
=\frac1{BG}\sum_{i,j}\frac{\operatorname{sg}(A_{ij})}{L_{ij}}
\sum_{t=1}^{L_{ij}}
\log\pi_\theta(y_t^{(i,j)}\mid x^{(i)},y_{<t}^{(i,j)}).
$$

对它求导，便得到上一节的梯度估计。PyTorch 优化器通常执行梯度下降，所以返回它的负数：

$$
\boxed{
\mathcal L_{\mathrm{GRPO}}(\theta)
=-\frac1{BG}\sum_{i,j}\frac{\operatorname{sg}(A_{ij})}{L_{ij}}
\sum_{t=1}^{L_{ij}}
\log\pi_\theta(y_t^{(i,j)}\mid x^{(i)},y_{<t}^{(i,j)})
}
$$

这样，普通梯度下降 $\theta\leftarrow\theta-\alpha\nabla\mathcal L$ 就对应着沿 $\hat g_{\mathrm{GRPO}}$ 的上升更新。实际代码使用 AdamW，更新还包含优化器的自适应缩放。

这里需要同时记住三个层次：

- 原始 $J(\theta)$：我们希望提高的期望奖励。
- $\hat g_{\mathrm{GRPO}}$：经过组统计和长度归一化修改后的更新估计。
- $\mathcal L_{\mathrm{GRPO}}$：固定本批采样数据后，用来让自动微分产生所需更新的代理 loss。

因此，不能把这个 loss 的数值直接当成“数学题错误率”。样本、奖励和优势都在随轮次变化，它也不需要像固定数据集上的交叉熵那样持续下降；出现负值也不自动表示出错。

当前函数中，先把每条回答的 advantage 整理成长度 $N$ 的向量，再计算：

```python
result = -einsum(advantages, policy_log_probs, 'b,b t -> b t')
```

这里 `advantages` 的形状是 $(N,)$，`policy_log_probs` 是 $(N,S)$，返回 `result` 也是 $(N,S)$。对每个位置，它做的就是：

$$
\ell_{n,t}=-A_n\log\pi_\theta(\text{目标 token}\mid\text{前缀}).
$$

正 advantage 的该项推动已采样 token 的概率提高；负 advantage 的该项推动其概率降低。整批梯度叠加后，实际变化还受到不同样本的梯度相互作用影响。

### 3.11 为什么需要 labels 右移和 response mask

`tokenize_prompt_and_output` 的参数分别是 prompt 字符串列表、回答字符串列表和 tokenizer；返回值是包含输入、预测目标和 mask 的字典。

你先分别编码 prompt 和 response，设置 `add_special_tokens=False`，再直接拼接，不额外插入 BOS、EOS 或分隔符。这样明确保留两部分的 token 边界，方便构造回答 mask。

假设 prompt 有三个 token，response 有两个 token：

```text
拼接后：       [p0, p1, p2, r0, r1]
input_ids：    [p0, p1, p2, r0]
labels：       [p1, p2, r0, r1]
response_mask：[ 0,  0,  1,  1]
```

`p2` 所在位置预测的 label 是第一个回答 token `r0`，所以那个位置的 mask 必须为 1。你的代码用 $P-1$ 个 0 加 $L$ 个 1 来实现这个对齐，其中 $P$ 是 prompt 长度。

随后右侧 padding，并对 padding 对应的 label 位置补 0。

`get_response_log_probs` 按以下顺序计算：

1. `model(input_ids).logits` 得到 $(N,S,V)$ 的 logits，$V$ 是词表大小。
2. `log_softmax(..., dim=-1)` 得到每个词的 log-prob。
3. `labels.unsqueeze(-1)` 把目标编号变成 $(N,S,1)$。
4. `gather` 取出实际目标 token 的 log-prob。
5. `squeeze(-1)` 得到 $(N,S)$ 的结果。

这里一次带因果 mask 的前向计算，就能算出已有回答所有位置的条件概率，不需要在训练时逐个 token 重新生成回答。

最后聚合时：

$$
L_n=\sum_t M_{n,t},\qquad
\mathcal L=\frac1N\sum_n
\frac{\sum_t M_{n,t}\ell_{n,t}}{L_n}.
$$

对应你当前的顺序是 `loss * mask`、沿 token 求和、除以 `mask.sum`、再对回答求均值。prompt 仍作为条件参与模型计算，只是不把预测 prompt token 本身计入策略 loss。

### 3.12 一次训练更新如何保持 on-policy（4.2.2—4.2.4）

当前程序分别维护用于训练的 Hugging Face 模型和用于生成的 vLLM 副本。它们的角色不同，但采样时应对应同一版本的策略参数。

你的实际顺序是：

1. `server.sync_policy_weights(model)`：先把当前权重同步到 vLLM。
2. 抽取 $B$ 道题，每道题采样 $G$ 个回答。
3. 得到三条对齐的列表：prompt、模型回答、参考最终答案。
4. tokenize，计算奖励和整组 advantage。
5. 把回答批次切成若干 microbatch，重新前向计算 log-prob 并反向传播。
6. 所有 microbatch 完成后，裁剪梯度范数。
7. 只执行一次 `optimizer.step()`，然后 `optimizer.zero_grad()`。
8. 下一轮重新同步更新后的权重，并重新采样。

on-policy 的关键是：**本轮数据来自当前策略，并在这批数据对应的一次参数更新中使用。** 多次 `backward()` 只是累积同一次更新的梯度，只要中途没有 `optimizer.step()`，参数就没有因这些反向传播而提前改变。

如果在同一批回答上继续执行第二次、第三次参数更新，样本就来自较旧的策略，这正是 Section 6 要研究的 off-policy 情形。单纯给函数增加 `old_log_probs` 参数，并不意味着已经实现了相应的校正算法。

### 3.13 为什么 microbatch loss 要按回答数量缩放

设整批有 $N$ 条回答，第 $k$ 个 microbatch 有 $m_k$ 条。因为标准 GRPO 对回答做平均：

$$
\mathcal L_{\mathrm{batch}}
=\sum_k\frac{m_k}{N}\mathcal L_{\mathrm{micro},k}.
$$

所以每次应该反向传播 $\frac{m_k}{N}\mathcal L_{\mathrm{micro},k}$。

如果恰好分成 $K$ 个等大的 microbatch，$m_k/N=1/K$，就可以每次对 `loss / gradient_accumulation_steps` 调用 `backward()`。

你当前设置是：

$$
B=32,\quad G=8,\quad N=256,\quad K=64,\quad m=4.
$$

因此当前每个 microbatch 的均值 loss 除以 64，累积起来与整批回答均值的归约相匹配。组内 advantage 在分 microbatch 之前就已算好，所以即使每组 8 条回答跨越两个 microbatch，也仍然使用完整组统计量。

如果以后 $N$ 不能整除 $K$，就不能直接假设每个 microbatch 都占 $1/K$；需要按实际 $m_k/N$ 处理。这是当前实现使用等大 microbatch 的适用前提。

当前 `batch_loss` 累加的是 `detach()` 后的值，只用于日志；真正产生梯度的是原始 microbatch loss 的 `backward()`。

`max_grad_norm` 对累积完成后的参数梯度向量做范数裁剪。这与 Section 6 中对新旧策略概率比的裁剪是两个不同的操作。

### 3.14 Section 4.3 如何验证这个算法有效

讲义要求观察验证奖励、格式奖励、回答长度、梯度范数、token 熵和生成样例，并比较多次运行的差异。

当前代码中，token entropy 计算的是每个位置整个词表分布的熵：

$$
H_t=-\sum_{v=1}^{V}p_{t,v}\log p_{t,v}.
$$

你使用 `torch.no_grad()`、分块计算和 `detach()`，避免为日志中的熵额外保留反向图和整个词表的额外概率张量。训练循环再只对 response mask 中的 token 汇总熵。它是观测指标，当前没有作为熵奖励加进策略 loss。

当前脚本默认运行 50 次更新，采用 seed 0，并每 10 次更新验证一次；验证设置会改成 `temperature=0.0, n=1`。这些是当前脚本的配置，不代表本次已重新运行它，也不等于已经完成讲义要求的多随机种子正式实验。

当前验证的贪心解码与训练时 temperature 1 的采样分布不同。因此，验证正确率可以用来观察该解码设置下的表现，但不能直接当成训练采样目标 $J(\theta)$ 的无偏估计。

讲义的标准实验建议使用 200 次更新、4 次不同随机种子的运行，并报告跨运行的变化；50 次更新用于前期检查。它还要求学习率实验和提示词消融。推导告诉我们更新从哪里来，这些实验用来检查更新在真实模型和数据集上是否有效。

## 4. 小例子推演

### 4.1 一道题、四个回答，优势怎么算

取 $B=1,G=4$，四个回答的奖励依次为：

$$
r=[1,0,1,0].
$$

组均值：

$$
\mu=(1+0+1+0)/4=0.5.
$$

按当前 `torch.std()` 默认样本标准差计算：

$$
s=\sqrt{\frac{4\times0.5^2}{4-1}}
=\sqrt{1/3}\approx0.5773502692.
$$

取 $\epsilon=10^{-6}$：

$$
A\approx[0.866023904,-0.866023904,0.866023904,-0.866023904].
$$

令 $a=0.866023904$，再假设回答长度分别为 $[2,2,4,4]$：

| 回答 | 奖励 | advantage | loss 中每个回答 token 的显式系数 $-A/L$ |
|---|---:|---:|---:|
| 1 | 1 | $+a$ | $-a/2$ |
| 2 | 0 | $-a$ | $+a/2$ |
| 3 | 1 | $+a$ | $-a/4$ |
| 4 | 0 | $-a$ | $+a/4$ |

最外层还会统一除以 $BG=4$。因此，同样是正 advantage，较长回答中的单个 token 获得的显式系数更小。

这不表示评分程序逐个判断了推理 token 的正确性。它只给整条回答一个奖励，再把相同的 advantage 分配给该回答所有有效 token。

如果改用讲义展示的总体标准差，$\sigma=0.5$，忽略 epsilon 后会得到 $\pm1$；与代码默认样本标准差得到的结果不同。手算时必须说明使用哪个定义。

### 4.2 算出一个完整标量 loss

延续上例，假设四个回答的 token log-prob 分别为：

```text
回答1：[-1, -1]             平均为 -1
回答2：[-2, -2]             平均为 -2
回答3：[-1, -2, -1, -2]     平均为 -1.5
回答4：[-3, -3, -3, -3]     平均为 -3
```

这些是为手算构造的 log-prob 数值，不是实际模型输出。每条回答的平均 loss 为 $-A\times\text{平均 log-prob}$：

$$
\mathcal L_1=a,\quad
\mathcal L_2=-2a,\quad
\mathcal L_3=1.5a,\quad
\mathcal L_4=-3a.
$$

整批 loss 为：

$$
\mathcal L=\frac{a-2a+1.5a-3a}{4}
=-0.625a\approx-0.541264940.
$$

loss 可以为负，因为负 advantage 的回答会产生负的 loss 项。我们关心的是这些项导出的参数更新，而不是要求它像普通非负交叉熵一样取值。

### 4.3 一组全错或全对时会怎样

若 $r=[0,0,0,0]$，则 $\mu=0,s=0$，所有 advantage 都是 $0/(0+\epsilon)=0$。

若 $r=[1,1,1,1]$，则 $\mu=1,s=0$，所有 advantage 同样为 0。

在本节当前的策略 loss 中，这些组没有通过 advantage 产生策略梯度贡献，因为组内没有奖励差异。当前 AdamW 有动量状态时，整批 advantage 为零也不能保证参数数值完全不动；“该项梯度贡献为零”和“优化器一定不更新参数”需要分开理解。

## 5. 关键性质、适用条件与当前限制

### 5.1 每一步是否保持原始梯度期望

| 步骤 | 与 $\nabla J$ 的关系 | 原因 |
|---|---|---|
| REINFORCE | 在采样及求导条件下无偏 | log-derivative 恒等式与样本均值 |
| 减去与当前回答无关的 baseline | 保持梯度期望 | score function 的期望为 0 |
| 减去包含自身奖励的组均值 | 变成 $(G-1)/G$ 倍 | 当前样本在组均值中占 $1/G$ |
| 再除以组标准差 | 通常不再只有固定比例 | 随机分母与同组样本有关 |
| 再除以回答长度 | 再次改变权重 | 长度由回答决定 |

因此，完整 GRPO 包含原始策略梯度推导，也包含后续算法设计。不要把整条链都称为“对原目标完全等价的变形”。

### 5.2 实现时必须保持的不变量

- 同一道题的 $G$ 条回答必须连续，且 prompt、response、ground truth 三个列表对齐。
- 归一化在同题组内完成，不能把整个 batch 混在一起计算一个均值和标准差。
- 对整组先计算 advantage，再切 microbatch，避免分批改变组统计量。
- `response_mask` 对齐的是右移后的 `labels`；第一个回答 token 不能漏掉。
- 分母是有效回答 token 数，不能把 prompt 或 padding 长度算进去。
- 每个回答的 advantage 在该回答全部有效 token 上共享。
- 奖励与 advantage 是固定权重，梯度主要经过当前模型的 token log-prob。
- 所有 microbatch 反向传播完成后，才裁剪梯度范数、执行一次参数更新。
- 下一轮采样前同步新权重，保持本节的 on-policy 流程。

### 5.3 当前代码实际支持到哪里

本次静态阅读看到的实现对应：

```text
baseline = "mean"
advantage_normalizer = "std"
importance_reweighting_method = "none"
loss_normalization = "sequence"
```

虽然函数签名包含后续变体参数，但不能据此认为那些分支已经实现：

- `compute_group_normalized_rewards_grpo` 对其他 baseline／normalizer 分支目前只有 `pass`，相关列表可能为空。
- `compute_policy_gradient_loss_on_policy` 当前始终计算 `-advantage * log_prob`，没有按参数实现新旧策略概率比和裁剪。
- `aggregate_loss_across_microbatch_sequence` 当前始终进行逐回答长度平均，尚未按 `constant` 分支实现不同归约。

这些是现有实现的范围说明。本文没有修改代码，也没有把后续改进写成现有功能。

### 5.4 边界条件与容易忘记的细节

1. **标准差加 epsilon 的位置。** 当前是 $s+\epsilon$，不是 $\sqrt{s^2+\epsilon}$，两者数值不同。
2. **组大小需要大于 1。** 默认样本标准差在 $G=1$ 时没有有效自由度，可能得到 NaN；给 NaN 加 epsilon 不能修复它。
3. **回答长度需要大于 0。** 当前序列聚合会除以有效 token 数，空回答需要额外处理才安全；本文未增加这种处理。
4. **microbatch 等分前提。** 当前按 `N // K` 切分、每次除以 `K`，适用于当前能够等分的设置。
5. **同组优势和为 0，不代表模型梯度和也为 0。** 各回答的 log-prob 梯度向量通常不同。
6. **中间推理也会被训练。** 即使只在最后评分，回答中间 token 仍在序列 log-prob 和 mask 中；这不等于逐步监督这些推理的正确性。
7. **梯度范数裁剪与概率比裁剪不同。** 当前已调用的是前者，不能据此声称已实现 Section 6 的 PPO/GRPO clipping。
8. **记录熵不等于优化熵。** 当前熵仅用于日志，没有额外加入 entropy bonus。
9. **on-policy 要匹配生成分布。** 理论上要求采样策略与计算 log-prob 的策略一致；讲义设置 temperature 1、top-p 1，并要求先同步参数。实际不同推理后端还可能存在数值差异。

## 6. 验证与复习

### 6.1 本次实际做了什么验证

本次读取并对照了讲义推导、Algorithm 1、代理 loss，以及 `tool.py`、`training_step.py` 和测试适配器的当前实现；同时渲染检查了推导所在页面，确认关键公式的分母、求和范围和归一化定义。

另外执行了不写入代码文件的小规模数学核对：

- 使用标准库 `fractions.Fraction` 精确计算 $p\in\{1/4,1/2,2/3,3/4\}$、$b\in\{0,p,1-p\}$ 的 Bernoulli 梯度均值和方差，检查结果与本文公式一致。
- 对上述概率，以及 $G\in\{2,3,4\}$，枚举同题全部二元回答组合，检查只减组均值时的期望确为 $(G-1)/G$ 倍原始梯度。
- 数值核对 $[1,0,1,0]$ 的样本标准差和包含 $10^{-6}$ epsilon 的 advantage。

这些检查用于核对代数和小例子，不替代一般性证明，也不等于项目单元测试通过。本次没有重新运行 pytest、调用 vLLM 或开展 GPU 训练。

### 6.2 有针对性的代码验证命令

需要检查 Section 4 的标准路径时，可以运行下面的测试选择；这是复习时可执行的命令，本次未执行：

```bash
uv run pytest tests/test_grpo.py -k 'test_tokenize_prompt_and_output or test_get_response_log_probs or test_compute_rollout_rewards or test_compute_group_normalized_rewards_grpo or test_compute_policy_gradient_loss_on_policy or test_aggregate_loss_across_microbatch_sequence or test_grpo_train_step_standard_on_policy'
```

这些测试通过后，仍需要按 Section 4.3 检查真实生成样例、验证奖励变化和多次运行差异。后续变体测试覆盖的是更大的范围，不能只凭标准 on-policy 路径就推断它们也会通过。

### 6.3 不看代码时，尝试自己回答

1. 为什么不能只对某次抽样得到的 0/1 奖励直接反向传播？
2. 为什么 $\nabla\pi=\pi\nabla\log\pi$ 能把梯度改写成可采样的期望？
3. 为什么与回答无关的 baseline 项期望为 0？
4. 为什么包含当前奖励的组均值会引入 $(G-1)/G$？
5. 为什么默认样本标准差与讲义公式里的总体标准差会给出不同手算值？
6. 为什么逐回答平均和全部 token 一起平均不相同？
7. 为什么一个 sequence-level 奖励能作用到整段推理的 token 上？
8. 为什么调用 64 次 `backward()` 仍然可以是一次 on-policy 更新？
9. 当前函数里的 `labels` 来自模型采样回答，还是来自数据集参考推理？
10. 为什么 loss 为负、或者没有单调下降，本身不能证明训练出错？

最后复习时，优先记住下面几句话：

- **原始目标：提高模型生成正确回答的期望奖励。**
- **REINFORCE：用奖励加权已经采样回答的 log-prob 梯度。**
- **组内 baseline：同题比较；包含自身奖励会带来固定缩放。**
- **标准 GRPO：再除组标准差，再对每条回答的 token 平均。**
- **实际 loss：负 advantage × token log-prob，经 mask 和两层平均后反向传播。**
- **一次更新：组内统计先算完整，microbatch 只分计算，最后统一更新参数。**
- **本文完整推导到 Section 4 的 on-policy 版本；后续概率比和裁剪要单独推导。**
