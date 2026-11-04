# WandB（Weights & Biases）使用手册：结合 CS336 Assignment 5

整理日期：2026-11-04。

对应代码：[training_step.py](../cs336_alignment/training_step.py)、[modal_utils.py](../cs336_alignment/modal_utils.py)、[pyproject.toml](../pyproject.toml)。

本文介绍 W&B 的实验记录功能：如何登录、记录配置与指标、查看训练曲线、保存回答表格、比较多次实验，以及处理常见问题。这里的 WandB、wandb 和 W&B 都指 Weights & Biases；本文主要使用 Experiments，不涉及 Weave 的应用调用追踪。

本文根据整理时的实际代码和官方文档编写。当前脚本的行为与改进建议分别说明；示例代码没有写入你的 Python 文件。本次没有启动 GPU 训练、登录账号或上传实验，也没有验证当前训练脚本已经能够完整运行。官方网页目前可能显示 CoreWeave Forge 品牌，项目与 run 的入口以终端输出的实际链接为准。

## 1. 整体思路：W&B 在训练中负责什么

你的程序负责生成回答、计算奖励、反向传播和更新参数；W&B 负责把你交给它的数据记录下来，并提供查看、比较和导出的界面。

```text
训练程序计算 loss、reward、entropy 等
                  ↓
run.log({...}, step=step)
                  ↓
记录到本地 wandb/，在线模式下同步到服务端
                  ↓
在网页查看曲线、配置、日志和回答表格
                  ↓
比较不同 seed、学习率或算法的实验
```

`wandb.init()` 不会替你计算 reward，也不会自动改变学习率或执行梯度更新。W&B 可以自动采集部分系统信息，但业务指标仍需要你的代码显式记录。[官方日志说明](https://docs.coreweave.com/products/wandb/track/log)

### 1.1 核心概念

| 名称 | 含义 | 在本作业中的例子 |
|---|---|---|
| entity | 实验所属的用户或团队空间 | 你的个人或课程团队空间 |
| project | 收纳相关实验的项目 | `cs336-grpo` |
| run | 一次独立的实验记录 | 用一个 seed 跑一次训练脚本 |
| run name | 方便阅读的显示名称，可以重复 | `grpo-r1zero-lr1e-5-seed0` |
| run ID | 标识 run 的唯一 ID | 用于定位、同步或恢复某条记录 |
| config | 本次实验的配置 | 学习率、seed、group size |
| metric | 随训练变化的指标 | `train/reward`、`validation_reward` |
| step | 指标对应的位置 | 已完成的参数更新次数 |
| summary | 实验的汇总值 | 最后一次或指定聚合方式得到的指标 |
| Table | 结构化表格 | 问题、模型回答、参考答案和奖励 |
| Artifact | 可以版本化管理的文件或目录 | 显式上传的模型 checkpoint |

在你当前实现中，一轮 rollout 后执行一次 `optimizer.step()`，因此训练 step 与 rollout 轮次一一对应。这个对应关系来自你的训练代码。

## 2. 安装、登录与运行环境

### 2.1 在实际执行训练的环境中安装和登录

项目的 `gpu` extra 已包含 `wandb>=0.26.0`，`plots` extra 也包含 W&B。历史运行的依赖记录中出现过 `wandb==0.26.1`，实际版本应在你当前环境中查询。

在已经安装 W&B 的 Python 环境中，可以执行：

```bash
python -c "import wandb; print(wandb.__version__)"
wandb login
```

如果使用本项目的 Linux GPU 环境和 uv，可以执行：

```bash
uv sync --extra gpu
uv run --extra gpu wandb login
```

以上是供你手动使用的命令，本文编写过程中没有执行安装或登录。

根据登录提示，从你的 W&B/Forge 账号设置中获取 API key，并在终端输入。API key 用于认证，不应写进 `.py` 文件、Git 提交或实验 config。[官方入门说明](https://docs.coreweave.com/products/wandb/quickstart)

### 2.2 本地登录与远程登录的区别

浏览器已经登录账号，不代表远程训练进程已经取得认证信息。训练在哪台机器、哪个容器中执行，就需要让那个环境能够获得凭证。

常见方式有：

- 在训练机交互式执行 `wandb login`。
- 由远程平台的 Secret 功能注入 `WANDB_API_KEY`。
- 在可信的运行环境中配置 `WANDB_API_KEY` 环境变量。

你的 `modal_utils.py` 使用了名为 `wandb` 的 Modal Secret，并通过 `secrets=[wandb_secret]` 传入远程函数。使用这条路径时，平台中对应的 Secret 需要提供有效的 `WANDB_API_KEY`；本地浏览器登录不会自动替代这一步。当前文件还保留了 `SUNET_ID = "TODO"` 检查，使用 Modal 前需要按课程约定处理该配置。

### 2.3 第一次检查记录链路

先用一份很小的模拟日志确认账号、网络和项目位置，再投入完整训练。下面是独立示例，不加载模型，也不使用 GPU；其中的 loss 和 reward 都是人为构造的数据。

`wandb.init()` 返回本次实验的 `Run` 对象，保存在 `run` 中；后续操作都针对这条 run。

```python
import wandb

with wandb.init(
    project="cs336-grpo",
    name="logging-check",
    config={"purpose": "logging-check", "seed": 0},
) as run:
    print("Run URL:", run.url)
    for step in range(3):
        run.log(
            {
                "demo/loss": 1.0 / (step + 1),
                "demo/reward": 0.1 * step,
            },
            step=step,
            commit=True,
        )
```

这个例子只能验证日志链路。网页上出现这两条模拟曲线，并不代表 GRPO 实现已经通过验证。

## 3. 初始化实验：`wandb.init()`

### 3.1 参数和返回值

| 参数或返回值 | 含义 |
|---|---|
| `project` | 把 run 放入哪个项目 |
| `entity` | 把 run 放入哪个用户或团队；省略时采用账号默认设置 |
| `name` | run 的显示名称 |
| `config` | 要保存的配置字典 |
| `group` | 将同一实验条件下的多次运行归为一组 |
| `tags` | 便于筛选的标签 |
| `mode` | 如 `online`、`offline`、`disabled` |
| `id` | 指定 run ID，常用于恢复已有日志 |
| 返回的 `run` | `wandb.Run` 对象，用来记录指标、表格和汇总值 |

下面这些参数的详细行为以当前安装的 SDK 为准。[官方初始化 API](https://docs.coreweave.com/products/wandb/ref/python/functions/init)

### 3.2 你的当前写法

整理时，`main()` 中的初始化代码是：

```python
with wandb.init(
    project="cs336-grpo",
    name=f"standard-on-policy-seed-{seed}-experiment-{1}",
    config={
        "seed": seed,
        "n_eval": n_eval,
        "rollout_steps": rollout_steps,
        "microbatch_size": 4,
    },
) as run:
    ...
```

执行顺序是：创建一条 run，记录 config，然后进入缩进块运行训练。退出 `with` 时，SDK 会结束这条 run 并处理剩余日志。

这里的 `experiment-{1}` 总是产生 `experiment-1`，不会自动给实验编号。每次单独启动脚本通常仍然会创建新的 run ID，因此相同名称的 run 可以同时存在。

### 3.3 config 是记录，不会自动控制你的训练变量

如果本地写了 `learning_rate = 1e-5`，但 config 中写了另一个值，而优化器仍读取本地变量，训练就仍然采用本地变量。W&B 不会自动把 config 的内容注入优化器。

你当前主要采用“本地变量控制训练，config 保存这些变量”的方式。建议保持两者一致。下面是可选的记录示例，需要放在对应变量和 `training_data` 已经定义之后；它尚未替换当前代码。

```python
experiment_config = {
    "model_id": model_id,
    "prompt_name": "r1_zero",
    "algorithm": "grpo_on_policy",
    "seed": seed,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "adam_betas": list(betas),
    "question_batch_size": B,
    "group_size": G,
    "response_batch_size": N,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "microbatch_size": m,
    "rollout_steps": rollout_steps,
    "n_eval": n_eval,
    "training_pool_size": len(training_data),
    "sampling_temperature": sampling_params["temperature"],
    "sampling_max_tokens": sampling_params["max_tokens"],
    "max_grad_norm": max_grad_norm,
    "dtype": "bfloat16",
}
```

记录 `len(training_data)` 可以反映实际数据池大小。仅仅声明 `n_train`，并不说明数据池已经被限制到那个数值。

官方说明：[配置实验](https://docs.coreweave.com/products/wandb/track/config)。

## 4. 记录指标：`run.log()`、step 与 commit

### 4.1 参数和返回值

```python
run.log(data, step=None, commit=None)
```

| 内容 | 含义 |
|---|---|
| `data` | 字典，键是指标名称，值是数字、表格等可记录对象 |
| `step` | 这批数据对应的 W&B 步数 |
| `commit` | 是否结束当前 step 的记录 |
| 返回值 | `None`；结果写入 run，不通过返回值提供 |

你当前用 `step = k + 1` 表示已完成的更新次数。它不是 microbatch 编号，也不是 token 位置。

### 4.2 训练指标的记录示例

下面的片段需要已有 `batch_loss`、`metadata` 和 `step`，不能独立运行。它使用当前训练函数实际返回的 `mean_entropy` 键：

```python
run.log(
    {
        "train/loss": batch_loss.item(),
        "train/grad_norm": metadata["grad_norm"].item(),
        "train/token_entropy": metadata["mean_entropy"],
        "train/reward": metadata["mean_reward"],
        "train/format_reward": metadata["mean_format_reward"],
    },
    step=step,
)
```

这里有两层名称：

- `metadata["mean_entropy"]` 是你的 Python 字典里的键。
- `"train/token_entropy"` 是网页里显示的指标名称。

两者可以不同，但读取字典时，键必须真实存在。`train/` 前缀便于 W&B 把相关指标组织在一起。将标量 tensor 转成 `.item()` 得到的 Python 数字，也更方便确认记录的是单个值。

### 4.3 不传 step 时，W&B 会自己推进计数

连续调用两次未指定 step 的 `run.log()`，默认会产生两个日志 step。它不知道这两次调用是否属于同一次模型更新。

因此，你的脚本显式传入更新次数，是为了让训练指标和验证指标在同一条训练进度轴上对齐。

### 4.4 同一个 step 可以记录多次，但不能回写已完成的旧 step

SDK 的默认规则是：

- 不传 `step` 时，默认 `commit=True`。
- 显式传 `step` 时，默认 `commit=False`，允许向当前 step 累积数据。
- 推进到后续 step 后，不应再试图向已完成的旧 step 补写日志。

因此，你当前在第 10 次更新后先记录训练指标，再用同一个 `step=10` 记录验证指标，这种模式本身可以成立；不能仅因为同一步调用了两次 `run.log()` 就认定验证数据会丢失。

如果希望明确表达“先累计，再完成这一行”，可以采用下面的写法。数值只是示例：

```python
run.log({"train/reward": 0.30}, step=10, commit=False)
run.log({"validation_reward": 0.25}, step=10, commit=True)
```

一旦开始记录 step 11，就不要再把新的记录写回 step 10。对于初学者，一次把同一步的指标合并到一个字典中记录，也很容易检查。

显式 step 的未完成记录可能到下一步或 run 结束时才提交，所以网页短暂落后于终端，不一定代表日志丢失。[官方 `Run.log()` 说明](https://docs.coreweave.com/products/wandb/ref/python/experiments/run#method-runlog)

### 4.5 自定义横轴：可选功能

如果以后训练和评估采用不同频率，可以额外记录 `update_step`，再用它作为曲线横轴：

```python
run.define_metric("train/*", step_metric="update_step")
run.define_metric("validation_reward", step_metric="update_step")
run.log({"update_step": 10, "validation_reward": 0.25})
```

`step_metric` 表示把哪个已记录的指标作为横轴；它不负责更新模型参数。自定义横轴与 W&B 内部日志步数是两个概念，当前作业暂时不需要同时采用很多种计数方式。[官方自定义横轴说明](https://docs.coreweave.com/products/wandb/track/log/customize-logging-axes)

## 5. 本作业记录哪些指标，以及如何读曲线

以下是当前脚本计划记录的指标。整理时存在第 11 节列出的键名问题，需要先处理，不能把这张表理解为“当前版本已经成功记录了全部指标”。

| 网页中的键 | 来源及统计范围 | 阅读重点 |
|---|---|---|
| `train/loss` | 一次更新中累计记录的策略梯度 loss | 与 reward、entropy 一起观察；不要求像普通交叉熵一样单调下降 |
| `train/grad_norm` | 梯度裁剪前，所有参数梯度的整体 L2 范数 | 大于 `max_grad_norm` 不代表裁剪失败；看是否长期异常或出现尖峰 |
| `train/token_entropy` | 所有有效回答位置的平均预测熵，排除 prompt 和 padding | 表示预测分布的不确定性；高低本身不能单独判断模型好坏 |
| `train/reward` | 当前训练 batch 的平均总奖励 | 受抽到的问题和随机生成影响，可能波动较大 |
| `train/format_reward` | 当前训练回答的平均格式奖励 | 判断模型是否学会了评分器要求的格式 |
| `validation_reward` | 固定验证子集的平均总奖励 | 是比较训练效果的重要指标；保持评分器和验证集一致 |
| `validation_format_reward` | 同一验证子集的平均格式奖励 | 与总奖励一起区分格式改善和答题能力改善 |
| `validation_response_length` | 验证回答的平均 token 数 | 更长不自动代表更好，要结合正确率和回答内容 |

在当前 `r1_zero_reward_fn` 中，总奖励要求格式通过且答案正确；格式不通过时，总奖励为 0。因此 `validation_reward` 衡量的是当前评分规则下的成功比例，不应忽略格式约束，把它解释成完全不受格式影响的数学答题能力。

### 5.1 你的验证频率与采样设置

当前代码会：

- 在更新前做一次 `step=0` 的验证。
- 每完成 10 次更新做一次验证。
- 最后一轮更新后做一次验证。
- 每次使用 `validation_data[:n_eval]`，目前 `n_eval=1024`。
- 验证时设为 `temperature=0.0, n=1`；训练采样则使用温度 1.0、每题 8 个回答。

因此，训练奖励和验证奖励的采样方式不同，不要求两条曲线数值相同。验证点之间的连线也不意味着中间每一步都进行了验证。

### 5.2 一个阅读例子

假设格式奖励从 0.2 上升到 0.9，但总奖励一直在 0.05 附近。这个现象说明模型更经常满足格式要求，但没有显示出同等程度的答题成功率提升。接下来应查看回答表格，而不能只凭格式曲线宣布训练成功。

如果训练奖励上升而验证奖励没有相应改善，应检查训练与验证条件、数据抽样和实际回答，再判断是否存在泛化问题。

## 6. 保存模型回答：`wandb.Table`

曲线帮助看趋势，表格帮助看具体回答。你当前在每次验证时，取固定验证集的前 4 条问题，保存 prompt、回答、参考答案和奖励。

### 6.1 参数与返回对象

| 内容 | 含义 |
|---|---|
| `columns` | 列名列表 |
| `data` | 行列表；每行元素顺序与列名一致 |
| 返回的 `table` | W&B 表格对象，可以传给 `run.log()` |
| `sample_rows` | 你从验证结果中构造的 4 行数据 |

下面的片段需要已有 `sample_rows`、`metrics`、`step` 和 `run`：

```python
table = wandb.Table(
    columns=["prompt", "response", "ground_truth", "reward"],
    data=sample_rows,
)
run.log(
    {**metrics, "validation_samples": table},
    step=step,
)
```

`**metrics` 表示把验证指标字典展开并加入同一个日志字典。`validation_samples` 是表格在 run 中的名称。[官方 Table 说明](https://docs.coreweave.com/products/wandb/track/log/log-tables)

### 6.2 查看表格时重点比较什么

选择同一条 run 的训练前后记录，检查同一道题的：

- `<think>`、`<answer>` 等格式是否符合当前评分器要求。
- 最终数字是否正确，是否出现重复、截断或无关内容。
- 推理过程和最后答案是否一致。
- 回答是否变长，以及增加的内容是否真的有帮助。

你的指标基于最多 1024 个验证问题，表格目前只有前 4 条。表格适合做定性检查，不能代表整个验证集的成功率。

你当前每次验证都会新建一个 Table，适合保存当次结果。以后若改为复用并修改同一个表格对象，需要同时了解 SDK 的表格更新与日志模式，不能指望修改本地对象就自动更新云端记录。

## 7. 在网页中查看实验

### 7.1 定位正确的项目和 run

优先打开终端输出的 run 链接，也可以在代码中查看 `run.url`。在线 run 通常有网页链接，离线 run 在同步前没有可访问的在线 run 页面。

进入项目后，先确认：

1. 所属用户或团队正确。
2. project 是 `cs336-grpo`。
3. 当前选中的 run 名称、seed 和创建时间正确。
4. 没有被项目筛选条件隐藏。

### 7.2 常用页面与用途

| 页面或区域 | 用途 |
|---|---|
| Workspace / Charts | 查看指标曲线和表格 |
| Runs | 选择、筛选、比较多次运行 |
| Overview / Config | 查看配置、名称、状态等信息 |
| Logs | 查看终端输出、错误和 traceback |
| System | 查看采样得到的 GPU、CPU 等系统指标 |
| Files / Artifacts | 查看已同步的文件和显式记录的产物 |

菜单名称可能随界面版本变化。本文说明操作目的，不依赖某个固定的按钮坐标。

### 7.3 设置曲线

在 Workspace 中找到目标指标，例如 `validation_reward`。需要新建时，使用添加面板功能选择折线图，并在图表设置中选择横轴和纵轴。

本作业初期建议：

- 横轴使用训练 step。
- 纵轴每张图先放一个主要指标。
- 比较同一个指标的不同 run，先看原始曲线。
- 如果开启 smoothing，记录所用方式，并保留查看未平滑数据的渠道。

平滑只改变展示方式，不会重新训练模型。不要用强平滑掩盖不稳定，也不要把跨 seed 的波动与单条曲线的平滑混为一谈。[官方折线图说明](https://docs.coreweave.com/products/wandb/app/features/panels/line-plot)

### 7.4 系统显存指标与 PyTorch 峰值

W&B 可以采样 GPU 利用率和显存占用，但能否看到所有 GPU，取决于运行环境、权限和设备可见性。对你的双卡程序，GPU 0 训练、GPU 1 推理，两张卡的曲线需要分别看。

系统指标是按时间采样的，可能错过很短的显存尖峰。分析 OOM 时，应结合报错位置和 PyTorch 的峰值统计，不能只看系统图上某一个采样点。[官方自动采集说明](https://docs.coreweave.com/products/wandb/track/log)

另外，你的 `wandb.init()` 位于 Hugging Face 模型加载之后。该 run 的 `_runtime` 不包含初始化之前已经发生的加载时间；之后的 vLLM 启动、验证、生成和训练则会计入。因此 `_runtime` 不等于纯 GPU 训练耗时。

## 8. 比较多次实验：seed、学习率和算法

### 8.1 每条 run 要能说明自己做了什么

可采用下面的命名方式：

```text
grpo-r1zero-lr1e-5-seed0
grpo-r1zero-lr1e-5-seed1
grpo-r1zero-lr1e-5-seed2
grpo-r1zero-lr1e-5-seed3
```

名称便于人读，config 便于筛选和核对，两者都应与实际训练设置一致。把名称改成 `seed1`，并不会自动修改随机种子。

当前脚本将 `seed=0` 写在 `main()` 中，尚未提供 `--seed` 的命令行解析；不能仅在运行命令后追加 `--seed 1` 就认为实际种子已经改变。

### 8.2 用 group 管理同条件下的多次运行

下面是可选的初始化示例，`seed` 需要由你实际运行的程序提供：

```python
with wandb.init(
    project="cs336-grpo",
    name=f"grpo-r1zero-lr1e-5-seed{seed}",
    group="grpo-r1zero-lr1e-5",
    tags=["assignment5", "on-policy"],
    config={"seed": seed, "learning_rate": 1e-5},
) as run:
    ...
```

同一组可以放 4 个 seed 的 run。`group` 只组织记录，不会替你启动 4 次训练。[官方初始化参数说明](https://docs.coreweave.com/products/wandb/ref/python/functions/init)

### 8.3 多 seed 曲线与作业图表

在项目 run 列表中选择需要比较的运行，再在图表的 Grouping 设置中选择分组与聚合方式。对同一条件的多个 seed，可以查看均值，并显示标准差等变化范围。

要区分：

- 标准差：不同 seed 的结果分散到什么程度。
- 标准误或置信区间：关于均值不确定性的另一种统计表达。
- min/max：所选 run 的最小、最大值范围。

图里采用哪一种就写明哪一种。比较学习率时，应按学习率或对应 group 分组；不能把所有学习率混合后的均值当成某个学习率的结果。

比较前还要确认验证集、评分器、prompt、更新次数和横轴含义一致。按训练 step 比较回答学习进度的问题，按实际时间比较回答运行速度的问题。[官方分组曲线说明](https://docs.coreweave.com/products/wandb/app/features/panels/line-plot#visualize-average-values-on-a-plot)

### 8.4 导出与 Sweeps

图表菜单和 run 数据页面提供相应的导出选项。导出后，应注明指标、横轴、seed 数量、平滑与聚合设置。`wandb-summary.json` 只有汇总信息，不能替代完整训练曲线；需要完整时间序列时，应使用官方的数据导出或 Public API 功能。

Sweeps 用于自动安排超参数搜索。你的当前脚本没有将学习率等参数接入 Sweep 配置读取，因此仅创建一个 Sweep 不会自动改变本地写死的学习率。先做好单次运行的配置记录和手动对比，再考虑接入这项功能。[官方配置说明](https://docs.coreweave.com/products/wandb/track/config)

## 9. 本地文件、离线模式与恢复

### 9.1 本地 `wandb/` 目录

常见结构如下；具体文件会随 SDK 版本和运行方式变化：

```text
wandb/
  run-时间戳-runID/
    run-runID.wandb
    logs/
    files/
      config.yaml
      wandb-summary.json
      wandb-metadata.json
      requirements.txt
      media/
```

| 内容 | 用途 |
|---|---|
| `config.yaml` | 实验配置记录 |
| `wandb-summary.json` | 汇总指标，通常包含各指标最后记录的值 |
| `wandb-metadata.json` | 环境和硬件等元数据 |
| `requirements.txt` | 依赖版本记录，是否生成取决于设置 |
| `media/` | 表格等媒体数据 |
| `.wandb` 数据文件 | SDK 用于记录、同步 run 数据的文件 |
| `logs/` | SDK 运行与诊断日志 |

不应通过手动编辑这些文件来伪造曲线或修复训练结果。汇总文件中的某个值，只说明该值被记录过，不证明训练算法正确。

### 9.2 离线记录，再同步

可以为单次命令设置离线模式，先保留本地日志：

```bash
WANDB_MODE=offline uv run --extra gpu python -m cs336_alignment.training_step
```

这个命令会执行完整训练，不是轻量日志测试；当前脚本还应先处理第 11 节中的问题。

联网并完成认证后，可以同步某条本地 run。下面的路径是占位示例，需换成真实目录：

```bash
uv run --extra gpu wandb sync "./wandb/offline-run-YYYYMMDD_HHMMSS-RUN_ID"
```

传入完整的 run 目录，而不是仅传其中的 `files/`。远程容器离线运行时，还需要让本地日志目录在容器结束后保留下来；仅在临时容器中写了日志，不代表退出后仍可找回。

`mode="disabled"` 适合临时关闭 W&B，它与会保留待同步日志的 offline 模式不同。[初始化模式说明](https://docs.coreweave.com/products/wandb/ref/python/functions/init)、[同步命令说明](https://docs.coreweave.com/products/wandb/ref/cli/wandb-sync)

### 9.3 恢复日志与恢复训练是两件事

使用相同 run ID，并根据需要设置 `resume="must"` 或 `resume="allow"`，可以让 SDK 继续已有的在线实验记录。`must` 要求目标记录存在；`allow` 在目标不存在时允许创建。

但这不会自动恢复模型参数、优化器动量、训练步数或随机数状态。真正继续训练，还需要你的程序保存并加载对应 checkpoint，再从正确步数继续记录。当前 `training_step.py` 尚未实现完整的 checkpoint 恢复流程。

## 10. 汇总结果与模型文件

### 10.1 最后一次结果与最好一次结果

默认情况下，一个指标最后一次通过 `run.log()` 记录的值会进入 summary。最后一次验证奖励不一定等于整个训练过程中最高的验证奖励。

如果希望单独保留一个明确命名的最终结果，可以在获得最终 `metrics` 后记录：

```python
run.summary["final_validation_reward"] = metrics["validation_reward"]
```

如果希望让 SDK 维护最高验证奖励，可以在开始记录该指标之前设置：

```python
run.define_metric("validation_reward", summary="max")
```

这两种记录回答不同问题：前者强调最终状态，后者强调记录过程中的最大值。不要在作业报告中混用。[官方 summary 说明](https://docs.coreweave.com/products/wandb/track/log/log-summary)

### 10.2 记录曲线不会自动保存模型权重

`run.log({"train/loss": ...})` 只是在记录指标。要保存模型，应先由你的代码生成 checkpoint 或 `save_pretrained()` 输出，再决定是否作为 Artifact 上传。

下面的可选示例要求 `output_dir` 已经是存在的模型文件目录，`run` 是正在运行的 W&B 实验：

```python
artifact = wandb.Artifact("olmo2-grpo-policy", type="model")
artifact.add_dir(str(output_dir))
run.log_artifact(artifact)
```

这段代码尚未集成到当前训练脚本。模型文件可能有数 GB，上传会增加传输时间和存储用量；只需要分析曲线时，不必同时上传模型权重。[官方 Artifact API](https://docs.coreweave.com/products/wandb/ref/python/experiments/run#method-runlog_artifact)

## 11. 当前代码中特别需要对照的地方

以下结论来自整理时的静态阅读，文件后续改变后需要重新核对。

### 11.1 熵指标的字典键不一致

训练函数当前返回：

```python
metadata["mean_entropy"] = (entropy_sum / response_mask.sum()).item()
```

但 `main()` 的日志代码读取：

```python
{
    "train/token_entropy": metadata["token_entropy"],
}
```

这会在构造日志字典时触发 `KeyError`，W&B 还没收到这批训练指标。建议保持网页指标名不变，把读取侧改为实际存在的键：

```python
{
    "train/token_entropy": metadata["mean_entropy"],
}
```

这只是修改建议，没有自动应用。

### 11.2 config 的 microbatch 大小与实际计算不一致

当前计算是：

```text
B = 32
G = 8
N = 256
gradient_accumulation_steps = 32
m = 256 / 32 = 8
```

但 config 中写的是 `"microbatch_size": 4`。这会让网页显示的实验条件不准确。建议记录变量 `m`，而不是另外写一个固定数字。

### 11.3 声明的训练样本数与实际数据池

当前 `n_train = B * rollout_steps` 得到 6400，但加载过程尚未利用它截取或选择训练数据。每轮仍从完整的 `training_data` 随机抽取 32 个问题，不同轮可能重复。

因此，记录数据池大小时应以实际 `len(training_data)` 为准。6400 次问题抽取、6400 个不重复问题和 6400 条数据组成的池子，是三个不同的统计口径。

### 11.4 当前实现与可选功能的边界

| 功能 | 整理时的状态 |
|---|---|
| 使用 `wandb.init()` 初始化 run | 已有代码 |
| 验证前后同步 vLLM 权重 | 已有代码 |
| step 0 验证，每 10 步及末尾验证 | 已有代码 |
| 验证指标和前 4 条回答表格 | 已有代码 |
| 训练指标记录 | 已有代码，但熵键名不一致会使该处报错 |
| 完整超参数 config | 目前只记录部分，且 microbatch 字段不一致 |
| 自动运行 4 个 seed | 尚未实现 |
| Sweep 驱动超参数 | 尚未实现 |
| 模型 Artifact 上传和完整断点恢复 | 尚未实现 |

## 12. 常见问题排查

| 现象 | 优先检查 |
|---|---|
| `ModuleNotFoundError: wandb` | 运行脚本的 Python 环境是否安装了 W&B；不要只检查另一个终端环境 |
| 没有 API key、登录超时 | 实际训练机或容器是否取得认证信息，网络是否可用 |
| 权限错误或项目找不到 | entity、团队权限、账号默认项目位置是否正确 |
| 新 run 页面没有曲线 | 是否真的执行到 `run.log()`；是否仍在启动或初始验证；是否处于离线模式 |
| `KeyError: token_entropy` | 当前 `metadata` 返回的是 `mean_entropy`；这是 Python 字典问题 |
| 提示向更小的 step 写入 | 是否向已提交的旧 step 补写；是否混用了自动步数和显式步数 |
| 名称相同的 run 越来越多 | name 可重复；每次独立启动通常产生新 ID |
| 验证曲线只有少量点 | 你每 10 次更新才验证一次，先确认是不是预期频率 |
| 网页 microbatch 与预期不符 | 当前 config 写死为 4，实际变量 `m` 已是 8 |
| 网页有 loss，但找不到模型文件 | 指标日志不会自动保存权重；检查是否执行了 checkpoint 保存和上传 |
| GPU 图没有显示最高显存 | 系统采样可能错过峰值，结合 PyTorch 峰值统计与 OOM traceback |
| run 显示 Finished，但奖励很差 | Finished 表示运行完成，不表示模型质量达标 |
| run 异常结束 | 先看 traceback 与运行日志；不能仅凭状态判断是 W&B、GPU 还是算法问题 |

如果怀疑 W&B 拖慢训练，先检查是否在每个 token 或每个 microbatch 都记录大量数据、是否频繁上传大表格或模型。你当前每次更新记录一次标量、验证时记录少量回答，属于比较容易管理的日志方式。[官方日志性能说明](https://docs.coreweave.com/products/wandb/track/limits)

## 13. 每次实验的使用顺序

1. 确认实际训练环境已安装 W&B，认证与项目位置正确。
2. 确认实际超参数、run 名称和 config 一致。
3. 对照第 11 节检查当前脚本的日志问题；本文没有自动修复它们。
4. 新环境先运行第 2 节的模拟示例，确认能看到记录。
5. 启动真实训练，先检查 step 0 的验证和回答表格。
6. 训练期间同时观察 reward、entropy、grad norm 与生成内容。
7. 结束后检查最后的验证点、run 状态和本地/云端同步情况。
8. 比较多 seed 时，统一横轴、分组口径和验证条件。
9. 导出作业需要的图表，并注明均值、波动范围和所用配置。
10. 如果后续还要继续训练，另外保存模型、优化器及必要的训练状态。

## 14. 复习与参考资料

最值得记住的几点：

- `init()` 创建一次实验，`config` 记录条件，`log()` 记录过程，`Table` 记录具体例子。
- `run`、显示名称与 run ID 各有作用；名称相同不等于同一条记录。
- W&B 只会记录你实际提供的训练指标，记录配置也不会自动改变训练变量。
- 你的 step 当前对应参数更新次数，microbatch 不应另行推进同一条训练进度轴。
- 梯度范数超过 1 可能是裁剪前的正常记录；训练 loss 也不必单调下降。
- 看曲线时结合真实回答，看多次实验时结合 seed 波动。
- 日志同步、模型保存和断点恢复需要分别处理。

本次检查包括实际代码阅读、官方文档核对和本文结构检查；没有执行本文中的训练、登录、同步或上传示例。之后代码变化时，优先重新核对第 11 节。

官方参考：

- [快速入门与登录](https://docs.coreweave.com/products/wandb/quickstart)
- [初始化 run](https://docs.coreweave.com/products/wandb/ref/python/functions/init)
- [实验配置](https://docs.coreweave.com/products/wandb/track/config)
- [记录指标与媒体](https://docs.coreweave.com/products/wandb/track/log)
- [Run API：step、commit、finish 等](https://docs.coreweave.com/products/wandb/ref/python/experiments/run)
- [自定义横轴](https://docs.coreweave.com/products/wandb/track/log/customize-logging-axes)
- [记录 Table](https://docs.coreweave.com/products/wandb/track/log/log-tables)
- [折线图与多 run 比较](https://docs.coreweave.com/products/wandb/app/features/panels/line-plot)
- [Summary 汇总指标](https://docs.coreweave.com/products/wandb/track/log/log-summary)
- [离线记录的同步命令](https://docs.coreweave.com/products/wandb/ref/cli/wandb-sync)
- [日志规模与性能](https://docs.coreweave.com/products/wandb/track/limits)
