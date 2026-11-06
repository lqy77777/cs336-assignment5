# 连续实验的 vLLM 资源清理修复

日期：2026-11-06。

## 修改范围

仅修改 `cs336_alignment/vllm_utils.py`：新增标准库 `ctypes` 导入，完善 `VLLMServer.stop()`。另新增本记录。没有修改训练算法、采样参数、变体配置或调参循环。

`tuning_grpo_varients.py` 原有每轮调用 `server.stop()` 的代码会直接使用修复后的逻辑。

## 原问题与真实验证发现

原来的 `stop()` 只结束 vLLM 服务子进程，没有显式释放训练侧 NCCL 权重同步通信器；`start()` 注册的 `atexit` 回调也会继续持有服务器对象。这是资源生命周期缺口；此前没有实测证明它已经导致某次 reward 异常或显存不足。

本次最初尝试在关闭服务前调用 `weight_sync_group.destroy()`。模拟测试通过，但真实验证在第一轮生成完成后停在清理阶段。核对安装源码和 NCCL 文档后确认：`ncclCommDestroy` 需要同一节点上的参与者共同执行，而当前 vLLM 权重传输引擎关闭时只移除引用。只在训练端调用 destroy 会等待。因此没有保留这一版本，阻塞测试进程及其服务已主动终止并确认释放 GPU。

最终采用：先停止自己启动的 vLLM 服务，确认退出后，再用 `ncclCommAbort` 释放训练侧剩余通信器。NCCL 文档说明 abort 会终止未完成操作并释放通信器资源；这里用于推理端已退出、训练侧不再传输权重的清理阶段。[NCCL 通信器管理文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html)

## 具体修改

新增 `import ctypes`，修改后的方法如下：

```python
def stop(self) -> None:
    stop_server(self.process, timeout=self.shutdown_timeout)
    self.process = None
    group = self.weight_sync_group
    if group is not None:
        if group.available and not group.disabled:
            # destroy() waits for peer ranks; the inference process is gone.
            # vLLM's wrapper does not expose abort, so bind the NCCL API.
            abort = group.nccl.lib.ncclCommAbort
            abort.argtypes = [ctypes.c_void_p]
            abort.restype = ctypes.c_int
            with torch.cuda.device(group.device):
                group.nccl.NCCL_CHECK(abort(group.comm))
            group.available = False
            group.disabled = True
        self.weight_sync_group = None
    # Retain the exit callback if cleanup raises, so cleanup can be retried.
    atexit.unregister(self.stop)
```

- `stop_server(...)` 保留原有正常终止、超时强制终止及等待退出的逻辑。
- `group` 是训练侧通信器，`comm` 是原生 NCCL 句柄。当前 vLLM 没有封装 abort 方法，因此使用已加载动态库中的公开 NCCL API；没有修改第三方库。
- `ctypes.c_void_p` 声明句柄参数，`ctypes.c_int` 声明返回状态码，避免错误的指针转换。
- 在通信器所属 CUDA 设备上下文中调用，并通过已有 `NCCL_CHECK` 检查状态码，失败时不静默忽略。
- 成功后标记通信器不可用并清空引用，防止重复释放；最后注销 `atexit` 回调，允许服务器对象回收。
- 服务停止失败时保留服务与通信器句柄；abort 失败时保留通信器句柄。失败均不会执行最后的回调注销，以便重试。

调参脚本后续的 `del optimizer`、`del model`、`gc.collect()`、`torch.cuda.empty_cache()` 保留原样，负责训练模型和 PyTorch 缓存清理。

## 本次验证

### 最终实现的模拟检查

直接导入修改后的类，使用模拟通信器及服务停止函数，结果如下：

- 先停止服务，再 abort：通过。
- 重复停止时，同一通信器只 abort 一次：通过。
- 正常清理后用弱引用确认服务器对象不再被退出回调保留：通过。
- abort 返回错误时异常传播、保留通信器，并可在模拟故障解除后重试：通过。
- 服务停止失败时保留句柄、不提前 abort，随后可重试：通过。
- 未启动、未初始化或通信器已不可用时停止：通过。

### 两轮真实服务验证

已完成，测试进程退出码为 0，输出 `ALL_CHECKS_PASSED`。GPU 0 加载 `allenai/OLMo-2-0425-1B`，GPU 1 运行 vLLM；每轮初始化 NCCL 权重同步、实际同步权重、生成一条回答，再重复调用两次 stop 并检查资源。两轮在同一个 Python 进程中连续执行，不执行训练更新。

| 验证项 | 第一轮 | 第二轮 |
|---|---|---|
| 服务启动、NCCL 初始化和权重同步 | 通过 | 通过 |
| 生成一条回答 | 通过，4 tokens | 通过，4 tokens |
| 服务退出、重复 stop、属性清空 | 通过 | 通过 |
| 弱引用确认服务器对象已回收 | 通过 | 通过 |
| 清理耗时（包含模型回收及 empty_cache） | 约 1.22 秒 | 约 2.85 秒 |
| 清理后 GPU 0 / GPU 1 显存 | 851 / 3 MiB | 851 / 3 MiB |
| 清理后两张卡的 PyTorch memory_allocated | 0 / 0 bytes | 0 / 0 bytes |

两轮清理后的显存占用一致，没有观察到逐轮增长。测试 Python 进程退出后，`nvidia-smi` 显示两张 GPU 显存均为 0 MiB、利用率均为 0%。进程存活期间的少量非张量显存随进程退出释放。

最终 `git diff --check` 通过。代码 diff 为原文件净新增 17 行；项目改动只有这个 Python 文件和本 Markdown 文档。

## 验证边界

修复针对当前安装的 vLLM 0.19.1、NCCL 2.27.5，以及本项目自行启动并停止服务的用法。升级 vLLM 后，应重新检查封装字段以及是否已经提供官方 abort 方法。

模拟故障检查不等同于真实硬件故障注入；不承诺处理所有驱动或 GPU 挂死场景。两轮服务验证也不等于全部 15 组训练已经通过。
