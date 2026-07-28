# 异步推理功能开发过程

> 分支：`dev/async-inference`（A10_new 子仓与 openpi 主仓同名分支）
> 日期：2026-07-28
> 目标：在 A10 机械臂 + openpi VLA 推理链路上实现异步推理，消除 batch 间的 GPU 空闲停顿。

## 一、背景与问题

### 现有架构（同步）

```
机器人(A10_new C++ TCP:8080)  ←TCP JSON→  Python 桥接器(client/)  ←WebSocket msgpack→  openpi 推理服务(:8000)
```

- A10_new：500Hz RT 关节插值执行 `SET_JOINTS_BATCH`；30Hz 状态线程刷新 `robot_q_`；batch 执行期间冻结状态。
- openpi：`serve_policy.py` → WebSocket → `Policy.infer()` 同步阻塞，返回 `(action_horizon=10, 7)` action chunk。
- 桥接器：`采图 → infer(阻塞) → SET_JOINTS_BATCH → wait idle(阻塞, GPU 空闲) → 下一轮`。

### 核心问题

1. **推理与执行串行**：执行 batch 期间 GPU 空闲，batch 间有「推理延迟」大小的停顿 → 机器人卡顿。
2. **观测陈旧**：batch 执行期间 `robot_q_` 冻结，下一次观测其实是 batch 结束时刻的状态。
3. **相机 exclusive 模式慢**：双相机每帧 open/close 耗时 100–300ms × 2，采图比执行还慢。

## 二、方案选型（与用户确认）

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 异步策略 | **A：双缓冲流水线** | 推理 chunk_{k+1} 与执行 chunk_k 重叠，简单稳健 |
| 桥接器位置 | **openpi 仓库 `client/`** | 复用 openpi-client，与 A10_new 同仓维护 |
| A10 修改 | **两处小改**（为「好效果」） | 刷新状态 + STOP_POLICY |
| 分支名 | `dev/async-inference` | — |
| 观测陈旧度 | v1 接受 1-chunk 陈旧，receding horizon 留 v2 | open-loop chunk 执行的标准权衡 |
| 同步回退 | 保留 `--mode sync` | 对比与回退 |
| 停止方式 | Ctrl+C + `STOP_POLICY` 命令 | 异常退出时安全停机器人 |

## 三、双缓冲流水线设计

### 时序

```
启动: obs → infer → chunk_0 入 buffer
稳态每轮:
  chunk = buffer.pop()           # 上一轮已推理好的
  push chunk 到机器人             # 开始执行(耗时 T_exec)
  ── 重叠窗口 ──
  obs = get_follower_state()     # 当前 chunk 开始时的状态(A10 已改为 batch 期间也刷新, 故新鲜)
  next_chunk = infer(obs)         # 后台推理, 与执行重叠(耗时 T_infer)
  buffer.append(next_chunk)
  wait_policy_idle()             # 等 chunk 执行完
  ── 下一轮 ──
```

### 关键性质

- **无 GPU 空闲间隙**：执行 chunk_k 期间 GPU 推理 chunk_{k+1}。
- **观测陈旧约 1 chunk**：`next_chunk` 用「当前 chunk 开始时」的观测，而它在当前 chunk 执行完后才推送。模型逐 chunk 续接，机器人忠实跟踪即可（ACT/Diffusion Policy 标准 open-loop 做法）。
- **action 为绝对位置**：已验证 `A10Outputs` 输出绝对关节位置（`AbsoluteActions` 把 delta 转回绝对），chunk 间无积分漂移。
- **线程安全**：WebSocket 仅 Inferencer 单线程使用；`RobotTcpClient` 持锁覆盖 send+recv，多线程调用原子串行。

### 饥饿处理

若 `T_infer > T_exec`，Pusher 取 buffer 时为空 → 等待条件变量 → 机器人短暂停顿（不崩溃）。日志输出 buffer 深度便于观测。

## 四、实现内容（分步提交）

### A10_new 子仓（7 个提交）

1. **仓库优化: client/ 包化** — 新增 `__init__.py`；`run_bridge.py` 始终注入 `client/` 到 `sys.path` + 平铺导入，兼容 `python -m client.run_bridge` 与直接运行。
2. **仓库优化: a_passport.txt 密码移出** — 明文密码改为 `${A10_LOGIN_PWD}` 占位符，真实凭据走环境变量或 gitignored 的 `a_passport_local.txt`。
3. **A10 C++: 新鲜观测与优雅停止**
   - `main.cpp`：状态线程去掉 batch 期间跳过逻辑，始终刷新 `robot_q_` 查询缓冲（`send_set_joints` 只更新查询缓冲，不下发电机指令，与 RT 互不干扰）→ 异步推理在 batch 末尾也能取新鲜观测。
   - `a10_tcp_server.cpp`：新增 `STOP_POLICY` 行命令，复用 `A10PolicyTcpCliStop` 语义（置位 `g_a10_policy_tcp_stop_requested` + `clear_policy_tcp_targets_nrt`），运行中的 RT 驱动下一拍退出，机器人停在当前位置。
4. **Python: ThreadedCamera** — `usb_camera.py` 新增后台线程持续抓帧类，`read_rgb` 立即返回最新帧；保留 `USBCamera`(exclusive) 作 fallback。
5. **Python: robot_tcp_client 修复** — `wait_policy_idle` 缺键改为快速失败；新增 `stop_policy()`。
6. **Python: bridge 异步双缓冲** — `--mode async|sync`、`--camera-mode threaded|exclusive`、`--buffer-target`；`_shutdown` 实现 Ctrl+C 优雅停止。
7. **docs: README 更新**。

### openpi 主仓（1 个提交）

- 将 `third_party/A10_new` 注册为 git submodule，跟踪 `dev/async-inference` 分支，锁定到上述最新提交。

## 五、启动方式

```bash
# GPU 端
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_a10_finetune \
  --policy.dir=checkpoints/pi05_a10_finetune/Reach_5_9_1/130000

# 机器人端
./kaanhbin 5866    # CLI 里执行 policy

# 桥接器（机器人 PC）
cd third_party/A10_new/client
python3 run_bridge.py \
  --policy-host <GPU_IP> --policy-port 8000 \
  --robot-host 127.0.0.1 --robot-port 8080 \
  --mode async --camera-mode threaded \
  --hz 10
# 对比: --mode sync   fallback: --camera-mode exclusive
```

## 六、验证状态

- Python `py_compile` + AST 全部通过；无 lint 错误。
- C++ 改动小且复用现有函数，`g_a10_policy_tcp_stop_requested` 通过 include 头文件可见。
- **未做实机测试**（无硬件）。建议：先 `--mode sync --camera-mode exclusive` 跑通原闭环，再切 `--mode async --camera-mode threaded` 对比流畅度。

## 七、未做（留 v2）

- **append 语义**：`SET_JOINTS_BATCH` 当前整批替换，batch 间有 ~5ms 轮询间隙；追加语义可无缝衔接。
- **receding horizon**：每 k 步重推理取更新鲜观测（v1 接受 1-chunk 陈旧）。
- **temporal ensembling**：重叠预测加权融合平滑。
