# F1 从 12700 续训行走中双向转弯

## 本轮依据

同场景、固定 5 ms 的纯前进回放中，`12700` 左脚承重侧倾绝对均值为 5.23°，`11999` 为 16.28°。固定前进速度纠偏时，`12700` 的横向误差 RMS 为 1.581 m，`11999` 为 0.169 m。选择 `12700` 作为续训父模型，保留 `11999` 作纠偏性能对照。来源为 `exp-20261119-006/analysis/review_12700/existing_replays_review.json` 的 `cases.*.feet.left.loaded_abs_mean_deg`，以及同目录 `fixed_vx_comparison.json` 的 `*.navigation.feedback.cross_track_rms_m`。

在相同的直行前缀之后，`12700` 收到零、正、负三种转向指令，实测净偏航角速度依次为 +0.0776、+0.0698、+0.0605 rad/s。两种转向指令都比零指令时更偏向负方向，负指令下仍在向正方向转。用同实验 `output/intermediate12700_{straight,wz_left,wz_right}/trajectory.csv` 的 `t`、`yaw` 字段计算，对 `10.5 < t <= 23` 内的航向角解缠后取净变化除以实际采样时间跨度。

## 训练改动

沿用上一轮从 `12700` 续训的两个目标环境组和原有站立比例。按环境索引模 10 分组，组 0 和组 1 中的非站立环境先接收 5 秒 `(vx, vy, wz) = (0.5, 0, 0)`，随后保持 `vx = 0.5 m/s`、`vy = 0`，分别施加 `wz = +0.1`、`−0.1 rad/s`，直至本次 10 秒指令周期结束。重采样后先恢复直行。5 秒是本轮选用的半周期，转向幅度来自既有匹配回放和父模型指令范围。站立和其他环境沿用原采样规则。配置见 `training_delta.json` 的 `forward_exposure`，原采样周期与站立比例见父模型内嵌 `training_objective.command_sampling`。

两个目标组单独提供航向输入。恢复 `12700` 的奖励、指令范围、策略和 Adam 状态，沿用 r36 的转向采样。

## 动作异常处理

r36 的 `TASK_20261123_107` 因同个环境的原始动作连续放大而中止，依据为 `exp-20261123-002/raw/main_107.log` 的 `[F1_ROLLOUT_OUTLIER]`。阈值继续用 URDF 各关节完整行程除以动作缩放后的平方和，同时检查动作幅值和单步差值，以提早捕获持续放大。

有限越界时，只对异常环境把本步作用于仿真的动作替换为默认姿态目标。该环境经 Isaac Lab 的失败终止项复位。PPO 记录原始采样和对数概率，显式去掉异常行的超时 bootstrap。正常环境仍走原动作路径。恢复逻辑仅在训练采样期间启用，保存后反馈使用原策略。恢复事件保存到 `params/rollout_recovery.jsonl`。

原始动作范数、PPO 值或采样对数概率非有限时直接中止。每次 PPO 更新容许的异常环境步数上限取实际 `num_steps_per_env`，也就是平均每个采集步至多一个环境。超限前写 `params/rollout_outlier.json` 并中止。PPO 原有有限值检查继续启用。

## 训练接入

- 本地训练根目录为 `6_sim-train/F1-29DOF/sim_walking`。正式请求已写入 `configs/run_request.json`，训练增量见 `configs/training_delta.json`。请求沿用前一轮的 1000 次 PPO 更新，来源为 `exp-20261120-002/input/publish/configs/run_request.json` 的 `max_iterations`。
- 父模型取 `oma-runs/f1-29dof-walking/.oma/experiments/exp-20261119-006/models/model_12700.pt`（`TASK_20261119_307`）。请求中固定了 SHA256、迭代号及优化器状态。随代码提交 `configs/parent_producer_12700/` 的校验快照，模型文件按现有模型交付流程提供。
- 接手方为本次正式续训登记实验号，核对推送分支实际提交中的 `configs/launch.json` 已启用 training，再走现有 `scripts/rsl_rl/gm_entry.py`。入口按哈希查找父模型。使用 Isaac Lab 原生入口时，通过 `scripts/rsl_rl/train.py` 提供 `--resume --checkpoint <model_12700.pt的绝对路径>`，并使用请求指定的环境数、迭代数和随机种子。
- 启动日志核对 `[F1_RESUME]`、`[F1_FORWARD_EXPOSURE]`、`[F1_ROLLOUT_RECOVERY_CONFIG]`。训练后检查恢复事件是否集中于同类指令、PPO loss 和中间模型。存档后反馈窗口按配置取 300 个策略步，覆盖预定转向切换点。CPU 测试覆盖采样与终止交接，原生 GPU 运行由训练接手方验证。

## 回放与保留候选的条件

使用服务器 MuJoCo，沿用 `12700` 已测的固定 5 ms 场景、初始化和控制配置。用 `exp-20261119-006/output/intermediate12700_*` 的直行和转向指令时序，以及 `exp-20261120-001/input/r30_12998_fixed_vx_feedback_protocol.json` 的固定前进纠偏和停车配置。

| 观察项 | 本轮候选条件 |
|---|---|
| 带前进的正负转向 | 三条回放采用相同的直行前缀和初态。实测净偏航率满足 `正指令 > 零指令 > 负指令`，且正、负指令分别实际转为正、负方向 |
| 路线纠偏 | 横向误差 RMS、平均绝对航向误差均低于同模型的纯前进结果，并较 `12700` 固定速度纠偏改善。对照 `11999` 的同协议结果 |
| 步态与速度 | 直行和纠偏全程无跌倒、持续交替迈步。对照 `12700` 的前进速度 MAE、左脚承重侧倾和首触地侧倾，检查抖动、侧倾长尾和是否靠减速获得纠偏 |
| 停车 | 最后 2 秒双脚持续接触。对照 `12700` 双侧踝关节角速度 RMS 和基座水平速度 RMS。采样窗见 `existing_replays_review.json` 的 `last_2s_ankle_pitch_rms_deg_s` 与 `fixed_vx_comparison.json` 的 `stop_ankle_pitch_rms_deg_s` |

纯直行允许残余偏航。路线纠偏同时验证航向和横向误差最终能否收回，静止横移和原地转向不计入本轮。
