# F1 全身行走 v2 — 用户指定基线

来源：requirements.md 的 v2 增量、owner_training_profile.json 及当前聊天明确继续 Loop。Framing A，沿用 design-f1-fullbody-walk-v1.md 的原始 G1 velocity + PPO 映射；没有新论文或架构搜索。

| 组件 | 当前映射 | 验证 |
|---|---|---|
| 资产 | F1 SSOT 训练副本，关节29项显式映射，原mesh不变 | 哈希、FK、实际Isaac导入 |
| 观测/动作 | actor传感器可得项，critic保留速度真值；显式policy组和关节顺序 | runtime导出、新权重反馈、CPU实际加载 |
| PD与被动项 | 用户文本与截图；训练URDF friction0/damping0.05；不加额外armature/辨识/交付滤波 | owner配置、effective_control、原生运行时有效增益 |
| 奖励/PPO/课程 | 原始G1代码，非Option B | source_manifest、实际生效配置 |
| 参数辨识 | pending/null，后期用户测试后注入 | 不用工程设置冒充辨识 |
| 平台交付 | 单PT内嵌契约和实际反馈，按本轮注册计数下载 | task身份、实际torch.load、有限性 |

r1仅接入：100更新、model99、新权重加载、100步真实反馈；不是G1效果结论。训练上限是工程验证设置。

后续首版计划：接入通过后，从零按冻结G1 checkpoint_iter10999对应的11000次更新运行（10999+1；来源configs/reference/g1_11k_manifest.json）；这是与参考版本对齐的有限训练长度，不是收敛保证或费用估计。仍以实际账户剩余额度为资金边界，不新增充值和无限账号轮转。训练结束回收新模型，再进入同指令速度/姿态/足接触和视频对比；定量行为阈值尚无G1同协议测量，不编造通过线，不把reward或完成轮数当作达标。

首次有效F1行走视频需owner确认方向；最终臂腿配合仍是目标，首版允许弱摆臂。用户负责真机。
