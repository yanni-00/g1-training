# 接入配置

| 文件 | 作用 |
|---|---|
| robot_contract.json | F1 原始资产、模型关节、link、限位、静态姿态；URDF 顺序不代表实际引擎/SDK 顺序 |
| parameter_identification.json | 后期逐关节注入；pending/null 表示没有结果 |
| deployment_contract.json | F1 实际观测、时序、动作保护、SDK 与状态机证据；当前待接入 |
| launch.json | 本地运行开关、明确授权引用与训练规模；当前全部未授权 |
| evaluation.json | 已确认行为和延后项目；真实指令时间表、量化阈值后续冻结 |
| reference/ | G1 方法快照及稳定版实际 runtime 设置，只读来源，不作为 F1 测量 |

辨识填写规范：status=validated；全部 joints 填写 stiffness（位置 P 增益）、pd_damping（控制器 D 增益）、armature（等效关节惯量），使用仿真器 SI 单位；sources 中逐项给 path、sha256、description。armature 可为实测零，stiffness 必须为正。数据必须绑定当前 URDF 哈希。
URDF 自带被动 damping/friction 保留，不能把辨识 total damping 直接再与 URDF 阻尼相加；若测试报告只有总阻尼，先明确拆分语义，再填 pd_damping。delivery_filter 当前 null，非空滤波契约尚未实现时入口明确拒绝。

参数或数据就绪本身不授予启动许可。未来运行还需新的用户授权、OMA 轮次/准入证据和实际 F1 部署闭环。当前不填预算、账号、seed、速度/行为门或运行规模的估计。
