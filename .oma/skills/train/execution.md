# 执行、通知与模型交付

由 `$train`、`$tune`、恢复和平台操作共用。`oma run` 只管理本地执行记录和只读看护；不 create / run / stop / retry 云任务，不发送聊天消息，不修改全局登录。平台适配仍使用既有 flux 路径。

## 状态与作用域

- 当前权限仅存 `loop.json.pausedScopes`，允许值 `loop / monitor / evaluation / retry`，空数组表示无暂停；不另建 pause.json。首次显式编辑权限时读取旧标志并替换为该列表，移除同义的 `loopPaused / monitoringEnabled / automaticEvaluation / automaticRetryEnabled`，不自动迁移其他实例。旧状态只读兼容；孤立的历史 pausedAt 不作为当前暂停依据，损坏权限拒绝自动恢复。
- `pause --scope loop` 暂停决策 / 下一轮，并暂停自动验收和重试；保留只读看护与已提交训练。`monitor` 只停止本地查询（运行中的 watcher 在下一次循环检查时退出），`evaluation` 和 `retry` 可分别控制。恢复只恢复指定作用域，其他暂停保留；例如允许收结果但不继续开轮，可单独恢复 evaluation。
- 云任务取消须有明确针对训练的授权，再经平台操作；“暂停 loop / 看护”、余额不足或会话中断不自动派生 task stop。外部看护接管时暂停本实例 monitor，不重建对方或其他聊天的自动化。
- 进入决策、create / run 或重试前重读 `oma resume --no-query --json` 的 control：loop 关闭不得新决策 / 启动下一轮，retry 关闭不得自动重建或续跑；monitor 关闭不重挂查询。`oma decision register` 强制拒绝暂停时登记，已授权结果的 backfill 可保留；直接 flux 命令仍由执行方遵守此检查，不声称平台 API 已被拦截。

```bash
oma run pause --scope loop --reason "owner pause"
oma run pause --scope monitor
oma run resume --scope evaluation
oma resume --no-query --json
```

## 统一训练准入：登记与启动

已有部署代码是设计约束来源。先反向检查 actor 信号的实际生产者、坐标 / 单位 / 顺序 / 历史 / 归一化 / 时序与有效条件，以及动作到执行器的处理和保护。维度相同、接口可拼接、ONNX parity 均不能代替信号可靠性和真实执行链验证；关键未知不因 loop / standalone 的 advisory 模式自动放行。critic / 奖励 / 评测可用真值，部署 actor 不得偷偷依赖仿真独有状态；可验证的实际估计器路线仍然允许。

本轮 config 的 deployment 描述部署约束，execution 描述实际运行链、请求及资源；设计 / impl 只引用，不再维护平行 ready 状态。新轮先分配 exp_id、落盘 config，再 `oma decision register`；登记检查部署范围与契约，lint 后复核新鲜度。登记不是启动许可。create / run / retry 前统一执行：

```bash
oma run preflight --exp-ref <exp-id> --json
```

非零退出即不能以当前配置启动。命令不访问平台、不执行验证脚本、不持久写“已通过”；它核对来源、身份、证据和权限。已运行任务的终态收集、失败 backfill 不受这道新轮准入阻断，旧实例缺字段保持未知，不回填假通过或改写历史。

### 安装一致性

`oma setup` 与 `oma doctor --installation --json` 共用资产生成逻辑。config.toolchain 只保存 platform / version / source_sha256，源码指纹涵盖运行 CLI、源码、技能、模板及工具；检查将实例实际文件与当前 CLI 所属包重新比较，不对旧文件自签后宣称同版本。首次 setup 创建目标 / memory，升级保留业务配置、目标、记忆、loop 与任务绑定；已有资产不一致时普通 setup 保留并失败，审阅后显式 --force 仅替换工具资产。用户自定义合约使用 --overlay 标记区。旧实例无来源时标未知，不自动升级。

### 最小真实闭环

execution.pipeline 描述稳定运行链；同配置可用 `{ref: ".oma/experiments/<原轮>/config.json#/execution/pipeline"}` 引用原定义，不串链。字段：sources（相关源码 / 环境锁定 / 资产 manifest 的 `{path,sha256}`）、environment（实际镜像 / SDK / 框架版本）、assets（实际资产与数据）、entry、normalization、delivery、rollout。来源路径相对实例根目录或明确外部路径；避免只钉住无关说明。bindings 为上述六类各自的非空 JSON pointer 数组，定位本轮 config 中实际消费的执行字段（例如镜像 / 版本、资产版本、入口参数、归一化开关、保存路径和回放设置）；不重复抄值，也不指向 pipeline 的描述或结果声明。适配器须审计映射确被启动入口消费。与 deployment 契约、解析后的实际字段值共同生成 preflight.pipelineSha256；环境、资产、归一化、保存下载、控制或保护变化使旧证据失效，普通优化器调参不强制重跑整个接入。

- 初次接入允许有界 probe：部署任务沿用 deployment.stage / probe；明确 simulation_only 的任务在 execution.stage / probe 声明相同的 scope、limit_ref、success_criteria、failure_action。实际启动参数必须消费同一上限，不把 probe 当性能验收或无限重试许可。
- formal 通过 execution.readiness_ref 引用原验证轮 `results.json#/assessment/readiness`；部署任务可以直接复用 deployment.proof_ref 所属 readiness，不复制证据。
- readiness.status 与 assessment.training.status 须 passed；源 config 保留同任务 finished 终态；实际新模型已下载、加载、有限性检查通过，身份及字节沿用 checkpointInspection / execution.delivery，不接受原样父模型。
- readiness.runtime 记录 mode=`end_to_end`、scope、pipeline_sha256、checks 与 evidence。checks 的 environment / assets / training / normalization / delivery / load / rollout 逐项通过；evidence 钉住实际日志、原生生效配置、下载加载与连续反馈回放数据。部署导向任务还须通过下述 deployment 子项。
- 从实际启动入口完成环境 / 依赖 / 资产读取、真实训练、保存上传下载、新权重加载、实际输入及动作处理的反馈回放后，才有这些证据。CPU 静态检查、平台 finished、接口 parity、文件存在或手填 passed 不代替闭环；策略不必在接入阶段达到最终步态目标。失败先修接入，不投入正式优化。

### 额度与账号绑定

execution.plan 是实际执行请求：provider=`gradmotion` 时 request 为实际平台请求对象（项目 / 资源 / 镜像 / 入口和执行上限）；provider=`local` 时 command 为实际参数数组。临时请求从它序列化到本轮 work，执行后只保留此原始绑定及 taskId，不复制新账本。凭据不进入 plan。

execution.resource 绑定 preflight 输出的 resourcePlanSha256（写入 plan_sha256）；指纹包含本轮配置和实际请求，排除运行时间、taskId、资源观察及账号绑定。更改训练范围 / 参数 / 请求后旧报价失效。先准备实际计划，运行 preflight 取得指纹，再补充有来源的 resource；未知时命令会非零退出并给出缺项。

| resource 字段 | 含义 |
|---|---|
| mode | 云计费为 metered；本地为 local，必须与 plan.provider 相符 |
| required | metered：amount 为完整注册执行需求的非负整数，unit 为明确最小计费单位，evidence 为原始报价 / 预留 / 有硬上限的计费依据 `{path,sha256}` 数组；不填经验估算或“典型训练费用” |
| candidates | 已授权账号的公开观察数组，每项 account、balance、reserved、unit、observed_at、valid_until、plan_sha256、request_check、evidence；plan_sha256 绑定当前计划，request_check=passed 须有该账号对确切项目 / 请求 / 资源的实际权限及 dry-run 依据，不能仅以余额选到无权访问旧项目的账号；金额同单位非负整数，占用未知不能当零；时限必须来自实际平台观察 / 适配规则，不设基座猜测 TTL |
| account | 每个候选的 `{id,profile,keyEnv,baseUrl}`：公开账号标识、命令级 profile、专用凭据环境变量名和实际服务端地址；不保存原始 key。需由同一账号的 whoami / 计费观察核对身份 |
| scope / evidence | local 必填，说明已分配的本地执行资源与实际依据；不能将云任务改写 local 绕过额度门 |

preflight 离线核对来源 SHA、时效、单位和 `balance - reserved >= required.amount`，优先保留足额的当前账号，否则选择另一个有实际依据的足额候选；不足 / 未知就阻塞。先 `oma run bind-account --exp-ref <id>` 写入同一 config.execution.account，再 preflight 放行创建。绑定只处理公开记录，不读账号池或凭据、不访问平台；create / run 前执行方刷新报价及余额 / 占用，必要时使用平台预留。离线快照不保证并发下原子预留，不新增本地计费调度器。

所有 create / run / query / logs / download 使用本轮 account；适配器复用基座 `resources.platformEnvironment(account)` 给每个子进程独立传 env：专用 keyEnv 值 → FLUX_API_KEY，profile → FLUX_PROFILE，baseUrl → FLUX_BASE_URL。此函数不改 process.env 或全局配置，不打印凭据，缺专用凭据时拒绝回退全局账号。watch / resume 已复用此环境构造；历史任务缺账号绑定时报告未知，不从当前全局登录猜测归属。

已创建 taskId 不跨账号改绑。若启动时额度发生变化，先核对原任务状态和实际原因；中断恢复保留原任务记录 / checkpoint，再为新账号建立新恢复任务并重新准入。余额变化不派生 stop；不能仅凭 failed 判耗尽，不把账户元数据 used 直接写成预算。缺平台报价 / 额度 API 或真实接入证据时报告具体未知，不能宣称本地门已替代真实适配。

### 部署契约

| deployment 字段 | 含义与必要条件 |
|---|---|
| target / basis | `deployment` 或明确的 `simulation_only`；basis 为 `{path, sha256}`，指向用户目标 / 范围的现有出处。不能为绕门把部署任务降格为仿真研究；目标歧义按目标对齐规约解决 |
| scope | simulation_only 必填：具体仿真研究 / teacher / 对照范围；此范围不产生部署成功结论 |
| stage | deployment 下为 `probe` 或 `formal`，不是统一大训练前的无条件 smoke 通行证 |
| contract | 首轮定义下述契约；后续用 `{ref: ".oma/experiments/<原轮>/config.json#/deployment/contract"}` 引用原定义，不复制、不串联引用链 |
| probe | 有界接口验证：scope、success_criteria、failure_action；limit_ref 为 config 内实际执行上限字段的 JSON pointer，必须指向正数。上限是注册设置，不是预计收敛时间；平台适配器须将同一字段绑定实际提交参数，不能另填更大的启动值 |
| proof_ref | formal 必填：`.oma/experiments/<验证轮>/results.json#/assessment/readiness/deployment`；必须是同一契约下实际新权重的部署闭环证据 |

contract 不存“已验证”布尔量，只描述不随普通调参变化的接口与假设：

- sources：实际训练与部署源码 / 相关运行配置的 `{role: training 或 deployment, path, sha256}`；路径相对实例根目录，或明确的外部代码绝对路径。选择当前实际执行的源文件 / 配置，不能只钉住无关脚本或过期副本；源码版本与原生实际生效配置由任务适配器核验。
- observations：逐项 name、audience（actor / privileged）、source（sensor / estimator / reference / simulation）、semantics、conditions。semantics 写明构造 / 维度顺序 / 坐标 / 单位 / 历史与归一化；conditions 写实际可得性、噪声 / 漂移 / 延迟 / 丢包条件和待验边界。不能用一行“完整状态”替代清单。
- pipeline：preprocessing、timing、action、guards、entry 的实际分工与一致性处理。覆盖参考 / 命令生成、采样与控制周期、动作缩放 / 关节映射 / 限位 / 变化率 / PD 包络、状态机进入 / 复位及退出保护。差异须在训练复现、移除不可用依赖或通过同条件验证解释，不能撤保护换通过。

首个可消费的新权重尽早使用实际部署代码离线回放：只能输入部署真实可得信号，使用实际归一化、动作后处理、控制周期和保护，物理反馈持续闭环；同步做原生有效输入 / 动作审计，避免修改了配置文件却没有实际生效。无需先学会完整行走，但关键接口差异未解释不能用“策略尚未成熟”跳过审查。纯真值 / 原始无限幅路径只作隔离诊断，不能作为正式部署训练的放行证据。

复用验证轮 `results.assessment.readiness`：其 status=passed 后，deployment 子项记录 mode=`deployment_closed_loop`、scope、contract_sha256（preflight 输出）、checks（observations / preprocessing / timing / action / guards / entry 各自 passed）和 evidence（实际动态数据 / 原生审计 / 源码执行核验等 `{path, sha256}`）。真实新权重身份复用 results.checkpointInspection；已有 `config.execution.delivery` 可直接使用，不再复制检查记录。字段仍按下文模型交付规约，早期权重保留自身实际 iteration，不冒充最终交付。

readiness 本身仍填实际 scope / evidence 供既有摘要使用；可引用 `results.json#/assessment/readiness/deployment`，具体哈希证据只在子项维护一次。其他分层判定不由该技术准入自动置为通过。

formal 核对来源文件现有哈希、验证轮任务 / config / 模型身份、非空且已加载有限值的新模型、契约指纹和各项闭环证据。它不认证报告内容真实性，也不自动证明估计器精度或实机安全；任务适配器必须实际验证注册适用域，不允许把静态测试填写成动态闭环。行为 / owner / 总体 / 真机授权仍按原门分别判断。

同一契约允许跨普通调参轮复用证据。观测来源、参考、归一化、时序、动作 / PD / 保护 / 状态机、相关源码或适用域变化后，旧证据失效，按差异重新完成有界验证。更新哈希本身不算重新验证；不得改验证轮原契约来伪造新模型血缘。直接调用 flux 不被 Node 门拦截，执行方仍须遵守 preflight 结果；不得为绕门手改账本或切换目标范围。

## 轮次与恢复

`experiments/{exp}/config.json.round` 是面向用户的显式 `rN` 标识；新轮按真实登记顺序及已有编号映射填写。它与 lap、日期型 exp_id、重试 taskId 分开；同一轮重挂看护不增号。loop / trajectory 可镜像同一标识，冲突或缺失时显示“轮次未知”，核对原记录，不用 lap 或日期后缀反推、不改历史编号。

README、恢复摘要、阶段消息使用 `任务名 | rN（第 N 轮） | 阶段 | 状态`，并附 exp_id 供追溯。`oma log / status / resume / run` 共用格式。恢复扫描是报告，不会自动恢复暂停的权限或执行云操作；`--apply` 仍仅修正本地断点 / 镜像，manual 时不写。

`oma resume --write-summary` 不查询平台，只从本地真源更新 README / `.oma/RESUME.md` 的 OMA:CURRENT 区块；保留区块外手写内容，写前核对来源指纹与并发修改。摘要不是新状态文件，缺层级结果为未知，目录顺序不用于跨任务卡猜当前实验。遇 manual 或损坏标记先处理来源问题；普通 resume 默认仍不写文件。

## 终态先通知，再独立验收

```bash
oma run watch --exp-ref <exp-id> --interval 300
# 可后台运行；已有看护不要重复启动。只观察一次可加 --once。
oma run status --exp-ref <exp-id> --json
# 外部看护提供已有平台终态快照时（须有匹配的 taskId / taskStatus）：
oma run watch --exp-ref <exp-id> --file <existing-task-info.json>
# 向用户实际发送终态消息后，记录发送出处（不是等待用户再次批准）：
oma run notified --exp-ref <exp-id> --note "sent message reference"
```

- 看护把平台 `1/2/3/4` 当作非终态，`5/6/7` 才是结束；查询失败 / 未知码不写训练失败。重试查询的连续失败计数在成功观察后清零；达到既有失败界限后退出并保留未知，交由当前 owner 处理。
- 终态先持久写入本轮 `config.json.execution.terminal`（taskId、状态、观察时间、通知记录），再输出通知请求。重启后待通知状态继续可见；实际发送后才运行 notified，重复执行不重复确认。发送与本地记录之间仍可能中断，不能保证消息系统的严格 exactly-once；核对已发消息后补记，不虚构发送。
- 状态 finished 不等于行为通过；先报告训练终态，再单独回收 / 验证模型与行为。失败或 stopped 同样先告知，是否重试遵守任务卡及 retry 权限，不能把行政中断统计成算法假设失败。
- `watch --file` 只读取已有、任务身份匹配的终态快照，不查询平台或改变暂停权限，供 owner / 外部看护交接使用；无来源的“已完成”不能伪造成平台快照。

## 最终模型交付门

启动前在本轮 config 注册 `checkpoint.expectedIteration`（框架实际保存编号，不是盲取最大编号）；续训还登记 `checkpoint.parent` 的实际 sha256 / iteration，以及可恢复的 optimizerStep / normalizerCount。缺失预期信息保留待核验，不用父模型、smoke 或同名其他任务权重代替。

训练任务自然结束后按平台适配器核对保存 / 列表中的确切模型，下载到本轮实验目录。框架适配器实际加载所下载文件、检查张量有限性与元数据，输出以下字段；可复用现有检查文件，或放到既有 results.json 的 `checkpointInspection`，不强制新增一份报告。

| 字段 | 依据 |
|---|---|
| taskId、path、sha256 | 实际源任务 / 模型列表或下载元数据、实验目录内模型路径、检查时文件哈希 |
| iteration、loaded、finite | 实际加载检查；不能仅因文件存在或后缀正确填 true |
| parentSha256 | 续训实际输入权重，与注册父模型一致 |
| optimizerStep、normalizerCount | 框架实际恢复 / 推进计数；注册了父计数就必须验证推进，未提供不能估算 |

```bash
oma run verify --exp-ref <exp-id> --file <existing-inspection-or-results.json>
oma run status --exp-ref <exp-id> --json
```

通用门核对任务、注册终端迭代、目录、文件 SHA、父子身份与已登记计数，并把通过的检查存回 `config.json.execution.delivery`；不新增交付状态 JSON。状态查询重新核对文件内容，文件改变、删除、换任务或父权重不符都会失效。

`canEvaluate=true` 需要：对应任务 finished、终态已实际通知并记录、当前文件交付通过、evaluation 权限允许。这只是进入行为验收的条件，不是总体成功或部署许可。通用 Node 层不加载任意 ML checkpoint；实际框架加载结论来自适配器，报告必须有真实来源，不把通用格式校验宣传成实跑加载。

该门用于本轮最终判定。已预注册的中途 checkpoint 诊断保留自身范围，不为它改写注册终端编号，也不把中途结果报成训练完成。

具体 SDK 保存 / 中继目录、平台模型列表、模型加载程序与远端 play 路径由任务适配器实现。保存了但平台不可列出 / 下载时记为交付失败，保留训练结束事实，先检查已验证上传路径，不断言文件永久丢失。
