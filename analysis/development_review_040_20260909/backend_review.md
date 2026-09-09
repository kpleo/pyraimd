> 归档说明：分项复核创建的脚本与结果已保存到当前目录，脚本的夹具路径已改为相对仓库定位。下列临时目录及计时数值来自原复核执行；主审已检查相应实现和结果，未重复整套后端反例。

Pyramid 0.4.0 后端复核（只读，R5 / R6）

仓库 `/Users/pengkang/Research/Project/Project_PYRAIMD/pyraimd2`；开始、结束均核实 HEAD 为 `9c7dc4baba3c88544b85e254547b17b84a49196c`，分支 `dev/0.4.0`。对照 `docs/development_reports/INDEPENDENT_REVIEW_20260909.md`、`REVIEW_FIXES_BACKEND_20260909.md`、`REVIEW_FIXES_CORE_20260909.md`。tracked/staged diff 均为空。只创建 /tmp 文件，没有安装、云任务、真实 SCF、全套测试或源码修改。

**结论：旧 R5 显性反例多数已真正修好；通用 ASE 身份仍有同根碰撞。R6 的 CLI/配置工作流正常重试路径已修，但 direct API 组合、实际启动边界和耗时汇总仍不能整体关闭。**

特别区分三类证据：

- **主审已确认的 CLI 19/19 生产记录**：本分项不重复审计、不否定该证据。
- **本分项实际复现通过的配置工作流**：两种 QE 后端的 CLI singlepoint、配置驱动 relax/MD 均正确计内部重试。
- **主审 direct API 反例**：直接构造 QeEngine 不注入 event_log，EnergeticRunner 有 event_log，仍启动 2 次但记 actual=1/failed=0。这与已修的 CLI 路径并不矛盾，详见 F2。

核心恢复、在线更新、能量契约理论和导出不在本次范围。恢复只静态查看 event_log 注入接线，不给恢复正确性背书。

**脚本与证据**

- [R5/后端执行边界脚本](pyramid_040_backend_tiny.py)；[JSON](pyramid_040_backend_tiny.json)；[原始输出](pyramid_040_backend_tiny.log)。数据目录 `/tmp/pyramid-040-backends-h71lms0g`。
- [R6 普通工作流脚本](pyramid_040_workflow_tiny.py)；[JSON](pyramid_040_workflow_tiny.json)；[原始输出](pyramid_040_workflow_tiny.log)。数据目录 `/tmp/pyramid-040-workflow-raeohgt3`，每个案例的 stdout.log 和 events.jsonl 均保留。
- [路径/输入预检脚本](pyramid_040_input_tiny.py)；[JSON](pyramid_040_input_tiny.json)。数据目录 `/tmp/pyramid-040-inputs-wieyzhr_`。
- 主审已复现、本人读取并核对接线原因：[direct API 证据](previous_probe_recheck.json)，其中仅使用 retry_accounting 部分。本次没有重复运行该主审反例。

在目标仓库目录运行（后二者导入第一个脚本中的 fake helper，三个文件须同在当前证据目录）：

```sh
UV_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python analysis/development_review_040_20260909/pyramid_040_backend_tiny.py
UV_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python analysis/development_review_040_20260909/pyramid_040_workflow_tiny.py
UV_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python analysis/development_review_040_20260909/pyramid_040_input_tiny.py
```

使用已有 Python 3.12.13 / ASE 3.29.0 / NumPy 2.5.2；fake 程序读取仓库 Si 输出夹具并计实际进程进入次数，不计算 DFT。脚本输出当前行为与反例，不以脚本 exit=0 冒充缺陷已修复。未运行 pytest，全套测试由主审负责。

**通过：旧问题的定向复核**

- **R5 成功门槛**：[qe_engine.py:570](../../src/pyraimd2/engines/qe_engine.py:570)、[ase_qe.py:310](../../src/pyraimd2/engines/ase_qe.py:310)。正常 fixture 两路径成功且标签 force_consistent=True；缺 JOB DONE、缺/坏 stress、重复原子编号均拒绝。这里确认“拒绝坏标签”，并不等于执行计数正确，ASE 缺 stress 的额外启动见 F3。
- **R5 目录续号**：[qe_engine.py:531](../../src/pyraimd2/engines/qe_engine.py:531)、[ase_qe.py:208](../../src/pyraimd2/engines/ase_qe.py:208)。两路径两次求值后重建实例，第三次求值成功；实际目录为 eval-000000/000001/000002，不再撞号。原子 mkdir 占位竞态防护仅静态确认，未跑并发压力测试。
- **R5 磁矩/电荷映射**：[qe_engine.py:292](../../src/pyraimd2/engines/qe_engine.py:292)、[ase_qe.py:217](../../src/pyraimd2/engines/ase_qe.py:217)。同一 Atoms 的磁矩 [1,-1]、净电荷 0.5，两路径实际写出 nspin=2、ntyp=2、两项 starting_magnetization 和 tot_charge=0.5；随后置零再算，相关设置均清除、ntyp=1。非共线磁矩两路径均在启动前拒绝，新增进程/attempt=0。
- **R5 确定性重试分类**：[qe_engine.py:556](../../src/pyraimd2/engines/qe_engine.py:556)、[qe_engine.py:943](../../src/pyraimd2/engines/qe_engine.py:943)。输出未收敛并 exit1，max_retries=3，两路径均仅启动一次；旧四次重复已经修复。
- **R5 暖启动与清理**：[ase_qe.py:272](../../src/pyraimd2/engines/ase_qe.py:272)、[ase_engine.py:104](../../src/pyraimd2/engines/ase_engine.py:104)。两路径首算 atomic、次算复制密度；注入密度失败后记录 atomic→density失败→atomic成功，总计三次，原文件/来源保留。不再出现 Espresso.reset AttributeError。ASE 非默认 timeout_s 在构造时拒绝，未虚称支持超时。
- **R5 相对路径及基本指纹**：[qe_engine.py:513](../../src/pyraimd2/engines/qe_engine.py:513)、[ase_engine.py:170](../../src/pyraimd2/engines/ase_engine.py:170)。相对 pseudo_dir 在构造后固定；改变父进程 cwd，fake 仍从其真实子进程 cwd 读到正确赝势。赝势内容变化改变指纹；直接 LJ epsilon 1/2 的指纹不同。通用包装计算器残留见 F1。
- **R6 CLI/普通成功路径接线**：[md.py:610](../../src/pyraimd2/workflows/md.py:610)、[md.py:676](../../src/pyraimd2/workflows/md.py:676)、[md.py:792](../../src/pyraimd2/workflows/md.py:792)。qe 与 qe-ase 的 CLI singlepoint 首次失败、重试成功：fake=2、attempt=2、logical=1、actual=2、failed=1。relax 同结果。普通 MD 两步、初始求值首次失败后成功：fake=4、attempt=4、logical=3、actual=4、failed=1；所有 attempt 都绑定已有 task。该证据说明修复确已接入普通成功路径。

**已复现残留（新 HEAD 行号）**

**F1 — [P1，R5/指纹] serializable parameters 不等于完整物理身份。**

位置：[ase_engine.py:89](../../src/pyraimd2/engines/ase_engine.py:89)。任何可 JSON 化的 parameters 都被认作完整身份；ASE 自带 SumCalculator 的 parameters={}，真正的子计算器/权重在其他成员中。

实际证据：`AseEngine(SumCalculator([LennardJones(epsilon=1)]))` 与 epsilon=2 的指纹相同，均为 `ase:sumcalculator:f169a6295e3718b4:force_consistent=False:stress=False`，相同 Ar2 能量分别 -0.3148571525343358 / -0.6297143050686717 eV。后端 JSON 的 `fingerprint_ase_sum`；直接 LJ 对照指纹已不同。这是旧“物理参数改变但身份不变”的同根残留，不是要求新增一种后端。

修正/最小停止条件：对未声明完整可识别状态的 wrapper/自定义 Calculator 返回 None 或要求显式 identity；如果支持组合计算器，则身份必须包括子计算器和权重。保留直接 LJ 已过的用例；SumCalculator 例不得再得到两个相同的非空“可信指纹”。不要求为所有 ASE 类新增序列化适配器。

**F2 — [P1，R6] direct API 有 Runner 日志、无 engine 日志时，签名被误当成上报保证。**

位置：[events.py:84](../../src/pyraimd2/runtime/events.py:84)、[qe_engine.py:765](../../src/pyraimd2/engines/qe_engine.py:765)、[energetic.py:988](../../src/pyraimd2/loop/energetic.py:988)。physical_attempt 仅检查 compute 接受 request_id，随后不再发事件；QeEngine._event_log=None 时 _emit_attempt 直接 return。Runner 构造仅把日志给 calculator，并不连接到内置 engine。

实际证据：主审 `/tmp/pyramid_040_main_probe.json` 的 retry_accounting：fake=2，records=[failed,success]，logical=1、actual=1、failed=0。源码因果与该证据一致；本分项的 CLI 对照得到正确 1/2/1。

支持范围判断：这是公开 Python 组合，QeEngine 的 event_log 参数为 optional（qe_engine.py:712、725），Runner 也未声明/检查必须双重传同一日志；energetic.py:318 还明确承诺有 event_log 时记录实际执行。不能在运行已接受之后把它追认为用户误用。

修正/最小停止条件：对内置 QE 显式连接相同事件接收器，或由调用接口传递 attempt sink/消费后端执行记录；不要仅凭 request_id 签名假定接线成功。若确实不支持某种组合，应在进程启动前明确拒绝。该 direct API 例必须变为 logical=1、actual=2、failed=1；CLI/主审 19/19 保持。只回退成“外层一条 attempt”仍掩盖内部重试，不能作为修复。

**F3 — [P1，R6/ASE] 一次 ASE 适配器调用仍不等于一次真实进程启动。**

位置：[ase_qe.py:309](../../src/pyraimd2/engines/ase_qe.py:309)、[ase_qe.py:352](../../src/pyraimd2/engines/ase_qe.py:352)、[ase_engine.py:190](../../src/pyraimd2/engines/ase_engine.py:190)。现在每次 super().compute 对应一条 attempt，未在真实执行处观测启动。

实际证据 A：缺少 stress 的完整数值输出，max_retries=0，AseQeEngine 实际 fake=2，却只有 1 个 failed attempt。原因是先读能量/力后 get_stress，ASE BaseCalculator.get_property 发现 stress 不在 results，会再次 calculate/execute；同一目录输出也被重复执行覆盖。最终拒绝标签是正确的，但不能称“只启动一次”。

实际证据 B：绝对路径不存在的可执行文件，真实启动=0，ASE 路径仍发出 1 个 physical_attempt；普通 singlepoint ledger 也记 actual=1、failed=1。后端 JSON `stress_missing`/`no_executable`，工作流 JSON `qe-ase/singlepoint/noexe`。

修正/最小停止条件：在 ASE 真正执行入口记录启动/结束；每个逻辑 attempt 明确执行一次、再读取整份结果，避免属性 getter 的隐式再算。输入写入失败或 exec 未发生应 actual=0；缺 stress 应拒绝且实际启动数、attempt 数一致，max_retries=0 时不得额外执行。保留失败诊断与每次实际执行的目录身份。

**F4 — [P2，R6] 全新日志的启动前失败被误判为旧日志中的一次 SCF。**

位置：[costs.py:58](../../src/pyraimd2/runtime/costs.py:58)、[costs.py:103](../../src/pyraimd2/runtime/costs.py:103)。以“整个日志是否存在任一 attempt”识别 legacy，不能区分新的零启动运行。

实际证据：配置驱动 `qe` singlepoint、可执行文件不存在。fake=0、attempt=0、task_failed=1，但 ledger actual=1、failed=1。QeEngine 正确没有发 physical_attempt，错误来自汇总 fallback；不能只修 ASE 的 F3。见工作流 JSON `qe/singlepoint/noexe`。

修正/最小停止条件：按明确 schema/事件协议或任务的执行记录方式区分新旧日志，不依据任一 attempt 是否曾经出现。新日志首个启动前失败必须 logical=1、actual=0、failed_attempts=0；保留独立 legacy 兼容测试。手写 Popen 的 FileNotFoundError 还应转换为契约内 EngineError，不必把预检失败做重试。

**F5 — [P2，R5/R6] SCF 完成后 density manifest 写入失败会漏掉物理 attempt。**

位置：[qe_engine.py:982](../../src/pyraimd2/engines/qe_engine.py:982)、[ase_qe.py:323](../../src/pyraimd2/engines/ase_qe.py:323)。写 manifest 位于成功判定之后、发 attempt 之前，却不在保证终结事件的异常保护范围内。

实际证据：fake 写出正常 SCF fixture，同时创建名为 density_manifest.json 的目录以注入写文件失败。两路径均实际启动=1、attempt_events=0，抛 IsADirectoryError；native record 为 failed/error=None，ASE record 仍为 running/error=None。见后端 JSON `manifestdir`。这是可复现的后处理 I/O 故障，不声称正常 QE 自然创建该目录。

修正/最小停止条件：一旦进程确已启动，无论解析或副产物写入如何失败，都终结并持久化其 attempt（失败分类可以区分进程成功与后处理失败），保留原错误且不残留 running。未启动则不能伪造 attempt。该注入必须有 1 条实际执行记录；不能依赖 legacy task fallback 碰巧补数。

**F6 — [P2，R6] 普通 relax/MD 首次求值失败缺失父逻辑任务。**

位置：[md.py:437](../../src/pyraimd2/workflows/md.py:437)、[md.py:555](../../src/pyraimd2/workflows/md.py:555)、[md.py:817](../../src/pyraimd2/workflows/md.py:817)、[md.py:888](../../src/pyraimd2/workflows/md.py:888)。MD 初始 _evaluate 位于 run 的 try 之前；relax 的逻辑 task 只在成功 on_evaluation 中写入。

实际证据：首次求值输出未收敛，qe 与 qe-ase 的 relax/MD 四个配置均 fake=1、attempt=1、task=0，ledger logical_requests=0、actual=1、failed=1，attempt.request_id='r6-task-1' 无父任务。旧缺陷“失败成本/逻辑请求完整记录”还未闭环。见工作流 JSON `relax/nonconv`、`md/nonconv`。

修正/最小停止条件：在每次逻辑请求的统一 try/finally 中保留预分配 task_id，成功或失败都写相同父 ID；实际执行日志与逻辑失败分开。四例应 logical=1、actual=1、failed=1 且无未绑定父 ID。md.py:500 的后续失败 `_new_task_id()` 还会换父 ID，这是同根静态风险，本轮未额外注入后续步失败。

**F7 — [P2，R6] 正确计数之后，耗时仍有漏计和重复计入。**

密度复制：代码先复制（[qe_engine.py:903](../../src/pyraimd2/engines/qe_engine.py:903)、[ase_qe.py:290](../../src/pyraimd2/engines/ase_qe.py:290)），之后才启动 attempt 时钟（native 917、ASE 307）。[costs.py:79](../../src/pyraimd2/runtime/costs.py:79) 却无条件排除 physical_io，理由是其已包含在 attempt 内。实际两个 warm-start 测试中，io.start+elapsed 明确早于相应 attempt.start；native attempt 总和 0.07915849899291061 s、copy=0.0003692920145113021 s，ledger 仅等于前者；ASE 同样漏掉 0.00036120900767855346 s。这是当前实现的**漏计**，不是再次断言旧版的嵌套重复计时。tiny 文件很小，数值只用于证明包含关系，不用于性能估计。

普通 MD cache hit：[md.py:453](../../src/pyraimd2/workflows/md.py:453) 即使没有计算，也沿用 last_label.wall_time_s。静止 Si 两步：真实启动=1、cache_hit=2、actual=1（计数正确），但 native attempt 耗时 0.02091837499756366 s 被写到两个 cache task，ledger 总耗时 0.06275512499269098 s，恰为 3 倍；ASE 为 0.022942250012420118→0.06882675003726035 s。见工作流 JSON `md/good`。

修正/最小停止条件：按实际互不重叠的时间段汇总；复制在 process span 外时单计 I/O，或调整父 span 并明确协议；cache task 用本次命中耗时，不能继承过去 SCF 耗时。补 real-event 时间区间关系断言，不能仅构造一个“声称已嵌套”的事件列表。计数和耗时两种断言都满足即可停止，不要求相加覆盖全部工作流管理开销。

**仅静态风险/未做运行时背书**

- ASE-QE 仍把绝对赝势名原样交给 Espresso（[ase_qe.py:147](../../src/pyraimd2/engines/ase_qe.py:147)），而手写路径在 pseudo_dir 内使用 basename（qe_engine.py:353）。本次 CLI fake 生成的 ATOMIC_SPECIES 行实际为 84 字符：`Si 28.085 /private/tmp/pyramid-040-workflow-raeohgt3/qe-ase-singlepoint-flaky/Si.UPF`。旧 WP08 曾记录过 QE 长路径卡片问题，但本次没有真实 QE，**不能据此宣布该输入实际运行失败**。最低收口是让两路径共享 basename 映射并做输入层断言，或保留未验证说明；不要求追加云计算。
- 普通 resume 的 event_log 已在 md.py:1053–1055 创建并传给后端；adaptive 接线在 986–998。本轮仅核对接线存在，未运行恢复，不将其写为恢复正确性通过。
- 没有追加并发压力、全系统断电、真实 QE 版本矩阵或新后端支持要求；不把这些当作本轮停止条件。

**交给主审/开发者的最小停止条件**

保留已通过的 R5 反例和 CLI 19/19；依次修正 F1 的不可信身份、F2 的事件接收器接线、F3/F5 的真实启动终结记录，再修 F4/F6/F7 的汇总与父请求。只回归这里已列的 tiny 反例，加主审负责的一次全套测试。验收应满足：可信身份不碰撞；底层实际启动数=physical_attempt 数；零启动=0；每个已发生逻辑请求有正确父 ID；复制和 cache 耗时按真实包含关系汇总。不需要扩展材料计算、后端种类或新功能。
