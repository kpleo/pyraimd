> 归档说明：原分项审阅仅写临时目录；本报告和脚本由主审归档。core_probe.log 是主审独立复跑该脚本的输出；原临时目录保留为溯源信息。

Pyramid 0.4.0 核心复核：R1 / R2 / R3

基线已核实：`dev/0.4.0`，HEAD `9c7dc4baba3c88544b85e254547b17b84a49196c`，仓库 `/Users/pengkang/Research/Project/Project_PYRAIMD/pyraimd2`。本轮只读，没有修改仓库源码、暂存或提交。依据 `INDEPENDENT_REVIEW_20260909.md`、`REVIEW_FIXES_CORE_20260909.md`、`REVIEW_FIXES_R3_20260909.md`，并读取最终真实 MACE 持久化证据。范围不包括能量契约、材料科学解释、CLI/导出/inspect、后端或成本汇总；这些由主审及其他审阅者覆盖。

结论：这一轮有实质修复，原审阅主要触发点多数已通过；不能再把旧九项全部原样挂起。但仍有 **2 项 P1、2 项 P2 的同根残留**，足以组成 0.4.1 的最小核心修正单。以下不是新功能需求，也不要求增加云计算或大套件。主审报告 458 项全套通过；本复核未重复执行全套，只运行 `/tmp` 中 0–3 步解析反例。

**修复通过：限实际验证的路径**

- **R1 / 旧 F01：pending 检查 RNG。** 新 [energetic.py:844](../../src/pyraimd2/loop/energetic.py:844) 恢复 proposal.check_rng_after。参考检查失败后恢复，再继续三个接受评估，draw 恢复为 `[0.2616121342493164, 0.2984911434141233, 0.8142257405942803]`，与连续分支完全相同；位置也逐位相同。旧“第二次重复第一随机数”已修好。
- **R1 / 旧 F02、F07：已入库标签的缓存重建与窗口内复用。** [energetic.py:668](../../src/pyraimd2/loop/energetic.py:668)、[energetic.py:1631](../../src/pyraimd2/loop/energetic.py:1631)。静止体系、p=1、n_label=2，在初始 checkpoint 后直接恢复或先完成两次缓存检查再恢复，均得到 n_consumed=1、n_updates=0、k=0.8，没有额外训练，也不再误报消费丢失。
- **R1 / 旧 F03：提交事件绑定具体 row_id 的机制已实现。** [energetic.py:1378](../../src/pyraimd2/loop/energetic.py:1378)、[store.py:171](../../src/pyraimd2/store/store.py:171)。故障窗口产生两行时，committed_row 正确选择新行；但后续取力还没全面使用该行，故整体闭环不能判通过，见 C1。
- **R2 / 旧 F04：JSON 模型工件内容篡改。** [energetic.py:804](../../src/pyraimd2/loop/energetic.py:804)。在 checkpoint 后 update 的测试工件中把 k 改成 9.9，现在立即抛出摘要不匹配的 ResumeError。旧的“同 model_id 静默加载不同 k”触发点已被阻止。
- **R3 / 旧 F05：训练、保护集、FD 的物理结构。** [online.py:169](../../src/pyraimd2/loop/online.py:169)、[online.py:194](../../src/pyraimd2/loop/online.py:194)。本地解析模型实际记录了所有训练/预测输入，晶胞体积 27 Å³、PBC=true、电荷 0.2、磁矩 1 均保留；不再重建为空晶胞孤立体系。
- **R3 / 旧 F06：内部能量—力一致性。** [online.py:296](../../src/pyraimd2/loop/online.py:296)。上一轮“二原子势能不变、力乘 1.1”的候选现在按 energy_force_inconsistent 拒绝，父模型恢复。未要求有限方向诊断成为全局保守性证明。
- **R3 / 旧 F08 的两个原触发点：候选推理异常、publisher OSError。** [online.py:408](../../src/pyraimd2/loop/online.py:408)、[energetic.py:1467](../../src/pyraimd2/loop/energetic.py:1467)。候选推理异常现在记录 validation_failed；publisher 失败后 k=0.8、n_updates=0、generation=0，父状态保留。异常保护域仍有缺口，见 C3。
- **R3 / 旧 F09：普通 tensor/array 落盘通路确已打通。** 本地用普通 NumPy 权重 `(2,3)` 验证 ModelRegistry JSON 占位＋NPZ 往返，值相等。最终 [committee_process2.json](../../docs/development_reports/contract_evidence/committee_process2.json) 与对应脚本记录了真实 MACE 独立进程恢复：消费计数 3、更新 1、成员权重/energy shifts 匹配、预测力差 `1.3877787807814457e-17`。承认这条已有证据，不再列为“真实 MACE 从未验证”。本机没有重跑真实后端；数组身份仍存在 C2。

以上本地通过项可执行：

```sh
cd /Users/pengkang/Research/Project/Project_PYRAIMD/pyraimd2
PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python analysis/development_review_040_20260909/core_probe.py passes
```

**已复现残留：按严重性排列**

1. **[P1，C1，R1 / 旧 F03] 选中正确提交行后，核心 resume 仍从孤立旧行取驱动力，覆盖有效 checkpoint 的完整步状态。**

   - 准确位置：[energetic.py:1992](../../src/pyraimd2/loop/energetic.py:1992)–1998：1992 用 committed_row，1994 却调用 `store.driving_label(run_id, last_eval - 1)`；后者在 [store.py:225](../../src/pyraimd2/store/store.py:225) 回到 `_row_at_step`，拿第一行。
   - 触发：无跨评估缓存的参考后端，参考路由的 DB 行已写入、evaluation_committed 尚未写入时中断；恢复重执行该参考并绑定第二行，再保存一个有效完整步 checkpoint。为让两个 payload 可辨，解析故障注入令重执行返回极小不同的合格 E/F。这里模拟的是重执行结果差异，没有要求新功能或宣称允许用户更改参考物理设置。
   - 实测：孤立行 row_id=2；提交行 row_id=3。提交/完整步 checkpoint 的力为 `-0.24136591053875858`，再次 resume 的缓存力却为旧行的 `-0.24116493975562892`。完整步动量相对刚保存的有效 checkpoint 偏差 `9.870373320342019e-07`（ASE 动量单位）。又在真正独立 Python 进程读取该目录，同样重现。
   - 后果：这次不是“可能影响物理轨迹”或导出显示问题，而是核心恢复实际改变了可复用驱动力和完整步动量；后续第一 half-kick 使用旧力。正常同值假后端的测试掩盖了错误。
   - 最小修正：从已验证的同一 row 直接取 driving payload；已有 checkpoint 覆盖的边界优先使用 checkpoint 自身完整状态，只有游标后事件才重建。将核心 label/update replay 的另外两个旧按步读取点一并替换，见静态风险 S1。无需改 legacy Store.driving_label 对其他旧调用者的公共语义。
   - 最小停止条件：保留孤立行、让重执行 E/F 与旧行不同，第二次及独立进程 resume 的位置/完整动量/力与最新有效 checkpoint 逐值相同；更新重锚的标签必须来自该 event 绑定行。
   - 可执行：`PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python analysis/development_review_040_20260909/core_probe.py orphan`。
   - 现成证据目录：`/tmp/pyramid-040-core-vqakoixj/orphan-force`。

2. **[P1，C2，R2/R3 / 旧 F09] 数组内容寻址忽略 dtype/shape，正常无损坏的状态也可静默恢复为另一张量；标量还会被升维。**

   - 准确位置：[models.py:63](../../src/pyraimd2/runtime/models.py:63)–64 只 hash `.tobytes()`；[models.py:120](../../src/pyraimd2/runtime/models.py:120)–130 以该 digest 共用 NPZ；[models.py:147](../../src/pyraimd2/runtime/models.py:147)–151 读取只核对 bytes digest，不核对占位里的 dtype/shape。模型工件 [models.py:232](../../src/pyraimd2/runtime/models.py:232)、checkpoint sink [energetic.py:591](../../src/pyraimd2/loop/energetic.py:591) 同样用 `np.ascontiguousarray`，将 0-D 数组升为 `(1,)`。
   - 触发：一个正常 updater state 同时含 `zeros((2,2))` 与 `zeros(4)`；或相同字节、不同 dtype 的两数组。无需编辑工件、无需崩溃。
   - 实测：声明 shape `(4,)` 的 vector 加载成 `(2,2)`。`uint64([4607182418800017408])` 与 `float64([1.0])` 的字节相同，先保存 float 后，整数状态加载为 `float64([1.0])`；所有当前摘要检查都通过。另经实际 ModelRegistry publish/read 保存 `np.array(2.)`，shape 从 `()` 变为 `(1,)`。
   - 后果：事件侧车重放可以改变权重/计数 buffer 的形状、类型和值；严格 load_state_dict 可能拒绝，宽松加载可能带错状态继续。现有真实 MACE 的一次干净 checkpoint/工件预测一致，并不能证明这个不同数组共用事件侧车的路径安全。本轮没有声称那条真实 MACE 记录已经损坏。
   - 最小修正：数组身份覆盖 dtype（含字节序）、shape 和规范化内容；保留 0-D shape。所有 source（content/dict/artifact）校验三者而非只 hash bytes。若需要兼容旧侧车，先根据原文件与占位明确验证/恢复，歧义时拒绝，不能静默 reshape 或转换值。
   - 最小停止条件：同字节不同 shape/dtype、空数组、标量，在 consumed-event/模型工件/checkpoint 三处都严格保持 dtype、shape、值；重开进程仍一致；侧车形状/类型不符应明确报错。使用现有 NumPy 环境即可，不需 MACE 或云任务。
   - 可执行：`PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python analysis/development_review_040_20260909/core_probe.py arrays`。
   - 现成证据目录：`/tmp/pyramid-040-core-vqakoixj/array-collision`、`/tmp/pyramid-040-core-vqakoixj/scalar-artifact`。

3. **[P2，C3，R3 / 旧 F08] 训练结束后的成功 task 写入不在回滚保护域内，日志 I/O 失败仍留下未发布候选。**

   - 准确位置：[energetic.py:1422](../../src/pyraimd2/loop/energetic.py:1422)–1428 位于 `on_label` 成功返回之后，但 publication 的 try/rollback 直到 [energetic.py:1467](../../src/pyraimd2/loop/energetic.py:1467) 才开始。
   - 触发：GuardedUpdater 训练和验证通过；随后的 training-success task 在 EventLog.append 中抛 OSError，例如磁盘写满。该点早于 model publisher。
   - 实测：父 k=0.8；失败停止后 k=0.81、updater.n_updates=1，但 generation=0、model_update 事件数=0。与上一轮泄漏候选状态同根，只是修复覆盖了 publisher 失败，遗漏了更早的持久化失败。
   - 后果：Runner 确实停止，因此仍定 P2，未声称失败后继续积分；但“任一持久化步骤失败都保留父状态”的声明仍不成立。
   - 最小修正：从 on_label 返回被接受候选开始，将成功 task 写入、state_dict 获取、工件写入、提交事件统一纳入候选回滚域；或让 GuardedUpdater 的候选真正隔离，明确 commit 后再激活。该工作不需要引入异步训练。
   - 最小停止条件：分别在 training-success task、候选状态导出、工件写入、MODEL_UPDATE 提交前注入异常，停止后模型参数、generation、n_updates、pending 队列一致；不把未提交候选记为活跃模型。
   - 可执行：`PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python analysis/development_review_040_20260909/core_probe.py log`。
   - 现成证据目录：`/tmp/pyramid-040-core-vqakoixj/post-training-log`。

4. **[P2，C4，R1/R3 / 上轮重校准窗口观察] 重校准已完成、proposal 已持久化、evaluation 未提交时恢复，会丢失校准完成状态。**

   - 准确位置：[energetic.py:1124](../../src/pyraimd2/loop/energetic.py:1124)–1125 增加 segment/n_calibrations；[energetic.py:1164](../../src/pyraimd2/loop/energetic.py:1164)–1173 应用新 anchor 并清理 deferred_origin；[energetic.py:839](../../src/pyraimd2/loop/energetic.py:839)–848 重建 tail 只恢复 pending.anchor、deferred_record 和 RNG，没有恢复上述已完成校准的全局状态。
   - 触发：n_label=1、p=1 的小解析更新势；完成 step 1 后，step 2 已针对新模型校准且 proposal 已写，进入 `_finish` 前中断；从较早 checkpoint 恢复并继续到 step 3。
   - 实测：连续分支 n_calibrations=3、segment=3；恢复分支 n_calibrations=2、segment=2；两边 model_generation 都为 4，位置逐位相同。即完成校准的计数/segment 历史不一致，后续 segment ID 被重复使用。
   - 后果：没有观察到这个反例中的位置错误；实际缺陷是恢复没保留同一校准/段历史。上一轮 n_label=2 示例后来计数追回，故当时只列观察；本次 n_label=1 的同窗口使差异持续，已升级为复现残留。
   - 最小修正：把重校准完成当作可恢复状态变化，显式恢复 anchor、segment、n_calibrations、deferred_origin/record；或者独立提交校准完成事件，保证只应用一次。若暂不支持该窗口，应在恢复时清楚拒绝，不能声称已完整重放后静默丢计数。
   - 最小停止条件：同一窗口下 n_label=1/2 两例，连续/恢复的 segment、n_calibrations、anchor 模型代次、检查流、模型链一致；重复恢复不再次应用校准完成。
   - 可执行：`PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python analysis/development_review_040_20260909/core_probe.py tail`。
   - 现成证据目录：`/tmp/pyramid-040-core-vqakoixj/tail-calibration`、`/tmp/pyramid-040-core-vqakoixj/tail-control`。

**静态风险与证据边界：不另扩成新审计单**

- **S1，C1 的同根读取点。** `_replay_label_event` 在 [energetic.py:774](../../src/pyraimd2/loop/energetic.py:774) 与 [energetic.py:823](../../src/pyraimd2/loop/energetic.py:823) 仍以 `_row_at_step` 选择参考原点。因此 commit-bound row 的修复还应覆盖消费及模型更新后重锚。这里未另做两套端到端反例，列为静态同根风险，避免把已证实的动量错误重复计数。
- **S2，C2 的共同解析边界。** `dict_array_source`（models.py:166–171）和 `resolve_artifact_state`（197–202）也只验证字节摘要；即使事件内容寻址修好，shape/dtype 检查仍应在全部 source 中统一执行。本次对标量 ModelRegistry 路径和事件内容寻址做了实测，对所有侧车破坏排列没有穷举。
- 真实 MACE 证据支持所运行的干净 checkpoint 恢复与边界预测；`committee_persist_al.py` 没有再运行恢复后的下一次训练，也没有构造 C1/C3/C4 的故障窗口。这是验证范围说明，不否认已有真实集成，也不要求为本次最小修正再启动云任务。
- 本轮未重新审阅或重复主审已确认的 initial 半 kick 导出、relax 导出、孤立行重复导出、inspect 未完成尾帧问题。C1 虽也涉及 orphan，但证据指向核心 resume 的真实原子状态，责任边界不同。
- 不追加优化器中途恢复、额外后端、长轨迹、全局保守性证明或新的性能要求。未做 OS 断电/网络文件系统强耐久性验证。

**可直接执行的 0.4.1 最小核心修正顺序与停止点**

1. C1：让全部恢复消费者使用同一提交行，已有 checkpoint 边界不被旧行覆盖。
2. C2：完善 tensor-array 身份和 0-D 保真；这是序列化正确性修正，不是新训练功能。
3. C3：扩大候选回滚事务到训练后的所有持久化步骤。
4. C4：完整恢复校准完成状态，或明确拒绝尚不能正确恢复的窗口。

四项对应反例由“输出当前残留”改为验证不变量成立后停止；保留上面已通过的旧触发点。主审统一安排必要的最终回归，本分工不重复全套或材料算例。代码修正需另行由执行者完成，本审阅没有改动任何实现。

交付：

- [本报告](core_review.md)。
- [极小复核脚本](core_probe.py)：支持 `passes / orphan / arrays / log / tail`，不传参数则全部执行；所有输出目录位于 `/tmp/pyramid-040-core-*`。
- 当前实测目录：`/tmp/pyramid-040-core-vqakoixj`。所有失败注入和工件操作都仅作用于该类测试目录。
