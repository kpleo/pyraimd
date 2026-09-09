# 0.4.0rc1 独立审阅修复回报——核心组（R1/R2/R4/R6/§7）

- 日期：2026-09-09
- 审阅依据：`docs/development_reports/INDEPENDENT_REVIEW_20260909.md`（含轻量反例）
- 基线 commit：`1ef713a`（docs: review 0.4.0rc1 runtime and backend correctness）
- 修复 commit：与本报告同批的 `Fix:` 前缀系列提交（逐条 hash 见 git log）
- 分工说明：R3（在线更新/周期结构）与 R5/R2-引擎/R6-引擎（QE 两条适配路径、引擎侧 attempt 上报）由后端组完成，见 `REVIEW_FIXES_BACKEND_20260909.md`；本批不触碰 `engines/`、`surrogate/`、`backends/`、README/CHANGELOG/examples/reproducibility。R6 消费侧按后端组锁定的接口约定实现（事件类型 `"attempt"`、`record="physical_attempt"`、`request_id` 挂父逻辑请求；自报告引擎 = `compute`/`predict` 接受 `request_id` 关键字）。
- 新增实际 SCF 数：0（全部修复用解析假后端 hermetic 验证；无需也不应动用真实配额）
- 真实后端验证文件路径：无（本轮为运行时正确性修复，真实 QE/MACE 路径验收见后端组报告）

## R1. 恢复保持检查序列、标签消费和数据库/事件身份（F01/F02/F03/F07）

### 具体原因

- F01：检查抽签只在 accepted 评估提交时推进 RNG，未提交提案的"抽样后状态"不持久；崩溃窗重放时同一提案重新抽签，恢复后的检查流与连续运行分叉。
- F02：数值标签缓存为纯内存结构，恢复后冷启动；同一参考标签被分配新 label ID，恢复比连续多消费一个标签、多触发一次更新（k: 0.8→0.81）。
- F03：evaluation 提交事件不绑定数据库行；崩溃发生在"行已写入、提交未发生"之间时，孤立行在恢复后被当作权威记录。
- F07：重放窗口只统计窗口内的标签消费，窗口内被缓存复用的标签被误判为"消费丢失"。

### 行为改变

- `evaluation_proposed` 事件新增 `check_rng_after`（抽样后的 bit-generator 状态）；恢复未提交提案时复原该状态，恢复后的抽签序列与连续运行逐值一致。
- 恢复时从持久记录重建标签缓存：凡带 engine payload 与 `engine_label_id` 的提交行，完整决定（几何、参考身份、能量口径）→ label ID；冷启动不再重新消费。
- `evaluation_committed` 绑定 `row_id` 与 `row_digest`（内容摘要）；`Store.append(dedupe=True)` 按 (run_id, step, engine_label_id) 幂等；恢复边界与重放一律经 `committed_row` 解析行，孤立行永不作为权威。
- 重放改为收全量事件、内部按游标过滤；消费集合按全日志（LABEL_CONSUMED/MODEL_UPDATE）构建，窗口内缓存复用不再误判。

### 反例前后对照（全部 hermetic，解析谐振后端）

- F01 前：崩溃后恢复的检查抽签序列与连续运行不同；后：逐值一致（回归断言 `[0.2616121342493164, 0.2984911434141233, 0.8142257405942803]`，atol=1e-15）。
- F02 前：恢复运行 n_consumed=2、n_updates=1、模型 k=0.81；后：n_consumed=1、n_updates=0、k=0.8，与连续对照完全一致。
- F03 前：无指纹引擎场景孤立行被当作权威；后：重执行产生新行且提交绑定新行（该步 2 行，提交指向新行 row_id）；有指纹引擎场景孤儿行被收编（该步 1 行，verification 任务 1 success＋1 cache_hit，无第二次物理执行，`n_reference == 崩溃时已计费数 - 1`）。
- F07 前：窗口内缓存复用在恢复时被判消费丢失；后：恢复正常继续，n_consumed=1、n_updates=0。
- 连续/新进程对照：每项均以"连续运行"与"崩溃＋全新对象恢复"双运行逐字段比对（位置/动量/抽签/消费/更新计数）。

回归：`tests/unit/test_review_r1.py`（6 项）。

## R2. 模型工件与后端身份对应真实物理参数

### 具体原因

模型更新事件与 checkpoint 不绑定工件内容；替换模型文件或改动配置后仍可"恢复"成貌似的同一运行；plain surrogate 恢复不校验模型身份与积分步长。

### 行为改变

- `MODEL_UPDATE` 事件携带 `artifact_digest`（工件除写入时间外的 canonical 摘要）；checkpoint manifest 记录 `model_artifact_digest`；恢复入口与逐条更新重放都校验摘要、model_id、schema、代次与父链。
- plain 模式恢复：surrogate 校验 `model_id_for(backend, 0)` 与 checkpoint 的 model_id、timestep_fs；不符即 WorkflowError（提示新运行或 fork）。reference 此前已有指纹校验，不变。

### 反例前后对照

- 前：修改 resolved_config 的 k/timestep 后恢复照常进行；后：启动即拒绝并说明原因。工件摘要被篡改的 checkpoint/更新事件在恢复时拒绝（ResumeError）。

回归：`tests/unit/test_review_r2.py`（3 项）。

## R4. 配置链、完整步轨迹、空标签与约束口径

### 具体原因

- `_policy_kwargs` 漏传 `force_metric`，TOML 选择从未到达 runner/checkpoint。
- 正常 adaptive 导出行保存的是力评估时刻的半步动量（与完整步 checkpoint 差恰为 0.5·dt·F，反例实测 0.0019022701914672424）；inspect 温度同口径偏差（138.10867668813074 K vs 完整步 137.29843851213937 K）。
- 评估提交与积分步完成混为一谈：最后 half-kick 前崩溃时 STEP_COMPLETED=0，inspect 报完成 1 步、export 输出失败步帧。
- plain 静止平衡恢复时 ASE 命中计算器缓存，last_label 未恢复，reference/surrogate 两模式均写空 payload，导出报轨迹损坏。
- FixAtoms 报表口径错误：relax 以固定原子原始反力报告 final_fmax（反例：自由原子 0.02 已收敛、报告 1.0），driving 记录原始力；inspect 温度固定除以 3N（一半原子固定时温度差一倍）。

### 行为改变

- `_policy_kwargs` 补传 `force_metric`；TOML→runner→checkpoint→resume 链一致。
- 新增共享完整步读取接口 `Store.complete_step_frame`（energetic 行用驱动力按 ASE kick 重建完整步动量，不改写原始记录；帧带 `momenta_source` 三态标记）与 `Store.row_timestep_fs`；导出按 `step_completed` 事件过滤，初始评估帧保留并标 `integration_phase="initial_evaluation"`。
- inspect 的 `n_complete_steps` 以 STEP_COMPLETED 事件计数（当前 schema 日志严格计数；旧日志保留回退）；温度按完整步帧与受约束自由度（dof=3·(N−n_fixed)）计算，n_fixed 取自行内 constraint 记录。
- plain 恢复从边界提交行 payload 重建 `last_label`；`_PlainDriver._record_evaluation` 写库前拒绝空 driving/label（WorkflowError）。
- relax/singlepoint：driving 统一为约束投影后力，原始力留 backend payload 与 constraint 记录（`raw_forces_eV_A`）；`final_fmax_eV_A` 取优化器实际判据（约束投影后最大单原子力范数，即 `atoms.get_forces()`），原始全原子值另列 `raw_all_atom_fmax_eV_A`。

### 反例前后对照

- force_metric：前——TOML `all_atoms_max_atom`、checkpoint 仍 `active_dofs_max_atom`；后——链路三处一致（含 resume 后新 checkpoint）。
- 导出动量：前——与完整步差 0.0019022701914672424（=0.5·dt·F）；后——导出帧动量与"行记录＋0.5·dt·F_driving"逐值一致（atol=1e-12），原始行记录逐位不变。
- 未完成步：前——崩溃步被计数并导出；后——inspect 报 0 完整步，导出仅初始帧（标 initial_evaluation）。
- 空标签：前——静止恢复写空 payload；后——恢复从边界行重建标签，全部行 payload 非空，与连续对照逐行一致（reference 与 surrogate 两模式）。
- FixAtoms relax：前——final_fmax=1.0（固定原子反力）；后——final_fmax=0.02（判据同口径、converged=true），raw_all_atom_fmax=1.0 另列，driving 固定分量为 0、engine payload 保留原始 −1.0。

回归：`tests/unit/test_review_r4.py`（8 项）。

## R6. 成本账本区分逻辑请求与真实进程执行

### 具体原因

外层 task 事件把整个 `engine.compute` 调用当成一次物理执行：引擎内部首次失败、重试成功（真实启动两次）仍记 actual=1/failed=0；预检失败与缓存命中也被混计。普通 MD/relax 只取最后一次 wall_time；resume 后后端不再获得事件记录器。

### 行为改变

- 采用后端组锁定的接口约定（`REVIEW_FIXES_BACKEND_20260909.md`）：事件类型 `"attempt"`、`record="physical_attempt"`、字段含 `operation`/`purpose`/`request_id`/`attempt`/`status`/`started_unix`/`elapsed_s`/`returncode`/`directory`/`start`/`source`/`error`；`compute`/`predict` 接受 `request_id` 关键字的引擎每次真实启动自报告一条（QeEngine/AseQeEngine 由后端组实现）；不接受的引擎由调用方每次调用记一条（`runtime/events.py` 的 `physical_attempt`，含失败）。
- `summarize_tasks`：logical_requests=reference task 数（含 cache_hit）；actual_executions=attempt 数（success＋failed）；failed_attempts=attempt failed；cache_hits=task cache_hit。task 失败且无 attempt 子记录＝启动前失败，actual=0（逻辑失败仍可见）；无 attempt 的旧日志保持旧语义。task 有子记录时其耗时不再与子记录重复相加；`record="physical_io"` 的引擎 I/O task 嵌套在父 attempt span 内，不重复相加；崩溃丢失父 task 的孤儿 attempt 仍计入实际成本。
- energetic `_reference`、plain MD（含 ASE 缓存命中记 cache_hit、不只取最后 wall_time）、relax、singlepoint 全部接线；`resume_workflow`（adaptive 与 plain）在构建后端时传入事件日志，恢复后 attempt 继续入账（后端组已完成 adaptive 路径的引擎工厂注入；plain/resume 注入为本批工作）。
- 引擎侧上报（QeEngine 子进程、AseQeEngine ASE 执行、密度 I/O `physical_io`）由后端组实现并另有报告；本批交付消费侧：调用方接线、`request_id` 传递与账本汇总。

### 反例前后对照（假引擎经真实 compute/workflow 路径）

- 内部重试：前——logical=1/actual=1/failed=0；后——logical=1/actual=2/failed=1/success=1（与后端组 `test_logical_vs_physical_aggregation_convention` 同一原料口径）。
- 缓存命中：后——logical 含命中、actual 不含命中（actual = logical − cache_hits）。
- 启动前失败：后——actual=0、failed=0，逻辑失败任务仍可见。
- 无内部启动的引擎：每次调用恰好一条 attempt（成功或失败），logical=actual。
- 崩溃重放：恢复重放不新增任何 attempt（重放前后 attempt 数逐值相等）。
- 嵌套物理 I/O：父 task span 与 `physical_io` 拷贝均不重复相加，只计叶子 attempt 时间。
- 旧语义保持：无 attempt 事件的日志汇总结果与修复前一致（回归直接构造旧式事件序列验证）。

回归：`tests/unit/test_review_r6.py`（8 项）。

## §7. JSONL 尾部损坏的安全恢复

- 前：`EventLog` 与 inspect 读取遇到任何损坏行即 EventLogError，崩溃截断的最后半行（仍有有效 checkpoint）阻止一切读取与恢复。
- 后：仅最后一条未提交记录损坏时跳过并标记（`EventLog.torn_tail`），写入方打开时丢弃未提交的尾部字节（已提交事件永不改写）；中间已提交事件损坏仍按行号报错。inspect 与导出的只读路径同样跳过未提交尾记录。
- 回归：`tests/unit/test_event_log.py`（尾部/中间两场景、seq 复用、inspect 读取）。

## 文档修正

- `WP02_report.md`：删除"缓存消失只影响成本不影响正确性"判断（F02 已证伪）。
- `WP03_report.md`：标签缓存冷启动结论更正；补 RNG 持久化、提交绑定行、消费集合口径。
- `WP07_report.md`：FixAtoms 报表口径、空标签、温度自由度、完整步导出的修正记录。
- `WP08_report.md`：Al 收敛解释更正（ASE 3.29 为约束投影后最大单原子力范数；"分量最大值"解释不成立）。
- `WP09_report.md`：R3 完成度表述按审阅结论逐条修正。

## 验证

- `uv run pytest tests/unit tests/test_smoke.py -q`：417 passed, 1 deselected（基线 389 → ＋28 项回归）。
- `uv run ruff check src/pyraimd2`：2 个错误，与基线逐条相同（split.py RUF007、energetic.py TRY004 存量），新增 0。
- 最小核心依赖不变（NumPy＋ASE＋标准库），不导入 Torch/PySCF。

## 保留限制与待办

- R3 与 R5/R2-引擎/R6-引擎侧不在本批（后端组报告 `REVIEW_FIXES_BACKEND_20260909.md`）；QE 路径验收以后端组证据与合并后回归为准。
- Al 末帧原始力数组的固定（0–7）/自由（8–16）分列重算未完成（需原始记录，审阅时未获得）；WP08 的 58 次实际参考执行与分用途合计 57 不符，需按 attempt 重算后修正——两项均为材料记录校正，不猜测补数。
- 跨版本混合日志（恢复前旧段无 attempt）按段各自语义汇总，属既定兼容行为。
- 自定义 `direction` 回调不参与恢复校验（WP03 既定语义，未改）。
- 本轮未做 OS 断电、网络文件系统或 fsync 极端耐久性验证，不宣称覆盖这些情形。
