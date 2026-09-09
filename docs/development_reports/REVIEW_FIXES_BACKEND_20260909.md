# 独立审阅修复回报 — 后端部分（R5 / R2-引擎 / R6-引擎）

修复日期：2026-09-09。对应审阅：`INDEPENDENT_REVIEW_20260909.md`。
修复基线：`dev/0.4.0` 审阅基线 `5a1d105`；修复提交在分支 `dev/0.4.0-rfix`（顶端 `63585f3`），待核心部分（R1/R2/R4/R6）完成后合并回 `dev/0.4.0`。
本文件只覆盖引擎/后端侧；loop/runtime/workflows 侧见核心组的修复回报。

- 新增实际 SCF 数：0（全部为 fake executable 与输出夹具）。
- 测试：`420 passed, 1 deselected`（审阅基线 389 + 新增 31）；新增 27 项回归在修复前代码上实测全部失败、修复后通过；ruff 改动文件无新错误。

## R5.1 ASE-QE 成功契约（P1）

- 原因：ASE-QE 路径此前以"解析到能量和力"为准，rc=0 但无 `JOB DONE.` 或带 `convergence NOT achieved` 的输出也会返回标签；手写 QE 正确拒绝。
- 行为改变：新增共享判定 `check_qe_run_text(text, nat, require_stress=True)`（`engines/qe_engine.py`）：`JOB DONE.`、无未收敛标记、无 QE 错误横幅、完整解析（能量 + 有序 1..nat 力块 + 完整有限 stress 块）。手写 `_attempt` 与 `AseQeEngine.compute`（读 ASE 实际产出的 `espresso.pwo`）过同一道判定。
- 反例修复前后：审阅归档脚本 `pyramid_review_backends_5a1d105.py` 中 missing_done / not_converged 两例，修复前 ASE 路径返回标签、修复后两路径均拒绝。

## R5.2 ASE-QE 目录冲突（P1）

- 原因：ASE-QE 新实例从 `eval-000000` 重新计数，新进程首次求值即撞旧目录。
- 行为改变：共享 `allocate_run_dir(run_root, base)`：扫描续号 + `mkdir(exist_ok=False)` 原子占位、撞号重扫；两条 QE 路径共用；同时修复手写侧 scan-then-mkdir 的竞态窗口。
- 回归：两个实例/新进程场景均有用例。

## R5.3 手写 QE 静默丢弃初始磁矩（P1）

- 原因：相同 Atoms 的 `[1, -1]` 磁矩在 ASE 输入中产生 nspin=2 和两种 species，手写路径没有任何自旋设置——静默计算了另一个物理体系。
- 行为改变：`write_qe_input` 对共线磁矩写 `nspin=2`、按 (element, magmom) 分 species、`starting_magnetization(sidx)`（与 ASE 写出逐项一致）；净电荷写 `tot_charge`；非共线磁矩在运行前明确拒绝（无子进程、无 attempt 事件）。

## R5.4 未收敛重试分类（P2）

- 原因：`convergence NOT achieved` 加非零退出被当作可重试失败，max_retries=3 实际执行 4 次。
- 行为改变：`classify_qe_failure_text` 先于退出码分类：未收敛/错误横幅即使 exit≠0 也判确定性失败不重试。
- 反例修复前后：审阅归档脚本 `pyramid_review_qe_retry_5a1d105.py`，attempts 4 → 1。

## R5.5 暖启动 / 清理 / 路径 / 元数据（P2）

- AseQeEngine 接入与手写共用的密度验证/staging/atomic 回退；`timeout_s` 非默认值在 ASE 路径构造时明确拒绝（ASE FileIO 无超时接口）；`max_retries` 在 ASE 路径由引擎侧按同一分类实现。
- `clear_calculator_results`：对无 `reset()` 的 FileIO 计算器（Espresso）清理 results 字典，不再以调用不存在的方法掩盖原始异常。
- `normalize_config_paths`：构造时把 `pseudo_dir`/`density_source` 解析为绝对路径——fingerprint 哈希路径与子进程实际读取路径一致（审阅反例 relative_pseudo_dir 现已通过）。
- AseQeEngine 以 `force_consistent=True` 取能，标签元数据与 capabilities 一致（绝缘 energy / metallic free_energy）。
- parser 严格化：force 原子编号必须有序 1..N（重复/跳号拒绝）；`**********` 等不可解析数值统一 EngineError；声明 stress 的路径上 stress 块缺失/不完整拒绝；失败 attempt 记录不再残留 "running"。

## R2-引擎 ASE 指纹（P1）

- 原因：`calculator.name` 不含物理参数——LJ epsilon 1→2 能量翻倍而指纹不变。
- 行为改变：`calculator_identity`（`engines/ase_engine.py`）：类路径 + JSON 规范化 `parameters` + 参数中模型文件的内容 sha256；无法可靠识别的外部状态返回 `fingerprint=None`（诚实 unknown），或要求调用方传显式 `identity=`。`AseSurrogate` 透传。
- 回归：LJ epsilon 变化指纹必变；不可序列化参数的计算器指纹为 None。

## R6-引擎 物理执行 attempt 事件（P1）

- 原因：一次逻辑参考请求内 QE 首次失败、第二次成功，账本却把整个调用记为一次物理执行。
- 行为改变：QeEngine 每次真实子进程启动发射一条 `attempt` 事件（含超时被 kill、启动后处理异常；预检拒绝不发射）；AseQeEngine 每次 ASE 执行一条；`compute(..., request_id=None)` 可挂父逻辑请求；`EngineResult.wall_time_s` 为跨 attempt 物理总计；密度 I/O 事件带 `record="physical_io"` + `request_id`（嵌套口径，汇总不重复相加）。另发现并修复：ASE 几何缓存会让同构型第二次 compute 不启动子进程——AseQeEngine 每次 compute 强制真实执行。
- 与核心组的接口约定（已锁定）：事件类型 `"attempt"`，字段含 `record="physical_attempt"`、`operation`、`purpose`、`request_id`、`attempt`、`status`、`started_unix`、`elapsed_s`、`returncode`、`directory`、`start`（atomic/density）、`source`、`error`；逻辑请求 = 外层 `type=="task"` span，物理执行 = `type=="attempt"`。目标账本 logical=1 / actual=2 / failed=1 的原料已由 `test_logical_vs_physical_aggregation_convention` 锁定。
- 接线现状：adaptive 路径（workflows/md.py）已把 event_log 传给引擎工厂，attempt 事件即刻入 events.jsonl；plain singlepoint/relax 与 resume 路径的 event_log 注入、`summarize_tasks` 对 attempt 事件的消费属核心组范围。

## 保留限制（如实）

- 两条 QE 路径的密度 manifest 各记各的 fingerprint（`qe-subprocess` vs `ase-espresso`），跨路径密度复用被保守拒绝（CODATA 常数差 1e-7 量级，身份不相容）。
- 非共线磁矩两路径均明确拒绝（未实现 nspin=4/noncolin）；电荷只映射净电荷 `tot_charge`。
- AseQeEngine 不支持 `timeout_s`（非默认即拒绝）；超时杀进程组仅手写路径有（ASE 自身无进程组管理）。
- 通用 `AseEngine`（非 QE）不发射 attempt 事件、保留 ASE 几何缓存语义；`parameters` 不可序列化的计算器指纹为 None，依赖调用方显式 `identity=`。
- 审阅 R5 中"validate 明确拒绝"属 config/workflows 层（核心组）；引擎构造层已拒绝（timeout_s、非共线、缺赝势、奇异晶胞）。
