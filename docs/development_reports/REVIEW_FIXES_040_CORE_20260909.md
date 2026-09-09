# 0.4.0 第二轮独立审阅修复回报——核心（A1–A4、C1–C4、B2/B3 消费侧）

- 日期：2026-09-09
- 审阅依据：`docs/development_reports/INDEPENDENT_REVIEW_040_20260909.md`（基线 9c7dc4b / v0.4.0）
- 基线 commit：`7d47edd`（第二轮审阅报告入库）
- 修复 commit：`a49df45`（C1–C4）、`3c13c24`（A1–A4）、`3e1b813`（B2/B3 消费侧）、本报告随后
- 分工说明：B1（ASE 包装指纹）与 B2/B3 的引擎侧（真实启动、终态、耗时口径的引擎部分）由后端组在独立 worktree 修复；本批为消费侧与共享事件语义约定（见下）。未触碰 engines/、surrogate/、backends/、README/CHANGELOG/examples/、reproducibility/、analysis/、manuscript/。
- 新增实际 SCF 数：0（全部解析模型/假后端 hermetic 回归）

## 共享事件语义约定（与后端组一致，消费侧实现）

1. 物理 attempt 只对真实进程启动发射；零启动 = 零 attempt（启动前失败不记）。
2. 每次逻辑调用在统一 try/finally 内完成一条 task 记录，失败不换 task ID。
3. 每个已启动 attempt 必须有终态（success/failed/killed/post_processing_failed），不留 running。
4. 进程成功＋后处理失败也要终结记录，并与解析/程序成功分开描述。
5. 缓存命中用本次访问耗时；密度拷贝 I/O 按真实时间段另计；嵌套 span 不重复相加。
6. attempt sink 显式连接：包装层传 request_id 而引擎无接收器的组合在启动前拒绝，绝不回退成外层一条 attempt 掩盖内部重试。
7. ASE 包装引擎一次明确执行后读取整份结果，不隐式二次执行。

消费侧落点：`runtime/events.py` 的 sink 连接协议（`physical_attempt`＋`_SINK_ATTRIBUTES`）与 ledger 判定标记（`ATTEMPT_LEDGER_PHYSICAL_V1`），`runtime/costs.py` 的汇总口径，`loop/energetic.py` 与 `workflows/md.py` 的接线。

## C1 [P1] resume 选对行却从旧行取力

- 原因：恢复边界用 `committed_row` 取位置/动量，却用按 step 取首行的 `driving_label` 取力；孤立行与 committed 行同 step 时力来自孤立行（审阅实测差 9.87e-7）。
- 修复（a49df45）：`Store.driving_label_for_row` 从给定已验证行读取；恢复边界、`LABEL_CONSUMED` 重锚定原点与 `MODEL_UPDATE` 重锚定点三处全部改经 `committed_row`；plain resume 同路径。
- 反例修复后可核对输出：无指纹漂移引擎使孤立行与重执行行数值不同（driving 力相差 5e-6 可分辨）；修复前边界力等于孤立行，修复后逐值等于 committed 行（atol 1e-15），位置同样来自 committed 行。
- 新增回归：`tests/unit/test_review_c1.py`。
- 兼容：旧提交（无 row_id 绑定）回退到 step 查找，行为不变。

## C2 [P1] 数组侧车混淆 shape/dtype、标量升维

- 原因：内容寻址只哈希字节，加载只校验字节；`zeros((2,2))` 与 `zeros(4)` 互载，同字节 float64/uint64 互载；`np.ascontiguousarray` 把 0-D 升为 (1,)。
- 修复（a49df45）：占位身份升级为 scheme v2（摘要覆盖 dtype 含字节序、shape、内容）；事件侧车、模型工件、checkpoint 三个入口统一经 `_verify_placeholder` 校验；写出用保 0-D 的 `_contiguous`。0.4.0 旧占位（无 scheme）按字节摘要＋shape/dtype 字段校验，仍可解析。
- 反例修复后可核对输出：同字节矩阵/向量与 float64/uint64 各写各的文件（内容寻址键不同），交叉引用显式 ModelRegistryError；0-D 往返 shape == ()。
- 新增回归：`tests/unit/test_review_c2.py`。
- 兼容：旧工件/侧车可读（legacy 校验），新写入一律 v2。

## C3 [P2] 训练成功日志写入失败不回滚

- 原因：训练/验证通过后，success task 记录、state_dict 快照、UPDATE_REJECTED 记录在回滚域（原 try）之外；这些入口失败时模型/计数已改、代次未进、无发布事件。
- 修复（a49df45）：候选接受后的日志、快照、拒绝记录、工件持久化与提交事件纳入同一回滚域；任一失败恢复父模型与 updater 一致状态后抛出。
- 反例修复后可核对输出：在 success-task 写入、state_dict、提交事件三处逐一注入失败，停止时 generation=0、n_updates=0、模型为父态、pending 保留、无 MODEL_UPDATE 事件。
- 新增回归：`tests/unit/test_review_c3.py`。

## C4 [P2] 重校准后未提交 proposal 恢复丢段状态

- 原因：更新后的重校准（anchor/segment/计数/deferred 状态）只留在内存；其 proposal 已持久化但计数未随存；恢复后校准/段计数落后（审阅实测 3/3 vs 2/2）。另发现：重放 MODEL_UPDATE 会重新武装 deferred 原点，可能二次校准；窗口中间写的 checkpoint（状态含重建计数、游标在其提交前）会致下次恢复双计。
- 修复（a49df45）：proposal 持久化 segment 与 n_calibrations（重校准后的冻结值）；重建时若 deferred_record 存在则解除重放原点（不二次校准）；计数在重建评估**提交时**应用，恰好一次，且不会进入提交前写出的 checkpoint。
- 反例修复后可核对输出：n_label=1/2 两窗口、首次与二次恢复后，segment/n_calibrations/generation/anchor 段与连续运行逐项相等，提交事件流（route/检查序列）逐条一致。
- 新增回归：`tests/unit/test_review_c4.py`（其中二次恢复两项在修复前通过、锁定修复自身的恰好一次；首恢复两项复现反例）。
- 兼容：旧日志的窗口缺这两个字段则保持旧行为（已知限制，见报告末尾）。

## A1 [P1] 初始帧被多加半步动量

- 原因：`complete_step_frame` 对所有 energetic 行补 0.5·dt·F；初始 evaluation 存的是完整初始动量（审阅实测被改动 0.0011787233746156922）。
- 修复（3c13c24）：按 context phase（`initial`）与 step=-1 双重判定豁免初始行，`momenta_source` 标 `initial_evaluation_record`；真正 MD 中间行仍重建为完整步。
- 反例修复后可核对输出：初始帧动量与原始输入逐值一致（atol 1e-15），后续完整步帧仍等于行记录＋0.5·dt·F_driving。
- 新增回归：`tests/unit/test_review_a.py::test_a1_*`（同测初始帧与后续步）。
- 说明：该反例证明的是导出重建错误，未证明积分器改动初始动量（同审阅判断）。

## A2 [P2] relax 轨迹被 MD 完成步规则全部过滤

- 原因：`export_run` 无条件套 `completed_step_ids`；relax 无 STEP_COMPLETED，全部行被滤掉（10 行报"no committed evaluations"）。relax 帧原本连 commit 事件都没有。
- 修复（3c13c24）：按 RUN_START 的任务类型选提交语义——MD 类（plain-nve、adaptive）按完成步边界过滤；relax/singlepoint 每次提交即完整记录。relax 帧现写带 row 绑定的 commit 事件，终态重复帧按评估身份去重；无事件日志的历史运行回退为全部行（文档与代码一致：没有日志 ≠ 确定没有完成帧）。
- 反例修复后可核对输出：harmonic reference relax 导出全部已提交帧（≥2），singlepoint 1 帧，plain MD 仍为初始＋完成步过滤。
- 新增回归：`tests/unit/test_review_a.py::test_a2_*`（含三种 task 经 export_run 的验收）。

## A3 [P1] 权威行选择没接入导出

- 原因：导出遍历全部行只按 step 过滤；孤立行与 committed 行同 step 时重复导出（审阅实测 3 帧 evaluation_id=[0,1,1]）。
- 修复（3c13c24）：新增共享读取接口 `Store.iter_committed`（commit→row_id→行，单一入口），导出只产权威行；plain MD 与 singlepoint 的提交也绑定 row_id＋摘要（与 energetic 一致）。孤立行留在库中作审计记录，不自动成为轨迹。
- 反例修复后可核对输出：崩溃-重执行场景导出 evaluation_id=[0,1] 两帧，逐帧 row_id 等于 `committed_row` 身份（本例两次解析力数值不同，排除碰巧相同）。
- 新增回归：`tests/unit/test_review_a.py::test_a3_*`。
- 兼容：无日志运行回退旧行为（全部行），report 中注明。

## A4 [P2] inspect 时间/温度来自未完成尾帧

- 原因：步数按 STEP_COMPLETED 计，但时间/温度/能量取自数据库最后一行与最后评估 context（审阅实测 0 完成步却报 0.1 fs、温度与初态不同时刻）。
- 修复（3c13c24）：inspect 经同一 `iter_committed` 接口取权威行；轨迹观测量（时间/能量/温度/last_step）来自最后一个完成边界（或初始评估），温度按完整步帧＋受约束自由度；新增 `last_evaluation` 字段（evaluation_id、step、phase、时间、能量、complete 标记）单独呈现已计算未完成的状态。
- 反例修复后可核对输出：无完成步的崩溃运行报 0 步、0.0 fs、last_step=-1，尾评估以 phase=md_step、complete=False 单独列出。
- 新增回归：`tests/unit/test_review_a.py::test_a4_*`。
- 兼容：trajectory 增加 n_committed 字段，顶层加 last_evaluation；旧字段语义按完成态收紧。

## B2 [P1] Runner 有日志、engine 无日志漏记内部重试（消费侧）

- 原因：包装层仅凭 compute 接受 request_id 就认定引擎自报告，引擎无接收器时内部重试从账本消失（审阅实测 logical=1、actual=1、failed=0）。
- 修复（3e1b813）：`physical_attempt` 在引擎接收器未连接时把运行日志显式接为本次调用的 sink（attach point 按 attempt_sink→event_log→_event_log 顺序，用后恢复）；接受 request_id 却完全没有 sink 属性的组合在启动前以 EventLogError 拒绝（该拒绝不被包装成引擎失败）。
- 反例修复后可核对输出：sink 未连接的引擎经 runner 运行后记为 logical=1、actual=2、failed=1，引擎真实启动数与 logical 一致，调用后 sink 恢复未连接；无 sink 组合零启动拒绝。
- 新增回归：`tests/unit/test_review_b.py::test_b2_*`。
- 兼容：配置工作流与 Al 19/19 路径不变（引擎构造时带日志→直接自报告）。

## B3 [P1/P2] 账本口径残留（消费侧）

- 零启动误计：`summarize_tasks` 曾按"有无任意 attempt"判新旧日志。修复（3e1b813）：RUN_START 写 `attempt_ledger="physical_attempt_v1"` 标记，新日志按显式协议判定；无标记且无 attempt 的日志保持旧语义。回归：全新日志零启动失败 actual=0/failed=0；同内容无标记旧读法 actual=1/failed=1。
- 缓存命中耗时：plain MD 静止命中曾沿用旧 SCF wall_time。修复：命中记本次访问耗时（wall_time_s=5.0 假引擎下命中条目 <1 s，真实执行条目 =5 s）。
- 首次失败无父请求：relax 求值失败曾只写 RUN_END。修复：失败在统一出口关闭其唯一 task 记录，task_id 与 attempt 的 request_id 相同（回归逐一核对）。
- 嵌套 I/O 与 task span 不重复相加（0.4.0 R6 已立）；"已启动 attempt 必有终态""一次明确执行读整份结果"为引擎侧（后端组）。
- 新增回归：`tests/unit/test_review_b.py::test_b3_*`。

## 验证

- `uv run pytest tests/unit tests/test_smoke.py -q`：**481 passed, 1 deselected**（审阅基线 458 → ＋23 项回归；C 系 13、A 系 5、B 系 5）。
- 修复前/后：A 系 4 失败→5 通过（plain-MD 完成步项修复前后均通过）；C1/C2/C4 首恢复项与 B 系 5 项修复前失败、修复后通过；C3 中 success-log 注入项修复前失败。
- `uv run ruff check <改动文件>`：无新增错误（energetic.py TRY004 为存量基线，行号随内容平移）。
- 最小核心依赖不变（NumPy＋ASE＋标准库）；新测试 hermetic，无真实 SCF。

## 保留限制

- B1（ASE 包装计算器指纹）与 B2/B3 引擎侧（启动/终态/后处理/耗时）由后端组在 worktree 修复，本批不含；合并后建议跑合并回归确认 sink 协议端到端吻合（消费侧对三个 attach 点名做了兼容）。
- C4 对 0.4.0 旧日志窗口（proposal 无 segment/n_calibrations 字段）保持旧行为——该窗口的段状态无法事后重建，属已知限制；新日志完整覆盖。
- A2 的无日志历史运行导出全部行（回退语义），inspect 的 trajectory 增加 n_committed、顶层增加 last_evaluation，旧字段按完成态收紧语义。
- A1 修复的是导出重建；积分器本身未被该反例证伪（同审阅判断）。
- 本轮未做打包冒烟（wheel/sdist 仓库外最短工作流）——审阅收尾条件之一，建议在所有分项（含后端 B 系）合入后统一执行。
