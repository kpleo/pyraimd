# 第二轮独立审阅修复回报 — 后端部分（B1 / B2 / B3 引擎侧）

修复日期：2026-09-09。对应审阅：`INDEPENDENT_REVIEW_040_20260909.md`。
修复提交：分支 `dev/0.4.1-bfix` 顶端 `c8786c6`，已经 `dec5c93` 合并回 `dev/0.4.0`；联调修正 `b8c6a43`。
核心侧（A1–A4、C1–C4、B2/B3 消费侧）见 `REVIEW_FIXES_040_CORE_20260909.md`；共享事件语义约定七条以该文件为准。

- 新增实际 SCF 数：0（全部 fake executable 与解析输入）。
- 测试：合并后 `496 passed, 1 deselected`；后端 15 项新回归在修复前代码上实测全部失败、修复后通过；ruff 改动文件无新错误。

## B1（P1）ASE 包装计算器身份碰撞

- 原因：`SumCalculator([LennardJones(epsilon=1)])` 与 epsilon=2 的 `parameters={}`，指纹相同而同构型能量差一倍；JSON 可序列化不等于包含全部物理设置。
- 修复：`calculator_identity` 递归识别 ASE 包装结构（`.mixer.calcs`/`.weights` 或 `.calcs`），纳入子计算器与权重并带循环防护；任一子计算器不可识别则整体指纹 None（诚实 unknown）；显式 `identity=` 是逃生口。
- 反例修复后输出：SumCalculator epsilon 1 vs 2 指纹不同且非空可信；审阅归档脚本 `pyramid_040_backend_tiny.py` 的 fingerprint_ase_sum 碰撞消除。
- 回归：`test_ase_adapters.py` 新增 3 项（SumCalculator/MixedCalculator/不可识别子）。
- 保留限制：只递归识别 ASE mixing 包装结构；无 `parameters` 且无 `.calcs` 结构且参数为空 dict 的自定义包装器会得到基于空参数的可信指纹——已知边界，用显式 `identity=` 声明。

## B2（P1）Runner 有日志、engine 无日志漏记内部重试

- 原因：直接构造 QeEngine（不传 event_log）再向 EnergeticRunner 传日志，包装层仅凭 compute 接收 request_id 就认定后端会上报，实际启动 2 次账本 actual=1。
- 修复（两侧共同实现，按约定第 4/6 条）：
  - 引擎侧：每个接受 request_id 的引擎暴露 sink 属性（`attempt_sink`/`event_log`/`_event_log`，未连接为 None）；引擎无 sink 属性却收到 request_id 时启动前拒绝（零副作用）。无 request_id 时引擎自生成并在 `last_attempt_records` 留完整可消费记录（含 started_unix/failure_kind/returncode）。
  - 包装侧（核心组）：runner 持有的日志在调用期间显式接到引擎 sink、用后恢复；引擎完全无 sink 属性的组合启动前 EventLogError 拒绝。
- 反例修复后输出（联调后实测）：假程序首次失败再成功——启动 3 次（anchor 失败+重试 + 后续参考 1 次），账本 attempt 事件 3 条（failed/success/success），request_id 齐全。回归：`test_qe_reliability.py::test_runner_log_connects_sink_to_sinkless_engine`（合并时按"显式连接"语义改写原"必须拒绝"断言，见 `b8c6a43`）。

## B3（P1/P2）真实启动、失败终结与耗时统一（分项 F3–F7）

- F3（P1，ASE 隐式额外执行）：`AseEngine.compute` 改为一次显式 `calculator.calculate(atoms, properties, all_changes)` 后直接读整份 `results`，缺 energy/forces/stress 即拒，不再有 getter 触发的隐式重算；`AseQeEngine._attempt` 直接驱动 write_inputfiles→execute→read_results，启动边界精确。反例（缺 stress 输出）：实际启动 2→1、attempt 1 条、标签仍正确拒绝。
- F4（P2，零启动误计）：可执行文件不存在（FileNotFoundError）转为契约内 `QeEngineError("executable not found", retryable=False, failure_kind="executable_missing")`，不发 attempt；两路径一致。新日志按 `attempt_ledger="physical_attempt_v1"` 协议标记判定（消费侧，核心组）。
- F5（P2，后处理失败漏记）：density manifest 写失败 → attempt 终态 `post_processing_failed`，事件必发、原异常保留、不留 running；标签不交付（provenance 无法落盘则不默默继续）。
- F6（P2，首次失败无父请求）：引擎侧 request_id 逐字透传；task 统一 try/finally 由核心组在 workflows 实现。
- F7（P2，耗时口径）：attempt span 覆盖 staging+process+validation（计时起点移到密度拷贝之前），io 事件真实区间必然嵌套（测试断言真实区间关系）；attempt 事件另带 `process_elapsed_s`（纯进程段）；`wall_time_s` 为各 attempt span 总计；缓存命中记本次访问耗时（消费侧）。
- 终态词汇扩为 `success/failed/killed/post_processing_failed` + `failure_kind`（process/parse/timeout/post_processing/executable_missing/input_write）。消费侧已把 killed/post_processing_failed 计入 failed_attempts（合并回归覆盖）。
- 回归：`test_ase_qe.py` +5、`test_qe_reliability.py` +7（含 B2 runner 级用例）；审阅归档脚本复跑：stress_missing 两路径 calls=1/attempt=1/拒绝、manifestdir 两路径 post_processing_failed、no_executable 两路径 0 attempt + 契约错误、warm-start io 嵌套断言为真。

## 保留限制（如实）

- ASE-QE 仍不支持 `timeout_s`（构造拒绝）；超时 kill 进程组只手写路径有（ASE 无进程组管理）。
- direct API 用户不给引擎接日志又不带 request_id 时，引擎本地记录完整但账本聚合粒度为一次调用一条 attempt（不掩盖内部重试的推荐路径是接同一 EventLog）。
- 跨 QE 路径密度工件 fingerprint 不相容（qe-subprocess vs ase-espresso）维持保守拒绝。
