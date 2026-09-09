# WP02 回报 — 运行身份、数据与完整成本

## 本次范围

- 工作包／审阅节点：WP02（R1 的第三部分）
- 基线提交、分支、当前提交：基线 `4c33023`（WP01 完成点）；分支 `dev/0.4.0`；当前提交见文末 SHA。
- Python／ASE／NumPy 及有关后端版本：Python 3.12（uv）；ASE ≥3.29、NumPy ≥1.26；本轮不运行真实后端。
- 本次解决的用户问题：运行/评估/标签/任务缺少持久身份；无权威事件记录与完整成本账本；两种缓存未区分；无法离线复算独立检查；无可读运行状态与导出。
- 实际实现的功能：
  - `runtime/events.py` `EventLog`：单写者 JSONL 权威事件存储（run_start/evaluation_proposed/evaluation_committed/task/model_update/run_summary/run_end），单调 `seq` 即"已提交事件编号"，`iter_events(after_seq=)` 为重放游标；`append_once(key)` 幂等去重（键集跨重开持久）；`O_CREAT|O_EXCL` 锁文件拒绝并发写者（第二次并发启动同一 run 被拒）。
  - 运行身份：run/evaluation/segment/model/reference-settings/task/attempt/label 各有 ID（task `run-task-N`、attempt 分组重试、label `run-label-N` 持久耐用）；`run_start` 记录软件版本、reference_id、model_id、surrogate fingerprint、策略参数；首个评估事件记录 `input_hash`（全物理输入哈希）。
  - Store schema：`data["schema_version"]=2`（`STORE_SCHEMA_VERSION`），`data["engine_label_id"]`；旧行（无标记）可读：`driving_label`/`iter_labels`/`iter_observations`/`latest_state` 不变，`summary_csv` 以空白字段导出。
  - 成本账本：每次物理执行（reference anchor/refusal/probe/verification、inference proposal/probe/calibration、training/model_update、io/trajectory_append）一条 task 事件，字段含 task_id/attempt/status(success|failed|cache_hit)/started_unix/elapsed_s/cpu_cores(null)/gpu(null)/queue_s(null)/source/evaluation_id(父评估)/label_id；`summarize_tasks` 只汇总叶子任务，区分逻辑请求/实际执行/失败尝试/缓存命中；外层墙钟由 `EnergeticRunner.run` 直接测量写入 `run_summary`，不与叶子重复相加。账本只追加。
  - 数值标签缓存（§5.4，`runtime/labels.py`）：键=原子序数+坐标+晶胞+PBC+初始电荷/磁矩+参考 fingerprint+能量口径+属性，精确字节哈希，无近邻匹配；质量与速度故意排除；仅用于 verification 检查（命中则检查正常计数、物理 SCF +0）；引擎无 fingerprint 默认关闭。
  - `store/replay.py` `replay_verification`：以已冻结 driving force 与同定义 reference force 复算逐评估误差与违规，重放检查流重建 IndependentCheckBound；p=0 报关闭，p=1 给精确观测比例；逐行比对在线记录。
  - `runtime/inspect.py`：`inspect_run`（结构化 dict：完整步/物理时间/能量/温度/参考执行vs请求vs缓存命中/独立检查/最后 checkpoint(null, WP03)/失败原因/模型代次/事件游标）、`format_inspection`（同源可读文本）、`summary_csv`（逐评估 CSV）。
- 与原计划的偏离及原因：
  - 事件记录与缓存对既有用法为可选（`event_log=None` 默认关闭、引擎无 fingerprint 时缓存自动关闭）：旧测试与直接 Calculator 用法行为零变化；WP04 workflow 将默认接线事件存储。
  - 标签缓存仅用于 verification：§5.4 只明确授权检查复用；refusal/anchor/probe 目前总是真实执行（probe 重用留作后续显式决定）。
  - `EnergeticRunSummary` 字段未改（兼容）；富化摘要在 `inspect_run`，避免改动既有返回契约。
  - attempt 分组仅用于轨迹上标签（anchor/refusal/verification 的重试共享 task_id）；探针重试在 `_calibrate` 内整体重跑，按独立任务记录（§5.3 要求的只是同评估身份与同检查抽样，均已满足）。

## 改动清单

- 新增 `src/pyraimd2/runtime/events.py`：`EventLog`（append/append_once/iter_events/last_seq/close/上下文管理）、`EventLogError`、事件类型常量、`EVENT_SCHEMA_VERSION=1`。
- 新增 `src/pyraimd2/runtime/labels.py`：`label_key`（sha256 精确规范化）、`atoms_input_hash`、`LabelCache`（enabled 仅当 fingerprint 存在）。
- 新增 `src/pyraimd2/runtime/costs.py`：`summarize_tasks`（叶子汇总；reference 四分类计数；by operation×purpose 桶）。
- 新增 `src/pyraimd2/runtime/inspect.py`：`inspect_run`/`format_inspection`/`summary_csv`。
- 新增 `src/pyraimd2/store/replay.py`：`replay_verification`。
- `src/pyraimd2/store/store.py`：`STORE_SCHEMA_VERSION=2`；`append(..., label_id=None)` 写入 `data["engine_label_id"]` 与 `data["schema_version"]`；读取路径不变。`store/__init__.py` 导出常量。
- `src/pyraimd2/switch/base.py`：`LabelObservation` 追加可选 `label_id=None`（位置构造兼容）。
- `src/pyraimd2/loop/energetic.py`：
  - `EnergeticCalculator.__init__` 新增 `event_log=None`、`label_cache=True`；发 `run_start`；新增 `_emit/_emit_once/_new_task_id/_new_label_id/_emit_task`。
  - `_predict(atoms, purpose)` 与 `_reference(atoms, purpose, *, event_purpose, task_id, attempt) -> (label, label_id)`：任务事件（success/failed）、label_id 分配、缓存写入；参考口径查缓存用 `engine_capabilities(...).energy_kind`。
  - `_finish`：verification 先查标签缓存（命中记 cache_hit 事件、不计 reference_calls、检查照常计数）；轨迹标签重试共享 task_id 递增 attempt；store.append 包 io 任务事件；提交后 `evaluation_committed`（append_once 键 `evaluation:<run>:<id>`，eval0 带 input_hash）；`on_label` 包 training 任务事件，`LabelObservation` 带 label_id；代次递增后 `model_update`（append_once 键 `model-update:<label_id>:eval-<id>`）。
  - `calculate`：冻结后、外部计算前写 `evaluation_proposed`（含检查抽签，append_once 键 `proposal:<run>:<id>`）——§5.3 的"外部计算前持久化冻结提议及检查抽样"。
  - `_calibrate` 探针记录带 `label_id`；`_prepare_updated_model` 任务归属原点评估 ID。
  - `EnergeticRunner` 透传 `event_log`/`label_cache`；`run()` 成功写 `run_summary`（外层墙钟直接测量），失败写 `run_end(status=failed, reason)` 再抛。
- `runtime/__init__.py`、`store/__init__.py` 导出新符号。
- 新接口、配置字段和默认值：全部为可选关键字参数/带默认字段；无新配置文件。
- 旧接口／已有数据的兼容方式：旧 db 行无 schema_version 也可读可导出（专项测试）；`Store.append` 旧调用签名不变；`LabelObservation` 旧位置构造不变；`EnergeticRunSummary` 不变。
- 是否改变单位、力预算、时间、约束、随机检查或模型切换语义：否。决策顺序（冻结→决定→抽查→持久化→提交）未动；事件写入为旁路追加。

## 验收证据

```text
验收项：一条已知调用顺序的受控轨迹能手工对上账本
命令：uv run pytest tests/unit/test_cost_ledger.py::test_controlled_trace_matches_handcomputed_ledger -q
测试：eval0（anchor+4 probes）+ eval1（accepted+checked），手算期望值写死在断言里
通过标准：reference anchor/probe/verification=1/4/1，inference proposal/probe=2/4，io=2，label_id 6 个互异，logical==actual==6==engine.attempts，task 事件字段齐全（cpu/gpu/queue 为 null）
实际关键结果：全部相等；store 行带 schema_version=2 与 engine_label_id
状态：通过
```

```text
验收项：失败和探针均进入总成本
命令：uv run pytest tests/unit/test_cost_ledger.py::test_failed_attempt_and_probes_enter_total_cost tests/unit/test_cost_ledger.py::test_run_failure_recorded_with_reason -q
测试：fail_on 注入一次探针失败（随后重试成功）与一次检查失败（runner 失败）
通过标准：failed task 事件含 purpose=probe/verification、elapsed、error；reference 计数 successful=6/failed=1/actual=7=attempts/logical=7；run_end 记录失败原因；probe 桶 count=6 含 failed=1
实际关键结果：满足
状态：通过
```

```text
验收项：缓存命中不计实际 SCF；静止几何+p=1+缓存时每个新接受评估计入检查，重放不重复计数
命令：uv run pytest tests/unit/test_label_cache.py -q
测试：带 fingerprint 引擎 + 回调两次改模型：eval0 锚定标签入缓存；eval1/eval2 同几何新接受评估被 p=1 检查，verification 全部缓存命中（reference_calls["check"]==0，attempts 只随重锚定探针增长 1→5→9），accepted_count 1→2；回调返回 False 后的同几何请求为重放：计数/事件/attempts 全不变
通过标准：cache_hit 事件 purpose=verification、label_id 为原点耐用 ID；无 fingerprint 引擎对照组检查走真实 SCF（check==1）
实际关键结果：满足
状态：通过
```

```text
验收项：重复回调不重复写同一逻辑事件（label ID 去重）
命令：uv run pytest tests/unit/test_cost_ledger.py::test_duplicate_callback_cannot_rewrite_a_logical_event tests/unit/test_event_log.py::test_append_once_deduplicates_by_key_across_reopen -q
测试：回调内以 label_id 键重复 append_once（第二次返回 None）；model_update 每 (label,evaluation) 一条；去重集跨重开有效
通过标准：label_consumed/model_update 各 1 条；LabelObservation 携带 label_id
实际关键结果：满足
状态：通过
```

```text
验收项：验证重放与在线记录一致
命令：uv run pytest tests/unit/test_replay_verify.py -q
测试：含违规的 p=1 轨迹（accepted=detected=1）、干净接受 p=1（bound==0.0）、p=0（enabled=False）
通过标准：重放 accepted/detected/bound 与在线 metadata 逐行一致（matches_online，mismatches==[]）；p=1 bound==精确观测比例 1.0/0.0
实际关键结果：满足
状态：通过
```

```text
验收项：两次并发启动同一 run 被拒绝
命令：uv run pytest tests/unit/test_event_log.py::test_concurrent_writer_rejected_until_close tests/unit/test_cost_ledger.py::test_concurrent_run_start_rejected_by_lock -q
测试：活跃 EventLog 持锁时第二次打开（含挂 runner 的场景）抛 EventLogError；close 后可重开且 seq 连续
通过标准：并发写者被拒绝而非静默串行
实际关键结果：满足
状态：通过
```

```text
验收项：支撑性测试（身份/键/inspect/旧格式）
命令：uv run pytest tests/unit/test_event_log.py tests/unit/test_inspect.py -q
测试：游标/续写/损坏行报错；label_key 对坐标/晶胞/PBC/电荷/磁矩/元素/设置/口径/属性敏感、对质量与速度不敏感；inspect 结构（执行vs请求vs命中、检查、失败原因、last_checkpoint=None）、format_inspection、summary_csv；旧格式行 driving_label/iter_labels/CSV/inspect 兼容
通过标准：如断言
实际关键结果：19 个新文件测试全部通过
状态：通过
```

```text
验收项：受影响套件整体回归
命令：uv run pytest tests/unit tests/test_smoke.py -q
通过标准：全部通过
实际关键结果：221 passed（基线 202 + 新增 19）；ruff 全仓 13 个错误（与 WP01 后基线相同，新增 0；改动文件中仅存基线既有 TRY004 一处）
状态：通过
```

## 恢复与随机状态（相关工作包必填）

- 本 WP 不实现 resume（WP03），但交付其所需接口：事件 `seq` 为已提交事件编号（checkpoint 游标）；`iter_events(after_seq=)` 按原逻辑顺序重放；`evaluation_proposed` 在外部计算前持久化冻结提议与检查抽签；`append_once` 保证重放已提交事件不重复追加、不重新抽签；label_id 耐用且入 store 行；检查 RNG 语义未动（现有失败重试测试原样通过）。

## 算力与成本

- 新增实际参考执行总数：0（全部解析假后端）。
- 预算使用与剩余额度：WP08 配额未动用。

## 回归与交付

- 受影响测试通过情况：tests/unit + test_smoke 全绿（221）。
- 原有 energetic／legacy switching 示例：未改动；legacy Runner/SwitchingCalculator/online 文件零改动（仅 `LabelObservation` 增可选字段、`Store.append` 增可选参数）。
- wheel 安装及仓库外 CLI 测试：未执行（WP04/WP09）。
- 最小依赖是否仍不导入 Torch／PySCF：是（已断言导入 runtime/store.replay/loop 不加载）。
- 用户可复制的完整运行与恢复命令：同前；新库层 API 见 tests/unit/test_inspect.py 用法（`EventLog` + `EnergeticRunner(..., event_log=)` + `inspect_run(dir)`）。
- README／示例／支持矩阵更新位置：未更新（WP09）；接口文档为各新模块 docstring + 本报告。
- 已知故障、未执行验证与风险：
  - 锁文件在进程崩溃后成为 stale lock，需用户人工删除（文档已写明，属刻意的安全拒绝）。
  - inspect 的 run_id 推断：无 events.jsonl 且 db 多 run 时报错要求显式 run_id；run 目录布局（多 .db）在 WP04 固定后可能需要收窄 `_find_db`。
  - 标签缓存为进程内存级（不持久）；当时判断崩溃后缓存消失只影响成本不影响正确性（WP03 恢复不依赖它）。**2026-09-09 修正**：独立审阅反例 F02 证明该判断不成立——冷启动缓存给同一参考标签分配新 label ID，恢复比连续多消费一个标签并多触发一次更新。已在 R1-2 修复（恢复时从持久记录重建标签键→label ID 索引），见 REVIEW_FIXES_CORE_20260909.md。
  - probe 目前不使用标签缓存（见偏离）；cpu/gpu/queue 恒为 null（本环境无来源）。
- 需要负责人判断的具体决策及备选方案：
  - 并发策略选择"拒绝"（锁）而非串行化等待：恢复/分叉语义下静默排队更危险；如希望排队等待，可在 EventLog 上加显式 wait 选项。
  - `evaluation_proposed` 事件体量随 forecasts 增长（每方向一条）；如成为 I/O 负担，可在 WP03 裁剪为恢复必需子集。
- 下一工作包及其入口：WP03——`EnergeticRunner.resume`、`_EnergeticVerlet`、Store + 新 checkpoint 模块；直接使用本 WP 的事件游标（`last_seq`/`iter_events(after_seq=)`）、冻结提议事件、label_id、RNG 后状态记录点与代次身份。
