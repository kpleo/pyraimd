# WP03 回报 — 完整续算与故障恢复

## 本次范围

- 工作包／审阅节点：WP03（R1 的最后一部分）
- 基线提交、分支、当前提交：基线 `6608806`（WP02）；分支 `dev/0.4.0`；当前提交见文末 SHA。
- Python／ASE／NumPy 及有关后端版本：Python 3.12（uv）；ASE ≥3.29、NumPy ≥1.26；全部为解析假后端，无真实 DFT/torch。
- 本次解决的用户问题：energetic 运行无法续算；崩溃后无可靠恢复基点；模型更新与检查随机流状态不可持久化。
- 实际实现的功能：
  - `runtime/checkpoint.py` `CheckpointManager`：`checkpoints/<generation>/{state.json, arrays.npz, manifest.json}`，临时目录写入→逐文件刷盘→sha256 校验清单→`os.replace` 原子更新 `latest.json` 指针；保留最近 2 代；读取时从指针回退到最近**有效**代（截断/篡改校验失败即跳过）。
  - checkpoint 内容（§6 七项）：Atoms 数组/连续坐标/PBC/晶胞/质量/初始电荷磁矩（npz）＋边界**整步动量**与可复用驱动力；真实时间与下一评估 ID；锚点（位置/参考与基势能量和力/常数校正/responses/open_prefix/segment/代次，复用 `_anchor_record` 格式）；deferred 重锚定原点（含动量，供方向计算）；策略参数与 `_next_reason`；检查计数与**检查 RNG 完整 bit-generator 状态**；模型身份/代次与 updater 完整状态；`last_event_seq`（事件游标）与三份 schema 版本。
  - `EnergeticRunner.resume`：新进程（测试中全新 Python 对象）读取最近有效 checkpoint → 校验参考 fingerprint/模型链/updater → 从游标按原序重放事件（committed 评估按记录值应用：计数、检查流推进（每个 accepted 一抽）、bound 更新、锚点演化、label/task 计数器按最大后缀续号）→ 重建边界原子态（位置＋半步动量＋0.5·dt·F 的整步动量，ASE 踢不含质量）→ 若有未提交尾部提议则按原检查抽签重建 pending 并调度重放。不重热化、不重训、不重抽、不重复消费。
  - 更新协议：`runtime/updater.py` `StatefulUpdater`（`__call__`/`state_dict`/`load_state_dict`）；每次回调后持久化 updater 状态（`label_consumed`/`model_update` 事件载荷）；模型工件 `models/<model_id>/state.json` 在 model_update 事件提交**之前**原子写入（runner 的 `_publish_model_artifact`）；重放只在原边界加载已持久化状态，工件缺失/消费状态缺失/training 失败记录一律以 ResumeError 停止并报告待处理。
  - 探针即时持久化：每个成功探针一条 `probe_completed` 事件（含完整记录与 label_id）；校准中途崩溃后重放复用同评估同代次的已验证探针（按 direction/step/sign/代次精确匹配＋位移 1e-7 校验），只补跑缺失探针。
  - 信号：`handle_sigint=True` 安装 SIGINT 处理器（只置标志）；`request_stop()` 在下一完整步边界写 checkpoint 并停止（`run_end status=stopped`），正常控制流执行全部文件提交；停后同一 runner 可继续。
  - `fork`：从父运行最近 checkpoint 启动新运行目录/新 run_id，物理态与模型链继承，策略可覆盖（新检查段），记录 `forked_from`（parent id/dir/checkpoint 代/模型 id）；`resume` 对策略或后端身份不匹配一律拒绝。
  - 事件协议补全：`step_completed`（每步边界，相位显式）、`probe_completed`、`label_consumed`、`resumed`、`forked_from`；`evaluation_proposed` 富化为完整冻结记录（位置/预测/冻结能量力/open_prefix/anchor_record/deferred_record）；`_anchor_record` 增加力与 open_prefix 与代次。
- 与原计划的偏离及原因：
  - **get_property 语义细化（重要）**：WP01 的"模型变更使同几何缓存失效"规则与 runner+updater 流程冲突——ASE 的 `irun` 在步间会重读力（`_refresh_properties`），更新后该重读必须按 §5.2 作为已提交事实的重放返回，不能变成新评估。现按模式区分：积分器驱动的运行（首次 `_schedule` 后）中，未调度请求一律是重放；无调度的兼容层（直接 Calculator 用法）保持 WP01 规则不变（其验收测试原样通过）。这是对 WP01 规则的语义修正，两版各有测试。
  - 探针重试的进程内路径仍整体重跑（WP02 既有行为）；跨进程复用只发生在重放路径。
  - 崩溃前已执行但未应用的参考计算（标签已得、评估未提交）在恢复时重算：其成本已在账本中计费，逻辑事件不重发；§6 的"已完成但未应用"复用留给后续（需在任务事件中持久化标签值，超出本 WP 范围）。
  - `keep_generations=2` 固定写死（配置化属 WP04）。

## 改动清单

- 新增 `src/pyraimd2/runtime/checkpoint.py`：`CheckpointManager`（write/read_latest_valid/_prune/next_generation）、`Checkpoint` dataclass、`CheckpointError`、`ResumeError`、`rng_state_to_json`、`CHECKPOINT_SCHEMA_VERSION=1`。
- 新增 `src/pyraimd2/runtime/updater.py`：`StatefulUpdater` Protocol。
- `src/pyraimd2/runtime/events.py`：新增 `STEP_COMPLETED/PROBE_COMPLETED/LABEL_CONSUMED/RESUMED` 事件类型；`EventLog(..., force=False)`——`force=True` 为崩溃后刻意的锁回收（文档明确禁止用于排队写者）。
- `src/pyraimd2/runtime/inspect.py`：`last_checkpoint` 现在报告真实值（代次/步数/物理时间/事件游标/valid）。
- `src/pyraimd2/loop/energetic.py`：
  - `_anchor_record` 增加 `base/reference_forces_eV_A`、`open_prefix`、`model_generation`；新增 `_response_from_dict`、`_anchor_from_record`、`_id_suffix`、`_is_stateful`、`StopRequested`。
  - calculator：`_policy_dict`、`_checkpoint_payload`、`_apply_checkpoint_state`、`_replay_committed`、`_replay_label_event`、`_rebuild_pending`；`__init__` 增加私有 `_resume_state`（跳过 run_id 查重与 run_start）；`_calibrate` 发 `probe_completed` 并支持 `reused_probes` 计数与复用；`_finish` 回调后持久化 updater 状态（`label_consumed`/`model_update` 载荷）并在 model_update 前调用 `_model_publisher`；`calculate` 的 proposal 事件富化。
  - get_property：按 `_integrator_owned` 区分重放/新评估语义（见偏离）。
  - 模块级 `_replay_window`（含未消费标签的安全检测）、`_model_artifact`、`_check_resume_safety`。
  - `_EnergeticVerlet.step` 发 `step_completed`（幂等键 `step:<run>:<n>`）。
  - `EnergeticRunner`：`run_dir`/`checkpoint_interval_steps`/`handle_sigint` 参数；`_write_checkpoint`/`_maybe_checkpoint`（attach 观察器）、`_publish_model_artifact`、`request_stop`/`close`；`run()` 处理 `StopRequested`；类方法 `resume`（真实实现）与 `fork`。
- `src/pyraimd2/runtime/__init__.py`：导出 checkpoint/updater 符号。
- `tests/unit/test_energetic_loop.py`：`test_explicit_restart_rejected_without_changing_legacy_runner` 的 resume 断言由 NotImplementedError 更新为"无 checkpoint 目录被 ResumeError 拒绝"（resume 已实现，原断言前提失效；直接 Calculator 的 run_id 查重拒绝保持不变）。
- 新接口、配置字段和默认值：均为可选关键字参数；checkpoint schema 1、事件 schema 1、store schema 2 均记录于 checkpoint manifest。
- 旧接口／已有数据的兼容方式：不传 `run_dir`/`event_log` 的旧用法零变化（全部旧测试通过）；`_anchor_record` 新增键为纯增量。
- 是否改变单位、力预算、时间、约束、随机检查或模型切换语义：否（get_property 的模式区分见偏离，兼容层行为不变）。

## 验收证据

```text
验收项：连续 100 步 vs 40 步＋新进程恢复 60 步
命令：uv run pytest tests/unit/test_resume.py::test_continuous_100_vs_40_resume_60 -q
测试：同一解析世界两种走法（恢复世界 checkpoint 间隔 7，最后 checkpoint 在 35，重放 36..40）
预先确定的通过标准：逐行 route/reason/check_draw/checked/violation/model_id/label_id/检查计数严格相等；位置/动量/驱动力 atol=1e-12；计数/updater(n_consumed/n_updates/k)/verification 严格相等；终态位置动量 1e-12
实际关键结果：101 行逐项相等； counters 全部相等
状态：通过
```

```text
验收项：中断位置覆盖锚点后、接受检查后、模型更新边界
命令：uv run pytest "tests/unit/test_resume.py::test_interruption_points_resume_identically" -q
测试：探针运行找出三类边界步（重锚定锚点/被检查接受/模型更新），各在中点处 interval=1 停-恢复-跑完，与连续 100 步对照
通过标准：三种中断各自逐行一致（assert_same_run 全字段）
实际关键结果：3 个参数化用例全部通过
状态：通过
```

```text
验收项：checkpoint 间隔>1 的窗口崩溃重放（不重复抽样/训练/消费/计数，账本只追加）
命令：uv run pytest tests/unit/test_resume.py::test_crash_window_replay_without_double_effects -q
测试：interval=10，40 步后再跑 5 步（41..45 无 checkpoint）弃置（锁泄漏模拟崩溃），force 恢复后跑到 100 与连续对照
通过标准：恢复点 n_evaluations==46；轨迹逐行一致；updater 消费/更新计数严格一致；崩溃前 task 事件数只增不减
实际关键结果：满足
状态：通过
```

```text
验收项：故障注入（截断 checkpoint／外部参考失败／更新失败／重复恢复）
命令：uv run pytest tests/unit/test_resume.py -q -k "truncated or reference_failure or failed_update or concurrent"
测试与标准：
 截断：gen 11 截断→回退 gen 10＋重放 step 10（resumed 事件记 checkpoint_generation==10），后续与连续一致；
 参考失败：fail_on 注入窗口内一次失败→runner 拒绝继续（半步不继续）→恢复后与无故障连续运行逐行一致；失败 task 仍在账本；committed 事件 101 个无重复；
 更新失败：updater 第三次消费抛错→ResumeError("failed model update")拒绝恢复→fork 从 checkpoint 成功（模型链继承，forked_from 记录父 run）；
 重复恢复：并发第二次 resume 被锁拒绝；顺序 resume→run→close→再 resume 仍与连续逐行一致。
实际关键结果：全部满足
状态：通过
```

```text
验收项：参考失败后的半步状态不直接继续积分
命令：同参考失败用例
通过标准：失败 runner 的 run() 抛 RuntimeError；失败评估无 store 行；恢复后轨迹与从未失败的连续运行一致（含同一检查抽签——由持久化提议重建）
实际关键结果：满足
状态：通过
```

```text
验收项：探针即时持久化与中途崩溃复用
命令：uv run pytest tests/unit/test_resume.py::test_verified_probes_reused_after_mid_calibration_crash -q
测试：重校准第 3 支探针注入失败（2 支已验证并持久化）→崩溃→恢复
通过标准：重校准记录 reused_probes==2，仅补跑 2 支探针（引擎执行数吻合）；轨迹与连续对照一致
实际关键结果：满足
状态：通过
```

```text
验收项：信号停止与边界 checkpoint；恢复前置拒绝
命令：uv run pytest tests/unit/test_resume.py -q -k "stop_request or sigint or missing_model_artifact or requires_the_updater"
测试与标准：request_stop 与真实 SIGINT（os.kill 自身）各在下一步边界停止并写 checkpoint，可继续且与连续一致；删除游标后模型工件→ResumeError(pending)；无状态导出的纯回调消费标签后→ResumeError(updater)
实际关键结果：满足
状态：通过
```

```text
验收项：受影响套件整体回归
命令：uv run pytest tests/unit tests/test_smoke.py -q
通过标准：全部通过
实际关键结果：240 passed（基线 221 + 新增 19）；ruff 全仓 13 个错误（与基线相同，新增 0）
状态：通过
```

## 恢复与随机状态（相关工作包必填）

- 连续轨迹与分段恢复轨迹的长度、完整步位置：100 步 vs 40＋恢复 60（另含 55/56/57 等参数化边界）。
- 是否新进程恢复：以全新 Python 对象（surrogate/updater/engine/Store/EventLog）模拟新进程，除文件系统外无共享可变状态；同 CPU 解析模型确定性成立。
- 位置／动量差及单位：位置 ≤1e-12 Å、动量 ≤1e-12（ASE 动量单位），逐行比对。
- route、evaluation、segment、model ID、检查抽样／计数是否一致：严格一致（含 label_id 与 model_id 字符串）。
- 准入前缀和检查上界是否一致：open_prefix 与 accepted/detected/bound 逐行一致。
- 故障注入位置与恢复结果：见验收证据 4–6。
- checkpoint 之后持久事件的重放范围、积分阶段及 RNG 后状态：游标后全部 committed 事件按序重放；每个 accepted 评估推进一次检查 RNG；`step_completed` 标记完整步相位；重放后下一次抽签与连续运行逐值一致。
- 检查／模型更新已提交而新 checkpoint 未写入时的恢复证据：test_crash_window（interval=10，窗口 5 步）。
- 重复 label／probe／update 的处理：label_id/task_id 按事件最大后缀续号；append_once 键防重；已验证探针按键复用。
- 模型身份不匹配、破损快照或缺工件的处理：ResumeError 拒绝（fingerprint/模型链/updater/工件四道校验）。
- 已完成但未应用的计算如何计费、保存和复用：已计费（task 事件只追加）；WP03 不重用之（恢复时重算），复用留待后续（需在任务事件中持久化标签值）。

## 算力与成本

- 新增实际参考执行总数：0（全部解析假后端）。
- 预算使用与剩余额度：WP08 配额未动用。

## 回归与交付

- 受影响测试通过情况：tests/unit + test_smoke 全绿（240）。
- 原有 energetic／legacy switching 示例：未改动；legacy Runner/SwitchingCalculator 零改动。
- wheel 安装及仓库外 CLI 测试：未执行（WP04/WP09）。
- 最小依赖是否仍不导入 Torch／PySCF：是。
- 用户可复制的完整运行与恢复命令：库层 API——`EnergeticRunner(..., event_log=EventLog(dir), run_dir=dir, checkpoint_interval_steps=k)` → `EnergeticRunner.resume(dir, model, engine, updater=updater)`；`EnergeticRunner.fork(old_dir, new_dir, new_id, ...)`。
- README／示例／支持矩阵更新位置：未更新（WP09）。
- 已知故障、未执行验证与风险：
  - 崩溃锁需 `event_log_force=True` 刻意回收（resume 默认 force=False，并发写者仍被拒绝）。
  - 标签缓存为内存级，恢复后冷启动（正确性不受影响，首个检查会真实执行）。
  - 第一步尚未完成即崩溃（无任何 checkpoint）时无恢复基点，ResumeError 拒绝——需从头开始，属既定语义。
  - 自定义 `direction` 回调不参与恢复校验（恢复方需自行保证同一 callable；默认速度方向完全覆盖）。
  - WP01 兼容层与积分器模式的 get_property 语义差异已写入类 docstring 与本报告；若负责人希望统一为单一规则，需要改动 WP01 兼容层验收测试。
- 需要负责人判断的具体决策及备选方案：见上 get_property 偏离；`keep_generations=2` 固定值待 WP04 配置化。
- 下一工作包及其入口：WP04——`config.py`/`cli.py`/`workflows/`。留给 WP04/WP06 的接口：
  - `EnergeticRunner(..., run_dir=..., checkpoint_interval_steps=...)` / `.resume(run_dir, model, engine, updater=...)` / `.fork(...)` / `.close()` / `.request_stop()`：CLI run/resume/inspect 的直接包装对象。
  - `StatefulUpdater` 协议与 `_model_publisher` 钩子（`models/<model_id>/state.json` 工件格式 `{model_id, updater_state, written_unix}`）：WP06 的候选模型→验证→发布在此处接入，不形成依赖环。
  - checkpoint manifest 的 `last_event_seq` 为恢复游标；`EventLog.iter_events(after_seq=)` 为重放口。
  - `inspect_run` 已含 `last_checkpoint`。
