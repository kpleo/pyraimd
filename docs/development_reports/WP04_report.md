# WP04 回报 — 配置、CLI 与正常用户流程

## 本次范围

- 工作包／审阅节点：WP04（R2 的第二部分）
- 基线提交、分支、当前提交：基线 `895897b`（dev/0.4.0 顶端，含 WP00–WP05）；分支 `dev/0.4.0-wp04`（worktree `pyraimd2-wp04`，与主检出的 WP06 并行工作隔离）；当前提交见文末 SHA。
- Python／ASE／NumPy 及有关后端版本：Python 3.12（uv 管理）；ASE 3.29.0、NumPy ≥1.26（uv.lock 锁定）；本机无 pw.x/torch/pyscf，全流程验收均为 builtin 解析后端；QE/MACE 只到 create_backend 参数与路径校验。
- 本次解决的用户问题：CLI 只打印版本提示；没有配置文件；普通用户无法不改源码完成"配置→运行→停止→续算→检查→导出"流程。
- 实际实现的功能：
  - `config.py`：标准库 `tomllib` + dataclass 的配置层。`schema_version`；`[run]` id/directory/seed；`[task]` kind/mode；`[structure]` file；`[dynamics]` ensemble(nve)/timestep_fs/steps/temperature_K/velocity_seed；`[reference]`/`[surrogate]` backend+选项；`[policy]` energetic 参数；`[verification]` probability/failure_probability/tilt/seed；`[checkpoint]` interval_steps/keep_generations；`[output]` 两个输出频率。单位在字段名中。未知段/字段（含近似名建议）、负步长、零步数、probe_steps 不递增、transverse_cap 越界、零检查非法组合（probability=0 时仍设 failure_probability/tilt）、keep_generations≠2、mode×section 不匹配、run.id 含路径分隔符、schema_version 不符均及早报错，错误含点分字段名与补救方式。相对路径一律以配置文件目录为基准解析（换 cwd 结果位置不变）；路径型后端选项（`*_path`/`*_file`/`*_dir`/`model`/`density_source`/`pseudos`+`pseudo_dir`）同样解析并在 validate 时查存在性。`resolved_dict()`/`load_resolved_config` 供 resume 重建配置（纯模式不生效的段记 null）。schema migration 思路写入模块 docstring 与 docs/configuration.md（显式迁移函数 + 严格字段，不靠宽松吞字段）。
  - `cli.py` + `__init__.py` 的 `main()`：`pyramid --version`、`init`（模板写出 run.toml+structure.extxyz）、`validate`（默认不跑 SCF/推理/下载；`--probe-backends` 才做一次小型后端自检）、`run`、`resume`（--steps 为额外步数，打印当前步与目标步；--force-unlock 刻意回收崩溃锁）、`inspect`（人读 + --json 同源）、`export`（--force-source driving|reference|base）、`backends`（列出注册工厂）。--help 离线可用，重模块按命令惰性导入。退出码 0/1/2/130。
  - `workflows/`：`run_workflow`/`resume_workflow`/`validate_setup`/`export_run`/`RunOutputs`/`write_template`，Python API 与 CLI 同一套实现。adaptive 走 WP03 EnergeticRunner（完整 checkpoint/resume）；reference/surrogate 走共享的 ASE VelocityVerlet 小驱动器 `_PlainDriver`，写同构 store 行与事件（run_start/task/evaluation_committed/step_completed/run_summary/run_end，context 语义与 energetic 一致），inspect/export 对三模式一致可用。运行目录按 §6：config.toml 原件、resolved_config.json、manifest.json（含配置/结构 sha256 与后端 fingerprint）、trajectory.db、events.jsonl、checkpoints/、models/、calculations/（QE run_root）、summary.json/csv、trajectory.extxyz（可再生派生物，间隔配置驱动，resume 时自愈重建）。SIGINT 处理器在 run 前保存、结束后恢复。
  - 后端：`backends/harmonic.py`（harmonic-reference/harmonic-surrogate 两个解析 builtin，bias 项计入能量因而如实 force_consistent）；registry 新增 builtin `harmonic-reference`/`harmonic-surrogate`/`mace`（惰性，构造不触发 torch）与公开函数 `backend_factory`；工厂约定：声明 `run_root`/`event_log` 的工厂由 workflow 注入（QE 由此获得 calculations/ 目录与事件日志）。
  - validate 的检查面：TOML/schema、结构可读性与基本物理量（非空、有限、质量、无约束——约束明确报 WP07）、路径型选项存在性（含 QE 结构物种对应赝势在 pseudo_dir 下的存在性，含默认表）、backend 经 create_backend 构造（参数校验，QE 用临时 run_root，不在用户目录落文件）、reference/surrogate 能量口径契约（与 runner 同一 assert）、重复 run_id／目录复用拒绝。
  - export：三种 force-source；缺参考标签的帧写 `forces_available=F` + 全 NaN 力 + 无 energy 键（绝不零补齐），有标签帧带 `reference_label_id`；每帧记录 run_id/step_id/evaluation_id/physical_time_fs/route/force_source；原子级 tmp+replace 写入；默认输出 `export-<source>.extxyz`，覆盖需 --force。
  - 示例与文档：`examples/harmonic_adaptive/`（可离线运行，与 init 模板同内容）、`examples/qe_mace_skeleton/`（QE+MACE 配置形状，非验证过的材料配方，WP08 才交付配方）、`docs/configuration.md`（用户文档：命令、字段、目录布局、路径规则、种子、resume 语义、导出缺失标记、schema migration 政策、当前限制）。
- 与原计划的偏离及原因：
  - **builtin 谐振后端用连字符命名**（`harmonic-reference`/`harmonic-surrogate`）而非计划 §8 的下划线名：WP05 的示例插件已用下划线名注册 entry-point，registry 对 builtin/插件重名是硬错误（其测试 monkeypatch 这两个名字）。连字符名与 `qe-ase` 风格一致，且不会与已安装插件冲突。
  - **`keep_generations` 只接受 2**：WP03 在 `runtime/checkpoint.py` 固定 KEEP_GENERATIONS=2，而 runtime/ 不在本 WP 文件区域。配置字段保留、默认 2，其他值报明确错误（可配置保留数留待后续）；已在报告与文档中如实标注。
  - **`policy.force_metric` 未实现**：计划 §8 示例中的该字段在当前 energetic 策略里不存在对应参数（自由坐标投影属 WP07/FixAtoms），schema 不包含它；写入即按未知字段报错。
  - **builtin 谐振 surrogate 的能量包含 bias 项**（E = ½k|dr|² + bias·Σcos(dr)），使其 force_consistent=True 如实成立；WP05 示例插件的声明（能量不含 bias）未改动。
  - **plain 模式（reference/surrogate-only）不支持 resume**：其续算需要 energetic 的整步 checkpoint 协议（WP03 为 adaptive 专用），本 WP 拒绝时给明确消息，不用 trajectory.db 猜状态；export/inspect 对其可用。
  - **QE 引擎在 resume 路径不接 event_log**：`EnergeticRunner.resume` 自行打开 EventLog，外部无法传入；resume 的 QE 运行少了 density io 任务事件（正确性不受影响，账本缺一类 io 计数），如实记录。
  - **`[output]` 语义**：trajectory_interval_steps 是 run 目录预览 trajectory.extxyz 的抽帧间隔（trajectory.db 永远全量），summary_interval_steps 是 summary.json/csv 刷新节奏（结束必写）；两者都在 resume 时全量自愈重建。
  - **种子默认值**：velocity_seed 与 verification.seed 默认取 run.seed（文档写明；计划示例中三者显式独立，模板保持显式）。
  - **mace 注册为 builtin surrogate 工厂**（WP05 只注册了三个引擎）：MaceSurrogate 模块级不导入 torch，构造与能力/口径校验均惰性，满足"缺可选依赖有清楚提示"（选择时/探针时报 pip extra）。

## 改动清单

- 新增 `src/pyraimd2/config.py`：`PyramidConfig` 及各段 dataclass、`load_config`/`load_resolved_config`/`parse_config`、`ConfigError`、`CONFIG_SCHEMA_VERSION=1`。
- 新增 `src/pyraimd2/cli.py`：argparse 入口与各命令、`build_parser`、`main(argv)->int`（退出码约定）。
- `src/pyraimd2/__init__.py`：`main()` 接 CLI（惰性导入，名字与入口不变）；`__version__` 不变。
- 新增 `src/pyraimd2/workflows/`：`__init__.py`（API 导出）、`md.py`（`run_workflow`/`resume_workflow`/`WorkflowResult`/`_PlainDriver`/`_BackendCalculator`/`_SigintGuard`）、`setup.py`（`validate_setup`/`create_configured_backend`/`build_backends`/`load_structure`/`prepare_run_directory`/`RunOutputs`/`WorkflowError`）、`export.py`（`export_run`/`frames_from_store`/`frame_from_row`/`write_extxyz`/`ExportError`）、`templates.py`（harmonic 模板与 `write_template`）。
- 新增 `src/pyraimd2/backends/harmonic.py`：`HarmonicReference`/`HarmonicSurrogate` 与带 `backend_kind` 的工厂。
- `src/pyraimd2/backends/registry.py`：`_BUILTINS` 增加 `mace`、`harmonic-reference`、`harmonic-surrogate`；新增公开 `backend_factory`；`backends/__init__.py` 同步导出。
- 新增测试 `tests/unit/test_config.py`（31）、`tests/unit/test_workflows.py`（11）、`tests/unit/test_cli.py`（22）。
- 新增 `examples/harmonic_adaptive/`（run.toml + structure.extxyz + README）、`examples/qe_mace_skeleton/`（run.toml + 2 原子 Si extxyz + README）、`docs/configuration.md`。
- 新接口、配置字段和默认值：见上；配置字段均带文档默认值（docs/configuration.md 逐字段列出）。
- 旧接口／已有数据的兼容方式：不改 loop/runtime/store/engines/surrogate 任何文件；既有 283 测试全绿；`pyraimd2:main` 入口名不变；Store 行格式、事件协议零改动（plain 驱动只按既有 schema 写入）。
- 是否改变单位、力预算、时间、约束、随机检查或模型切换语义：否。配置层只做参数到既有 runner 参数的映射；自适应行为全部由 WP01–WP03 的既有实现决定。

## 验收证据

```text
验收项：从模板到运行、恢复、inspect、导出的全流程（tmp_path，含空格目录名，换 cwd 不移动结果）
命令：uv run pytest tests/unit/test_cli.py::test_full_flow_space_directory_and_cwd_independence -q
测试：目录 "my run" 内 init→validate→run→重复 run 拒绝→inspect（人读+--json 同源断言）→三种 force-source 导出→resume --steps 5→inspect；validate/run 在另一 cwd 下执行
预先确定的通过标准：全部命令退出码正确；结果落在配置目录（非 cwd）；run 目录布局齐全（config.toml/resolved_config.json/manifest.json/trajectory.db/events.jsonl/checkpoints/summary.*/trajectory.extxyz）；重复 run.id 报错含字段名与补救；resume 打印 "at complete step 20" 与 "5 additional steps (target 25)"；最终 25 步
实际关键结果：全部满足（真实子进程复跑计划 §8 流程同样通过：run 20 步 16 次实际参考执行，resume 至 40 步，导出 41 帧）
状态：通过
```

```text
验收项：validate 及早报错（未知字段／负步长／零检查非法组合／重复 run_id／缺模型文件），含字段名与补救方式
命令：uv run pytest tests/unit/test_config.py tests/unit/test_cli.py -q -k "unknown or negative or zero or duplicate or missing"
测试：未知字段给近似名建议；timestep_fs=-0.5；probability=0 仍设 tilt/failure_probability；运行后重复 run.id（validate 与 run 两条路径）；mace model 指向不存在文件
通过标准：退出码 2；错误文本含点分字段名（dynamics.timestep_fs、verification.*、run.id、surrogate.model）与补救动作
实际关键结果：5 个场景全部满足；另覆盖未知段、未知 backend、keep_generations≠2、mode×section 冲突、缺赝势目录/文件
状态：通过
```

```text
验收项：export 三种 force-source 行为正确，缺标签用缺失标记
命令：uv run pytest tests/unit/test_cli.py::test_export_marks_missing_reference_labels tests/unit/test_workflows.py -q -k "export or surrogate_mode or reference_mode"
测试：adaptive 运行导出 driving/reference/base；reference 源下无标签帧（accepted 未检查）断言 get_forces 全 NaN、calc 无 energy、forces_available=F，有标签帧有限且带 reference_label_id；surrogate-only 运行 reference 源 6/6 帧全部缺失标记；覆盖保护（--force）
通过标准：无零补齐；缺失标记存在；三源帧数与已提交评估一致
实际关键结果：adaptive 运行 12 帧有标签（含 1 帧被检查 accepted）、9 帧缺失标记；surrogate-only 全部缺失标记；driving/base 全有限
状态：通过
```

```text
验收项：--help 离线可用；import pyraimd2 不加载 torch/pyscf
命令：uv run pytest tests/unit/test_cli.py -q -k "help or light or backends"
测试：主帮助与 7 个子命令帮助（SystemExit 0）；--help 后 sys.modules 无 torch/pyscf；import pyraimd2.cli/pyraimd2.workflows 同样不含；`pyramid backends` 列表不含 torch
通过标准：帮助文本含 usage；torch/pyscf 不出现在 sys.modules
实际关键结果：满足（本机无 torch/pyscf，WP05 的三级导入断言亦保持）
状态：通过
```

```text
验收项：resume --steps 语义（额外 N 步，打印当前/目标步）；Ctrl-C 停止后续算
命令：uv run pytest tests/unit/test_workflows.py -q -k "resume or sigint"
测试：8 步运行 + resume 4（打印 8→12）；连续 12 步 vs 8 步+新对象恢复 4 步逐行比对；真实 SIGINT（os.kill 自身，等 run_start 后 0.2 s 发出）在下一步边界停止并写 checkpoint，resume +3；plain 模式 resume 明确拒绝；崩溃锁 resume 拒绝、--force-unlock 放行
通过标准：当前/目标步打印正确；连续与分段轨迹位置/动量 atol=1e-12、route/checked/检查计数严格一致；SIGINT 后 stopped_early=True 且可续算
实际关键结果：全部满足（详见"恢复与随机状态"节）
状态：通过
```

```text
验收项：QE/MACE 后端配置经 create_backend 接通（参数校验，不跑真实后端）
命令：uv run pytest tests/unit/test_cli.py -q -k "qe_backend or mace or optional_dependency"
测试：qe 配置缺 pseudo_dir 报错含 reference.pseudo_dir；赝势文件就位后 validate OK（构造 QeEngine、能力与契约通过、本机无 pw.x 不执行）；mace model 为裸名（small）时 validate OK 且 torch 未入 sys.modules；model 为路径形态且不存在时报 surrogate.model 缺文件；pyscf --probe-backends 缺依赖时报 pip extra 提示
通过标准：全部在工厂/配置层拦截，无 SCF、无推理、无下载
实际关键结果：满足；examples/qe_mace_skeleton/run.toml 手工 validate OK（占位赝势，未提交）
状态：通过
```

```text
验收项：Python API 与 CLI 同一套配置与 workflow
命令：uv run pytest tests/unit/test_workflows.py -q
测试：全部工作流测试直接调用 load_config+run_workflow/resume_workflow（不经过 CLI）；CLI 测试经 cli.main 走同一函数
通过标准：库级结果（布局、检查计数、导出）与 CLI 级一致；plain 三模式（adaptive/reference/surrogate）同一代码编排
实际关键结果：满足
状态：通过
```

```text
验收项（计划列出但本轮未执行的项）
- singlepoint/relax/FixAtoms 联合验收：未执行——属 WP07；本 WP 在 schema 接受枚举值、validate/run 均以含 WP07 指引的错误拒绝，不静默降级。
- 真实 QE/MACE 后端运行：未执行——本机无 pw.x/torch；只到 create_backend 参数/路径/能力校验（计划本轮即如此限定）。
- wheel 安装后仓库外 CLI 测试：未执行——属 WP09；本轮以 uv run 在项目环境内完成真实子进程验收。
状态：未执行（原因如上）
```

## 恢复与随机状态（相关工作包必填）

- 连续轨迹与分段恢复轨迹的长度、完整步位置：12 步连续 vs 8 步＋恢复 4 步（tests/unit/test_workflows.py::test_continuous_vs_stop_resume_are_identical）；另 20＋恢复 5/20（CLI 流程与真实子进程）。
- 是否新进程恢复：恢复侧全部对象（engine/surrogate/Store/EventLog）由 resume_workflow 从 resolved_config.json 经 create_backend 重建，除文件系统无共享可变状态；真实子进程 resume 亦验证。
- 位置／动量差及单位：逐行 atol=1e-12（位置 Å，动量 ASE 单位）。
- route、evaluation、segment、model ID、检查抽样／计数是否一致：route 与 checked 逐行严格相等；inspect 的 checks.accepted_count 相等；model_id 链一致（无更新，g0）。
- 准入前缀和检查上界是否一致：检查流由 WP03 checkpoint RNG 状态+事件重放保证；本 WP 未新增破坏面（同上计数相等）。
- 故障注入位置与恢复结果：真实 SIGINT（运行中发送）→ 下一步边界 checkpoint＋run_end(stopped) → resume +3 到目标步；崩溃锁（手工放置 events.jsonl.lock）→ resume 退出码 2 报 active writer，--force-unlock 后续算成功；重复 run → 退出码 2 报 run.id。
- checkpoint 之后持久事件的重放范围、积分阶段及 RNG 后状态：沿用 WP03 机制（事件游标重放）；本 WP 不重实现，恢复正确性由上两条轨迹同一性证据覆盖。
- 检查／模型更新已提交而新 checkpoint 未写入时的恢复证据：本 WP 无模型更新（WP06）；检查窗口恢复由 WP03 测试覆盖，本 WP 的连续 vs 分段对照间接覆盖 interval=3 的窗口。
- 重复 label／probe／update 的处理：无更新器；label_id 由 runner 分配（持久），重放不重复（WP03 机制）。
- 模型身份不匹配、破损快照或缺工件的处理：backend 参数改动 → fingerprint 变化 → EnergeticRunner.resume 的 ResumeError 原样上抛给 CLI（退出码 2）；其余由 WP03 四道校验覆盖。
- 已完成但未应用的计算如何计费、保存和复用：账本只追加（WP02/WP03 机制），本 WP 不改变；plain 模式每次评估一条 task 事件（reference/inference, purpose=md）。
- 本 WP 新增可恢复面：resume 从 resolved_config.json 重建后端与策略（checkpoint interval 取自配置）；输出派生物（trajectory.extxyz/summary）在 resume 开始即全量重建，崩溃窗口造成的预览空洞自愈；事件日志/数据库真源不受影响。

## 算力与成本

- 新增实际参考执行总数：0（全部 builtin 解析谐振后端；QE/MACE/pyscf 未真实执行）。
- 预算使用与剩余额度：WP08 配额未动用。
- 端到端墙钟（演示量级，本机）：harmonic 模板 20 步 adaptive ≈0.1 s；21 CLI 测试 ≈0.9 s；全套 347 项 ≈12 s。

## 回归与交付

- 受影响测试通过情况：`uv run pytest tests/unit tests/test_smoke.py -q` 347 passed（基线 283 + 新增 64）；`uv run ruff check` 本 WP 全部新增/改动文件 0 错误（仓库存量 4 处既有问题未触碰）。
- 原有 energetic／legacy switching 示例：未改动；loop/runtime/store/engines/surrogate 零改动。
- wheel 安装及仓库外 CLI 测试：未执行（WP09）；项目环境内真实子进程全流程已执行（init→validate→run→inspect→resume→export）。
- 最小依赖是否仍不导入 Torch／PySCF：是（新增 sys.modules 断言覆盖 cli/workflows；config/cli/workflows 只依赖 NumPy+ASE+标准库；mace/pyscf 工厂保持惰性）。
- 用户可复制的完整运行与恢复命令：
  `pyramid init --template harmonic --output my_run` → `pyramid validate my_run/run.toml` → `pyramid run my_run/run.toml` → `pyramid inspect my_run/runs/harmonic-demo [--json]` → `pyramid resume my_run/runs/harmonic-demo --steps N` → `pyramid export my_run/runs/harmonic-demo --force-source driving|reference|base --output traj.extxyz`。Python：`load_config` + `run_workflow(config)` / `resume_workflow(run_dir, N)`。
- README／示例／支持矩阵更新位置：docs/configuration.md（用户文档）；examples/harmonic_adaptive/、examples/qe_mace_skeleton/；README 首页重组属 WP09。
- 已知故障、未执行验证与风险：
  - resume 路径 QE 引擎不接 event_log（density io 任务事件缺失，正确性不受影响）；需 loop/ 提供注入点，留待后续。
  - keep_generations 固定 2（runtime/ 不在本 WP 区域）；配置字段先到位，放开需一行 runtime 改动。
  - plain 模式不可 resume（WP07）；其 run 目录 export/inspect 可用。
  - validate 不检查"QE 需要非奇异晶胞"这类后端×结构物理相容（--probe-backends 才暴露）；文档已写明。
  - 进度/摘要观察器对派生物的写入全部容错（警告不中断运行）；真源（db/事件）由 runner 自身保证。
- 需要负责人判断的具体决策及备选方案：
  - builtin 谐振命名（连字符 vs 下划线）：若更希望与计划 §8 字面一致，可让示例插件改名腾出下划线名（需动 WP05 插件与测试）；当前选择零冲突、零改动存量。
  - plain 模式事件/schema 复用程度：当前复用 store 行与主要事件类型使 inspect/export 一致，但未发 evaluation_proposed（无冻结提议概念）；若 WP07 需要 plain resume，可在 _PlainDriver 上加整步 checkpoint 而非新起 schema。
- 下一工作包及其入口：WP06（模型更新）——本 WP 未提供更新器配置（adaptive 运行均为冻结模型）；`EnergeticRunner.resume` 在 checkpoint 引用 updater 状态时会 ResumeError，WP06 需要在 config/workflow 接更新器配置与工件链（接口位：`_policy_kwargs` 与 `resume_workflow` 的 updater 参数位置已留空）。WP07——workflows/md.py 的 kind 分发点与 `_PlainDriver` 是 singlepoint/relax/FixAtoms 的接入位置；`load_structure` 的约束拒绝信息即 WP07 指引。WP08——examples/qe_mace_skeleton 可作配方起点（未验证）。配置/CLI/工作流 API 面：`pyraimd2.config`（load_config/load_resolved_config/PyramidConfig/ConfigError）、`pyraimd2.workflows`（run_workflow/resume_workflow/validate_setup/export_run/write_template/WorkflowError/WorkflowResult）、`pyraimd2.backends.backend_factory` 与工厂 run_root/event_log 注入约定。

提交：`6693dd9`（配置/CLI/工作流/后端/测试）、`64a2d99`（严格化+示例+文档）、`ad4c3a7`（MACE 名验证测试）、本报告提交为 `dev/0.4.0-wp04` 分支顶端。
