# WP05 回报 — 后端工厂、QE 与外部进程

## 本次范围

- 工作包／审阅节点：WP05（R2 的第一部分）
- 基线提交、分支、当前提交：基线 `6608806`（WP02 完成点）；开发分支 `dev/0.4.0-wp05`（worktree `pyraimd2-wp05`，与主检出的 WP03/WP04 并行工作隔离）；当前提交见文末 SHA。
- Python／ASE／NumPy 及有关后端版本：Python 3.12（uv 管理）；ASE 3.29.0（uv.lock 锁定，Espresso 接口签名以此为准）；NumPy ≥1.26；本机无 pw.x/torch/pyscf，全部验收为 hermetic（fake pw.x + fixture）。
- 本次解决的用户问题：QE 参考路径的 XC/dispersion 硬编码、赝势与能量口径身份缺失、SCF 成功只看 returncode、超时留下孤儿进程、重试不区分错误类型；唯一目录下 startpot 每次先故意失败一次（WP00 遗留）；没有后端工厂/能力注册机制；ASE 接入缺少 QE 示例与口径对照。
- 实际实现的功能：
  - QE 产品化：`QeConfig` 新增 `xc`（默认 pbe）、`dispersion`（默认 grimme-d3，可 None）、`density_source`、`max_retries`；输入显式写 `input_dft`/`vdw_corr`；引擎名按配方生成（默认仍为 `qe-pbe-d3`）。赝势文件名+内容 sha256 进 fingerprint（不可读时降级为 null 并记录）；能量口径按 smearing 如实声明（metallic→free_energy，两者均 force_consistent=True）。SCF 成功 = returncode 0 + `JOB DONE.` 标记 + 无 `convergence NOT achieved` + 解析完整（能量+完整力块+原子数+有限性）的组合判定。超时用 `start_new_session` + `killpg(SIGKILL)` 清掉整个进程组。重试有界分类：QE `Error in routine` 横幅与同输入未收敛不重试，崩溃/超时/截断输出可重试，密度启动失败总是给一次 atomic 重试（不同输入）。
  - 密度热启动：每个成功 attempt 写 `density_manifest.json`（reference fingerprint/nat/species/.save 相对路径/来源）；下次计算先验证来源已知（manifest）且参考设置相容（fingerprint/nat/species + .save 存在），相容则复制进本次可写目录并记录来源、字节数与耗时（`last_attempt_records` + 可选 EventLog 的 io 任务事件），不相容或缺失则在启动前直接写 atomic 输入——不再有故意失败的第一次。任何两次计算不共用可写 .save（源只读，逐次复制）。
  - 后端工厂：新包 `pyraimd2.backends`（`registry.py`）：builtin 名称（`qe`、`qe-ase`、`pyscf`）惰性加载；`pyraimd2.backends` entry-point 组发现插件，与 builtin 重名或插件间重名即 `BackendRegistryError`；`create_backend(..., kind=..., require=EngineCapabilities|SurrogateCapabilities)` 在工厂层验证协议与能力契约（unknown 能力不满足显式要求），不匹配抛 `CapabilityMismatchError`，全部在任何计算之前。
  - ASE 扩展入口：新 `engines/ase_qe.py`——`espresso_input_data`/`make_espresso_calculator`（与手写 writer 同一 QeConfig、同一物理映射）与 `AseQeEngine`（Engine 协议、唯一目录、能力/fingerprint 以 `ase-qe-` 前缀区分路径）。Espresso 接口签名按锁定的 ASE 3.29 实测（`EspressoProfile(command, pseudo_dir)`、`Espresso(profile=, directory=, input_data=, pseudopotentials=, kpts=)`），有签名锁定测试。
  - 示例插件：`examples/backends/pyraimd2_harmonic`（独立 pyproject，entry-point 组注册 `harmonic_reference`/`harmonic_surrogate`），不改核心即可作为 reference/surrogate 使用；hermetic 测试经真实 `EntryPoint.load()` 机制验证，真实 `uv pip install` 验证见验收证据。
- 与原计划的偏离及原因：
  - **参考 fingerprint 不再覆盖平台/执行字段**（`pw_cmd`/`timeout_s`/`max_retries`/`density_source`/`startpot_file` 被排除）：launcher 或重试策略变化不改变收敛 SCF 的物理标签身份，且密度 manifest 的相容性判定要求"同一参考设置"在换 launcher 后仍成立（科学参数与平台 profile 分离，§4.3/§5.1）。科学字段（xc/dispersion/截断/k 点/nbnd/smearing/收敛与混合参数/赝势哈希）全部保留。WP01 引入的 fingerprint 契约（属性形态、稳定性、随设置变化）不变。
  - **metallic 配置的 energy_kind 由 energy 改为 free_energy**：pw.x 加 smearing 时 `!` 行是变分自由能，力是其梯度；此前声明 energy 是不准确的口径，本 WP 如实修正（WP01 的 EnergyKind.FREE_ENERGY 正为此设）。
  - **ASE 与手写路径的能量差 ~1e-7 相对**：ASE espresso 读取器用 CODATA-2006 的 Ry（13.6056919328）而手写 parser 用现行 `ase.units`（13.6056930122）。口径（单位/符号/Voigt 顺序）一致，量级差是常数版本差，测试按 rel=1e-6 断言并在 test 文档中写明，不做静默吸收。
  - `test_startpot_fallback_writes_atomic_start_and_keeps_failed_attempt` 被改写为 `test_density_start_fallback_keeps_failed_attempt`：原测试编码的正是 WP00 遗留的"故意失败一次"行为，本 WP 按计划消除；回退路径（密度启动失败→atomic 重试+保留诊断）用真实密度源重新覆盖。

## 改动清单

- `src/pyraimd2/engines/qe_engine.py`（重写，parser 逻辑与签名不变）：
  - `QeConfig`：新增 `xc="pbe"`、`dispersion="grimme-d3"`（None=显式无色散）、`density_source=None`、`max_retries=1`；其余字段、默认值与类属性访问（`QeConfig.conv_thr` 等，experiments/hpc 脚本在用）不变。
  - `QeEngineError(EngineError)`：带 `retryable` 分类。
  - `write_qe_input`：写 `input_dft`，`vdw_corr` 按配置条件写入；`prefix` 提为模块常量 `QE_PREFIX`。
  - `parse_qe_output`：未改（保留 WP00 成组解析与 D 指数处理）。
  - `pseudo_identities`/`_pseudo_sha256`（按 path+mtime+size 缓存，fingerprint 每次评估被读，不能重复哈希 MB 级 UPF）；`settings_payload`（科学字段+赝势身份，排除平台字段）。
  - `QeEngine`：`name` 改为按配方派生的属性；`capabilities` 按 metallic 声明 energy/free_energy；`fingerprint` = `<recipe>:<digest16>`；`compute` 密度解析+有界重试循环；`_attempt` 进程组运行、组合成功判定、成功写密度 manifest；`_kill_process_group`；`_stage_density`（复制+计时+字节+可选 io 事件）；`_resolve_density`/`load_density_source`/`write_density_manifest`；`last_attempt_records`/`last_density_decision` 诊断属性。
  - `create_qe_engine` 工厂函数（registry 用）。
- `src/pyraimd2/engines/ase_qe.py`（新）：`espresso_input_data`、`make_espresso_calculator`、`AseQeEngine`（继承 AseEngine，唯一目录轮转，能力/fingerprint 镜像 QE 配方）、`create_ase_qe_engine`。
- `src/pyraimd2/backends/__init__.py`、`registry.py`（新包）：`BackendRegistration`、`BackendRegistryError`、`available_backends`、`create_backend`、`backend_capabilities`、`assert_capabilities_satisfy`、`ENTRY_POINT_GROUP="pyraimd2.backends"`；builtin 表 `qe`/`qe-ase`/`pyscf`（惰性 `_lazy` 导入）。
- `src/pyraimd2/engines/__init__.py`：导出 `AseQeEngine`、`QeEngineError`。
- `examples/backends/pyraimd2_harmonic/`（新）：独立可安装插件包（`pyproject.toml` 声明 entry-point 组、`src/pyraimd2_harmonic/__init__.py` 两个解析后端+带 `backend_kind` 的工厂、README）。
- 测试：
  - `tests/unit/test_qe_engine.py`：改写 startpot 回退测试（真实密度源）；新增 `test_parse_missing_force_block_raises`、`test_parse_empty_force_block_raises`、`test_compute_engine_error_on_atom_count_mismatch`；`_fake_pwx` 支持子目录、新增 `_MAKE_SAVE` 辅助；更新一个过时 docstring。
  - `tests/unit/test_qe_reliability.py`（新，19 项）：配方显式性/命名；赝势内容哈希与不可读降级；平台字段不进 fingerprint；metallic/绝缘口径；无 `JOB DONE.` 拒绝；QE 错误横幅与同输入未收敛不重试；瞬态崩溃在预算内重试且有界；超时进程组清理（真实孙进程 pid 验证）；密度 manifest、复制与 provenance、无密度直接 atomic、不相容拒绝并记原因、密度 I/O 进 EventLog 账本。
  - `tests/unit/test_ase_qe.py`（新，7 项）：ASE 签名锁定；input_data 与手写 writer 物理映射一致；PBE-only 路径；profile command/pseudo_dir；fixture 口径一致（能量/力/stress 符号与 Voigt 顺序）；非零力口径；AseQeEngine 身份与唯一目录。
  - `tests/unit/test_backends.py`（新，11 项）：builtin 注册与惰性（无 torch/pyscf 入 sys.modules）；选择时加载；未知名报错含可用列表；kind 不匹配；能力要求在工厂层拒绝（含 unknown 不满足、surrogate uncertainty）；entry-point 发现/创建/能力拒绝；与 builtin 冲突、插件间冲突均报错；工厂返回错误协议被拒。
  - `tests/unit/test_backend_plugin_example.py`（新，3 项）：示例插件经真实 `EntryPoint.load()` 被发现；作为 reference+surrogate 跑通真实 `EnergeticRunner` 3 步；能力/类型不匹配被拒。
- 新接口、配置字段和默认值：如上（全部为带默认值的新增字段/可选关键字）；无 TOML 配置字段（WP04 之后接线）。
- 旧接口／已有数据的兼容方式：`QeConfig`/`QeEngine`/`write_qe_input`/`parse_qe_output` 公共签名不变；experiments/hpc 脚本所用 kwargs 全部保留；`Engine` 协议、`fingerprint` 属性形态、WP01/WP02 契约不变（全量回归覆盖）。
- 是否改变单位、力预算、时间、约束、随机检查或模型切换语义：否。单位/符号契约不变；metallic 能量口径是如实修正声明（此前同一物理量被错标为 energy）。

## 验收证据

```text
验收项：QE parser 覆盖（成功/未收敛/多块/D 指数/缺力/原子数错误）
命令：uv run pytest tests/unit/test_qe_engine.py -q
测试：fixture 成功、convergence NOT achieved、双块成组、D 指数、新增缺力（无块/空块）、引擎层原子数不符
预先确定的通过标准：16 项全过；原子数不符报 "force shape ... != (2, 3)"；缺力报 EngineError
实际关键结果：16 passed
状态：通过
```

```text
验收项：两次参考计算不共用可写目录；startpot 不再先故意失败
命令：uv run pytest tests/unit/test_qe_engine.py::test_compute_unique_directories_and_absolute_input tests/unit/test_qe_reliability.py -q -k "unique or warm_start or incompatible or density_copy or manifest"
测试：相对 run_root 下连续两次 compute 两个目录；有相容密度时复制并记录来源（源树不被改写，复制树独立可写）；无密度时第一次即 atomic（fake 仅被调 1 次，无 attempt-2）；fingerprint/nat 不相容直接 atomic 并记原因
预先确定的通过标准：如上断言 + manifest 字段齐全（fingerprint/nat/species/来源）
实际关键结果：全部通过（19 项 reliability 中含）
状态：通过
```

```text
验收项：密度搬运 I/O 计入成本（WP02 接口只调用）
命令：uv run pytest tests/unit/test_qe_reliability.py::test_density_copy_is_charged_to_the_cost_ledger -q
测试：QeEngine(event_log=EventLog) 两次 compute（第二次热启动复制）
通过标准：恰好 1 条 io/density_copy 任务事件（success、elapsed≥0、provenance 含来源/字节）；summarize_tasks counts.io == 1
实际关键结果：满足
状态：通过
```

```text
验收项：SCF 组合成功判定与有界分类重试；超时进程组清理
命令：uv run pytest tests/unit/test_qe_reliability.py -q -k "job_done or banner or nonconvergence or transient or budget or process_group"
通过标准：rc=0 无 JOB DONE 拒绝；QE 错误横幅/同输入未收敛只调 1 次（max_retries=3 下）；瞬态崩溃重试成功且计数为 2；超预算后抛错；超时后 fake 孙进程（写 pid 文件）在 5s 内消失
实际关键结果：6 项全过（进程组清理实测孙进程被 SIGKILL）
状态：通过
```

```text
验收项：同一状态经手写 QE 与 ASE QE 路径口径一致（parser/单位层）
命令：uv run pytest tests/unit/test_ase_qe.py -q
测试：fixture 与非零力改造块同时过 parse_qe_output 与 ase.io.espresso.read_espresso_out
预先确定的通过标准：能量/力 rel≤1e-6（CODATA-2006 vs 2014 常数差 ~8e-8，已在测试文档注明）；stress 6 分量符号与 Voigt 顺序一致（等静压均为负，剪切为零一致）；输入层面两路径写同一物理设置
实际关键结果：7 passed
状态：通过
```

```text
验收项：后端工厂（builtin 惰性、能力拒绝、entry-point 冲突）
命令：uv run pytest tests/unit/test_backends.py -q
通过标准：available_backends 含 qe/qe-ase/pyscf 且 torch/pyscf 不在 sys.modules；require 不匹配（含 unknown 不满足显式要求）抛 CapabilityMismatchError；插件与 builtin 重名、两插件重名均 BackendRegistryError；未知名报错列可用后端
实际关键结果：11 passed
状态：通过
```

```text
验收项：第三方示例工厂不改核心即可作为 reference/surrogate 使用
命令 A（hermetic）：uv run pytest tests/unit/test_backend_plugin_example.py -q
结果：3 passed——经真实 EntryPoint.load() 发现；harmonic_reference+harmonic_surrogate 跑通 EnergeticRunner 3 步；require stress 被拒、kind 不匹配被拒
命令 B（真实安装）：uv pip install ./examples/backends/pyraimd2_harmonic && uv run python -c "available_backends/create_backend/能力拒绝" && uv pip uninstall pyraimd2-harmonic
实际关键输出：harmonic_reference: {'kind': None, 'origin': 'entry-point:pyraimd2-harmonic'}；compute 返回 (2,3) 力；CapabilityMismatchError 触发；torch 不在 sys.modules；卸载后 available_backends 回到 3 个 builtin，套件 264 全绿
状态：通过
```

```text
验收项：import pyraimd2 不变重
命令：uv run python -c "import pyraimd2 / pyraimd2.backends / pyraimd2.engines; 检查 sys.modules"
通过标准：torch、pyscf 均不出现（注册表发现与 builtin 创建同样不引入）
实际关键结果：三级导入后均无 torch/pyscf
状态：通过
```

```text
验收项：外部命令规范
实现：QeEngine 以 argv 数组 Popen（无 shell）、cwd=attempt_dir（不依赖全局 cwd）、输入路径 resolve 为绝对路径；科学参数（QeConfig 科学字段）与平台 profile（pw_cmd/timeout/retries）在 fingerprint 中分离；无账号/机器路径/凭据模板
测试：test_compute_unique_directories_and_absolute_input（相对 run_root + fake 真实读 -in 文件）
状态：通过
```

```text
验收项：受影响套件整体回归
命令：uv run pytest tests/unit tests/test_smoke.py -q
通过标准：全部通过
实际关键结果：264 passed（基线 221 + 新增 43）；uv run ruff check <改动文件> 0 错误（修复 6 个新增小问题后）
状态：通过
```

```text
验收项（计划列出但本轮无法执行的项）
- 真实 pw.x 单点对照（手写 vs ASE Espresso 实跑）：未执行——本机无 QE；计划验收原文即限定 ASE 侧"只到构造/写输入层面"。
- TOML→工厂接线与 pyramid validate 的能力拦截：未执行——属 WP04 之后（任务书明确本 WP 只到工厂 API+测试）。
- WP07 联合验收（singlepoint/relax 模式经工厂创建后端）：未执行——WP07 未完成。
- NPT 前的 stress 符号远程集成测试（QE 引擎 docstring 既有前置条件）：未执行——无 QE 环境，维持 NVT-only 声明。
状态：未执行（原因如上）
```

## 恢复与随机状态（相关工作包必填）

- 本 WP 不实现 checkpoint/resume（WP03）。与之相关的两点：失败 attempt 目录与 pw.out 全部保留供恢复诊断（不删除）；事件日志接口只追加 io 任务事件，不触碰事件协议；`max_retries` 重试在同一评估身份内由引擎完成，对 loop 表现为一次 compute 调用（WP02 账本语义不变）。
- 已完成但未应用的计算如何计费、保存和复用：密度工件（.save + manifest）保留在各自 attempt 目录，作为后续热启动来源；复制 I/O 记 io 任务；失败 attempt 的目录/输出保留、其成本由 loop 的 failed 任务事件记录（本 WP 不改 loop）。

## 材料与后端（相关工作包必填）

- 参考程序版本、XC／dispersion、赝势身份、收敛参数和能量口径：fixture 为既有 QE 7.5 Si 体相输出（tests/data/qe_si_scf.out）；配方默认 PBE+grimme-d3（显式写出），可配置；赝势身份=文件名+内容 sha256（不可读降级 null）；能量口径按 smearing 声明 energy/free_energy 且 force_consistent=True。
- 哪些能力只是接口支持，哪些完成了真实后端验证：QE 两条路径（手写/ASE）完成输入层与 parser 层验证；真实 pw.x 执行未验证（本机无 QE）；pyscf/MACE 工厂只到注册层（不触发后端导入）。

## 算力与成本

- 新增实际参考执行总数：0（全部 fake pw.x + 既有 fixture；插件为解析谐振势）。
- 预算使用与剩余额度：WP08 配额（两案例各 ≤100 次参考执行）未动用。

## 回归与交付

- 受影响测试通过情况：tests/unit + test_smoke 全绿（264）；ruff 改动文件 0 错误。
- 原有 energetic／legacy switching 示例：未改动；loop/store/runtime/surrogate 零改动（任务边界）。
- wheel 安装及仓库外 CLI 测试：未执行（WP09）。
- 最小依赖是否仍不导入 Torch／PySCF：是（三级导入断言 + 注册表测试）；`pyraimd2.engines` 现导入 `ase.calculators.espresso`（ASE 为核心依赖，符合最小核心定义）。
- 用户可复制的完整运行与恢复命令：QE 引擎用法不变（`QeEngine(QeConfig(...), run_root)`，新增可选 `event_log=`）；工厂用法见 `examples/backends/pyraimd2_harmonic/README.md` 与 test_backends.py。
- README／示例／支持矩阵更新位置：未更新（WP09 统一处理）；接口文档为各模块 docstring + 本报告。
- 已知故障、未执行验证与风险：
  - `max_retries=0` 会同时关闭密度启动失败后的 atomic 回退（用户显式选择不重试）；默认值 1 保持 WP00 行为。
  - 超时清理覆盖同一进程组；mpirun 若把 rank 放进别的 session（如 orted 守护模式）则超出 killpg 范围——本地启动器能做到的边界，docstring 已注明。
  - ASE Espresso 实际执行依赖 ASE 的 FileIO 机制（字符串命令、shell），与我们的 argv 规范不同；AseQeEngine 执行语义（唯一目录已轮转，重试/进程组不在 ASE 侧）如实只在构造/输入层验证，真实执行对照留待有 QE 的环境。
  - `available_backends()` 对插件显示 kind=None（kind 声明在工厂的 backend_kind 上，发现期不加载工厂以保持惰性；创建时强制校验）。
- 需要负责人判断的具体决策及备选方案：
  - fingerprint 排除平台字段（含 `startpot_file`）：若负责人希望热启动开关本身参与标签身份（更保守），可一行加回；当前选择保证换 launcher/重试策略不使已收标签与密度工件失效。
  - metallic→free_energy 的口径修正会改变 metallic 配置报告的 energy_kind（此前错标 energy）；若已有按旧口径记录的运行数据，应在 fork/新运行中使用新口径，不回填旧数据。
- 下一工作包及其入口：WP06——`loop/online.py`、`surrogate/committee.py` 的更新器适配（与本 WP 无文件重叠）；本 WP 为其就位：后端可由工厂创建、能力在创建期拦截、参考身份含赝势内容哈希、密度工件带 provenance。
