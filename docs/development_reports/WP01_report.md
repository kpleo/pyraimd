# WP01 回报 — 稳定契约与显式时钟

## 本次范围

- 工作包／审阅节点：WP01（R1 的第二部分）
- 基线提交、分支、当前提交：基线 `0a4544b`（WP00 完成点）；分支 `dev/0.4.0`；当前提交见文末 commit SHA。
- Python／ASE／NumPy 及有关后端版本：Python 3.12（uv 管理）；ASE ≥3.29、NumPy ≥1.26（uv.lock 锁定）；本轮不运行 MACE/PySCF/QE 真实后端（其 capabilities/fingerprint 为纯元数据属性，测试在导入后端前断言）。
- 本次解决的用户问题：评估身份与物理时间混用（评估索引当时钟）、结果缺少能量口径与能力元数据、模型变更后同几何缓存/旧校准可被误用、单位/能量口径不匹配在昂贵计算后才暴露。
- 实际实现的功能：
  - 新包 `pyraimd2.runtime`：`EvaluationContext`（run_id/step_id/evaluation_id/phase/physical_time_fs/model_id，含校验与 `for_probe()`/`as_dict()`）与 `fingerprint_of`/`model_id_for` 身份助手。
  - `EngineResult`/`SurrogatePrediction` 增加带默认值的可选字段 `energy_kind`（energy/free_energy/unknown）与 `force_consistent`（True/False/None=未知），位置参数构造完全兼容。
  - `EngineCapabilities`/`SurrogateCapabilities`（后者继承前者并加 `uncertainty_available`）＋ `engine_capabilities()`/`surrogate_capabilities()` 回退助手（未声明 = 全 unknown，绝不当作已支持）。
  - `CapabilityMismatchError(ValueError)` 与 `assert_compatible_energy_contract()`：已声明的口径冲突（energy_kind 不同、force_conservative=False、force_consistent=False）在任何 SCF/推理之前失败。
  - `EnergeticCalculator`/`EnergeticRunner` 接入显式上下文：积分器（`_EnergeticVerlet`）按步数×步长提供显式物理时间；探针共享父评估身份不推进时钟；每个已提交评估的 context 写入 store metadata；模型代次（generation）键控决策缓存（pending 提议、anchor、ASE 结果缓存）；参考设置 fingerprint 中途变化在昂贵调用前拒绝。
- 与原计划的偏离及原因：
  - 未新增"单位"字段：包对外单位由 §5.1 契约唯一固定（Å/eV/eV·Å⁻¹/fs/ASE Voigt stress），只有一种受支持单位制，不存在可声明的第二单位制；"不匹配在昂贵计算前失败"由能量口径（energy_kind/force_consistent/forces_conservative）检查覆盖。若未来引入第二单位制，需在 capabilities 中加 units 字段。
  - 能力元数据放在 `engines/base.py`/`surrogate/base.py` 而非 `runtime/`：它们是结果类型的契约元数据，与结果类型同层可避免 layering 倒置（backend 层不依赖运行服务层）；`runtime/` 按 §4.2 只放评估上下文与模型身份。
  - 参考设置身份的中途守卫（fingerprint 变化即拒绝）是任务书"参考设置身份"的最小可执行实现；旧 anchor 与新参考混用在语义上不允许，拒绝（而非静默重锚定）与 §5.3 顺序一致。

## 改动清单

- `src/pyraimd2/runtime/__init__.py`、`context.py`、`identity.py`（新包）：
  - `EvaluationPhase`（StrEnum：initial/md_step/probe/single_point/optimization_trial；后两个供后续 workflow 使用）。
  - `EvaluationContext`：frozen dataclass，`__post_init__` 校验（step_id ≥ -1，evaluation_id ≥ 0，physical_time_fs 有限且 ≥ 0，phase 合法，run_id/model_id 非空；类型错误 TypeError、取值错误 ValueError）；`for_probe()` 只改 phase、保留同一身份与物理时间；`as_dict()` 产出 JSON 安全记录。
  - `fingerprint_of(obj)`：读后端声明的 `fingerprint` 属性或零参方法，未声明返回 None（绝不猜测）；`model_id_for(model, generation)` 组成 `<fingerprint-or-class>#g<N>`。
- `src/pyraimd2/engines/base.py`：
  - `EnergyKind`（StrEnum：energy/free_energy/unknown）；`CapabilityMismatchError(ValueError)`；`EngineCapabilities`（energy_kind、force_consistent、forces_conservative、stress_available，`__post_init__` 归一化校验）。
  - `EngineResult` 追加 `energy_kind="unknown"`、`force_consistent=None`（默认值=不声明口径，位置构造兼容）；`engine_capabilities()` 回退全 unknown；`Engine` Protocol 文档化 `capabilities` 约定（运行时可选，必须经回退助手读取）。
- `src/pyraimd2/surrogate/base.py`：
  - `SurrogatePrediction` 同样追加两个字段；`SurrogateCapabilities(EngineCapabilities)` 加 `uncertainty_available`；`surrogate_capabilities()`；`assert_compatible_energy_contract(engine_caps, surrogate_caps)`（双方均已声明且 energy_kind 不同 → 拒绝；任一方显式 force_consistent=False 或 forces_conservative=False → 拒绝；unknown 不构成冲突、放行并保持记录为 unknown）。
- `src/pyraimd2/engines/ase_engine.py`（`AseEngine`）：`capabilities` 属性——`force_consistent=True` ↔ `energy_kind="free_energy"` + `force_consistent=True`；默认 ↔ `"energy"` + 一致性 None（取决于被包计算器，不虚构）；`stress_available=include_stress`；`forces_conservative=None`。`fingerprint` 属性含计算器名与两个标志。`compute` 把这两个元数据写入返回值。
- `src/pyraimd2/engines/pyscf_engine.py`：capabilities（energy/一致/保守/无 stress）；fingerprint（functional/basis/conv_tol）；compute 写入元数据。
- `src/pyraimd2/engines/qe_engine.py`：capabilities（energy/一致/保守/stress 可用）；fingerprint = `qe-pbe-d3:<sha256(json(QeConfig))[:16]>`；`_attempt` 最终结果写入元数据（`parse_qe_output` 是原始解析助手，保持 unknown 默认）。
- `src/pyraimd2/surrogate/ase_surrogate.py`：capabilities 透传引擎四字段 + `uncertainty_available=False`；fingerprint 前缀 `ase-surrogate:`；predict 透传结果元数据。
- `src/pyraimd2/surrogate/mace_surrogate.py`：capabilities（energy/一致/保守/stress 可用/uncertainty 不可用——单一冻结模型诚实标 NaN）；fingerprint（model/device/dtype）；predict 写入元数据。
- `src/pyraimd2/surrogate/committee.py`：capabilities（energy/一致/保守/**stress 不可用**/uncertainty 可用）；fingerprint（成员数×backbone 集合/seed/扰动/filters）；predict 写入元数据。
- `src/pyraimd2/engines/__init__.py`、`src/pyraimd2/surrogate/__init__.py`：导出新符号（顺带被 ruff --fix 修正了既有的 I001 导入排序）。
- `src/pyraimd2/loop/energetic.py`：
  - `_Anchor`/`_Pending` 追加 `model_generation`；`_Pending` 追加 `context`（默认 None，不影响既有位置构造）。
  - `__init__`：在首次昂贵计算前执行 `assert_compatible_energy_contract`；快照 `_engine_fingerprint`；新增 `_model_generation`、`_scheduled_time_fs`、`_results_model_generation` 状态。
  - 新属性 `model_generation`/`model_id`；新方法 `_context_for(index)` 构造上下文（有调度时间用调度时间，否则按兼容层固定步长假设 index×timestep_fs）。
  - `get_property`：已提交结果仅在产生它的模型代次内有效——代次变化后同几何请求清空缓存结果，强制成为一次新的逻辑评估（新决定），而不是旧决定的重放。
  - `_predict`/`_reference`：保留预测/标签的 energy_kind 与 force_consistent；`_reference` 在 `engine.compute` **之前**检查参考 fingerprint，中途变化即 ValueError。
  - `_calibrate`：每条探针记录带 `phase="probe"` 与父评估 `evaluation_id`；calibration 记录带 `model_id`；新 anchor 记当前代次。
  - `_prepare_updated_model`：重锚定探针用原点评估的身份与物理时间、新模型代次构造 PROBE 上下文。
  - `_freeze`：代次不符的 anchor 防御性置 None（旧校准不得用于新评估）；冻结的 pending 携带上下文与代次。
  - `_finish`：metadata 增加 `context`（as_dict）与 `reference_id`；driving 预测的口径元数据取自实际能量来源（accepted→prediction，否则→label）；提交时记录 `_results_model_generation`；回调返回非 exactly-False 时 `_model_generation += 1`。
  - `calculate`：重试校验增加代次一致要求（失败重试用同一评估身份，不重新抽签的语义不变）。
  - `_schedule(index, positions, *, physical_time_fs=None)`：接受显式物理时间（有限且 ≥0）；`_EnergeticVerlet.step` 按 `(nsteps+1)×timestep_fs` 显式传时——物理时间来自积分器，评估计数来自计算器，两者分离。
- 新接口、配置字段和默认值：均为带默认值的可选字段/属性；无新配置字段；无必需参数变化。
- 旧接口／已有数据的兼容方式：`EngineResult(energy, forces, stress, wall_time_s)` 与 `SurrogatePrediction(energy, forces, stress, uncertainty)` 位置构造不变（新字段默认 unknown/None）；store metadata 旧键（`evaluation_index`、`time_fs`、`timestep_fs` 等）逐字保留；旧 fake（无 capabilities/fingerprint）行为完全不变（conftest 未动，全部旧测试原样通过）。
- 是否改变单位、力预算、时间、约束、随机检查或模型切换语义：否。固定步长下 `context.physical_time_fs` 数值等于旧 `time_fs`；检查抽签、计数、路由、力预算定义均未动；直接 Calculator 用法的时钟含义不变（docstring 明确其固定步长假设为兼容层）。

## 验收证据

```text
验收项：同一 evaluation 先取 energy 后取 forces 只产生一次决定
命令：uv run pytest tests/unit/test_runtime_contract.py::test_energy_then_forces_is_a_single_decision -q
测试：accepted+checked（p=1）评估先 get_potential_energy 再 get_forces
预先确定的通过标准：第二次读取后 n_evaluations、reference_calls、检查 accepted/detected 计数、engine/surrogate 调用数、store 行数全部不变；check_draw 存在且 verification.accepted_count == 1
实际关键结果：全部计数不变，1 次决定 1 次检查抽签
状态：通过
```

```text
验收项：不动坐标的新步仍推进物理时间
命令：uv run pytest tests/unit/test_runtime_contract.py::test_unchanged_coordinates_new_step_advances_physical_time -q
测试：零速度 EnergeticRunner 跑 3 步（坐标恒不变），逐行断言 context
通过标准：step_id == [-1,0,1,2]、evaluation_id == [0,1,2,3]、physical_time_fs == [0,0.1,0.2,0.3]、phase 初值 initial 其后 md_step；旧 time_fs 字段数值不变
实际关键结果：全部相等
状态：通过
```

```text
验收项：探针不推进物理时间
命令：uv run pytest tests/unit/test_runtime_contract.py::test_probes_do_not_advance_physical_time -q
测试：首次评估含 1 anchor + 4 probes（force_call_count=5），随后一步新评估
通过标准：4 条探针记录 phase="probe" 且共享父 evaluation_id=0；calibration 带 model_id；下一评估 physical_time_fs 恰好 +1 个 timestep（0.1 fs）而非随参考调用数增长
实际关键结果：0.0 → 0.1，n_evaluations 1 → 2
状态：通过
```

```text
验收项：变更模型使同几何的缓存/提议失效
命令：uv run pytest tests/unit/test_runtime_contract.py::test_model_change_invalidates_same_geometry_decision_cache -q
测试：on_label 改 k=0.8→0.9 后，在同几何上连续三次 get_forces
通过标准：第二次调用产生新评估（n_evaluations 1→2、新 store 行、context model_id 从 #g0 变 #g1、重锚定记录的 calibration model_id 为 #g1），且 surrogate 载荷来自新模型（-0.9×0.2）；第三次（模型未再变）合法重放，不计数
实际关键结果：全部满足；旧 anchor/pending/ASE 结果缓存均按代次失效
状态：通过
```

```text
验收项：单位或能量口径不匹配在昂贵计算前失败
命令：uv run pytest tests/unit/test_runtime_contract.py -q -k "mismatch or inconsistency or reference_identity"
测试：engine 声明 free_energy + surrogate 声明 energy（CapabilityMismatchError）；surrogate 声明 force_consistent=False；参考 fingerprint 中途变化
通过标准：构造期/参考调用前抛出，engine.attempts == 0 且 surrogate.calls == 0（或参考调用数不变）——没有任何 SCF/推理发生
实际关键结果：三个用例全部在昂贵计算前失败；双方 unknown 或口径一致时正常构造
状态：通过
```

```text
验收项：契约与后端声明（支撑性测试）
命令：uv run pytest tests/unit/test_runtime_contract.py -q -k "positional or capabilities or undeclared or context_validation or backend or pyscf_qe or mace_and_committee"
通过标准：位置构造兼容且默认 unknown；非法字段值报错；未声明后端读作全 unknown；AseEngine 的 force_consistent 标志与元数据一致（free_energy↔True，默认↔None）；MACE uncertainty 不可用、委员会 stress 不可用；PyscfEngine/QeEngine/MaceSurrogate/CommitteeSurrogate 的 fingerprint 稳定且随设置变化（均无需导入 torch/pyscf/pw.x）
实际关键结果：7 个用例通过
状态：通过
```

```text
验收项：受影响套件整体回归
命令：uv run pytest tests/unit tests/test_smoke.py -q
通过标准：全部通过
实际关键结果：202 passed（基线 187 + 新增 15）；ruff 全仓 13 个错误（基线 15；新增 0，我重写的两个 __init__.py 被 --fix 顺带修正了既有 I001）；改动文件中仅存 1 个基线既有 TRY004（energetic.py 的 check_seed 校验，非本次引入）
状态：通过
```

## 恢复与随机状态（相关工作包必填）

- 本 WP 不涉及 checkpoint/resume（WP03）。检查抽签流、失败重试不重抽、代次递增均不改变既有 RNG 语义；`test_failed_check_retries_same_decision_without_redrawing` 等旧测试原样通过。

## 算力与成本

- 新增实际参考执行总数：0（全部为解析假后端）。
- 预算使用与剩余额度：WP08 配额（两案例各 ≤100 次参考执行）未动用。

## 回归与交付

- 受影响测试通过情况：tests/unit + test_smoke 全绿（202）。
- 原有 energetic／legacy switching 示例：未改动；switching/Runner/online/store 文件零改动。
- wheel 安装及仓库外 CLI 测试：未执行（WP09）。
- 最小依赖是否仍不导入 Torch／PySCF：是；`runtime/` 只用标准库；新测试中对 MaceSurrogate/CommitteeSurrogate/PyscfEngine/QeEngine 的断言全部在构造/属性层，不触发后端导入。
- 用户可复制的完整运行与恢复命令：同 WP00（resume 仍未实现，WP03）。
- README／示例／支持矩阵更新位置：未更新（WP09 统一处理）；接口文档以本报告 + 各模块 docstring 形式交付（runtime/context.py、engines/base.py、surrogate/base.py、loop/energetic.py 的模块与类 docstring 已同步改写）。
- 已知故障、未执行验证与风险：
  - 模型代次递增以 `on_label` 返回值（非 exactly-False）为准——这是既有"只有回调可以改模型"规则的代次化；绕过回调直接改写模型对象仍无法被检测（与 WP00 前相同），文档已写明只允许回调更新。
  - `_Pending`/`_Anchor` 的代次守卫在正常控制流中不可达（代次只在提交后变化），属防御性检查；未为其单独构造白盒测试。
  - 决策缓存键尚不含"时间、动量、锚点、策略代次"的完整 §5.4 形式（当前为 evaluation_id + 代次 + 几何/动量同一性检查）；WP02 事件存储落地时一并规范化。
- 需要负责人判断的具体决策及备选方案：
  - 参考设置 fingerprint 中途变化选择"拒绝并要求新运行"（而非自动重锚定）：自动重锚定会静默混用两种参考口径的 anchor 历史；若负责人希望支持受控的参考切换，建议作为显式 segment 操作放入后续 WP。
- 下一工作包及其入口：WP02——`store/store.py`、`EnergeticRunSummary` 与新增 runtime 记录模块。本 WP 已为其就位：`EvaluationContext` 提供 run/step/evaluation/phase/model 身份，`reference_id`/`model_id` 已进入每行 metadata，`energy_kind`/`force_consistent` 在结果类型上可序列化；数值标签缓存（几何+设置键）与决策缓存（本 WP 的代次键）的边界已在 docstring 中划清。
