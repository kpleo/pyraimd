# WP07 回报 — 普通材料计算、固定原子与 NVE

## 本次范围

- 工作包／审阅节点：WP07（R2 的一部分）
- 基线提交、分支、当前提交：基线 `445aaf0`（WP04 合并点，含 WP00–WP06）；分支 `dev/0.4.0`；当前提交见文末 SHA。
- Python／ASE／NumPy 及有关后端版本：Python 3.12（uv）；ASE ≥3.29、NumPy ≥1.26；全部 builtin 解析后端，无外部程序。
- 本次解决的用户问题：只有 MD 一种任务可用；无固定原子工作流；plain 模式不可续算；周期坐标/约束语义未落实。
- 实际实现的功能：
  - 三个任务：`singlepoint`（reference 或 frozen surrogate 单次评估）、`relax`（ASE FIRE/BFGS 固定模型优化，用户指定 fmax）、`md`（既有三模式）。`adaptive` 仍只允许 `kind="md"`——energetic Calculator 不进优化器（无固定势能面），config 层既有拒绝保留。
  - FixAtoms 第一版：`loop/constraints.py` 的 `FixAtomsProjection`——固定自由度从探针方向与动力学误差范数中投影出去（active_dofs 模式的校准残差与观察误差都只计自由坐标），driving force 在两条力路径上统一投影（固定分量置零，参考与快速路径同一约束语义，约束在 workflow 层施加且只施加一次）；每个评估记录原始物理力、投影后驱动力与实际约束位移；原始全原子残差始终保留为诊断。`force_metric` 默认 `active_dofs_max_atom`（预算控制真正传播的自由坐标），`all_atoms_max_atom` 为显式备选，不静默改定义。多 FixAtoms 合并；RATTLE/含能/运动约束与变胞一律明确拒绝（NPT 经 ensemble choices 拒绝）。
  - 周期坐标：动力学内部保留连续展开坐标（ASE Verlet 不 wrap）；store 行保持 unwrapped；`frame_from_row/frames_from_store` 新增 `wrap` 输出选项（wrapped 帧带 coordinates 标记，store 不改写）。
  - plain 模式 resume：`_PlainDriver` 加整步 checkpoint（复用 WP03 `CheckpointManager`，同一目录布局与 schema：state.json/arrays.npz/manifest.json，每 interval 与停止请求时写入）；`resume_workflow` 对 plain 模式从最近有效 checkpoint 恢复、从最后一行已提交记录重建边界（plain 行记的是整步动量，无需补半踢）并续跑；`resume_workflow` 新增 `updater` 透传参数（接 WP04 留的接口位，传给 `EnergeticRunner.resume`）。
  - 配置新增：`[relax]`（optimizer/fmax_eV_A/steps）、`[constraints]`（fix_atoms_indices，与结构自带 FixAtoms 合并）、`policy.force_metric`；旧 resolved_config 无这些节时取默认（向后兼容）。
  - `docs/configuration.md` 同步（task/relax/constraints/policy.force_metric/resume 语义、当前限制）。
- 与原计划的偏离及原因：
  - plain 行的动量语义与 energetic 行不同：plain 驱动在步完成后记录（整步动量），energetic 在步中记录（半步动量）；plain resume 因此直接取整步动量，不做 energetic 的半踢重建。已在代码注释与测试容差（1e-12 逐行一致）中固定。
  - `resume --steps` 对 `singlepoint`/`relax` 拒绝（无动力学可续）；relax 未收敛如实报 `stopped_early=True`。
  - POSCAR selective dynamics 是结构自带 FixAtoms 的已验证入口；其它带约束格式未逐一验证（投影层对非 FixAtoms 一律拒绝，安全方向）。

## 改动清单

- 新增 `src/pyraimd2/loop/constraints.py`：`FixAtomsProjection`（from_atoms 合并 FixAtoms、力/位移/方向投影、按 metric 的误差范数、最大固定位移）、`ConstraintError`、`validate_constraints`；`FORCE_METRICS` 取自 config 的单一词表。
- `src/pyraimd2/loop/energetic.py`：`_check_identity` 允许 FixAtoms 并冻结集合（中途变更拒绝）；`_directions`/`_calibrate` 残差（active 模式）/driving 力投影；`_observed` 按 metric 计误差并记录 metric 名与原始残差；metadata 增 `constraint` 记录（原始力/投影定义/实际约束位移）；`_EnergeticVerlet` 调度期望位置按约束漂移；checkpoint/payload/policy 带 constraint 与 force_metric（resume/fork 复原 FixAtoms）；`EnergeticCalculator`/`EnergeticRunner` 增 `force_metric` 参数。
- `src/pyraimd2/config.py`：`RelaxConfig`/`ConstraintsConfig`/`policy.force_metric` 与解析、resolved_dict、`FORCE_METRICS`/`RELAX_OPTIMIZERS` 常量。
- `src/pyraimd2/workflows/setup.py`：`load_structure` 接受 FixAtoms（结构自带＋配置合并，其余类型明确拒绝）；`validate_setup` 放开 kind 校验。
- `src/pyraimd2/workflows/md.py`：`run_workflow` 按 kind 分发；新增 `_run_singlepoint`、`_run_relax`（ASE FIRE/BFGS、评估任务事件、收敛与最终 fmax 记录）；`_BackendCalculator` 增评估钩子；`_PlainDriver` 投影（driving 投影、constraint 记录）、整步 checkpoint、`start_step` 续跑；`_resume_plain`；`resume_workflow` 增 `updater` 参数与 plain 分支。
- `src/pyraimd2/workflows/export.py`：`frame_from_row/frames_from_store` 增 `wrap` 选项。
- `tests/unit/test_cli.py`、`tests/unit/test_workflows.py`：两个"plain 不可 resume"的旧断言更新为"plain resume 续跑"（WP07 实现使旧前提失效）。
- `docs/configuration.md`：同步新字段与语义。
- 新接口、配置字段和默认值：见上；全部带默认值，旧配置/旧 resolved_config 兼容。
- 旧接口／已有数据的兼容方式：无 constraint 时行为与 WP06 前完全一致（全部旧测试通过）；`force_metric` 默认即旧的"全原子"语义（无投影时 active==all）。
- 是否改变单位、力预算、时间、约束、随机检查或模型切换语义：无约束运行完全不变；FixAtoms 下力预算定义由 `force_metric` 显式给出（默认自由坐标），报告写明。

## 验收证据

```text
验收项：固定层逐步位置不动；自由原子的残差定义与报告一致；去掉约束恢复原行为
命令：uv run pytest tests/unit/test_fixatoms.py tests/unit/test_constraints.py -q
测试：4 原子团 [0,1] 固定跑 10 步；耦合参考（固定原子耦合两个自由原子）对比两种 metric；去约束对照
预先确定的通过标准：每行固定原子位置逐位不变；driving 在固定分量恒为零、原始力非零、max_fixed_displacement==0；active 模式接受（自由误差 0.015<0.02 无违规），all-atom 模式全 reference（固定残差 0.03 超预算）；observed 记录 metric 名与 4 原子原始残差；去约束后 driving 全分量恢复、constraint 记录为 None
实际关键结果：全部满足
状态：通过
```

```text
验收项：跨 PBC 的展开位移和功连续
命令：uv run pytest tests/unit/test_fixatoms.py::test_pbc_unwrapped_displacement_and_work_stay_continuous -q
测试：单原子以 v=1.0 穿过 [3,3,3] 晶胞边界 6 步
通过标准：store 位置单调越过 3.0（unwrapped，无回绕），步间位移连续（<0.2）；residual_work 跨界与同胞内位移一致；export wrap=True 帧回胞内、unwrapped 保持越界
实际关键结果：positions[-1] > 3.0 且连续；功一致；wrapped/unwrapped 输出标记正确
状态：通过
```

```text
验收项：固定模型优化的最终力满足用户指定收敛阈值
命令：uv run pytest tests/unit/test_relax_singlepoint.py -q -k "relax"
测试：harmonic surrogate（FIRE，fmax=0.05）与 reference（BFGS，fmax=0.02）各一例；fmax=1e-9 的超紧例
通过标准：最终行 max|F| 小于各自阈值；超紧例 stopped_early=True（如实报未收敛）
实际关键结果：满足
状态：通过
```

```text
验收项：一般约束和 NPT 被明确拒绝
命令：uv run pytest tests/unit/test_constraints.py::test_unsupported_constraints_rejected_explicitly tests/unit/test_relax_singlepoint.py -q -k "npt or adaptive_relax"
测试：Hookean/FixBondLength 投影层拒绝；ensemble="npt" config 拒绝；adaptive+relax config 拒绝
通过标准：各报含类型名/字段名的明确错误
实际关键结果：ConstraintError/ConfigError 均按预期抛出
状态：通过
```

```text
验收项：singlepoint/relax 三种模式经 CLI 全流程可跑（builtin 解析后端）
命令：uv run pytest tests/unit/test_relax_singlepoint.py -q -k "cli or singlepoint"
测试：cli_main(["run", cfg]) 跑 singlepoint(reference/surrogate) 与 relax(surrogate/reference)；singlepoint 结果与直接后端评估逐值一致
通过标准：退出码 0、run directory 与 trajectory.db 生成、energy/forces 与直接评估一致
实际关键结果：满足
状态：通过
```

```text
验收项：plain MD 的 checkpoint/resume 连续一致性
命令：uv run pytest tests/unit/test_plain_resume.py -q
测试：reference 连续 40 vs 30＋resume 10；surrogate 连续 24 vs 14＋resume 10；checkpoint 间隔 10 的窗口续跑；FixAtoms plain resume
通过标准：逐行 route/label_id/位置/动量/驱动力一致（位置动量 atol=1e-12）；窗口续跑步数正确；固定层在 resume 后仍不动
实际关键结果：满足
状态：通过
```

```text
验收项：受影响套件整体回归
命令：uv run pytest tests/unit tests/test_smoke.py -q
通过标准：全部通过
实际关键结果：383 passed, 1 deselected(slow)（基线 361 + 新增 22）；ruff 全仓 11 个错误（与 WP06 后基线相同，新增 0）
状态：通过
```

## 恢复与随机状态（相关工作包必填）

- plain resume：连续 40 vs 30＋恢复 10（reference）、连续 24 vs 14＋恢复 10（surrogate）；位置/动量 ≤1e-12（plain 行记整步动量，与 energetic 半步语义不同，已在代码注释固定）。
- checkpoint 游标：plain 复用 `CheckpointManager` 与 `last_event_seq`；窗口状态从最后一行已提交记录重建（无锚点/RNG/更新器状态需要重放，plain 无决策态）。
- 故障注入：plain 的截断回退与并发锁由共享的 CheckpointManager/EventLog 语义覆盖（WP03 已验收）；本 WP 验证 checkpoint 间隔>1 的窗口续跑。
- updater 透传：`resume_workflow(..., updater=...)` 接 WP04 留位传入 `EnergeticRunner.resume`。

## 算力与成本

- 新增实际参考执行总数：0（全部 builtin 解析后端）。
- 预算使用与剩余额度：WP08 配额未动用。

## 回归与交付

- 受影响测试通过情况：tests/unit + test_smoke 全绿（383）。
- 原有 energetic／legacy switching 示例：未改动；无约束路径行为与 WP06 完全一致。
- wheel 安装及仓库外 CLI 测试：未执行（WP09）；CLI 经进程内 `cli_main` 验收。
- 最小依赖是否仍不导入 Torch／PySCF：是。
- 用户可复制的完整运行与恢复命令：`pyramid run run.toml`（kind=singlepoint|relax|md），`pyramid resume <run_dir> --steps N`（adaptive 与 plain 均可），`pyramid export <run_dir>`。
- README／示例／支持矩阵更新位置：`docs/configuration.md` 已同步；README 与支持矩阵留 WP09。
- 已知故障、未执行验证与风险：
  - FixAtoms 与 adaptive＋updater 的组合未做恢复专项（投影在 checkpoint 中有记录，resume/fork 复原；WP03 恢复机制本身不变）。
  - relax 的评估级成本只按任务事件计（与 plain MD 同类）；优化器内部 ASE 行为（FIRE/BFGS）未改写。
  - 结构自带约束只验证了 POSCAR selective dynamics 入口。
- 需要负责人判断的具体决策及备选方案：plain resume 边界取"最后一行已提交记录"而非 checkpoint 数组（窗口情形等价且更简单）；若希望严格走 checkpoint＋事件重放，可在后续统一。
- 下一工作包及其入口：WP08——两条周期材料配方（体相 Si 用 plain/adaptive MD、固定层表面用 `[constraints]`+FixAtoms）；本 WP 的 FixAtoms 投影、plain resume、singlepoint/relax 与 CLI 即配方的运行底座。
