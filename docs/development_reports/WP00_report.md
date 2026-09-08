# WP00 回报 — 已定位的基础正确性修复

## 本次范围

- 工作包／审阅节点：WP00（R1 的第一部分）
- 基线提交、分支、当前提交：计划基线 `362696a`（0.3.0）；main 实际已前进到 `dd30ef4`，软件源代码与基线一致（diff 仅新增 4 个测试文件、README 删 1 行、QE fixture 改 2 行），未强制回退。开发分支 `dev/0.4.0`（基于 main `dd30ef4`）。
- Python／ASE／NumPy 及有关后端版本：Python 3.12（uv 管理）；ASE ≥3.29、NumPy ≥1.26（uv.lock 锁定）；本轮不触碰 MACE/PySCF/QE 真实后端。
- 本次解决的用户问题：计划 WP00 列出的 6 个已定位正确性问题。
- 实际实现的功能：6 个问题的定向回归测试（修复前均可暴露问题）＋局部修复；无仓库级重构。
- 与原计划的偏离及原因：
  - `Hookean` 约束参数名按本环境 ASE 版本实际签名（`a1/a2`）书写。
  - QE 唯一目录采用 `run_root/<label>-NNNNNN/attempt-N/` 两层结构（调用级目录 + 尝试级目录），因此更新了 `test_compute_with_fake_pwx` 的目录断言；这是 WP00"唯一目录"要求的直接结果。
  - `startpot_file=True` 的密度链语义：唯一目录下首次 attempt 不再有可复用的 `.save`，startpot 调用会先失败一次再回退 atomic start。WP05 的"密度工件复制/克隆 + 工件身份"将消除这次故意失败，本报告将其列为 WP05 入口依赖。

## 改动清单

- `src/pyraimd2/engines/ase_engine.py`（`AseEngine.compute`）：能量改用 `apply_constraint=False`，与已有的未投影力／stress 口径一致；docstring 声明"后端只返回原始物理量，约束由 workflow 施加且只施加一次"。
- `src/pyraimd2/surrogate/mace_surrogate.py`（`MaceSurrogate.predict`）：能量/力/stress 三个 getter 同步 `apply_constraint=False`（复制输入的既有行为不变）。
- `src/pyraimd2/engines/qe_engine.py`：
  - `QeEngine.compute(atoms, label=None)`：每次调用分配唯一目录 `<label|eval>-NNNNNN`；`_attempt` 接收独立 `attempt-N` 子目录与显式 `QeConfig`；输入路径一律 `resolve()` 为绝对路径（修复相对 run_root + 子进程 cwd 的组合错误）。
  - startpot 回退：首次失败后用 `dataclasses.replace(config, startpot_file=False)` 生成明确的 atomic-start 输入重试一次；保留失败 attempt 目录与 pw.out 诊断，不再 `rmtree(tmp)`。
  - `parse_qe_output`：能量取最后一个 `! total energy`，力取其后第一个连续原子力块，stress 取该力块之后的 stress 块——同一完整 SCF 块成组解析；新增 `_to_float` 处理 Fortran D/d 指数；能量/力/stress 非有限时抛 `EngineError`。
- `src/pyraimd2/surrogate/committee.py`：
  - `__init__`：`device != "cpu"` 明确抛 `ValueError`（0.4 只支持 CPU 委员会数据路径；单 MACE GPU 路径属 MaceSurrogate）。
  - `load_state_dict`：加载前完整校验 model_specs、n_members、训练 recipe（seed/perturbation/epochs/lr/force_weight/trainable_filters）、member_state_dicts 数量、energy_shifts 长度；任一不符抛 `ValueError`，且不做任何部分修改。
- `src/pyraimd2/loop/energetic.py`（`EnergeticCalculator`）：`_validate_atoms` 拆出 `_check_identity`（不含计划位置检查）；新增 `get_property` 覆写——ASE 缓存命中跳过 `calculate` 时（质量/约束变化不在 ASE cache 失效集合内）仍先校验不可变物理状态；校验失败清空 `results` 再抛出（保持"被拒求值不留陈旧结果"的既有不变量）。
- 新接口、配置字段和默认值：`QeEngine.compute` 的 `label` 默认值由 `"step"` 改为 `None`（自动命名 `eval-NNNNNN`）；目录布局变为 `<label>-NNNNNN/attempt-N/{pw.in,pw.out,tmp/}`。无其他新配置。
- 旧接口／已有数据的兼容方式：Engine/Surrogate 协议签名不变；QE 目录布局变化只影响运行产物路径，不影响 API。
- 是否改变单位、力预算、时间、约束、随机检查或模型切换语义：否。只统一了"后端返回原始物理量"的口径（此前 ASE 适配器能量端可能混入约束能量项）。

## 验收证据

```text
验收项：WP00-1 带能量项约束的能量/力口径一致
命令：uv run pytest tests/unit/test_ase_adapters.py -q
测试：test_constraint_with_energy_term_returns_raw_energy_and_forces（Hookean k=5.0，含 adjust_potential_energy）
预先确定的通过标准：engine/surrogate 返回的能量等于 apply_constraint=False 的原始能量，且与调整能量不相等（否则测试无意义）；力为未投影力
实际关键结果：修复前能量端混入约束能量项（测试失败）；修复后 6/6 通过
状态：通过
```

```text
验收项：WP00-2 QE 唯一目录与绝对输入路径
命令：uv run pytest tests/unit/test_qe_engine.py::test_compute_unique_directories_and_absolute_input -q
测试：fake pw.x 真实校验 -in 文件存在；run_root 故意用相对路径
通过标准：连续两次 compute 不共用目录（2 个目录）；输入文件可从 run 目录内读取
实际：修复前输入路径随 cwd 失效（exit 7）且两次调用同落 "step/"；修复后通过
状态：通过
```

```text
验收项：WP00-3 startpot 回退生成 atomic-start 输入并保留诊断
命令：uv run pytest tests/unit/test_qe_engine.py::test_startpot_fallback_writes_atomic_start_and_keeps_failed_attempt -q
通过标准：attempt-2 的 pw.in 不含 startingpot；attempt-1 目录与 pw.out（含失败输出）保留
实际：修复前重试仍写 startingpot='file' 而密度已删，必然二次失败；修复后通过
状态：通过
```

```text
验收项：WP00-4 成组解析与 D 指数
命令：uv run pytest tests/unit/test_qe_engine.py -q -k "groups_last_scf_block or d_exponents or si_fixture"
通过标准：双块输出只取最后一块的能量/力/stress；D 指数正常解析；原 Si fixture 数值不变
实际：修复前力行为两块拼接（(4,3)）、D 指数 float() 崩溃；修复后通过，Si fixture 结果不变
状态：通过
```

```text
验收项：WP00-5 委员会恢复校验与 CPU-only 声明
命令：uv run pytest tests/unit/test_committee_state.py -q
通过标准：recipe/成员数/偏移长度不符在加载前抛 ValueError（无需 torch）；device='cuda' 明确拒绝
实际：修复前不匹配被忽略或 zip 静默截断；修复后 5/5 通过
状态：通过
```

```text
验收项：WP00-6 缓存前的物理状态检查
命令：uv run pytest tests/unit/test_energetic_loop.py -q
通过标准：仅改质量或新增约束（ASE cache 不失效）时取缓存力也抛 ValueError；被拒求值不留陈旧 results；既有 Verlet/调度行为不变
实际：修复前缓存力被静默返回；修复后 24/24 通过（含既有 test_invalid_atomic_changes_rejected 4 例）
状态：通过
```

```text
验收项：受影响套件整体回归
命令：uv run pytest tests/unit tests/test_smoke.py -q
通过标准：全部通过
实际：187 passed（基线 175 + 新增 12）；ruff 未引入新错误（存量 15 个为基线已有）
状态：通过
```

## 算力与成本

- 新增实际参考执行总数：0（全部为解析假后端与 fake pw.x）。
- 参考、推理、训练、I/O 墙钟及计时口径：无真实后端执行。
- 预算使用与剩余额度：WP08 配额（两案例各 ≤100 次参考执行）未动用。

## 回归与交付

- 受影响测试通过情况：tests/unit + test_smoke 全绿（187）。
- 原有 energetic／legacy switching 示例：未改动；unit 套件覆盖的既有行为全部保持。
- wheel 安装及仓库外 CLI 测试：未执行（WP09）。
- 最小依赖是否仍不导入 Torch／PySCF：是；新 committee 校验测试在加载模型前触发，无需 torch。
- 用户可复制的完整运行与恢复命令：同 WP04 前的既有用法；QE 目录布局变化见上。
- README／示例／支持矩阵更新位置：未更新（WP09 统一处理）。
- 已知故障、未执行验证与风险：
  - `startpot_file=True` 在唯一目录下首次 attempt 无密度可读，会先失败后回退 atomic start；WP05 用密度工件复制消除（本 WP 的计划内依赖）。
  - MaceSurrogate 的约束口径修复未跑真实 MACE 推理（torch 不在最小开发环境）；代码路径与 AseEngine 相同，slow 回归套件（test_mace_surrogate.py）留待有 MACE 环境时执行。
- 需要负责人判断的具体决策及备选方案：无（均为计划已列明的局部修复）。
- 下一工作包及其入口：WP01——`engines/base.py`、`surrogate/base.py`、`loop/energetic.py` 加入 EvaluationContext 与能力/元数据契约。
