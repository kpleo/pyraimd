# 能量契约细化、短 Al adaptive 与真实 MACE 持久化集成回报

- 日期：2026-09-09
- 审阅依据：`docs/development_reports/INDEPENDENT_REVIEW_20260909.md` §5（能量契约修正）、§6 第 5/6 条、§7 相关停止条件；`REVIEW_FIXES_R3_20260909.md` 的 tensor-artifact-v1 约定
- 基线 commit：`ee3a43f`（全部审阅修复合入，456 测试全绿）
- 执行环境：本地（契约与 hermetic 测试）＋ Neimeng A 集群（QE 7.5、MACE-MPA-0 medium float64 CPU、torch 2.6.0 CPU；与 WP08 同一环境）
- 证据包：`docs/development_reports/contract_evidence/`（配置、事件日志、attempt 汇总、inspect/验证 JSON、作业脚本与日志；张量权重只记内容摘要，未复制模型文件）
- 新增实际 SCF 数：**35**（预算 ≤30，超出 5 次——见"预算核算"逐条分解，不掩饰）

## 任务 A：能量契约细化（§5）——通过

### 契约推导对应的实现点

审阅给出的恒等式 d/dt[K+Φ_R] = Ṙ·(F_drive−F_R) 要求：参考标量势与参考力一致、替代模型与自身势能力一致、锚定差值明确所用参考泛函并保留身份；不要求两侧 `energy_kind` 字符串相等。实现（`surrogate/base.py: assert_compatible_energy_contract`，返回组合模式）：

- 任一侧声明 `force_consistent=False` 或 `forces_conservative=False`：仍拒绝（不变）。
- 两侧声明**不同** energy_kind（如 QE metallic `free_energy` × MACE `energy`）：仅当两侧都严格声明 `force_consistent=True` 且 `forces_conservative=True` 才允许——任一侧为 unknown（未声明或 None）即拒绝，unknown 永不当作已验证（审阅红线）。
- 其余 unknown 组合保持原有"通过但记录为 unknown"语义；validate 报告按 `same_kind / cross_kind / unknown` 如实写组合模式（`compatible` / `compatible_cross_kind` / `undeclared`）。
- 锚定差值所用参考泛函即运行固定的参考引擎：其 fingerprint（含 smearing 类型/宽度、XC、色散、ecut、k 点、nbnd、电子求解设置）绑定进 run_start、label 缓存键（另含初始电荷/磁矩）与 checkpoint 身份校验。核对确认 WP05 的 `settings_payload` 已覆盖上述字段＋赝势内容哈希（`engines/qe_engine.py:492`），无需改动。
- 未把 MACE `energy` 改名 `free_energy`，未把 unknown 当作已验证；未通过验证的组合仍拒绝（回归覆盖）。

### 差分验证（云端，Al(111) 17 原子 slab，配方原样设置：PBE+D3、50/400 Ry、4x4x1、MV 0.02 Ry、conv_thr 1e-8）

- 预设容差（先于执行写入 `contract_fd_al.py`，未后调）：每方向残差 ≤ 5e-4 eV/Å（conv_thr 能量噪声 ~1.4e-4 eV/Å @ h=1e-3 Å 的约 3.5 倍余量）。
- 结果（5 次 SCF：1 基态＋2 自由层方向 × ±h）：
  - atom 8, x：FD −0.0111566687 vs −F·u −0.0111519037，残差 **4.77e-6 eV/Å** ✅
  - atom 16, y：FD 0.0279596989 vs −F·u 0.0279954589，残差 **3.58e-5 eV/Å** ✅
  - 引擎 `energy_kind=free_energy`，fingerprint `qe-pbe-d3:9a4b6b847f82ff3b`，两个方向均通过：QE metallic 的变分自由能与其力一致。
- 证据：`contract_evidence/contract_fd_result.json`（含预算 n_scf=5 与容差）。

### hermetic 契约回归（本地解析后端）

`tests/unit/test_runtime_contract.py`：双侧验证的跨类组合允许（free_energy×energy）、单侧未验证的跨类组合拒绝、显式 `force_consistent=False` 拒绝、组合模式三态、旧 unknown 组合行为不变。旧"跨类即拒绝"测试按新契约改写（该测试本身是被审阅修正的旧计划产物）。

## 任务 B：短 Al adaptive 案例（§6-6）——通过

契约验证通过后执行（同一 slab、同一配方设置，`run-md-adaptive-c1.toml` = 配方 adaptive 配置改 run id 与 4 步）：

- validate 输出 `energy contract: compatible_cross_kind`（此前为契约拒绝；旧拒绝记录保留在 `wp08_evidence/al/contract-evidence.txt`）。
- 运行：4 完整步（评估 5、接受 3、违规 0；参考调用 13＝anchor 2＋probe 8＋check 3）；**新进程 resume +2**：至 6 完整步（评估 2、接受 1、违规 0；参考调用 6＝anchor 1＋probe 4＋check 1）。合计 6 完整步、7 评估、4 接受（57%）、0 违规，预算 0.15 eV/Å 未放松。
- 账本一致（R6 产线实证）：logical_requests=19 = actual_executions=19（physical attempts），failed=0，cache_hits=0。
- 有真实接受与独立检查（p=1.0），无需针对"无接受"诊断。
- 证据：`contract_evidence/{validate-c1.txt, inspect-c1.json, runs/al-adaptive-c1/{events.jsonl, summary.json, resolved_config.json, export-driving.extxyz}}`；Al README 与支持矩阵支持级别已按此结果更新。

## 任务 C：真实 MACE 持久化集成（§6-5）——通过

Al(111) slab 上的一次真实更新：`CommitteeSurrogate`（MACE-MPA-0 medium，2 成员，epochs=2，CPU）＋ `GuardedUpdater`（n_label=2）＋ QE metallic 参考，energetic runner 跑初始评估＋2 步（check p=1.0）：

- 一次**被接受的更新**（n_consumed=3、n_updates=1、n_rejected=0、代次 0→1；两成员损失 0.2507→0.2452、0.2354→0.2301）；tensor 工件按 tensor-artifact-v1 落盘（`state.json` 全占位＋`state_arrays.npz`）。
- **独立新进程恢复**：全新 CommitteeSurrogate/GuardedUpdater/QeEngine `EnergeticRunner.resume`——消费计数一致（3/1）、guard payload 带 cell 与 pbc=(T,T,F)、成员权重逐值相等、energy_shifts 逐值相等、同一边界预测能量一致（|ΔE| ≤ 1e-10 eV）且力最大差 1.4e-17 eV/Å。全部检查 `passed: true`。
- 委员会优化器状态（Adam 矩、训练 RNG）**不含**在持久化内（既定语义：fine-tune 续训重启优化器而非续接，`committee.py: state_dict` 文档已写明，本报告如实标注）。
- 证据：`contract_evidence/{committee_process1.json（权重仅记 sha256）, committee_process2.json, runs/al-committee/events.jsonl}`。

## 预算核算（如实）

- 事前估算：FD 5 ＋ adaptive ≈12 ＋ committee ≈5 ≈ 22。
- 实际：FD **5** ＋ Al adaptive **19**（13＋6）＋ committee **11** ＝ **35**，超出预算 5 次（17%）。原因：energetic 循环每次 anchor 附带 probe（本批共 anchor 3、probe 12），低估在 probe-per-anchor 开销与 p=1.0 检查密度（沿用配方 verification 设置而非削减）。所有执行均为验收行为本身，无填充性计算；为不再增加开销，不重跑精简版。后续同类运行的估算口径应按 anchor×(1+n_probes) 计入。
- 简历 resume 的参考调用均为完成验收（中断恢复）所需，未为省数而裁掉验收行为。

## 验证

- 本地 `uv run pytest tests/unit tests/test_smoke.py -q`：**458 passed, 1 deselected**（基线 456 → ＋2：契约模式与跨类拒绝各一）。
- 远端同文件契约回归（py312 环境）：17 passed。
- `uv run ruff check <改动文件>`：无新增错误。
- 最小核心仍只依赖 NumPy＋ASE＋标准库（张量序列化不导入 torch）。

## 支持级别更新

- README 支持矩阵：metallic（smeared）参考的 adaptive 由"0.4.0 不支持"改为"契约验证后支持"（双侧力—能量一致性验证＋完整参考身份），并写明验证数值；未验证组合仍拒绝、unknown 不算已验证。
- `examples/al_surface_qe_mace/README.md`：adaptive 小节按实测结果重写（命令、接受率、账本）。
- `CHANGELOG.md`：0.4.0rc1 变更说明更新（契约细化、tensor 持久化）。
- WP08 的 58 次参考执行记录校正已完成（`cf5cf80`，另一位工程师）；本批 35 次与之无关。

## 保留限制

- 差分验证覆盖 Al(111) 此一组 smearing 设置（MV 0.02 Ry）下的两条方向，是该组合的诊断门槛，不是对所有 QE smearing 口径的全局证明；其它 smearing 类型/宽度/XC 组合需各自验证（审阅同口径）。
- 冷展宽宽度不等于真实电子温度（审阅引 QE 文档），契约不对此做物理声明。
- Al adaptive 为 6 步机制验证（接受 4/7），不构成稳态接受率或加速结论；金属领域支持级别仅限"契约验证的组合"。
- 委员会优化器状态不持久（既定语义）；MACE 权重全文不进 git（证据只记摘要与工件路径）。
- 预算超支 5 次已如实列出，未以重跑掩盖。
