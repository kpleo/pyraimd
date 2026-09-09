# WP08 回报 — 两条周期材料配方与分级测试

## 本次范围

- 工作包／审阅节点：WP08（R2 的材料集成部分）
- 基线提交、分支、当前提交：基线 `3f2f5b8`（WP07）；分支 `dev/0.4.0`；当前提交见文末 SHA。
- 本地环境：Python 3.12（uv）、ASE ≥3.29、NumPy ≥1.26。
- 云端环境（Neimeng A，9242 分区，96 CPU 节点，单节点串行）：QE 7.5（conda envs/qe）、Python 3.12 环境（torch 2.6.0 CPU、mace 0.3.16、ase 3.29.0、numpy 2.5.1）、MACE-MPA-0 medium（本机缓存模型文件，float64）、pw_cmd 48 进程。
- 本次解决的用户问题：需要两条真实材料工作流（体相 Si、固定层金属表面）验证 QE+MACE 的完整闭环，外加一个无外部软件的周期 CI 示例。
- 实际实现的功能：
  - 配方 A（examples/si_bulk_qe_mace/）：8 原子 Si 金刚石常规胞，QE 参考＋MACE 替代。单点→固定模型 FIRE 优化（10 步收敛，fmax 0.037<0.05）→固定晶胞 NVE 6 步（reference-only 7 SCF/80.3 s；surrogate-only 0 SCF/1.7 s；adaptive 7 评估 4 接受(57%)、19 次参考执行）→中断恢复（4 步＋resume +2 至 6 步；主运行 6→8 步 resume 2 评估全接受）→导出（9 帧 driving extxyz）。
  - 配方 B（examples/al_surface_qe_mace/）：Al(111) 2×2×4 ＋ 1 吸附原子（17 原子），底部两层 FixAtoms（indices 0–7）。adaptive 在 validate 阶段即被能量口径契约拦截（metallic 体系报告 free_energy 与 MACE 的 energy 不匹配，证据存 contract-evidence.txt）；plain 部分全部完成：surrogate 单点、FIRE 优化（16 步）、reference-only 6 步（9 SCF/271.8 s）、surrogate-only 6 步、reference-only 的 plain checkpoint/resume +2 步、导出。
  - CI 示例（examples/periodic_lj/ + examples/backends/pyraimd2_lj）：32 原子 fcc Lennard-Jones 固体，无外部软件，覆盖 singlepoint/relax/三种 MD 模式/恢复/导出，4 个 hermetic 测试通过。
  - 通用 Slurm 模板与填写说明（hpc/templates/）；敏感现场信息（账号、路径、凭据）未入库，填充后的作业脚本留本地。
  - 两处引擎级最小修复（见"设计偏离"单列）。

## 改动清单

- 新增 `examples/si_bulk_qe_mace/`：make_structure.py、README、run-singlepoint/relax/md-{reference,surrogate,adaptive}.toml。
- 新增 `examples/al_surface_qe_mace/`：make_structure.py、README、同类 run-*.toml（含 [constraints] fix_atoms_indices=[0..7]）。
- 新增 `examples/periodic_lj/`：make_structure.py、README、同类 run-*.toml（LJ 玩具配方）。
- 新增 `examples/backends/pyraimd2_lj/`：LJ 后端插件包（lj_reference/lj_surrogate，能力与指纹如实声明）。
- 新增 `hpc/templates/slurm_generic.sbatch`（占位符模板）与 `hpc/templates/README.md`（填写说明）。
- 新增 `tests/unit/test_periodic_lj.py`（4 个用例：singlepoint/relax、plain 两模式、adaptive 接受＋恢复一致、wrapped/unwrapped 导出）。
- 引擎修复（src 最小改动，单列）：
  - `src/pyraimd2/engines/qe_engine.py` `write_qe_input`：pseudo_dir 内的赝势在 ATOMIC_SPECIES 中写 basename（QE 卡片行缓冲有限，长绝对路径静默截断成不可读文件名——真实材料运行中复现，CRASH 记录显示 `file .../S<垃圾> not found`）。
  - `src/pyraimd2/engines/qe_engine.py` `QeEngine` 目录编号改为按 run_root 扫描续号（resume 后新进程不再撞 `eval-000000`；保持 WP00 起的全局编号约定，label 不重置编号）。
  - 配套测试：`tests/unit/test_qe_engine.py` 新增 2 个用例；`test_density_start_fallback_keeps_failed_attempt` 的期望目录按新续号约定更新（chain-000001）。
- 旧接口／已有数据的兼容方式：basename 写法只改变 pseudo_dir 内文件的行内容（QE 实际打开同一路径）；目录续号不改变单进程行为。
- 是否改变单位、力预算、时间、约束、随机检查或模型切换语义：否。

## 验收证据

```text
验收项：配方 A（体相 Si）全流程与至少一个真实 surrogate 接受/重新锚定/恢复案例
命令（Neimeng A，sbatch，单节点 48 进程）：见 examples/si_bulk_qe_mace/README.md
测试或结果文件：远端 runs/si-*/{events.jsonl,summary.json}，inspect-adaptive.json / inspect-adaptive-short.json（已回收）
预先确定的通过标准：三模式各 6 完整步；adaptive 有真实接受与检查计数；中断恢复可继续
实际关键结果：
  pilot 单点：Si8 SCF 12.5 s（48 进程）、MACE 首次推理 49.3 s；
  reference-only 6 步 7 SCF（80.3 s）；surrogate-only 6 步 0 SCF；
  adaptive 6 步：7 评估 4 接受(57%)、违规 0、参考执行 19（anchor 3、probe 12、check 4，墙钟 302 s）；
  中断恢复：4 步＋resume +2（评估 5→7、接受 3→4、检查 4、边界 0.0）；主运行 6→8 步 resume 2 评估全接受、2 次检查；
  inspect 计数与在线记录一致（actual=logical，cache_hits=0，1 次失败尝试入帐）
状态：通过
```

```text
验收项：配方 B（固定层 Al 表面）工作流与约束语义
命令：examples/al_surface_qe_mace/README.md 的 plain 部分命令
测试或结果文件：runs/al-*/events.jsonl、inspect-{reference,surrogate}.json、contract-evidence.txt
通过标准：单点/relax/reference-only/surrogate-only 与 plain resume 可跑；固定层语义成立；adaptive 的处置如实
实际关键结果：FIRE 16 步（ASE 收敛判据通过，最终按原子范数 max|F|=0.055）；reference-only 6 步 9 SCF（含 resume +2，271.8 s）；surrogate-only 6 步 0 SCF；plain resume +2 至 8 步；导出 9 帧。adaptive 被 WP01 能量口径契约在 validate 阶段明确拦截（QE metallic 报 free_energy、MACE 报 energy），未伪造通过——见"未通过项"
状态：通过（plain 部分）；adaptive 见下
```

```text
验收项：CI 周期解析示例
命令：uv run pytest tests/unit/test_periodic_lj.py -q
测试：4 个用例（singlepoint/relax、plain 两模式、adaptive 接受与恢复逐行一致、导出 wrap 选项）
预先确定的通过标准：全部通过且 adaptive 有 ≥2 个接受
实际关键结果：4 passed；adaptive 8 步中接受 3 步、参考执行含 anchor/probe/check
状态：通过
```

```text
验收项：受影响套件整体回归
命令：uv run pytest tests/unit tests/test_smoke.py -q
通过标准：全部通过
实际关键结果：389 passed, 1 deselected(slow)（基线 383 + CI 示例 4 + QE 修复 2）；ruff 改动文件全部通过
状态：通过
```

## 参考执行配额与墙钟（计划 §9-L2）

- 总参考执行（实际 SCF）：**58 ≤ 200**。配方 A 49（先导探针 1、reference-only 7、adaptive 21、adaptive-short 20 含 1 失败尝试），配方 B 9（plain reference 6 步＋resume +2）。每条均 ≤100。
- 分用途：anchor 6、probe 24、check 9、plain-MD 16、先导 1、失败尝试 1。
- 缓存命中：0（QeEngine 无 fingerprint 可共享标签缓存——无，如实记录）；逻辑请求 == 实际执行（无缓存路径）。
- 墙钟：Si8 SCF 约 11.5–12.5 s（48 进程）；Al17 板 SCF 约 27–30 s；MACE-MPA-0 首次推理 49.3 s、逐步约 0.3–0.5 s；adaptive 6 步总墙钟 302 s（SCF 主导）。
- 失败尝试 1 次（resume 期间旧目录编号碰撞，已计费并触发 WP08 引擎修复，修复后同一恢复路径通过）。
- 分配资源：每配方 1 节点×48 CPU（9242 分区）；无 GPU；排队约 25 min（因账号节点上限改投 9242）。

## 未通过项与原因

- **配方 B 的 adaptive 模式：未通过（契约拦截，非实现缺陷）。** QE 在 metallic+smearing 下报告 free_energy（WP05 起如实声明），MACE 报告 energy，WP01 能量口径契约在 validate 阶段明确拒绝混合。处置：plain 部分全部完成；报告该组合在 0.4.0 的真实能力边界。下一步最小建议（交负责人判断）：(a) 契约细化为"两侧均 force_consistent 时允许跨 kind 锚定"并配套 endpoint work 口径说明；或 (b) 接受该限制并写入支持矩阵。
- **配方 A 的稳态接受率**：6–8 步内接受率 57–100%（短程示例，不构成稳态或加速主张；未放松任何容差）。

## 恢复与随机状态（相关工作包必填）

- 连续 vs 中断恢复：si-adaptive-short 4 步＋resume +2 与 si-adaptive 6 步＋resume +2，均在新进程（新 Slurm 作业）完成；检查序列、accepted/detected 与在线记录一致（inspect 复核）。
- 故障注入与恢复证据：旧版 QeEngine 目录编号碰撞导致 resume 首个 check 失败（1 次失败尝试入帐）；修复后同路径恢复通过。
- 成本账本：事件账本只追加；失败尝试、探针、检查均入帐；plain reference 的 resume 成本亦入帐。

## 算力与成本

- 新增实际参考执行总数：58（本工作包全部）。
- 参考、推理、训练、I/O 墙钟及计时口径：见上（任务事件为准，外层墙钟直测）。
- 预算使用与剩余额度：本批两案例共用 58/200；剩余 142。

## 回归与交付

- 受影响测试通过情况：tests/unit + test_smoke 全绿（389）。
- 原有 energetic／legacy switching 示例：未改动；QE 既有测试随新续号约定更新一处期望。
- wheel 安装及仓库外 CLI 测试：未执行（WP09）。
- 最小依赖是否仍不导入 Torch／PySCF：是（CI 示例为纯 NumPy+ASE；MACE 只在材料配方环境使用）。
- 用户可复制的完整运行与恢复命令：见两个配方 README（含 pseudo_dir 短路径注意事项与 pw_cmd 说明）。
- README／示例／支持矩阵更新位置：README 与支持矩阵留 WP09；素材见下。
- 已知故障、未执行验证与风险：
  - 集群 mpirun 需要 `--map-by :OVERSUBSCRIBE`（2026-09 现场；8 月的快照已过期），launcher 细节留本地，通用模板不含现场 hack。
  - Al relax 最终全原子 max|F|=0.055 而 ASE 判据通过。当时解释为"ASE 收敛判据用分量最大值、配方打印原子范数"；**2026-09-09 修正**：独立审阅用本机 ASE 3.29 证明该解释不成立——ASE 的判据是**约束投影后的最大单原子力范数**（(0.04,0.04,0) 在 0.05 阈值下并不收敛）。正确口径下，"0.055 且判据通过"指向固定层（indices 0–7）反作用力被计入报表值；R4 已把 final_fmax 改为判据同口径（投影后），原始全原子值另列 `raw_all_atom_fmax_eV_A`。Al 末帧原始力数组仍需按固定/自由分列重算以确认自由原子（8–16）是否低于 0.05——审阅时未获得该数组，重算列入 REVIEW_FIXES_CORE_20260909.md 待办。
  - MACE 模型文件与赝势不进 git（输入非代码）；配方引用 recipe 相对路径。
  - 填充后的 sbatch（含现场路径）留本地未入库，符合敏感信息约定。
- 需要负责人判断的具体决策及备选方案：metallic/free-energy 参考与 energy 替代的 adaptive 组合是放宽契约（a）还是写入支持矩阵（b）。
- 下一工作包及其入口：WP09 收尾——支持矩阵素材：QE 7.5+MACE 0.3.16 的 Si bulk adaptive 已验证（接受/重锚/恢复）；metallic smearing 参考的 adaptive 在 0.4.0 不支持（契约）；plain 三模式与 FixAtoms 在真实材料上验证；LJ 周期 CI 示例在最小依赖下通过。

## 设计偏离（单列）

- 引擎修复 1（write_qe_input basename）：源于配方运行的真实失败（QE 卡片行缓冲截断），带回归测试。
- 引擎修复 2（QeEngine 目录续号）：源于 resume 的真实失败（eval-000000 碰撞），保持 WP00 全局编号约定，带回归测试；一处 WP05 测试期望相应更新。
- 配方 B 只选 Al(111) 一种并固定；adaptive 部分按契约如实报未通过，未改用替代材料。
