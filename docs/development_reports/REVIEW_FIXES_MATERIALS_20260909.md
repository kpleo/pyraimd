# 0.4.0rc1 独立审阅修正回报——材料记录组（审阅 §4 逐条）

- 日期：2026-09-09
- 审阅依据：`docs/development_reports/INDEPENDENT_REVIEW_20260909.md` §4（材料报告需要修正）与 §8（回报格式）
- 基线 commit：`842242d`（dev/0.4.0 顶端）；分支 `dev/0.4.0-matfix`（worktree `pyraimd2-matfix`）
- 修复 commit：与本报告同批的 `Docs:` 前缀提交（逐条 hash 见 git log）
- 分工说明：§4 的材料记录校正（本批）与阶段 1 的核心修复（`REVIEW_FIXES_CORE_20260909.md`，含 relax 报表口径 final_fmax 改判据同口径、raw 另列）相互衔接；本批不改 `src/`、`tests/`、README/CHANGELOG、其他 WP 报告、reproducibility/analysis/manuscript。
- 新增实际 SCF 数：**0**（§4 明确"先重算记录，不重跑大计算"；云端只读回收与本地重算，未提交任何新任务）
- 真实后端验证文件路径：`docs/development_reports/wp08_evidence/`（2026-09-09 从 Neimeng A 只读回收，见下）

## §4.1 Al 收敛解释改正（重算完成）

### 具体原因

WP08 把 relax 末帧全原子 max|F|=0.055 > 0.05 解释为"ASE 用最大分量、Pyramid 用单原子范数"；审阅已证伪（ASE 3.29 判据为约束投影后最大单原子范数）。审阅要求按固定（0–7）/自由（8–16）分列重算原始力数组，区分"固定层反作用力/报表问题"与"自由原子仍超限"。

### 重算结果（原始记录）

末帧取自 `wp08_evidence/al/runs/al-run-relax/trajectory.db` 最后一行（step 16、route ml、WP08 时期 driving 即原始 MACE 力数组）：

- 全原子原始 max |F| = **0.054590 eV/Å**（与 events 中 `run_summary.final_fmax_eV_A = 0.05458950919183557` 一致）
- 固定层（indices 0–7）max = **0.054590**（atoms 4–7 为 0.052802–0.054590，约束反作用力）
- 自由原子（indices 8–16）max = **0.048156 < 0.05**（最大为吸附原子 16）

结论：自由原子已按 ASE 判据（约束投影后最大单原子范数）收敛；0.055 全部来自固定层反作用力的报表口径问题——属审阅预期的第一种情形（报表问题），不是自由原子超限。修正位置：`WP08_report.md`（验收证据与"已知故障"条）、`examples/al_surface_qe_mace/README.md`（新增 FixAtoms 报表口径说明）。

证据：`wp08_evidence/analysis/al_relax_final_fmax.txt`（逐原子范数）、`wp08_evidence/analysis/al_relax_final_frame.extxyz`（末帧结构+原始力数组，可独立复核）。

## §4.2 SCF 数记录校正（重算完成）

### 具体原因

WP08 总数 58 与分用途 6+24+9+16+1+1=57 对不上；"旧目录编号碰撞"失败是否真正启动 SCF 未核实；"QeEngine 无 fingerprint"与实现不一致。

### 重算结果（全部来自原始 events.jsonl 任务记录与 slurm 日志）

- **总数**：逻辑参考请求 **58**（运行事件内 57＋先导 `validate --probe-backends` 1 次真实 SCF：slurm 7625052 日志记录 energy −5087.018956 eV、12.521 s）；实际 pw.x 启动 **57**（56 次成功任务＋先导 1）。
- **那次失败是否启动 SCF**：**未启动**。`si-adaptive-short-task-33`（verification，eval 5）在创建 attempt 目录时 FileExistsError(17)，elapsed 0.05 s（该体系一次 SCF 约 13–24 s；修复后重试 task-34 成功、18.1 s）。按阶段 1 后 logical/actual 口径：logical=1、actual=0。WP08 当时按"任务=执行"旧口径计为 1 次失败尝试，现如实标注。
- **分用途（校正后）**：anchor 2、refusal 4、probe 24、verification 成功 10（＋1 失败仅逻辑）、plain-MD 16（Si 7＋Al 9）、先导 1。原报告两处口径错误的来源：anchor 6 是 runner 打印摘要把 refusal 并入 anchor 的计数（3＋3）；"check 9"把失败的 verification 同时计入 check 数与失败尝试（重复计数）。
- **指纹/缓存**："QeEngine 无 fingerprint"不成立。两侧 run_start 的 `reference_id` 与 manifest.json 均记录 WP05 设置指纹：Si `qe-pbe-d3:d23205a4a5469f4f`、Al `qe-pbe-d3:b5880bfa70564c2d`（按配方/设置正确区分）。标签缓存当时处于启用状态，命中为 0 的实情：每个被检查几何都是新构型（MD 不重访同一构型），且 WP02 标签缓存为进程内存级（resume 后冷启动）。logical==actual 是零命中的结果，不是缓存路径缺失。

证据：`wp08_evidence/analysis/scf_recount.md`（逐运行/逐段/逐用途重算与出处）、`wp08_evidence/analysis/scf_recount.json`（机器可读）、各 run 的 events.jsonl（包内 verbatim）。

## §4.3 证据包路径（已回收入库）

审阅未定位 WP08 声称已回收的 inspect JSON/contract-evidence。已从 Neimeng A（`~/cloud_projects/pyraimd2/experiments/wp08/`）只读回收轻量包至 **`docs/development_reports/wp08_evidence/`**：

- `si/`、`al/`：配方配置（含 interrupted 演示用的 run-md-adaptive-short.toml）、make_structure.py、输入结构；每个 run 的 config.toml / resolved_config.json / manifest.json / events.jsonl / trajectory.db / summary.* / 导出轨迹（存在时）；inspect-adaptive{,-short}.json、inspect-{reference,surrogate}.json、contract-evidence.txt。
- `analysis/`：scf_recount.{md,json}、al_relax_final_fmax.txt、al_relax_final_frame.extxyz、si_adaptive_final_frame.extxyz。
- 身份：软件 0.4.0.dev0（远端 torch 2.6.0 CPU / mace 0.3.16 / ase 3.29.0 / numpy 2.5.1，QE 7.5）；参考指纹（上）；赝势 `Si/Al.pbe-n-kjpaw_psl.1.0.0.UPF`；模型身份 `mace-mp:.../macempa0mediummodel:cpu:float64`（模型文件本身 MB 级未入库）。
- 未入库（敏感 profile 与大型产物）：填充后 sbatch、现场 launcher（pwx.sh 及 mpirun map-by 现场 hack）、波函数与 calculations/ 大目录（留云端）。run 记录为 verbatim，文件内容保留运行时现场路径，包 README 已注明。

## §4.4 README 修正（完成）

- `examples/al_surface_qe_mace/README.md`：不再引导执行 adaptive（原命令块含 `pyramid run run-md-adaptive.toml` 与 al-adaptive 的 resume/inspect/export）。新增"Adaptive mode: not supported for this recipe in 0.4.0"一节：说明 QE metallic 报 free_energy 与 MACE 报 energy 在 validate 阶段被 WP01 口径契约拦截（指向 contract-evidence.txt），解除限制需审阅 §5 的契约细化；命令流改为 plain 两模式＋plain resume/inspect/export，`validate run-md-reference.toml` 为预检命令；保留 `validate run-md-adaptive.toml` 作为"按设计失败"的可复现演示。
- `examples/si_bulk_qe_mace/README.md`：resume 写法改为 **6+2**（与所附 6 步配置一致：主运行 6 步→resume +2 至 8）；另说明 WP08 的 interrupted 演示用的是单独的 4 步配置（改 steps=4 的副本→resume +2 至 6，runs/si-adaptive-short）。
- 两个 README 均新增"stages are independent demonstrations"：所有阶段读同一 `make_structure.py` 输出，优化终态**未**自动传给后续 MD；如需以优化终态起步，显式导出并在 `[structure] file` 引用。
- Si README 另补"what this demonstration does and does not show"（集成可运行≠加速，见 §4.5）。

## §4.5 不加速主张复核（保持并加强）

- 全库检索 "57%|节省|saving|speedup|加速"：WP08/WP09 均未把 57% 接受率写成成本节省；WP09 已有"本报告不作任何加速主张"。hpc.md 中两处 "saving" 是 8 月科研战役的密度链记录，非 WP08 主张，不在本轮范围。
- `WP08_report.md` 在"配方 A 的稳态接受率"条下加强为：同一 6 步测试点 reference-only 7 SCF/80.3 s vs adaptive 19 次参考执行/302 s——该测试点 adaptive 总成本**更高**；57% 接受率不是节省 57% DFT 成本；当前案例证明集成可运行，加速评估需探针开销可摊薄的更长场景。

## 验证

- 重算全部在本地对回收记录执行（脚本未入库，结果文件见 `wp08_evidence/analysis/`）；记录数值与 events/manifest 原值逐项一致（final_fmax、reference_id、任务计数）。
- `cd /Users/pengkang/Research/Project/Project_PYRAIMD/pyraimd2-matfix && uv run pytest tests/unit tests/test_smoke.py -q`：**448 passed, 1 deselected**（本批为文档与证据改动，未动可执行内容）。
- 新增实际 SCF 数：0。

## 保留限制

- 云端 `calculations/` 大目录、波函数、密度工件未回收（不需要也不轻量）；若需进一步核对 attempt 目录内容，记录在集群原路径仍在（未改动）。
- `al-reference` 叶子任务耗时合计（439.7 s）大于两段墙钟之和（379.1 s）：WP08 时期 plain 驱动计时口径（引擎自报 wall_time 优先）的既知现象，阶段 1 后已改"只取真实 attempt"口径；历史记录不重改，如实保留。
- `al-run-relax/trajectory.db` 末帧存了两行（同 evaluation_id，WP08 时期 relax 驱动怪癖）；`run_summary.n_evaluations=17` 只计一次。重算取末行，两行力相同，不影响 §4.1 结论。
- WP08 报告正文的历史数字（19/21/13 等 runner 打印口径）未逐句改写为 events 口径；校正集中在"配额与墙钟/算力与成本/已知故障/验收证据"四处并指明重算文件，避免把报告改写成第二套账。
- Al adaptive 仍按契约不支持（本批不改契约）；metallic smearing 的 free_energy 口径联合验证属审阅 §5 的后续项。
