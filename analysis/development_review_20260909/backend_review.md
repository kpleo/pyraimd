> 归档说明：以下为独立分项复核记录，代码位置针对 5a1d105。原复核仅写入临时目录；主审已将脚本及本报告归档于当前目录，并将脚本依赖改为可迁移的相对位置。文中临时目录是原始执行的溯源信息；交付的同名日志来自归档脚本的复核执行。总体验收与优先级以 docs/development_reports/INDEPENDENT_REVIEW_20260909.md 为准。

独立只读审阅：ASE/QE 后端、约束与普通工作流

审阅对象为 `/Users/pengkang/Research/Project/Project_PYRAIMD/pyraimd2`，分支 `dev/0.4.0`，HEAD `5a1d1054d7cdd9444bdd85a62afeba6a5fc607d5`，未使用 `.release/pyramid`。已核对 WP00/WP05/WP07 报告及 DEVELOPMENT_PLAN_zh.md；按追加要求核对 WP08 对 Al 优化的解释。未修改或提交仓库代码，未安装、联网下载或运行云端/真实 DFT。复现使用现有 Python 3.12.13、ASE 3.29.0、NumPy 2.5.2。结束时 HEAD 不变、tracked/staged diff 均为空；原有 untracked 文件保留。

不重复主审负责的 `_policy_kwargs` 丢失 `force_metric` 传参；不审 core energetic checkpoint/update、CLI 发布和真实材料验收。本报告涉及的 resume 问题均来自普通工作流或后端目录。

以下 P1 表示应在候选版放行前修复的标签、恢复或物理身份问题；P2 表示明确可复现的功能/契约缺陷。每项均列出现有代码位置、触发、实测后果和修正建议。代码行号针对上述 HEAD。

复现材料（仅创建于 /tmp）：

- [后端脚本](pyramid_review_backends_5a1d105.py)，[完整输出](pyramid_review_backends_5a1d105.log)。输出数据目录 `/tmp/pyramid-review-backends-cloyrheg`。
- [普通工作流脚本](pyramid_review_workflows_5a1d105.py)，[完整输出](pyramid_review_workflows_5a1d105.log)。输出数据目录 `/tmp/pyramid-review-workflows-s67bni9e`。
- [QE retry/账本/路径脚本](pyramid_review_qe_retry_5a1d105.py)，[完整输出](pyramid_review_qe_retry_5a1d105.log)。输出数据目录 `/tmp/pyramid-review-retry-cszal4tv`。

在目标仓库目录下分别执行：

```sh
UV_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python analysis/development_review_20260909/pyramid_review_backends_5a1d105.py
UV_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python analysis/development_review_20260909/pyramid_review_workflows_5a1d105.py
UV_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python analysis/development_review_20260909/pyramid_review_qe_retry_5a1d105.py
```

这些是展示当前错误行为的复现脚本，不是声称修复后的测试通过。每次执行会创建新的 /tmp 数据目录；fake pw.x 只读取现有输出 fixture，根本不计算 DFT。所有下面的数值均来自实际本地执行或安装版本源码。

1. **[P1] 通用 ASE 指纹碰撞：改变物理参数仍是同一后端/模型身份。**

   位置：[ase_engine.py:60](../../src/pyraimd2/engines/ase_engine.py:60)，[ase_surrogate.py:33](../../src/pyraimd2/surrogate/ase_surrogate.py:33)。实际指纹只包含 calculator.name、force_consistent 和 include_stress，完全没有 calculator.parameters 或模型工件身份。

   触发/证据：同一 Ar2、距离 1.5 Å，`LennardJones(epsilon=1)` 与 `epsilon=2` 的 AseEngine 指纹均为 `ase:lennardjones:force_consistent=False:stress=False`，能量却分别为 `-0.3148571525343358` 与 `-0.6297143050686717 eV`；AseSurrogate 同样碰撞。对应后端日志 `ase_fingerprint_collision`。

   后果：runtime 把该字符串当作已知身份，而非 unknown。参考设置变化检查、label cache key 和 model_id 无法识别此类物理改变，有复用旧标签/旧模型身份的路径；见 runtime/labels.py:42–60、loop/energetic.py:869–875。该复现直接证明碰撞，没有声称另跑了完整 energetic 缓存审计。

   修正：提供明确、可序列化的设置/工件指纹接口；对可识别 ASE Calculator 哈希类路径、有效物理参数和模型文件内容。不能可靠识别的外部状态应返回 None 或要求用户提供身份，不应以 calculator.name 伪造稳定身份。回归覆盖同类不同参数、同路径模型内容变化和参数更新前后。

2. **[P1] 普通 surrogate MD 恢复不校验模型身份，允许跨势能面续算。**

   位置：[md.py:899](../../src/pyraimd2/workflows/md.py:899)。899–904 只在 reference 模式比对 engine_fingerprint；checkpoint 已保存的 model_id 在 surrogate 模式未核对，随后用新后端继续旧边界力。

   触发/证据：解析 surrogate 跑 2 步并保存 checkpoint；仅在 /tmp 运行副本的 resolved_config 中把 k 从 1 改为 2，再 resume 1 步。恢复成功到 step 3，checkpoint model_id 为 `harmonic-surrogate:4d610b8235e8#g0`，新行却变成 `harmonic-surrogate:324c74d29cdb#g0`。同样改动 reference k 会被正确拒绝。见工作流日志 `changed_backend_resume`。

   后果：普通 resume 静默变成未声明的模型切换，恢复首个 half-kick 仍使用旧模型边界力。违反计划 §6 的同一物理设置/模型链要求。

   修正：构建后端后、任何续算写入前，比对冻结 surrogate fingerprint/model_id；同时核验 checkpoint 中的 timestep 等不可变动力学设置。拒绝不匹配并指示 fork。回归覆盖 surrogate 参数、模型文件内容改变，及原模型连续/恢复对照。

3. **[P1] 零位移边界的普通 MD resume 会写出空标签并污染轨迹。**

   位置：[md.py:954](../../src/pyraimd2/workflows/md.py:954)、[md.py:364](../../src/pyraimd2/workflows/md.py:364)、[md.py:387](../../src/pyraimd2/workflows/md.py:387)。恢复只装入 Calculator.results/atoms，没有恢复 `_BackendCalculator.last_label`。

   触发/证据：两个解析谐振原子均处于 r0=0，动量为零，连续运行 2 步后 resume 1 步。ASE 缓存因坐标未变而命中，不调用 calculate，last_label 仍为 None。reference 与 surrogate 两种模式都实际产生 `ExportError: store row has no 'dft'/'ml' payload to drive from; the trajectory database looks corrupt`。见日志 `stationary_resume`；原始临时运行 DB/events 保留。

   后果：_record_evaluation 已经把 None 作为 engine/surrogate/driving 写入行，后续导出失败；这不是单纯预览错误。本轮实际复现为无约束平衡静止构型；FixAtoms 或坐标舍入造成零位移的情况也需补回归，不能依赖坐标改变来填充 last_label。

   修正：从恢复边界的原始 backend payload 重建 last_label，并与 results 保持一致；已有边界数据可复用，不必重算昂贵参考。写行前强制校验原始标签/驱动力非空且形状正确。补零速度平衡态、FixAtoms 不动构型的连续与恢复测试，断言无空行且重新导出/再次恢复均可用。

4. **[P1] ASE-QE 把未完成/明确未收敛输出当作合格参考标签。**

   位置：[ase_qe.py:159](../../src/pyraimd2/engines/ase_qe.py:159)、[ase_engine.py:72](../../src/pyraimd2/engines/ase_engine.py:72)。该路径仅依赖 ASE 读取能量、力、stress，并没有手写 QE 的组合成功门槛。

   触发/证据：fake 程序 rc=0，输出同一 Si fixture 的有效数值，但删掉 `JOB DONE.`；AseQeEngine 仍返回 `-1271.308088206284 eV`。另在 fixture 插入 `convergence NOT achieved` 也被接受。相同 fake 输出经 QeEngine 均被 QeEngineError 拒绝。见后端日志 `missing_done` 和 `not_converged`。

   后果：切换 backend 从 qe 到 qe-ase 会降低参考标签有效性标准，截断或失败结果能进入普通/自适应计算。不能用“只做过输入层测试”豁免运行时的 Engine 契约。

   修正：返回 ASE-QE 标签前验证该次输出的完成标志、SCF 状态、QE error banner、完整数值块及原子数，并将失败转为 EngineError。两条路径共享成功判定，新增 fake 实际执行测试，而非仅对 fixture 调用两个 parser。

5. **[P1] ASE-QE 新实例仍从 eval-000000 开始，已有运行不可正常 resume。**

   位置：[ase_qe.py:127](../../src/pyraimd2/engines/ase_qe.py:127)、[ase_qe.py:152](../../src/pyraimd2/engines/ase_qe.py:152)。内存 `_call_counter=0`，compute 使用 `mkdir(exist_ok=False)`。

   触发/证据：同一 run_root，第一实例完成 fake SCF；构造第二个同配置实例并 compute，立即 `FileExistsError: .../ase-restart/eval-000000`。见 `ase_qe_fresh_instance`。

   后果：普通/energetic 恢复都会重建后端；第一次真正重新求值就会撞旧目录。手写 QeEngine 已在 WP08 改成扫描续号，但 ASE 路径遗漏。

   修正：为两条 QE 路径共用不碰撞的目录分配，至少在新实例扫描现有编号并原子创建目录。回归应使用两个实例/新进程及真实 workflow resume，不仅在一个实例上调用 fresh_directory 两次。

6. **[P1] 手写 QE 忽略输入磁矩，两个适配器对同一结构计算不同自旋设置。**

   位置：[qe_engine.py:224](../../src/pyraimd2/engines/qe_engine.py:224)、[qe_engine.py:247](../../src/pyraimd2/engines/qe_engine.py:247)、[ase_qe.py:103](../../src/pyraimd2/engines/ase_qe.py:103)。手写 SYSTEM 没有 nspin/starting_magnetization，也按元素而非磁矩拆分 species。

   触发/证据：相同 Si2、相同 QeConfig，Atoms.initial_magmoms=[1,-1]。手写输入只有 `ntyp=1`，无自旋设置；ASE 实际写出的输入为 `nspin=2`、`ntyp=2`、`starting_magnetization(1)=1.0`、`(2)=-1.0`。见 `spin_input_mapping` 和临时 `spin-handwritten.in`/`spin-ase/espresso.pwi`。这是输入写入复现，不是磁性 Si 的材料计算。

   后果：输入身份/标签键虽然记录了磁矩，手写后端实际没有实施该输入；换适配器也无法保持所承诺的同一物理设置。该问题不依赖真实材料能否收敛。

   修正：要么在支持的自旋范围内与 ASE 共享 species/磁矩映射及有效设置身份；要么对非零磁矩明确在计算前拒绝并声明限制。不要静默变为默认非自旋计算。电荷等其他未映射的电子态输入也应明确支持或拒绝，但本轮没有把电荷映射另列为已跑实测结论。

7. **[P2] FixAtoms relax 保存/打印原始力为 driving 和收敛 fmax，丢失实际约束投影口径。**

   位置：[md.py:740](../../src/pyraimd2/workflows/md.py:740)、[md.py:761](../../src/pyraimd2/workflows/md.py:761)；singlepoint 同类记录见 [md.py:644](../../src/pyraimd2/workflows/md.py:644)。ASE optimizer 使用已投影力，Store driving 却直接取 raw label，constraint metadata 也没有保存。

   触发/证据：k=1、r0=0 的解析势，固定原子位于 (1,0,0)，自由原子位于 (0.02,0,0)，阈值 0.05。ASE FIRE 在第 0 步正确打印 fmax=0.020000 并收敛；Pyramid run_summary 却写 `converged=true, final_fmax_eV_A=1.0`，driving 中固定原子的力为 (-1,0,0)，constraint metadata 为 null。singlepoint 同样把未投影标签写为 driving。见 `fixed_relax`/`fixed_singlepoint`。

   后果：优化轨迹记录和 force-source=driving 导出不能代表优化器实际使用的力；打印的终态力无法验证用户阈值。原始全原子反作用力可以大于阈值，但必须另列诊断，不代表优化未收敛。

   修正：保留 engine/surrogate 的原始力；统一生成约束投影后的 driving，记录约束集合及位移；final_fmax 用 `atoms.get_forces()` 的最大单原子范数，并另存 `raw_all_atom_fmax`。singlepoint 无推进时也应明确 driving 的定义。回归覆盖固定原子反作用力大、自由原子已收敛，以及自由原子范数超阈值的情形。

   WP08 的“ASE 按最大分量收敛”解释错误。安装的 ASE 3.29.0 在 [ase/utils/abc.py:38](../../.venv/lib/python3.12/site-packages/ase/utils/abc.py:38) 使用 `np.linalg.norm(forces, axis=1).max() < fmax`；[ase/optimize/optimize.py:37](../../.venv/lib/python3.12/site-packages/ase/optimize/optimize.py:37) 的梯度取自带约束的 atoms.get_forces。实际测试 (0.04,0.04,0) 的最大分量为 0.04，范数为 0.0565685424949238，阈值 0.05 时 **不收敛**。须修正 [WP08_report.md:110](../../docs/development_reports/WP08_report.md:110)。

   本轮没有读取到 Al 最终原始力数组，因此不声称已证明 0.055 来自固定层。主审应在实际最终帧按固定 indices 0–7 及自由 indices 8–16 分别计算最大原子范数；若 free<0.05 而 raw_all=0.055，则是反作用力/报表口径问题。若 free 也超限，则须继续核对记录帧与 optimizer 实际终态，不能用“最大分量”解释。

8. **[P2] QE 非零退出的 SCF 未收敛被误分类为可重试。**

   位置：[qe_engine.py:637](../../src/pyraimd2/engines/qe_engine.py:637)。代码先处理 returncode!=0，仅按 Error in routine banner 决定 retryable；检查 convergence NOT achieved 在后面的 645 行，已不可达。

   触发/证据：fake 输出 `convergence NOT achieved after 200 iterations` 并 exit(1)，max_retries=3。实际执行 4 个 attempt，全部 retryable=true。见 `nonconvergence_nonzero_exit`。现有 `test_nonconvergence_of_identical_input_is_not_retried` 的 fake 最后运行 sed，退出码为 0，所以未覆盖此分支。

   后果：确定性的同输入 SCF 未收敛耗尽重试额度，与 WP05 声称的分类规则相反。

   修正：先从输出确定 SCF 未收敛/确定性输入错误，再结合退出码分类；仅密度→atomic 这种输入改变保留独立回退。补 rc!=0+未收敛、rc=0+未收敛和瞬态崩溃三种对照。

9. **[P1，后端接口观察] 内部 QE 多 attempt 与外层 cost ledger 不对应。**

   分工更新：主审已独立确认相同的两次进程执行/一条成功账本问题。本条保留为后端接口交接证据，由主审统一汇总，不重复作为独立新增发现；density I/O 嵌套计时、WP08 分用途合计及旧目录冲突是否真正启动 SCF，均不在本报告中代替主审下结论。

   位置：[qe_engine.py:529](../../src/pyraimd2/engines/qe_engine.py:529)、[qe_engine.py:680](../../src/pyraimd2/engines/qe_engine.py:680)、[md.py:635](../../src/pyraimd2/workflows/md.py:635)。QE 把失败 attempt 留在内存 last_attempt_records，成功返回的 wall_time_s 只有最后一次成功执行；引擎本身只发 density I/O task，不发每次 SCF 的 task。energetic 的 `_reference` 也是围绕整个 compute 发一条 task（loop/energetic.py:884–909）。

   触发/证据：通过真实 singlepoint workflow，让第一次 fake pw.x exit139，第二次输出正常 fixture。实际目录 `eval-000000/attempt-1`、`attempt-2` 共 2 次执行；events 只有一条 reference/attempt=1/success。summarize_tasks 输出 `logical_requests=1, successful_executions=1, failed_attempts=0, actual_executions=1`。见 `inner_attempts_outer_ledger`。

   后果：物理 SCF 次数和失败次数被低报，不能据此验证 WP08 总参考执行预算。singlepoint/energetic 外层 elapsed 包含整次 compute 耗时，不能把这里笼统描述为“所有墙钟都漏了”；普通 MD/relax 使用 label.wall_time_s 时另有只取成功 attempt 的风险。

   修正：定义并接通后端 attempt 事件/返回记录接口，持久化每次真实进程启动、耗时、状态及其 logical evaluation/task 身份；外层逻辑 span 与物理 task 分开，避免重复计费。QE density I/O 与这些 span 的嵌套口径交给主审统一核对。本轮没有重做完整成本审计。

10. **[P2] ASE-QE 暖启动配置未经 staging，空目录仍写 startingpot='file'。**

    位置：[ase_qe.py:86](../../src/pyraimd2/engines/ase_qe.py:86)、[ase_qe.py:159](../../src/pyraimd2/engines/ase_qe.py:159)。AseQeEngine 接受完整 QeConfig，但既不检查/复制 density_source，也不解析前次成功密度，不做 atomic 回退。

    触发/证据：startpot_file=True、无密度的首次 compute。fake 实际读取生成的 input，并在 startingpot=file 且 .save 不存在时 exit2；ASE 路径失败，手写 QE 路径同条件直接 atomic 成功。见 `warm_start_no_density`。

    修正：接入与手写 QE 共用的密度验证/staging/atomic fallback；若当前阶段不支持，应在构造或 validate 时明确拒绝这些参数，不能接受后在每个新目录触发缺密度失败。ASE 路径未实现的 timeout_s/max_retries 也应明确拒绝/声明，不能让用户误以为这两个字段已控制其运行。

11. **[P2] ASE 失败清理假定 reset() 存在，会用 AttributeError 替换原计算异常。**

    位置：[ase_engine.py:82](../../src/pyraimd2/engines/ase_engine.py:82)。`except Exception` 内无条件调用 self.calculator.reset()；当前 ASE Espresso/GenericFileIOCalculator 没有该方法。

    触发/证据：第 10 项失败实际抛出 `'Espresso' object has no attribute 'reset'`，不是约定的 EngineError；本地 `hasattr(Espresso, 'reset')` 为 False。

    后果：失效/超时/坏结果等错误无法按 EngineError 契约处理，清理还未完成就发生新异常。现有失败测试仅使用继承 Calculator 且有 reset 的类。

    修正：适配 ASE BaseCalculator/GenericFileIOCalculator 的缓存清理接口，在 finally/异常转换中保护原始异常，确保 results 清空且最终统一 EngineError。补真实 Espresso fake 失败测试。

12. **[P2] Python API 相对 pseudo_dir 的哈希对象与子进程实际输入路径不一致。**

    位置：[qe_engine.py:222](../../src/pyraimd2/engines/qe_engine.py:222)、[qe_engine.py:366](../../src/pyraimd2/engines/qe_engine.py:366)、[qe_engine.py:621](../../src/pyraimd2/engines/qe_engine.py:621)。pseudo_dir 按调用者 cwd 哈希，但原样写入 input，pw.x 在 attempt_dir 下解析它。

    触发/证据：调用目录下确有 `pseudos/Si.UPF`，QeConfig(pseudo_dir='pseudos') 得到非空内容 SHA256；fake 在真实 attempt cwd 读取该配置目录时找不到文件，抛 QeEngineError。见 `relative_pseudo_dir`。CLI 配置层通常会预先绝对化，此项明确针对公开 Python API。

    后果：能被识别/哈希的赝势仍无法启动；若另一个相对目录恰好存在不同同名文件，还可能与记录的身份不一致。

    修正：在引擎构造时固定所有路径型输入的解析基准，input 与 fingerprint 使用同一规范绝对路径/赝势映射；两条 QE 适配器共用。补调用者 cwd 与 subprocess cwd 不同的真实读文件测试。

13. **[P2] capabilities 与交付标签存在已实测不一致，parser 完整性门槛仍不足。**

    位置：[ase_qe.py:132](../../src/pyraimd2/engines/ase_qe.py:132)、[ase_engine.py:85](../../src/pyraimd2/engines/ase_engine.py:85)、[qe_engine.py:309](../../src/pyraimd2/engines/qe_engine.py:309)。

    ASE-QE 的 capabilities.force_consistent=True，但父类以默认 force_consistent=False 初始化；实际正常 fixture 标签 `force_consistent=None`。这不是已证明取错了 QE 能量数值，而是承诺与逐标签元数据不一致。应明确选择 force-consistent getter 并按实际口径填写结果，补绝缘/metallic 两种标签断言。

    手写 QE 声明 stress_available=True，却接受缺少 stress 的 fixture 返回 stress=None。stress 数值改为 `**********` 时，则由 _to_float 泄漏 ValueError，不会进入只捕获 EngineError 的解析失败/重试记录路径。重复 atom 1、缺失 atom 2 但总行数仍为 2 的 force block 也被接受。见后端日志 `missing_stress`、`bad_stress`、`duplicate_atom`。这些是故障注入门槛，不声称正常 QE 会自然打印重复 atom 编号。

    修正：当声明/request stress 时要求完整有限 3×3 块；统一解析异常到 EngineError，并验证 force 原子编号唯一、完整且顺序正确（或按编号还原）。如果 stress 确实可选，要让 capabilities/request/result 的可选语义一致。补失败 attempt 最终 status/error 的断言，不能留下 running。

验收判断与实际验证边界：

- WP00 约束能量原始口径、手写 QE 多块/D 指数/唯一目录等已有用例在本环境通过；没有把这些已修复点重新列为 bug。
- WP05 尚未达到“更换适配器保持输入/标签契约、可靠唯一目录、明确失败/重试、实际成本可复核”的完整门槛。ASE 构造/fixture 对照通过不等于其 compute 路径完成集成验收。
- WP07 固定层静止、普通移动轨迹恢复的已有测试通过；零位移缓存恢复、surrogate 身份验证和受约束优化驱动力/阈值记录没有达到门槛。不能把所有 plain resume/relax 统称通过。
- WP08 Al 的“最大分量收敛”描述须纠正；本报告未执行真实 Al/MACE 复算，是否自由原子已收敛由主审用保留的实际最终力数组判定。

所选现有测试命令：

```sh
UV_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest -q -p no:cacheprovider --basetemp=/tmp/pyramid-review-pytest-5a1d105 tests/unit/test_ase_adapters.py tests/unit/test_ase_qe.py tests/unit/test_plain_resume.py::test_plain_reference_resume_matches_continuous tests/unit/test_constraints.py::test_unsupported_constraints_rejected_explicitly tests/unit/test_fixatoms.py::test_fixed_layer_never_moves_and_driving_is_projected tests/unit/test_qe_reliability.py::test_warm_start_copies_density_and_records_provenance tests/unit/test_qe_reliability.py::test_nonconvergence_of_identical_input_is_not_retried tests/unit/test_qe_reliability.py::test_timeout_kills_the_whole_process_group tests/unit/test_qe_engine.py -k 'not compute_with_fake_pwx'
```

实际输出：`34 passed, 1 deselected, 10318 warnings in 2.98s`。警告来自 ASE 在 NumPy 2.5 下设置 array.shape 的 DeprecationWarning；这只是断言通过，不能写为 warning-clean。未运行全套测试、MACE/PySCF 或真实 QE 集成。
