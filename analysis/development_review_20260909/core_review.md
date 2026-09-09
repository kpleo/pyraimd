> 归档说明：以下为独立分项复核记录，代码位置针对 5a1d105。原复核仅写入临时目录；主审已将脚本及本报告归档于当前目录，并将脚本依赖改为可迁移的相对位置。文中临时目录是原始执行的溯源信息；交付的同名日志来自归档脚本的复核执行。总体验收与优先级以 docs/development_reports/INDEPENDENT_REVIEW_20260909.md 为准。

Pyramid 新实现独立审阅：energetic 恢复与模型更新

审阅基线：`/Users/pengkang/Research/Project/Project_PYRAIMD/pyraimd2`，`dev/0.4.0`，`5a1d1054d7cdd9444bdd85a62afeba6a5fc607d5`。本次没有使用 `.release/pyramid`。未修改任何仓库文件、暂存内容或提交；反例脚本与最终证据均放在 `/tmp`。使用现有 `uv run --no-sync`，实测 ASE 3.29.0 / NumPy 2.5.2，无 torch；没有安装、云计算或大套件运行。未审查 CLI、普通 MD、QE parser 或 workflow/export。

结论：开发者已经实现可工作的正常路径，但 WP09 对 WP02/WP03/WP06“完成”的概括超出了实际保障。应修正下面的已确认问题后重新验收恢复和更新。主要风险是故障窗口中 RNG、标签身份、数据库记录和模型工件不再对应同一条逻辑历史；不是简单增加测试数量能解决的问题。这里的 P1/P2 是缺陷严重性，不等同于计划的 P0 工作包优先级。

正向对照也做了：冻结模型连续 4 步，与 2 步 checkpoint 后启动真正独立 Python 进程再跑 2 步，位置、完整步动量逐位相同，检查序列相同。重复读取 energy/forces 不增加 evaluation；checked accepted 行的 driving force 保留候选值，与参考力不同，未被参考力事后替换。因此没有把正常冻结力路径本身列为缺陷。

以下 F01–F08 为实际运行极小解析反例确认；F09 为可定位的静态接口缺陷，未冒充真实 MACE 集成实测。

1. **[P1，F01] 未提交检查提议恢复后，RNG 仍停在该抽样之前。**

   - 位置：[energetic.py:676](../../src/pyraimd2/loop/energetic.py:676)（676–679 只对 committed accepted 事件推进 RNG）；[energetic.py:1815](../../src/pyraimd2/loop/energetic.py:1815)（1815–1832 重建 tail pending，却不恢复抽样后 RNG）；[energetic.py:1108](../../src/pyraimd2/loop/energetic.py:1108)（原运行抽样位置）。
   - 触发：checkpoint 在初始完整边界；下一个 accepted evaluation 已持久化 proposal/check_draw，但参考检查抛错，尚未提交 evaluation；从 checkpoint 续算。
   - 实测：seed=2、p=0.5，连续运行前三次 draw 是 `[0.2616121342493164, 0.2984911434141233, 0.8142257405942803]`；恢复变成 `[0.2616121342493164, 0.2616121342493164, 0.2984911434141233]`。真正独立 Python 进程也重现。第三次检查从不执行变为执行。
   - 后果：当前失败检查虽复用了原 draw，下一检查却重复该随机数；检查流、参考调用、后续标签/训练/模型链可能改变。不能宣称恢复了同一独立检查随机流。
   - 修复：在 proposal 中持久化抽样后 bit-generator state，恢复 pending 时应用；或严格区分 committed/tail 的已消费 draw 并恰好推进一次。补“选中检查后参考失败→新进程恢复→继续至少两个 accepted evaluation”的反例，核对后续 draw，而不只核对失败点自身 draw。
   - 复现：`uv run --no-sync python analysis/development_review_20260909/pyramid_core_review.py rng`；跨进程版 `analysis/development_review_20260909/pyramid_core_process.py`。

2. **[P1，F02] checkpoint 恢复丢失标签缓存，能改变消费与训练，不只增加 SCF 成本。**

   - 位置：[energetic.py:414](../../src/pyraimd2/loop/energetic.py:414)（恢复也新建空 LabelCache）；[energetic.py:516](../../src/pyraimd2/loop/energetic.py:516)（checkpoint payload 未保存缓存/标签键索引）；[energetic.py:1166](../../src/pyraimd2/loop/energetic.py:1166)（miss 后获取新 label ID）；[online.py:217](../../src/pyraimd2/loop/online.py:217)（消费只按 label ID 去重）。
   - 触发：静止解析体系，固定 x 方向探针，p=1；GuardedUpdater 的 n_label=2。初始标签消费后写 checkpoint，续算一个同构型 accepted evaluation。
   - 实测：连续分支命中原标签，`n_consumed=1, n_updates=0, k=0.8`；恢复分支重新参考并获新 label ID，`n_consumed=2, n_updates=1, k=0.81`。
   - 后果：即使干净停机、没有损坏文件，resume 也改变模型链。WP02/WP03 报告中“缓存冷启动只影响成本，不影响正确性”的判断在接入 GuardedUpdater 后不成立。
   - 修复：恢复经参考身份/构型/口径验证的 durable label 索引和原 label ID；可以从持久标签记录重建，而非必须把整个内存缓存塞入 checkpoint。区分物理执行 attempt 与逻辑 label 身份。验收比较消费 ID、训练次数和 model_id，不能只比这个静止反例中恰好不变的位置。
   - 复现：`uv run --no-sync python analysis/development_review_20260909/pyramid_core_more.py cachetrain`。

3. **[P1，F03] SQLite 行先于 authoritative commit 事件落盘，恢复会重复写该 evaluation，并读取旧行。**

   - 位置：[energetic.py:1233](../../src/pyraimd2/loop/energetic.py:1233)（1233–1284，先 append 数据库，再发 evaluation_committed）；[store.py:81](../../src/pyraimd2/store/store.py:81)（无 evaluation 唯一/幂等约束）；[store.py:197](../../src/pyraimd2/store/store.py:197)（按步读第一条）；[energetic.py:1510](../../src/pyraimd2/loop/energetic.py:1510)（重放依赖上述按步读取）。
   - 触发：SQLite append 已成功，evaluation_committed 尚未 append 时中断。反例在该 event 写入入口故障注入；恢复后完成同一 pending。
   - 实测：同一 `(run_id=r, step=0)` 出现两行，分别携带 `r-label-6` 和 `r-label-7`；唯一 committed event 携带 `r-label-7`，而 `_row_at_step` 返回 `r-label-6`。
   - 后果：权威事件和重放所读 payload 来自不同执行，逻辑 evaluation 不再 exactly-once；标签谱系和元数据已实际不一致。解析反例的力值相同，不据此声称已经测到错误轨迹；非逐位确定后端的两次参考结果还可能不同，扩大影响。
   - 修复：明确一个提交协议：event 绑定唯一 row ID/内容摘要，数据库以 evaluation ID 幂等存储并在恢复时对账孤立行；或让事件包含足够 payload、数据库仅为可重建视图。不能只给 JSONL 加 append_once，也不能简单取“第一行/最后一行”。补 SQLite 成功后至 event fsync 前的故障注入及重复恢复。
   - 复现：`uv run --no-sync python analysis/development_review_20260909/pyramid_core_review.py row`。

4. **[P1，F04] 模型工件只有 JSON 可解析检查，损坏的权重可以沿用原 model_id 静默加载。**

   - 位置：[models.py:39](../../src/pyraimd2/runtime/models.py:39)（工件未产生受事件保护的内容摘要）；[energetic.py:1453](../../src/pyraimd2/loop/energetic.py:1453)（只 json.loads）；[energetic.py:717](../../src/pyraimd2/loop/energetic.py:717)（717–723 直接加载 artifact.updater_state）。
   - 触发：checkpoint 后已有一个持久化 model_update；其工件发生仍能解析为 JSON 的内容损坏/误覆盖，然后恢复。
   - 实测：仅在 `/tmp` 测试工件里将 `updater_state.surrogate.k` 从 0.81 改为 9.9，保留事件原文；resume 成功，加载 k=9.9，model_id 仍为 `review-harmonic-v1#g1`。model_update 事件内其实还保存着 k=0.81，恢复没有发现二者矛盾。
   - 后果：模型 ID 不能保证对应实际模型内容，“同一模型链”检查可接受不同模型。当前不可覆盖检查只能约束通过 publish() 再次写入，不能验证读取时的工件完整性。
   - 修复：发布时计算工件内容 digest，把它连同 model_id/parent/generation 固定写入提交事件和 checkpoint；加载前校验摘要、schema、身份和链关系。工件与事件携带状态不一致时明确拒绝，不能任选其一加载。
   - 复现：`uv run --no-sync python analysis/development_review_20260909/pyramid_core_more.py artifact`。这是故障注入，仅改测试工件，未修改仓库中的任何模型。

5. **[P1，F05] GuardedUpdater 重建训练/保护集结构时丢失晶胞、PBC、电荷和磁矩。**

   - 位置：[online.py:154](../../src/pyraimd2/loop/online.py:154)（154–169 只保存 numbers/positions 和 E/F）；[online.py:235](../../src/pyraimd2/loop/online.py:235)、[online.py:307](../../src/pyraimd2/loop/online.py:307)（保护集和训练均使用丢失身份后的 Atoms）；[online.py:256](../../src/pyraimd2/loop/online.py:256)（FD 探针再次只重建 numbers/positions）。
   - 触发：周期或带电/磁矩结构产生一个训练标签并交给 GuardedUpdater；无需崩溃，即使当次训练也经过这套 payload 转换。
   - 实测：原 Atoms 为 3×3×3 Å 晶胞、PBC 全 true、初始电荷 0.2、磁矩 1；finetune 实际接收到零晶胞、PBC 全 false、电荷/磁矩全零。候选仍发布成功。
   - 后果：周期参考的标签被用于训练孤立结构；周期邻居关系和电子态输入改变，保护集也在错误物理结构上评分。解析位置势不依赖这些字段，正好解释现有解析测试为什么漏检；没有运行 MACE 来声称其具体数值损失。
   - 修复：使用完整且版本化的 Atoms 序列化，保存 cell/PBC/相关 arrays/info 与约束身份；训练、保护集、FD 扰动都从相同原结构复制。对无法保留的后端必需输入应预检拒绝。验收用确实依赖 cell/PBC/charge 的小解析模型。
   - 复现：`uv run --no-sync python analysis/development_review_20260909/pyramid_core_review.py cell`。

6. **[P1，F06] 候选能量—力检查只沿全体原子的刚体平移，能放过明显不一致的分子势。**

   - 位置：[online.py:250](../../src/pyraimd2/loop/online.py:250)（250–261，`direction = ones_like(positions)`）；[online.py:270](../../src/pyraimd2/loop/online.py:270)（据此通过/拒绝）。
   - 触发：平移不变二原子势，更新只把力乘以 1.1，能量不变，且参考力为原力的 1.2 倍，因此保护集力误差反而降低。
   - 实测：候选 `published=True`；相对原子位移的独立中心差分得到 `|dE/dx + F_x| = 0.10000000000038778`。当前检查沿整体平移，两端能量不变、总力为零，给出零一致性误差。
   - 后果：常见内部相互作用体系中，这个保护项只测净力，无法实现所声称的基本能量—力一致性筛查；明显不一致的候选可以进入依赖保守能量的 energetic work/response 路径。
   - 修复：在保护集上采用明确记录的内部相对位移方向/若干坐标分量，或受控随机方向并保存 RNG/固定方向；避免所有方向都落入平移零模。FD 中的能量和误差也须做 finite 检查。补二原子反例，不应只用单原子外部谐势检验此项。
   - 复现：`uv run --no-sync python analysis/development_review_20260909/pyramid_core_guard.py`。

7. **[P2，F07] 已消费缓存标签再次出现在 checkpoint 后窗口，会被误判为未消费而拒绝恢复。**

   - 位置：[energetic.py:1367](../../src/pyraimd2/loop/energetic.py:1367)（consumed event 仅以 label_id 去重）；[energetic.py:1488](../../src/pyraimd2/loop/energetic.py:1488)（窗口从空 unconsumed 集合开始）；[energetic.py:1515](../../src/pyraimd2/loop/energetic.py:1515)（1515–1524 对每条带标签 commit 重新等待 consumed/update event）。
   - 触发：初始标签已消费并覆盖进 checkpoint；之后静止构型 p=1 的两个检查都复用该 label ID；GuardedUpdater 正确返回 False，但 append_once 抑制了重复 label_consumed 事件。此时中断且没有新 checkpoint。
   - 实测：updater.n_consumed 一直为 1，两次 cache_hit 都是 `r-label-1`；resume 抛出 `labels ['r-label-1'] were committed but their consumption state was never persisted`。
   - 后果：正常且全部有记录的运行无法自动续算，误导用户按“消费丢失”处理。与 F02 不同，本例在恢复开始前就失败，尚未发生缓存冷启动后的新执行。
   - 修复：区分“evaluation 使用标签”与“首次消费标签”；重放应从 checkpoint 恢复已消费身份，或每个 evaluation 保存明确的消费/no-op 结果而不重复训练。不能要求缓存复用每次再产生首次消费事件。
   - 复现：`uv run --no-sync python analysis/development_review_20260909/pyramid_core_review.py cache`。

8. **[P2，F08] 候选验证异常与发布保存异常未覆盖回滚，失败后内存仍是未发布模型。**

   - 位置：[online.py:303](../../src/pyraimd2/loop/online.py:303)（303–321，仅 finetune 异常在 try/except 内；candidate _evaluate/_validate 抛错不恢复父状态）；[energetic.py:1337](../../src/pyraimd2/loop/energetic.py:1337)（1337–1354，模型已修改、generation 已递增，才保存工件）。
   - 触发 A：训练成功修改 k，随后候选在保护集 predict 时抛错。实测父状态 `{k:0.8,calls:0}` 变成 `{k:0.81,calls:1}` 留在对象中，n_rejected=0，没有回退。
   - 触发 B：候选通过保护集，但 model publisher 抛 OSError。实测模型同样留在 k=0.81，generation=#g1、updater.n_updates=1，工件数=0、model_update 事件数=0。
   - 后果：父模型继续可用的契约没有成立；调用者手里留下一个无法与持久化历史对应的模型/更新器。**Runner 确实停止，没有证据表明它继续用这个坏状态积分，所以这里定 P2，未夸大为静默轨迹污染。** WP06 允许“保存失败就停机”的设计偏离，不等于允许丢失内存父状态。
   - 修复：候选训练、保护集推理、数值验证、序列化和 publish 组成覆盖全部异常的事务；在工件和激活事件持久化成功前不提交 live generation/active model。失败恢复模型及更新器的受影响状态，记录失败尝试/拒绝原因。若采用候选副本，父对象应保持可用。
   - 复现：`uv run --no-sync python analysis/development_review_20260909/pyramid_core_review.py validation publish`。

9. **[P2，F09；静态确认，真实后端未运行] CommitteeSurrogate 的状态格式不能写入当前 JSON 持久化通道。**

   - 位置：[committee.py:407](../../src/pyraimd2/surrogate/committee.py:407)（407–409 保留 `torch.Tensor`）；[online.py:367](../../src/pyraimd2/loop/online.py:367)（原样嵌入 updater state）；[events.py:115](../../src/pyraimd2/runtime/events.py:115)、[models.py:66](../../src/pyraimd2/runtime/models.py:66)（无 serializer 的 json.dumps）。
   - 触发：CommitteeSurrogate + GuardedUpdater + 有事件日志的 EnergeticRunner。即使 n_label 尚未达到训练门槛，首次 label_consumed 都会尝试把模型 tensor 状态写入 JSON；更新/保存 checkpoint 同样走不兼容通道。
   - 后果：将触发 tensor 不可 JSON 序列化错误，无法完成宣称的持久化发布/恢复链；这不是“未保存 Adam 状态”的那项已披露限制。
   - 证据边界：本机无 torch/MACE，未执行真实模型；上述为代码类型与序列化调用的静态闭合证据。现有 [test_committee_guarded_update.py:42](../../tests/unit/test_committee_guarded_update.py:42) 只把内存 dict 交给另一个对象 load_state_dict，既不经过 JSON/模型工件，也不经过 EnergeticRunner.resume，不能覆盖此接口。
   - 修复：将权重存为有内容摘要的 tensor 工件，JSON 中只存版本、recipe、索引及工件引用；或提供显式安全的序列化适配器。在真实 MACE 环境补磁盘保存→新进程加载→后续更新的最小集成，不能用内存 dict roundtrip 代替。

补充观察与证据缺口（不计入上面已运行反例的结论）：

- 已实测在 events.jsonl 尾部加入一条未写完的 JSON 后，即使上一 checkpoint 有效，EventLog 也在 [events.py:143](../../src/pyraimd2/runtime/events.py:143) 拒绝打开。当前选择属于保守拒绝；未把它混同为静默错误。若“崩溃恢复”包括 append 中途进程终止，仍需明确最后未提交尾记录的识别、保留诊断和修复协议，不能只验 checkpoint 截断。复现 `analysis/development_review_20260909/pyramid_core_more.py torn`。
- 还有一次“更新后重校准已完成、tail proposal 已写、evaluation 未提交”的窗口：恢复完成该 tail 后 n_calibrations=1、segment=1，但 pending 携带的 anchor 来自第 2 次校准，deferred_origin 仍为 1；下一评估又处理该 origin。此反例最终计数追回且成功探针得到复用，未据此上报错误轨迹。需要执行者补充重校准完成状态的单独恢复验证；脚本 `analysis/development_review_20260909/pyramid_core_more.py recalibration`。
- Committee 的每次 finetune 在源码中重新创建 Adam 和按 seed 初始化训练抽样，这与“在训练调用之间恢复”的语义可以兼容；本审阅没有把缺 optimizer snapshot 单独认定为已发生的训练 RNG 缺陷。训练中断恢复、真实模型完整 recipe/权重身份、后续训练逐值一致性没有实际后端证据。
- 没有进行操作系统强杀/断电持久性实验，没有验证网络文件系统的 fsync/rename 语义；已报告的持久化窗口通过调用点异常与测试工件故障注入重现，不能扩写成这些平台都已实测。
- 不重复主审负责的 half-step/未完成步导出审查。本次只在恢复路径使用 Store，F03 是持久化/重放对账问题。

复现材料：

- [基础极小反例](pyramid_core_review.py)：RNG、已消费缓存标签、DB/event 提交窗口、保存/验证异常、训练结构。
- [补充恢复反例](pyramid_core_more.py)：模型工件损坏、缓存冷启动触发训练、重校准窗口、事件尾截断。
- [二原子能量—力反例](pyramid_core_guard.py)。
- [真正跨进程的正常对照和 RNG 反例](pyramid_core_process.py)。
- 运行命令前加 `PYTHONDONTWRITEBYTECODE=1`；cwd 为本次审阅仓库，全部使用 `uv run --no-sync python /tmp/<script>`。每个脚本只跑 0–4 步解析小体系，默认在 `/tmp/pyramid-core-review-*` 生成隔离运行目录。
- 基础证据目录：`/tmp/pyramid-core-review-xks2eoik`；补充证据目录：`/tmp/pyramid-core-review-3r4glj79`；跨进程证据目录：`/tmp/pyramid-core-review-6cxxad2u`。

建议执行顺序：先共同修复 F01/F02/F03/F07 的恢复事件与标签协议，再补 F04 模型内容校验和 F08 发布事务；随后修 F05/F06 的训练结构与保护集有效性，最后把 F09 接入真实持久化集成。以这些具体反例作为回归验收，逐项报告修改后的检查流、模型链和对应事件/工件，不以“原套件仍全绿”替代。
