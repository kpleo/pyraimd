# 0.4.0rc1 独立审阅修复回报——R3 在线更新

- 日期：2026-09-09
- 审阅依据：`docs/development_reports/INDEPENDENT_REVIEW_20260909.md` §3-R3（核心 F05/F06/F08/F09）与 §7 停止条件相关条目
- 基线 commit：`842242d`（阶段 1 全部修复合入，448 测试全绿）
- 修复 commit：与本报告同批的 `Fix:` 前缀提交（hash 见 git log）
- 分工说明：本批只覆盖 R3。文件区域：`loop/online.py`、`runtime/models.py`、`surrogate/committee.py`、`loop/energetic.py` 的最小钩子、`tests/unit/`、`docs/development_reports/`。未触碰 engines/、workflows/、store/、runtime/ 其他文件、WP08_report.md、README/CHANGELOG/examples/、reproducibility/、analysis/、manuscript/。
- 新增实际 SCF 数：0（全部解析模型 hermetic 验证）
- 真实后端验证文件路径：无（tensor 持久化的真实 MACE 集成为后续云端任务，见保留限制）

## R3-1. 训练/保护集丢失晶胞、PBC、电荷和磁矩（P1，F05）

### 具体原因

`GuardedUpdater` 的标签 payload 只序列化 numbers+positions；训练集、保护集和差分探针重建的 Atoms 全部退化为零晶胞、非周期、零电荷/磁矩，约束与标签能量口径也丢失——进入 finetune 的物理结构已经不是被标记的结构。

### 行为改变

- `_label_payload` 完整序列化：cell、pbc、masses、initial_charges、initial_magmoms、info（假定为 JSON 安全，否则在落盘时显式失败）、FixAtoms 约束（其余约束类型按既定语义显式拒绝）、`energy_kind` 与 `force_consistent`。
- `_payload_atoms` 按 payload 重建完整结构；旧版 payload（仅 numbers/positions）仍可读取，语义与当时记录的信息一致。
- 保护集评估与差分探针都从同一 payload 重建，探针用 `atoms.copy()` 改坐标，结构字段不再丢失。

### 反例前后对照（3 Å 立方晶胞、PBC=true、电荷 ±0.2、磁矩 ±1 的双原子结构）

- 前：finetune/保护集/探针看到的结构为 (volume=0.0, pbc=False, charge=[0,0], magmom=[0,0])，候选仍发布。
- 后：全部看到 (volume=27.0, pbc=True, charge=[0.2,−0.2], magmom=[1,−1])；payload 本身带 constraint indices 与 energy_kind/force_consistent。测试模型是能量依赖 cell 体积与电荷的解析模型（审阅指出单原子位置谐势无法发现此类问题）。

回归：`tests/unit/test_review_r3.py::test_training_guard_and_fd_probes_keep_the_physical_structure`。

## R3-2. 能量—力差分方向是整体刚体平移（P1，F06）

### 具体原因

保护集一致性检查的差分方向是 `ones/sqrt(N)`——整体刚体平移，任何合理势能的零模：平移不变势的力矢量和恒为零，差分检查在结构上无法发现"力不是能量的梯度"。审阅反例：力乘 1.1、能量不变的候选通过检查并发布，内部坐标 |dE/dx + F_x| ≈ 0.1。

### 行为改变

- 差分方向改为少量内部方向：相对坐标（呼吸）模 + 由结构内容播种的可复现伪随机方向（投影掉平移分量），逐方向取最差一致性。单原子无内部模，平移本身即物理坐标，直接检查。
- 差分能量与一致性值都检查有限性：非有限差分能量按无穷大不一致处理，候选被拒绝。
- 该检查是诊断门槛，不是全局保守性的数学证明（文档与代码注释均按此口径书写）。

### 反例前后对照

- 前：平移不变双原子势、finetune 后力 ×1.1 而能量不变——候选**发布**（保护集力误差 0.041 < 0.05 地板，平移方向一致性恰为 0）。
- 后：同一候选以 `energy_force_inconsistent` 被拒绝并回滚（内部方向一致性 ≈ 0.1）；力一致的候选（scale=1.0）照常发布。

回归：`test_fd_check_rejects_force_scaled_translation_invariant_candidate`、`test_fd_check_accepts_force_consistent_candidate`。

## R3-3. 验证或落盘异常未完整回滚（P2，F08）

### 具体原因

finetune 成功后，保护集评估若抛错（候选无法预测保护几何以至验证中断），异常直接逃出，内存模型停留在候选态；工件保存（OSError）发生在代次递增之后、发布事件提交之前——模型与代次已改变而没有发布事件，停止时的状态不满足原子发布契约。

### 行为改变

- `_attempt_update` 把候选评估与验证整体包进失败域：任何验证异常都回滚到父模型并记 `validation_failed` 拒绝（与 training_failed 同一原子语义）。
- 发布改为"候选隔离 → 验证 → 持久化 → 激活"：工件先持久化（返回实际存储的占位形式 payload，供提交摘要绑定），提交事件后，代次才递增；持久化或提交任何一步抛错，调用 `GuardedUpdater.rollback_accepted()` 恢复父模型、计数与 pending 队列后再抛出（runner 停止，但停止状态满足原子发布契约）。不支持回滚的旧式 updater 保持原行为（异常直接传播）。

### 反例前后对照

- 验证崩溃：前——模型停留在候选态（k=99）且无拒绝记录；后——k 恢复 0.8、`validation_failed` 入册、消费计数与历史一致。
- 工件保存 OSError：前——代次已 +1、模型为候选、无 MODEL_UPDATE 事件；后——代次不变、模型与 updater 完全回到父态（n_updates=0、pending 保留该标签）、无提交事件。

回归：`test_candidate_validation_crash_rolls_back_to_parent`、`test_artifact_save_failure_rolls_back_publish`。

## R3-4. 真实 tensor 状态持久化（P2，F09，静态缺口）

### 具体原因

`CommitteeSurrogate.state_dict()` 返回 torch tensor，而事件（LABEL_CONSUMED/MODEL_UPDATE 的 updater_state）、checkpoint `state.json`、模型工件 `state.json` 都用原始 `json.dumps`——含 tensor 的状态在任何一个落盘边界都会 TypeError。内存 dict 往返测试证明不了落盘可用。

### 行为改变（tensor-artifact-v1 约定）

- `runtime/models.py` 新增与 torch 无关的序列化层：`dump_state_arrays`/`load_state_arrays` 递归遍历状态，把 NumPy 数组或以 `detach().cpu().numpy()` 鸭子类型识别的 tensor 换成 JSON 占位（`{"__ndarray__": key, "sha256", "dtype", "shape"}`）；加载时逐数组校验 sha256，篡改显式失败。torch 从未被导入。
- 三处边界的侧车约定：模型工件用自带 `state_arrays.npz`（键 arr0..arrN，工件目录自包含，`ModelRegistry.publish` 返回占位形式 payload，`read(resolve=True)` 校验解析）；checkpoint 的 updater 状态数组搭自身 `arrays.npz`（`updater_state:arrN` 键，由 `_checkpoint_payload` 接线，resume/fork 解析）；事件 payload 用内容寻址存储 `models/state-arrays/<sha256>.npz`（同内容只存一次，重复消费零成本；重放时经同一存储解析）。
- `CommitteeSurrogate.load_state_dict` 兼容 NumPy 输入（`torch.as_tensor` 覆盖内存 tensor 与磁盘数组两路）。
- 工件摘要绑定占位形式（sha256 内含），提交摘要、checkpoint manifest 与重放校验同一形式，R2 的完整性链不变。

### 反例前后对照（NumPy 数组模型代替 tensor，序列化路径相同）

- 前：含数组的 updater_state 在事件写入处 `json.dumps` TypeError（任何含 tensor 的消费都无法记录）。
- 后：发布工件 `state.json` 全为占位 + `state_arrays.npz` 落盘；MODEL_UPDATE 事件摘要与工件一致；LABEL_CONSUMED 事件两次相同状态只产生一个数组文件；全新进程 resume 重放后 updater 的数组逐值恢复（连续/新进程对照）；侧车摘要被篡改时加载显式拒绝。

回归：`test_tensor_state_persists_through_artifact_and_replay`、`test_consumed_event_state_arrays_dedupe_and_replay`、`test_state_array_roundtrip_and_tamper_detection`；`tests/unit/test_models.py` 适配 `publish` 返回存储 payload。

### 真实 MACE 集成（如实标注：未执行）

真实 torch tensor 的端到端验证（一次标签消费 → 更新 → 落盘 → 新进程恢复）需在带 MACE 的环境执行一次小集成，为后续云端任务。本批交付：代码路径、tensor-artifact-v1 格式约定（上文三处边界）、`CommitteeSurrogate` 的双路加载兼容。本机无 torch，本报告不把此项写成已实测。

## 验证

- `uv run pytest tests/unit tests/test_smoke.py -q`：456 passed, 1 deselected（基线 448 → ＋8 项回归）。
- `uv run ruff check <改动文件>`：无新增错误（energetic.py TRY004 为存量基线，行号随内容平移）。
- 最小核心仍只依赖 NumPy＋ASE＋标准库；torch 不是必需导入。
- 反例前后证据：/tmp 独立脚本在修复前后各跑一遍（F05 字段 0.0→27.0/False→True、F06 发布→拒绝、F08 状态污染→一致、F09 TypeError→占位落盘），回归测试逐条固化。

## 保留限制

- 真实 MACE tensor 集成未执行（云端任务；格式约定与双路加载已就位）。
- 差分方向是诊断门槛，非全局保守性证明；不穷举全部自由度。
- 事件侧车存储按内容寻址、只增不改；磁盘配额下的长期清理策略未定义（与 events.jsonl 同为追加型运行记录）。
- 旧式（无状态）updater 的发布失败回滚不可用，异常直接传播（既定语义：无状态导出的运行本来不可恢复）。
- updater state 的 `info` 字段假定 JSON 安全；非 JSON 值在落盘时显式失败而非静默丢弃。
