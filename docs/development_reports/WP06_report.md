# WP06 回报 — 模型更新、标签消费和回退

## 本次范围

- 工作包／审阅节点：WP06（R2 的一部分）
- 基线提交、分支、当前提交：基线 `6b39ca1`（WP03）＋ `05bb4d6`（WP05）＋ `895897b`（reproducibility 固定）；分支 `dev/0.4.0`；当前提交见文末 SHA。
- Python／ASE／NumPy 及有关后端版本：Python 3.12（uv）；ASE ≥3.29、NumPy ≥1.26；本环境无 torch（MACE 集成测试 slow 标记，未执行）。
- 本次解决的用户问题：模型更新缺少候选—验证—发布/回退协议；更新记录不含父模型/标签集合/训练成本；纯记录回调在 WP01 语义下会触发不必要的重探针与伪更新；标签消费缺少跨进程去重；真实可保存更新器未通过恢复验收。
- 实际实现的功能：
  - `loop/online.py`：`GuardedUpdater`（候选→固定保护集验证→发布或回退，完整状态导出）、`UpdatePolicy`（阈值与数据划分的 Python/dataclass 配置）、`LegacyCallbackAdapter`（`TrainReport|None` → False/True 兼容桥）。原 `OnlineUpdater` 与回调 API 未动。
  - `runtime/models.py`：`ModelRegistry` 不可变工件仓库——每次更新写 `models/<model_id>/state.json`（generation、parent_model_id、label_ids、recipe、training、updater_state），原子写入，内容不同则拒绝覆盖（历史不可改写）。
  - 更新器适配：构造即要求 surrogate 有 `state_dict`/`load_state_dict`，否则拒绝（不满足显式状态接口的组合不给可靠续算，不静默降级）。
  - 标签消费按 durable label ID 去重（重送不重复消费/训练）；`train_on="full_history"` 为 recipe 显式安排的历史复用，不受去重限制。消费队列位置、保护集、报告与拒绝历史全部进状态（经既有 checkpoint 的 updater_state 字段持久化）。
  - 发布路径接 WP01/WP03 代次机制：发布后清缓存、旧响应拒绝、延迟重锚定；工件在 model_update 事件提交前持久化；恢复语义不变（重放只加载已持久化状态，不重跑训练；工件缺失/消费状态缺失/training 失败记录仍以 ResumeError 停止待处理）。
  - 回退路径：训练失败或保护集验证失败→恢复到精确的父状态（参数与历史），记 `update_rejected` 事件（reason/metrics/label_ids），父模型继续；训练成本照常在 task 事件计费（账本只追加）。保护集只取自最早消费的标签（绝不使用未来验收标签），检查有限输出、有限差分能量—力一致性、力误差异常增幅与可选能量漂移；单纯训练 loss 下降不作为放行依据。
  - `surrogate/committee.py` docstring 如实标注：state_dict 含权重/能量偏移/recipe（足以发布与恢复模型工件），不含优化器状态（Adam 矩与训练 RNG），微调续训将重建优化器而非续接。
- 与原计划的偏离及原因：
  - 回退后不停止运行：计划文字为"坏模型、训练失败或保存失败保留可用父模型，记录失败成本和处理结果"。训练失败与验证失败按"记录+父模型继续"实现（运行不中断）；**保存失败**（工件写入异常）仍按 WP03 语义中断运行——无法持久化的运行不可安全续算。
  - 兼容桥把 legacy `None` 映射为 `False`（无变化）：WP01 既有语义里 None 会触发代次递增与重校准（伪更新），正是验收项针对的问题；映射后纯记录回调成本与无回调基线完全一致。
  - `update_record()`/`pop_rejection()` 作为 StatefulUpdater 的可选钩子（getattr 探测），不改动既有 Protocol 方法签名。

## 改动清单

- 新增 `src/pyraimd2/runtime/models.py`：`ModelRegistry`（publish/read）、`ModelRegistryError`、`MODEL_ARTIFACT_FORMAT_VERSION=1`。
- `src/pyraimd2/runtime/events.py`：新增 `UPDATE_REJECTED` 事件类型。
- `src/pyraimd2/runtime/updater.py`：docstring 记录两个可选钩子。
- `src/pyraimd2/runtime/__init__.py`：导出 ModelRegistry/ModelRegistryError。
- `src/pyraimd2/loop/online.py`：新增 `UpdatePolicy`、`GuardedUpdater`、`LegacyCallbackAdapter`（`OnlineUpdater` 原样保留）。
- `src/pyraimd2/loop/__init__.py`：导出新符号（顺带修正存量 I001/RUF022）。
- `src/pyraimd2/loop/energetic.py`：
  - 回调块：发布前捕获 `parent_model_id`，经 `update_record()` 组装完整工件记录（generation/parent/label_ids/recipe/training/updater_state）交 `_model_publisher`；`pop_rejection()` 非空时发 `update_rejected` 事件（幂等键 `update-rejected:<label>:eval-<n>`）；model_update 事件增 `label_ids`。
  - `EnergeticRunner._publish_model_artifact` 改用 `ModelRegistry`（__init__ 与 resume 均建 `_model_registry`）；工件仍含 `updater_state` 键，WP03 重放兼容。
- `src/pyraimd2/surrogate/committee.py`：state_dict docstring 如实标注优化器状态不含。
- 新接口、配置字段和默认值：`UpdatePolicy`（n_label/guard_size/train_on/max_force_growth/force_growth_floor/max_energy_drift/FD 步长与容差）；工件格式版本 1；无配置文件改动（WP04 区域未碰）。
- 旧接口／已有数据的兼容方式：`OnlineUpdater`、`TrainReport`、回调签名、`models/<id>/state.json` 的 `updater_state` 键全部不变；WP03 恢复测试原样通过。
- 是否改变单位、力预算、时间、约束、随机检查或模型切换语义：否（发布/回退走既有代次与锚定机制）。

## 验收证据

```text
验收项：纯记录 callback 不触发不必要的重探针
命令：uv run pytest tests/unit/test_guarded_update.py::test_record_only_callback_triggers_no_unnecessary_reprobes -q
测试：LegacyCallbackAdapter 包 record-only（永远返回 None）跑 12 步，与无回调基线对照
预先确定的通过标准：无 model_update 事件、代次恒 0、reference_calls 与基线逐项相等、逐步 route 一致、回调确实观测到标签
实际关键结果：全部满足（基线成本 = 机制固有校准探针，无一次多余重探针）
状态：通过
```

```text
验收项：同一更新事件不因 label ID 重送而重复训练；recipe 安排的历史复用不受限
命令：uv run pytest tests/unit/test_guarded_update.py -q -k "redelivered or full_history"
测试：同 label_id 连送两次；full_history 模式下重送
通过标准：n_consumed 不重复、finetune 不重复；历史集中每个 label_id 恰好一次
实际关键结果：满足
状态：通过
```

```text
验收项：模型变化后不能接受旧响应
命令：uv run pytest tests/unit/test_guarded_update.py::test_stale_pending_rejected_after_model_change tests/unit/test_guarded_update.py::test_publish_produces_artifact_and_reanchors_with_new_generation -q
测试：代次递增后冻结的旧 pending 提交被拒（ValueError）；发布后下一评估以新代次重锚定（anchor.model_generation==1、segment 更新）
通过标准：如断言
实际关键结果：满足
状态：通过
```

```text
验收项：失败后模型哈希回到已知版本（回退）
命令：uv run pytest tests/unit/test_guarded_update.py -q -k rollback
测试：三条路径——力增幅超限（poison）、能量—力不一致（bias）、训练抛错
通过标准：各一条 update_rejected 事件（reason 分别为 force_growth/energy_force_inconsistent/training_failed）；无 model_update；模型 state_dict 精确回到父版本 {k:0.8, bias:0, finetune_calls:0}；label_consumed 事件的 updater_state 记录拒绝；训练 task 照常计费
实际关键结果：满足
状态：通过
```

```text
验收项：至少一个可保存的轻量更新器通过全过程恢复测试
命令：uv run pytest tests/unit/test_guarded_update.py -q -k "resume"
测试：GuardedUpdater＋解析可训练势：连续 60 vs 30＋新对象恢复 30；以及含 2 次回退的 40 步窗口恢复（数据驱动的触发计划，跨回退单调推进）
通过标准：逐步 route/check_draw/label_id/位置动量(1e-12)一致；updater n_consumed/n_updates/n_rejected 与模型 k 严格一致
实际关键结果：满足
状态：通过
```

```text
验收项：MACE 真实微调小型集成
命令：uv run pytest tests/unit/test_committee_guarded_update.py --runslow -q
测试：小委员会（n_members=2, epochs=2）经 GuardedUpdater 发布/回退与状态往返
预先确定的通过标准：发布或回退路径可执行、状态往返后消费位置连续
实际关键结果：未执行——本环境无 torch/mace（测试以 slow 标记，importorskip 保护）
状态：未执行
原因与后续处理：在有 MACE 的环境跑 --runslow；优化器状态不含在委员会 state_dict 中（已如实标注），微调续训重建优化器
```

```text
验收项：受影响套件整体回归
命令：uv run pytest tests/unit tests/test_smoke.py -q
通过标准：全部通过
实际关键结果：297 passed, 1 deselected(slow)（基线 283 + 新增 14）；ruff 全仓 11 个错误（基线 13；本区域 loop/__init__.py 的两个存量问题顺带修正，新增 0）
状态：通过
```

## 恢复与随机状态（相关工作包必填）

- 连续 vs 分段恢复：60 步 vs 30＋30（GuardedUpdater），位置/动量 ≤1e-12，整数与检查序列严格一致。
- 模型身份不匹配、破损快照或缺工件的处理：沿用 WP03 四道校验（fingerprint/模型链/updater/工件），ResumeError 停止待处理；本 WP 的工件格式向后兼容 WP03 重放。
- 重复 label／probe／update 的处理：durable label ID 去重在更新器内部生效（重送不重复消费）；append_once 键在事件侧生效；回退不重训（候选每次从头微调）。
- 已完成但未应用的计算如何计费、保存和复用：训练尝试（含失败）的墙钟在 training task 事件计费（append-only）；被拒绝候选不发布、不进入模型链。

## 算力与成本

- 新增实际参考执行总数：0（全部解析假后端；MACE 集成未执行）。
- 预算使用与剩余额度：WP08 配额未动用。

## 回归与交付

- 受影响测试通过情况：tests/unit + test_smoke 全绿（297，另有 1 个 slow 未执行）。
- 原有 energetic／legacy switching 示例：未改动；`OnlineUpdater` 与其 replay 路径零改动。
- wheel 安装及仓库外 CLI 测试：未执行（WP09）。
- 最小依赖是否仍不导入 Torch／PySCF：是。
- 用户可复制的完整运行命令：`GuardedUpdater(model, UpdatePolicy(...))` 作为 `EnergeticRunner(..., on_label=...)` 的更新器；恢复用 `EnergeticRunner.resume(dir, model, engine, updater=GuardedUpdater(...))`。
- README／示例／支持矩阵更新位置：未更新（WP09）。
- 已知故障、未执行验证与风险：
  - MACE/委员会真实微调未在本环境执行（slow 标记）；其优化器状态不可续接已如实标注。
  - 保护集内联于更新器状态（原子坐标与参考力），大体系长运行时状态体积随之增长；WP09 前如需可在工件中改为 label_id 引用。
  - `LegacyCallbackAdapter` 包无状态回调时按"不可恢复"处理（恢复时拒绝），属既定语义。
- 需要负责人判断的具体决策及备选方案：回退后继续运行 vs 停止（本 WP 选择继续，保存失败除外——见偏离）。
- 下一工作包及其入口：WP07——workflows/Runner/约束处理可复用 `GuardedUpdater` 与 `UpdatePolicy`；WP09 打包时委员会 slow 集成需在 MACE 矩阵环境执行并记录结果。
