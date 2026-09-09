# WP09 回报（第一阶段）— 文档、打包、兼容与 CI

本阶段只执行不依赖 WP08 材料结果的部分。支持矩阵的材料验证列、0.4.0 完成
条件逐项核对与候选发布决定留待 WP08 结果返回后进行，本报告对应条目一律
写"未执行"。

## 本次范围

- 工作包／审阅节点：WP09 第一阶段（R3 的可先行部分）
- 基线提交、分支、当前提交：基线 `3f2f5b8`（dev/0.4.0，含 WP00–WP07，383
  测试全绿）；分支 `dev/0.4.0-wp09`（worktree `pyraimd2-wp09`）；当前提交
  `9ee04a3`（本报告在其后单独提交）。
- Python／ASE／NumPy 及有关后端版本：本机验证 Python 3.12.13 与 3.13.12；
  ASE 3.29.0（pyproject 声明下限 `>=3.29.0`、uv.lock 锁定版与 PyPI 当前
  最新版三者目前相同）；NumPy 2.5.3；本机无 torch/pyscf，全部验收为
  builtin 解析后端与打包产物。
- 本次解决的用户问题：README 按模块而非用户任务组织、支持口径不分级；
  版本存在 pyproject 字面值、`__init__` 字面值与测试硬编码多处来源；无
  wheel/sdist 安装验证；CI 只有单 Python 单命令；无 API 参考文档。
- 实际实现的功能：
  - README 按用户任务重组：5 分钟入门（`examples/harmonic_adaptive`
    真实流程，命令逐条验证）、周期材料（指向 `qe_mace_skeleton`，如实
    标注需要 QE＋MACE 环境）、停止/续算/恢复、接自己的 ASE 计算器
    （Python 适配器与 entry-point 插件两条路）、开发新方法；支持状态分
    三级（接口可接入／已完成软件集成测试／已有材料验证），材料验证列
    留"待补充"占位。
  - 版本单一来源：`pyproject.toml` version = `0.4.0.dev0`（候选前开发
    标记）；`pyraimd2.__version__` 经 `importlib.metadata` 读取，源码树
    无安装元数据时回退字面值。
  - CHANGELOG 新增 `0.4.0 (in development)` 小节，按 WP00–WP07 报告归纳
    新增／修复／变更与兼容说明。
  - wheel/sdist 构建并在仓库以外干净 venv 验证：安装、`pyramid
    --version`/`--help`、harmonic 示例全流程（validate/run/inspect/
    resume/export）、核心安装无 torch/pyscf、示例插件可装可用。
  - CI 三个 job：core（Python 3.12/3.13 × ASE 固定最低版/最新版矩阵＋无
    torch/pyscf 断言＋确切版本记录＋核心套件＋energetic_loop 示例）、
    resume-regression（恢复相关 23 项独立成 job）、wheel（构建→仓库外
    安装 wheel 与 sdist→CLI 冒烟→隔离断言）。
  - `docs/api.md`：workflows/backends/engines/surrogate/runtime 公共入口
    参考（附 config/energetics/loop 指引），不引入 sphinx。
- 与原计划的偏离及原因：
  1. **触碰了文件区域外的三个文件**：`src/pyraimd2/__init__.py`
     （`__version__` 机制——版本单一来源是 WP09 明确条目，只能在这里
     实现）；`tests/test_smoke.py` 与 `tests/unit/test_cli.py` 中两处
     硬编码 `"0.3.0"` 的版本断言（版本前进必然使其失败；最小改动为
     断言与安装元数据一致，这本身是"单一来源"的测试化，语义不弱化）。
     其余 src/、tests/ 文件未动。
  2. **ASE 矩阵目前塌缩为一个版本**：3.29.0 既是声明下限也是 PyPI 最新
     版。矩阵机制保留（固定腿 `ase==3.29.0` vs 浮动腿 `latest`，确切
     版本在 CI 日志记录），新版 ASE 发布后浮动腿自动前移。
  3. **GitHub Actions 无法在本机运行**：YAML 已解析校验结构；每条命令
     均在本机逐项等价验证（证据见下），未上真实 runner。
  4. `docs/hpc.md` 未动：它是内部运维记录而非用户文档，README 未链接。

## 改动清单

- `README.md`（重写）：任务导向首页＋三级支持状态＋安装段（核心只带
  NumPy＋ASE，可选后端单独安装且惰性导入）。
- `CHANGELOG.md`：新增 `0.4.0 (in development)`。
- `pyproject.toml`：`version = "0.3.0"` → `"0.4.0.dev0"`（唯一来源）。
- `src/pyraimd2/__init__.py`（区域外例外，见偏离 1）：`__version__` 改
  为 `importlib.metadata.version("pyraimd2")`，`PackageNotFoundError`
  回退 `"0.4.0.dev0"`；`main()` 不变。
- `tests/test_smoke.py`、`tests/unit/test_cli.py::test_version`（区域外
  例外）：版本断言改为与 `importlib.metadata` / `pyraimd2.__version__`
  一致。
- `uv.lock`：`uv sync` 随版本号更新（锁内本包条目 0.3.0→0.4.0.dev0）。
- `.github/workflows/tests.yml`：core 矩阵 / resume-regression / wheel
  三个 job（action 沿用仓库既有 SHA 固定）。
- `docs/api.md`（新）：公共 API 参考。
- 新接口、配置字段和默认值：无新运行时接口、无配置字段、无第三方依赖
  （CI 用 `build` 为工作流环境工具，非项目依赖）。
- 旧接口／已有数据的兼容方式：`__version__` 仍为字符串，对调用方透明；
  其余行为不变。
- 是否改变单位、力预算、时间、约束、随机检查或模型切换语义：否。

## 验收证据

```text
验收项：README 五分钟入门命令逐条可运行（示例原文件，未修改源码）
命令／输入配置：
  cp -r examples/harmonic_adaptive /tmp/pyraimd_readme_check/
  cd /tmp/pyraimd_readme_check/harmonic_adaptive
  pyramid validate run.toml && pyramid run run.toml
  pyramid inspect runs/harmonic-demo
  pyramid resume runs/harmonic-demo --steps 10
  pyramid export runs/harmonic-demo --force-source reference --output reference.extxyz
预先确定的通过标准：全部退出码 0；run/resume 输出完成步数与参考调用结构
实际关键结果：validate: OK；run 完成 20 步（21 evaluations，accepted 18，
  reference 16 = anchor 3 + probe 12 + check 1）；resume 从 20 续到 30 步；
  export 31 帧（20 帧无参考标签，按 forces_available=F + NaN 标记，未零补）
状态：通过
```

```text
验收项：wheel/sdist 构建并在仓库以外干净 venv 安装运行
命令／输入配置：
  uv build  → dist/pyraimd2-0.4.0.dev0-py3-none-any.whl + .tar.gz
  uv venv --python 3.12 /tmp/pyraimd_wheel_check/.venv
  uv pip install <wheel>（工作目录 /tmp，仓库以外）
  pyramid --version / --help；cp examples/harmonic_adaptive 后跑全流程
预先确定的通过标准：安装成功；--version 打印 0.4.0.dev0；示例全流程退出码 0
实际关键结果：pyramid 0.4.0.dev0；validate/run/inspect/resume/export 全部
  通过（run 20 步、resume 至 30 步、export 31 帧）；sdist 同样安装成功
  （pip 从 tar.gz 经 uv_build 构建）并打印 0.4.0.dev0
状态：通过
```

```text
验收项：核心安装不带 torch/pyscf
命令／输入配置：wheel venv 内 pip list；python -c "import torch" / "import pyscf"
预先确定的通过标准：两者均 ModuleNotFoundError；环境只有 pyraimd2 及其
  声明依赖（ase/numpy 及 ASE 自身依赖 scipy/matplotlib 等）
实际关键结果：
  ase 3.29.0, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.1, pyraimd2 0.4.0.dev0
  ModuleNotFoundError: No module named 'torch'
  ModuleNotFoundError: No module named 'pyscf'
状态：通过
```

```text
验收项："接自己的 ASE 计算器"插件路径在 wheel 环境可用
命令／输入配置：
  uv pip install examples/backends/pyraimd2_harmonic（装入上述 wheel venv）
  pyramid backends
  python -c "from pyraimd2.backends import create_backend; create_backend('harmonic_reference', kind='engine', k=1.2)"
预先确定的通过标准：插件经 entry-point 组被发现，工厂可按 kind 构造
实际关键结果：backends 列表出现 harmonic_reference / harmonic_surrogate
  （origin entry-point:pyraimd2-harmonic）；create_backend 返回
  HarmonicReference，fingerprint harmonic-reference:73d8761e42ac
状态：通过
```

```text
验收项：版本单一来源
命令／输入配置：
  uv run python -c "import pyraimd2; print(pyraimd2.__version__)"
  uv run pyramid --version
  PYTHONPATH=src python3 -c "import pyraimd2; print(pyraimd2.__version__)"（系统 Python，无安装元数据）
预先确定的通过标准：前两者输出 0.4.0.dev0（来自 pyproject 经安装元数据）；
  无元数据时回退同一字面值
实际关键结果：三处均 0.4.0.dev0；smoke 测试断言 __version__ == 安装元数据
状态：通过
```

```text
验收项：CI core 矩阵的本机等价（Python 3.12/3.13、固定 ASE 3.29.0）
命令／输入配置：
  uv run pytest tests/unit tests/test_smoke.py -q            # py3.12.13, 锁定 ASE 3.29.0
  uv run --python 3.13 pytest tests/unit tests/test_smoke.py -q  # py3.13.12
  干净 venv: pip install -e '.[dev]' + pip install ase==3.29.0（pip 路径，同 CI 命令）
  uv run --python 3.13 python examples/energetic_loop.py
预先确定的通过标准：各环境 383 passed；示例退出码 0
实际关键结果：三个环境均 383 passed, 1 deselected；示例正常输出
  （ase 3.29.0 | numpy 2.5.3 | py 3.12.13 组合同样 383 passed）
状态：通过
```

```text
验收项：恢复回归子集独立可见（CI resume-regression job 的本机等价）
命令／输入配置：uv run pytest tests/unit/test_resume.py tests/unit/test_plain_resume.py tests/unit/test_checkpoint.py -q
预先确定的通过标准：全部通过
实际关键结果：23 passed
状态：通过
```

```text
验收项：CI wheel job 命令序列本机等价（python -m build 路径）
命令／输入配置：
  python -m build（隔离构建，uv_build 来自 PyPI）
  python -m venv 仓库外 + pip install dist/*.whl 与 dist/*.tar.gz
  pyramid --version/--help/init/validate/run/inspect/resume/export
  importlib.util.find_spec('torch'/'pyscf') 断言为 None
预先确定的通过标准：与 uv build 产物一致可用；命令序列零失败
实际关键结果：-m build 产出同名 wheel+sdist；两者安装后 --version 均
  0.4.0.dev0；init→export 全序列退出码 0；torch/pyscf 断言通过
状态：通过
```

```text
验收项：CI YAML 结构有效
命令／输入配置：PyYAML safe_load .github/workflows/tests.yml
预先确定的通过标准：可解析；job/step/矩阵结构符合设计
实际关键结果：jobs = core（8 步，矩阵 py 3.12/3.13 × ase 3.29.0/latest）、
  resume-regression（4 步）、wheel（7 步）
状态：通过（真实 runner 执行未执行——本机无 Actions 环境）
```

```text
验收项：回归基线保持
命令／输入配置：uv run pytest tests/unit tests/test_smoke.py -q
预先确定的通过标准：383 passed
实际关键结果：383 passed, 1 deselected（与 WP07 基线一致；ruff 对本次
  触碰的三个 Python 文件 All checks passed）
状态：通过
```

```text
验收项：QE+MACE 周期材料 validate --probe-backends 与材料验证列
状态：未执行
原因与后续处理：本机无 pw.x/赝势/torch；材料配方与验证是 WP08 交付物。
  已验证无赝势机器上 pyramid validate examples/qe_mace_skeleton/run.toml
  如实报缺（pseudo_dir 与 Si 赝势路径，含补救提示），README 按此如实描述。
```

## 恢复与随机状态（相关工作包必填）

本阶段未改动任何运行/恢复逻辑。恢复契约由既有测试保持并首次在 CI 独立
成 job：`test_resume.py`（连续 100 vs 40+恢复 60、三类中断点、探针复用、
故障注入）、`test_plain_resume.py`、`test_checkpoint.py` 共 23 项全部通过。
无新增 checkpoint/RNG/事件协议内容。

## 材料与后端（相关工作包必填）

- 本阶段无材料计算。后端状态如实写入 README 三级口径：
  - 接口可接入：任意 ASE 计算器（AseEngine/AseSurrogate）、entry-point
    插件、qe-ase 路径；
  - 已完成软件集成测试：builtin 谐振后端全工作流（含新进程恢复）、QE
    引擎 hermetic（fake pw.x＋输出夹具）、PySCF 单位/符号回归（有
    PySCF 的机器）、MACE 名称/委员会 slow 标记测试（需 torch 与本地
    模型）、配置/CLI/续算/导出三模式；
  - 已有材料验证：见文末"最终 0.4.0 完成度（R3 填写）"（WP08 已返回，
    体相 Si adaptive 与 Al 表面 plain 通过，metallic adaptive 不支持）。
- 哪些能力只是接口支持，哪些完成了真实后端验证：见上分级；无任何
  真实 QE/MACE 前向验证在本阶段被声明。

## 算力与成本

- 新增实际参考执行总数：0（全部解析后端；打包验证亦为解析示例）。
- 分用途：anchor / refusal / probe / verification / diagnostic：0。
- 失败尝试与重试数：0。缓存命中和逻辑请求数：0（除示例演示计数）。
- 参考、推理、训练、I/O 墙钟及计时口径：harmonic 示例 run 约 0.1–0.2 s
  （演示值，无生产口径）。端到端墙钟：CI 等价验证全序列分钟级。
- 分配 CPU 核／GPU 数、运行时长、资源时间：未知（本机工作站）。
- 预算使用与剩余额度：WP08 配额未动用。

## 回归与交付

- 受影响测试通过情况：`tests/unit` + `tests/test_smoke.py` 383 passed,
  1 deselected（Python 3.12.13、3.13.12、pip 固定 ASE 3.29.0 三种环境
  各验证一次）。
- 原有 energetic／legacy switching 示例：`examples/energetic_loop.py`
  在 3.12/3.13 均运行通过；`energetic_pyscf.py`、`minimal_loop.py`、
  `adaptive_h2o.py` 需要 pyscf/mace，本机未执行（缺依赖，不计通过）。
  `tests/regression` 的 PySCF/MACE L1 用例本机同样未执行
  （ModuleNotFoundError，与 WP09 前状态一致，非本次改动引入）。
- wheel 安装及仓库外 CLI 测试：通过（证据见验收项 2/3/8；wheel 与
  sdist 均覆盖）。
- 最小依赖是否仍不导入 Torch／PySCF：是（wheel 环境 pip list 与 import
  断言；CI 将该断言固化为步骤）。
- 用户可复制的完整运行与恢复命令：README"五分钟入门"与"停止、续算与
  恢复"两节，逐条验证过。
- README／示例／支持矩阵更新位置：`README.md`（任务导向＋三级状态）、
  `docs/api.md`（新）、`CHANGELOG.md`、`.github/workflows/tests.yml`。
- 已知故障、未执行验证与风险：
  - GitHub Actions 未在真实 runner 执行；YAML 经解析校验、命令经本机
    等价验证，首次推送后需关注 runner 差异（如 `python -m build` 联网
    拉取 uv_build）。
  - ASE 矩阵两腿当前同为 3.29.0（见偏离 2），新版发布后浮动腿才产生
    真实差异信号。
  - Python 3.13 为本机 uv 管理的 3.13.12；CI 用 setup-python 的 3.13，
    小版本可能不同。
  - `examples/minimal_loop.py` 等旧脚本需可选依赖且无跳过逻辑，缺依赖
    机器上直接报错（既有行为，未改动；README 已如实标注）。
- 需要负责人判断的具体决策及备选方案：版本标记取 `0.4.0.dev0`（候选前
  开发态）；若希望直接以 `0.4.0rc1` 标记候选，需在 WP08 验收后统一改
  pyproject 一处即可（单一来源已就位）。
- 下一工作包及其入口：WP08 材料案例结果返回后执行 WP09 第二阶段——
  填写 README/支持矩阵的材料验证列、按 §3 逐项核对 0.4.0 完成条件、
  决定候选发布（R3）。

## 最终 0.4.0 完成度（R3 填写）

### §3 完成条件逐项核对

- 可以用配置进行单点、固定模型结构优化、固定晶胞 NVE；支持
  reference-only、surrogate-only 和明确支持的 adaptive MD 模式：
  **完成。** `tests/unit/test_relax_singlepoint.py`（CLI 全流程）、
  `test_workflows.py`、`examples/si_bulk_qe_mace/`（真实材料三模式）、
  `examples/periodic_lj/`（解析周期示例）。adaptive 仅 `kind="md"`，
  metallic（smeared free-energy）参考的 adaptive 明确拒绝（WP08 证据）。
- energetic NVE 可以从完整步边界续算；冻结模型和受支持更新器的状态均可
  恢复：**完成。** `tests/unit/test_resume.py`（连续 100 vs 40+恢复 60，
  1e-12）、`test_plain_resume.py`（plain 模式）、`test_guarded_update.py`
  （可保存更新器全过程恢复）、`examples/si_bulk_qe_mace/`（真实材料在新
  作业中 resume +2，接受/检查一致）。
- 最小核心仍只需 NumPy、ASE 和标准库；不运行模型推理也能验证主要运行机
  制：**完成。** CI `tests.yml` 核心腿断言无 torch/pyscf（本报告验收项），
  wheel/sdist 安装冒烟同断言（0.4.0rc1 复验，见验收项"候选版本"）。
- 能替换 ASE 后端，并有 QE＋MACE 的周期体相与固定层表面案例：**完成。**
  `examples/si_bulk_qe_mace/`（体相 Si，QE 7.5＋MACE-MPA-0 medium）、
  `examples/al_surface_qe_mace/`（Al(111) 固定层，plain＋FixAtoms）、
  `examples/backends/{pyraimd2_harmonic,pyraimd2_lj}`（插件路径）、
  `qe-ase`（ASE 原生 QE 路径）。
- 每个新案例都有输入、后端版本、命令、预期检查和完整成本；不要求先得
  到论文级加速比：**完成。** 两个配方的 README（输入/设置/命令）、
  WP08 报告（58 次实际参考执行、分用途计数与墙钟、预算 58/200）。
- 支持固定原子约束 FixAtoms，定义并测试自由坐标的力预算及真实驱动力；
  复杂约束留到后续：**完成。** `tests/unit/test_fixatoms.py`、
  `test_constraints.py`、`examples/al_surface_qe_mace/`（真实固定层）；
  `policy.force_metric` 两档定义在 docs/configuration.md 写明；RATTLE 等
  复杂约束与变胞明确拒绝（测试覆盖）。
- 有一条真实的标签收集—模型更新—重新锚定—保存／恢复演示；生产可恢复
  模式只接受已实现状态导出的更新器：**完成（解析可保存更新器）。**
  `tests/unit/test_guarded_update.py`（发布/回退/去重/全过程恢复）；
  MACE 委员会路径的 slow 集成测试已就位但**未执行**（本机与 CI 均无
  torch；`tests/unit/test_committee_guarded_update.py`，WP06 报告如实
  标注），无状态接口的更新器恢复时明确拒绝。
- 关键错误会停止或进行有记录的有限重试，不会把失败结果当成成功标签：
  **完成。** WP00 定向修复套件、QE 重试与失败诊断（`test_qe_engine.py`、
  `test_qe_reliability.py`）、失败尝试入帐（WP08 报告的 1 次失败尝试）。

### 工作包逐项

- WP00 基础问题定向修复：**完成**（`WP00_report.md`，6 项问题各配暴露
  性回归测试）。
- WP01 物理时间、能力和模型身份：**完成**（`WP01_report.md`，
  EvaluationContext、能量口径契约、代次缓存键全部带验收测试）。
- WP02 事件、成本与检查重放：**完成**（`WP02_report.md`，权威事件日志、
  完整成本账本、数值标签缓存、验证重放与在线一致）。
- WP03 完整步 resume：**完成**（`WP03_report.md`，连续 vs 中断恢复逐项
  1e-12，故障注入、截断回退、重复恢复拒绝）。
- WP04 TOML／CLI：**完成**（`WP04_report.md`，init/validate/run/resume/
  inspect/export 全流程，含空格目录、离线 help）。
- WP05 后端与可靠 QE：**完成**（`WP05_report.md`，注册表、parser 边界、
  唯一目录、密度热启动、第三方插件）。
- WP06 更新／回退／恢复：**完成**（`WP06_report.md`；MACE 真实微调 slow
  测试未执行，优化器状态不含已如实标注）。
- WP07 普通任务和 FixAtoms NVE：**完成**（`WP07_report.md`，三任务分发、
  投影语义、plain resume、PBC 展开坐标）。
- WP08 两条周期材料工作流：**完成**（`WP08_report.md`；配方 A 含真实
  接受/重锚/恢复，配方 B adaptive 按契约如实报不支持，配额 58/200）。
- WP09 打包、文档、兼容和候选发布：**完成**（第一阶段报告＋本节；wheel
  与 sdist 的 0.4.0rc1 仓库外冒烟见验收项"候选版本"）。
- 已投稿复现标签和数值目标保持不变：**保持。** `reproducibility/` 自
  `895897b` 固定后未被任何 WP 提交触碰（git log 可复核）。
- 是否仍有 P0 未完成：**无。**（WP10–12 不在本轮，按计划不启动。）
- 建议：**可供外部试用**（0.4.0rc1 作为候选； metallic smearing 参考的
  adaptive 组合在支持矩阵与 README 中明示为 0.4.0 不支持）。

### 2026-09-09 独立审阅后的修正（R3 表述更新）

0.4.0rc1 经独立审阅（INDEPENDENT_REVIEW_20260909.md）发现上述部分"完成"
表述所依据的判断不成立，逐项修正如下，修复与证据见
REVIEW_FIXES_CORE_20260909.md：

- "WP02/WP03 完成"中的两处判断被反例推翻：恢复后标签缓存冷启动会改变
  消费与训练（F02）；崩溃窗口的恢复会改变检查抽签流、产生无身份的孤立
  数据库行（F01/F03/F07）。R1 已修复并加逐条回归，恢复现保持同一检查/
  标签/模型历史。
- "成本账本完整"的口径不区分逻辑请求与真实进程执行（引擎内部重试、
  预检失败、缓存命中被混计）。R6 已引入 attempt 事件并改造账本汇总；
  WP08 报告的 58 次实际参考执行需按 attempt 重算（58 与分用途合计 57
  不符），重算列入待办。
- "FixAtoms/约束口径完成"中，relax 的 final_fmax 报表口径与优化器判据
  不一致（固定原子反力被计入），inspect 温度固定除以 3N，导出动量为
  半步值且崩溃步被当作完成步导出。R4 已修正（判据同口径 fmax＋原始值
  另列、受约束自由度温度、完整步轨迹接口）。
- WP08 对 Al 收敛的解释（"ASE 用分量最大值"）被证明不成立，已在
  WP08_report.md 更正；Al 末帧力数组的固定/自由分列重算列入待办。
- 上述"完成"项在修复前不应作为验收证据引用；重新验收以
  REVIEW_FIXES_CORE_20260909.md 的逐条前后对照为准。

### 三句话结论

1. 软件机制：通过。389 项单元/冒烟测试在 Python 3.12/3.13 与 ASE
   3.29 固定腿全绿，wheel/sdist 仓库外 CLI 冒烟通过，全部 P0 工作包有
   逐项证据。
2. 真实材料集成：通过（按计划的两条配方）。QE 7.5＋MACE-MPA-0 medium
   的体相 Si adaptive 观察到真实 surrogate 接受、重新锚定与中断恢复，
   固定层 Al 表面的 plain 工作流与 plain resume 通过；metallic smearing
   参考的 adaptive 组合被契约明确拒绝并如实记录，不构成缺陷。
3. 物理精度与全成本证据是否足以支持加速主张：未开展。本批算例为 4–8
   步的机制验证（57–100% 的短程接受率不构成稳态或加速结论），稳态接受
   率与总成本—精度比较属下一阶段科研节点，本报告不作任何加速主张。

### 科学／架构决策（交研究负责人判断）

**议题：metallic（smeared）参考与普通能量替代模型在 adaptive 模式的组合。**

现状（实现 (b)）：QE 在 metallic smearing 下如实报告
`energy_kind = free_energy`（`!` 行即变分自由能，其梯度给出力），
MACE 等替代模型报告 `energy`。WP01 契约把已声明的口径不匹配视为不可
组合，在 validate 阶段明确拒绝（WP08 配方 B 证据）。0.4.0 因此不支持
该组合的 adaptive；plain 单后端工作流不受影响。

- (a) 放宽契约——"两侧均 force_consistent 时允许跨 kind 锚定"。要点：
  锚定公式 E_drive(X) = E_model(X) − c·(X−X₀) + const 只要求两个面各自
  与力一致（常数偏移由锚点吸收），endpoint work 用的是各自面内的差值，
  数学上可成立；风险是变分自由能对 smearing 参数（类型/宽度）敏感，
  温度增大时 E_free 与势能面的偏离增大，endpoint work 的物理解释随之
  漂移，而且"参考口径随参数而变"会污染长期可比性（换 degauss 即换
  标签口径）。若走此路，建议契约改为拒绝仅在 force_consistent=False 时，
  并在工件/检查记录中固定 smearing 参数进 reference fingerprint。
- (b) 维持不支持（当前实现）。要点：口径不匹配即拒绝，不静默降级；
  metallic adaptive 需要等"MACE 也以 free energy 口径训练/声明"或
  "无 smearing 的可靠金属参考协议"成立后再开。代价是 0.4.0 不能做金属
  体系的 adaptive。

本 WP 未自行修改契约；(a)/(b) 属科学判断，请研究负责人定夺后另起
工作包实施。

### 验收项：候选版本（0.4.0rc1）构建与仓库外冒烟

```text
命令：uv run --with build python -m build && 干净 venv 分别安装
      dist/pyraimd2-0.4.0rc1-py3-none-any.whl 与 .tar.gz，
      在仓库外执行 pyramid init/validate/run/inspect/resume/export
通过标准：两分发均安装成功，`pyramid --version` 输出 0.4.0rc1，
      CLI 全流程通过，wheel 环境无 torch/pyscf
实际关键结果：全部通过（`pyramid 0.4.0rc1` ×2；run 21 评估 18 接受；
      resume +10；export 31 帧；核心安装无 torch/pyscf 断言通过）
状态：通过
```
