# 半合成两层证据链流水线

## 1. 研究问题与适用范围

限功使实际功率成为潜在可用功率（PAP）的右删失观测。本流水线回答两个预声明的研究问题：

- **RQ1（第一层，统计）**：在控制诱导右删失下，删失感知似然（B6/BT）是否比点标签训练（B4）和删除删失样本（B1）更准确地预测前向 PAP 分布；比较链为 `B4_plain → B4 → B6/BT → ORACLE`。
- **RQ2（第二层，价值）**：同一半合成 A_true 驱动构网型风氢系统时，在相同缺额风险水平（Pareto 前沿匹配）下，更准的预测是否多产氢、更接近 Oracle、决策遗憾更小。

HOT 真实数据层、B8 支持度门控、重建真值与"评分-价值解耦"主张已全部退出主证据链，相关资产归档于 `archive/hot_real_layer/`；旧协议结果归档于 `archive/old_results/`。`--smoke` 产生的文件只验证代码路径，永远不能作为正式实验结果。

## 2. 第一层：冻结的预测协议

- 数据：Altahullion T11 的 1 min 半合成场景 S1–S5（`results/altahullion_audit/synthetic_{scenario}.parquet`），已知潜在真值 `A_true`、限幅 `C_syn`、观测 `Y_syn = min(A_true, C_syn)`；输入文件 SHA-256 写入协议清单。
- **共享事件掩码**：S1–S4 使用完全相同的 `policy_active/event_id`（由 `EVENT_SEED` 生成，S4 的每事件随机深度来自独立的 `CAP_SEED` 流），场景间唯一差别是限幅深度，深度剂量效应因此可识别；掩码一致性由 `scripts/10_altahullion_audit.py` 逐元素断言并记录在 `audit_summary.json`（`shared_event_mask_S1_S4`）。另生成 S0（无限功，`Y_syn = A_true`）供负对照使用，不进主协议。
- 任务：用过去 60 min 的可观测历史，预测 15 min 后的 PAP 分布。
- 输入通道：观测功率、仅由训练集估计参数的标准化风速、限幅值、删失指示量。
- 划分：按时间顺序以完整 `segment_id` 划分 70%/15%/15%；窗口不能跨段。
- 防泄漏：常规模型的 `fit()` 只能获得可观测标签视图（`fit_view()`）；只有声明 `requires_latent_truth` 的 ORACLE 适配器由运行器分派 `oracle_fit_view()`（标签换成 A_true、删失指示清零），其产物元数据记 `"oracle": true`。`predict()` 只能获得测试输入。
- 随机性：随机模型使用种子 `[42, 123, 256, 789, 1024]`；同架构模型必须通过初始化哈希检查。
- 支持域：`[0, 1.06] pu` 的 106 个概率格；9 个分位数 `[.05 .1 .2 .3 .5 .7 .8 .9 .95]`。

协议配置位于 `configs/pap_benchmark_semisyn.json`，输出目录 `results/pap_benchmark_semisyn/`。数据、划分、预测任务、训练和评价字段共同生成 `protocol_id`；修改这些字段后不能把新旧预测混在同一输出目录。

### 模型集合

| 模型 | 说明 | 在比较链中的角色 |
| --- | --- | --- |
| `B0_persistence` | 末值持续性 | 朴素下界 |
| `B0_climatology` | 训练集经验边际分布 | 朴素下界 |
| `B1` | 删除删失样本的 PMF-TCN | 删失似然的替代方案 |
| `B4_plain` | 点标签 PMF-TCN，仅见功率+风速两通道 | 控制状态输入的价值（对照） |
| `B4` | 点标签 PMF-TCN，见全部四通道 | 主参照 |
| `B6` | 右删失 survival 似然 PMF-TCN | 主挑战者 |
| `BQR` | 直接分位数回归 TCN | 分布形式对照 |
| `BT` | Tobit（删失高斯）TCN | 删失似然的参数化形式 |
| `ORACLE` | 用 A_true 训练的同结构 PMF-TCN | 完整标签训练参照，只进差距报告 |

`ORACLE` 是"完整标签训练参照（full-label reference）"而非理论上界：有限样本与随机优化下其数值不必然逐场景最优。任何"接近参照"的表述必须引用预声明等效界值（WIS 0.005）与 `comparisons.json` 中的 `equivalent_within_margin` 判定（95% 聚类 bootstrap CI 完全落入 ±0.005 才判等效），不得用"差异不显著"证明相等。

### 评价与推断

- 跨模型主指标为 WIS；同时记录精确离散 CRPS（PMF 模型）/有限分位数近似（分位数模型，`crps_kind` 隔离）、中位数 MAE/偏差、覆盖率、区间宽度和 PIT。
- 删失样本低估率：删失目标窗中 `q50 < A_true` 与 `q90 < A_true` 的比例（`metrics_by_run.csv`）。
- 分层评价 `metrics_stratified.csv`（预声明维度）：目标删失/未删失；60 min 历史窗删失比例 `{0, (0,0.3], (0.3,1]}`；限幅场景内三分位；风速档 `{<7, 7–11, >11 m/s}`。分层字段只进评分、不进模型输入。
- 正式配对比较：同一测试窗口先对种子取均值，再按完整段做聚类 bootstrap 与符号翻转检验；P 值在预声明 family 内 Holm 校正。预声明 family：
  - `RQ1_control_inputs_B4_vs_B4plain`（控制状态输入的价值）
  - `RQ1_censored_likelihood_primary`：B6 vs B4、BT vs B4、B6 vs B1（RQ1 主检验）
  - `RQ1_secondary_*`：B6 vs BT / BQR / persistence
  - `oracle_gap_report_not_confirmatory`：ORACLE vs B6，只作差距报告，不入 RQ1 主检验；附 `equivalence_margin: 0.005` 的等效性判定

### S0 无删失负对照（实现正确性诊断）

无删失时 B6 的 survival 似然应严格退化为 B4 的点似然，三者（B4/B6/ORACLE）的 WIS 差应在数值误差内。诊断配置 `configs/pap_benchmark_s0_control.json`（场景仅 S0、模型仅 B4/B6/ORACLE、5 种子共 15 次，输出 `results/pap_benchmark_s0_control/`），comparisons 为 family `S0_negative_control` 内 B6 vs B4 与 ORACLE vs B4，均带 `equivalence_margin: 0.005`。**通过判据**：两条比较 `equivalent_within_margin = true`。若 S0 中 B6 仍显著优于 B4，说明存在实现或优化流程差异，必须先修复再继续，主协议结果在此之前不得写入论文。该诊断不进主协议与主假设 family。

### 结论范围（预声明）

第一层结论限定于：Altahullion T11 单风机、单一时间划分（70/15/15 整段）、15 min 预测步长、五种预定义半合成限功机制。不外推到所有风机/场站/限功机制，不声称 B6 恢复了真实 PAP 分布。两条已知边界必须如实报告：

- B6 并非全场景最优删失方法（S5 中 BT 显著更好）；
- S5（未来风速触发的信息性删失）下 survival 似然仍显著改善 WIS，但无法恢复良好校准（删失目标 90% 覆盖率约 5%），参数化 Tobit 更稳健，且与完整标签参照仍有明显差距——删失感知不在任何机制下都自动解决可识别性问题。

## 3. 第二层：风氢价值实验协议

第二层直接使用半合成 A_true 驱动 `仿真模型/长时间风氢价值模型` 的 W2H 参考系统（6.66 MW 风机 + 4×1 MW PEM + 500 kW/1 MWh BESS）：

- **轨迹导出**（`scripts/40_semisyn_value_trajectories.py`）：从第一层产物读取 B0_persistence/B4/B6/BT/ORACLE 的全部 9 个分位列，取 5 种子分位数均值（预声明），pu × 6.66e6 缩放到模型瓦特，输出 `results/value_semisyn/<scenario>_trajectories.mat`。
- **时间对齐**：`issue_time = valid_time − 15 min`；EMS 在 t 时刻只用 valid_time = t 的预测，不做超出有效期的 ffill；每段前 75 min（60 窗 + 15 时距）无有效预测，不导出。
- **事件定义**：完整测试 segment，不切固定块、不跨缺口拼接；所有策略同一事件用相同 A_true 轨迹与相同初始 SOC/储氢/PEM 状态；前 300 s 暖机不计分。
- **公平性**：所有策略统一 `strategy_id=5`、`support_score≡1.0`（EMS 逻辑完全相同，唯一差别是馈入的预测分位数；B8 门控退出）。PerfectForesight（q05=q50=q95=A_true）只作上界，不入公平比较。
- **等风险扫描**：承诺分位从 `{q05, q10, q20, q30, q50}` 取五档，逐策略生成 H2-缺额 Pareto 前沿；仿真预算约定 S1、S5 全扫描，S2–S4 只跑默认档（q05）。
- **记录**：每 事件 × 策略 × 风险档 记 H2 kg、弃风 kWh、缺额 kWh、PEM 启停、BESS 吞吐、SOC 越界时长、频率/电压约束违反时长。
- **决策遗憾**：`Regret_m = J_perfect − J_m`；J 的权重在 `run_semisyn_value.m` 头部显式声明并在首次正式运行前冻结。
- **分析**（`scripts/41_value_analysis.py`）：事件（段）为聚类单元的 bootstrap 10000 次，报告等缺额 H2 差（前沿线性插值到共同缺额网格）、等 H2 缺额差、缺额 95% CVaR、B6 vs B4 与 B6 vs ORACLE 前沿间距；输出 WIS↔遗憾对应表；对 MATLAB 汇总的 J 做独立复算交叉校验。

## 4. 运行与验收

```powershell
# 代码级验证
python -m unittest tests.test_benchmark_v2 -v

# 数据重建（含 S1–S4 共享掩码断言与 S0 生成）
python scripts/10_altahullion_audit.py

# 第一层：冒烟（独立输出目录）→ 正式全量（9 模型 × 5 场景，随机模型 5 种子）
python scripts/run_pap_benchmark.py --config configs/pap_benchmark_semisyn.json run --smoke --force
python scripts/run_pap_benchmark.py --config configs/pap_benchmark_semisyn.json run

# S0 无删失负对照（诊断，B4/B6/ORACLE × 5 种子）
python scripts/run_pap_benchmark.py --config configs/pap_benchmark_s0_control.json run

# 只重新聚合已有预测（不训练）
python scripts/run_pap_benchmark.py --config configs/pap_benchmark_semisyn.json compare

# 第二层：轨迹导出（先 --selftest 再全量）
python scripts/40_semisyn_value_trajectories.py --selftest
python scripts/40_semisyn_value_trajectories.py --all

# 第二层：MATLAB 仿真（先冒烟：base workspace 中 semisyn_smoke = true）
# 在 MATLAB 中运行 仿真模型/长时间风氢价值模型/scripts/run_semisyn_value.m

# 第二层：分析与交叉校验
python scripts/41_value_analysis.py --selftest
python scripts/41_value_analysis.py
```

第一层正式结果只有在 `benchmark_status.json` 同时满足以下条件时有效：`complete` 与 `valid_for_final_comparison` 均为 `true`，`missing_artifacts` 和 `nonconverged_runs` 均为空。

主要产物：

- 第一层（`results/pap_benchmark_semisyn/`）：`protocol_manifest.json`、`last_run_manifest.json`、`data_summary.json`、`artifacts/<scenario>/<model>/<seed>/`（含 `"oracle"` 标记）、`metrics_by_run.csv`（含删失低估率）、`metrics_summary.csv`、`metrics_stratified.csv`、`comparisons.json`（含等效性字段与 `experiment_id`）；`benchmark_status.json` 中的 `experiment_id` 由 `protocol_id`、各场景输入数据的 SHA-256、固定分段及 comparisons 配置共同生成，并排除时间戳和本机绝对路径，用于稳定地区分同一 `protocol_id` 下的不同数据/比较计划；S0 诊断产物在 `results/pap_benchmark_s0_control/`；
- 第二层：`results/value_semisyn/<scenario>_trajectories.mat`（`provenance` 记录 `protocol_id`、`experiment_id`、`protocol_manifest_sha256`、每个源 `forecast.npz` 的 SHA-256 及场景/模型/种子聚合规则）、`仿真模型/长时间风氢价值模型/results/semisyn_value_results.mat`、`results/value_semisyn/value_summary.csv`、`pareto_frontier.csv`、`confidence_intervals.csv`、`wis_regret_table.csv`。

## 5. 新增模型

新增模型不需要修改运行器：

1. 实现 `ModelAdapter.fit()` 和 `ModelAdapter.predict()`，并用 `@register_model` 注册唯一名称；
2. `predict()` 返回统一分位数；若有 PMF，可同时返回；
3. 在 `model_modules` 中加入模块导入路径，在 `models` 中加入模型名；
4. 模型特有超参数写入 `model_settings.<模型名>`；
5. 只运行新模型，再聚合所有已有预测。

```powershell
python scripts/run_pap_benchmark.py --config configs/pap_benchmark_semisyn.json run --models new_model
python scripts/run_pap_benchmark.py --config configs/pap_benchmark_semisyn.json compare
```

中断后可按已声明种子增量补跑，例如 `--models B6 --scenarios S1_fixed_50pct --seeds 42 123`；已有且哈希兼容的产物默认复用。可运行的最小示例见 `src/censored_wind_power/benchmark/example_adapter.py`。

只有显式声明 `requires_latent_truth = True` 的适配器才能获得潜在真值标签视图，且其结果自动标记为 oracle、不进入 RQ1 主检验 family。新增模型的超参数只能根据训练/验证部分确定，并应在首次查看正式测试指标前冻结。
