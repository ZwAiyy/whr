# 两个 CSV 的清洗 + 四种方法分词

对 `source4_ctrip_raw.csv`（携程景点点评，526 条）与 `adb_dazhongdianping_raw.csv`
（大众点评景点点评，300 条）做统一清洗，并用**规则 / 统计 / 深度学习 / 大模型**
四种方法对同一批抽样文本分词，输出逐条对比结果与评价报告。

## 1. 快速开始

深度学习分词用 PyTorch，GPU 环境在 `C:\Users\Windy\Documents\Code\Python\WHR\.venv`
（已装 `torch 2.10.0+cu130`、`jieba`、`pandas`、`numpy`、`opencc`）。**推荐用它运行**：

```powershell
$PY = "C:\Users\Windy\Documents\Code\Python\WHR\.venv\Scripts\python.exe"

# 默认：清洗 + 四方法分词（对比样本 120 条）
& $PY .\clean_pipeline.py

# 自定义参与分词对比的条数（大模型调用量随之变化）
$env:SAMPLE_N="200"; & $PY .\clean_pipeline.py

# 只跑离线三方法（不联网、不消耗接口额度）
$env:SKIP_LLM="1"; & $PY .\clean_pipeline.py

# 跳过「大模型裁判」环节（四方法照常评测）
$env:SKIP_JUDGE="1"; & $PY .\clean_pipeline.py

# 强制使用纯 NumPy 回退后端（无需 torch 的环境）
$env:DL_FORCE_NUMPY="1"; & $PY .\clean_pipeline.py
```

`C:\Users\Windy\Desktop\7NLP作业9.23\.venv`（无 torch）同样可以运行：
深度学习部分会自动回退到 `seg_dl_numpy.py`，其余完全一致。

可调参数（`config.py` / 环境变量）：

| 参数 | 环境变量 | 默认 | 说明 |
| --- | --- | --- | --- |
| `SAMPLE_N` | `SAMPLE_N` | 120 | 参与四方法对比的文本条数（按来源分层采样） |
| `JUDGE_SAMPLE_N` | `JUDGE_SAMPLE_N` | 20 | 交给大模型裁判打分的文本条数 |
| `JUDGE_ENABLE` | `JUDGE_ENABLE` | 1 | 是否启用大模型裁判 |
| — | `SKIP_LLM` | 0 | 置 1 跳过大模型分词（用规则分词占位） |
| — | `DL_FORCE_NUMPY` | 0 | 置 1 强制使用纯 NumPy 后端 |
| `DL_EPOCHS` | — | 20 | BiLSTM 训练轮数 |
| `LLM_MODEL` | `LLM_MODEL` | deepseek-chat | 大模型分词所用模型 |

大模型接口凭据自动从 `C:\Users\Windy\.dsh\.credentials.yaml`（`DEEPSEEK_API_KEY`）读取，
也可用环境变量 `DEEPSEEK_API_KEY` / `LLM_API_KEY` 覆盖。

## 2. 输出文件

| 文件 | 内容 |
| --- | --- |
| `data/interim/ctrip_unified.csv`、`dianping_unified.csv` | 字段统一后的两来源数据 |
| `data/processed/clean_all.csv` | 清洗后语料（含 `text_clean`、`tokens`、`weak_label`） |
| `data/processed/clean_ctrip.csv`、`clean_dianping.csv` | 按来源拆分 |
| `data/processed/seg_compare_sample.csv` | **逐条四方法分词对比**（`tokens_rule/stat/deep/llm`） |
| `data/processed/clean_report.json` | 清洗质量报告（分步计数、分布、剔除样例） |
| `data/processed/seg_report.json` | 分词评价报告（全部指标原始值） |
| `outputs/reports/清洗报告.md` | 人读版清洗报告 |
| `outputs/reports/分词方法对比报告.md` | 人读版分词对比报告 |
| `outputs/models/bilstm_seg.npz` | BiLSTM 分词模型参数 |
| `outputs/cache/llm_seg_cache.jsonl`、`llm_judge_cache.jsonl` | 大模型结果缓存（可断点续跑） |

## 3. 清洗流程（S0~S11）

| 步骤 | 处理内容 |
| --- | --- |
| S0 | 载入两个 CSV，字段统一为同一套 schema（`doc_id/source/site/author/rating_star/text_raw/...`） |
| S1 | 缺失文本剔除 |
| S2 | HTML 标签与 HTML 实体清除 |
| S3 | URL / 邮箱 / @用户 / #话题# 噪声清除 |
| S4 | 表情占位符（`[大笑]`、`[Lv-种草]`）与 Emoji 清除 |
| S5 | 平台模板行清除（“通过大众点评消费”“2026-09出行\|游玩1-2小时\|情侣夫妻”“¥100”等） |
| S6 | 控制字符、零宽字符清除，空白归一，全角数字/字母转半角 |
| S7 | 繁体转简体（opencc，可选） |
| S8 | 长度过滤（<4 字剔除，>500 字按句末标点截断） |
| S9 | 无效内容过滤（纯数字/符号、标点占比 >85%、广告词） |
| S10 | 精确去重（清洗后文本完全一致） |
| S11 | 近似去重（去标点/表情后一致） |
| S13 | 弱标签映射（携程星级 / 大众点评评价档 → 正面/中性/负面/无法判断） |

## 4. 四种分词方法

| 方法 | 实现 | 代码 |
| --- | --- | --- |
| **规则分词** | jieba 精确模式（关闭 HMM 保证确定性）+ 领域用户词典 + 正则形态规则（数字/英文/标点）+ 单字兜底 | `seg_rule.py` |
| **统计分词** | LLR/PMI 无监督词发现 → 词频一元模型 `P(w)` → Viterbi 解码（`max Σ[log P(w)+β]`），λ/β 由开发集自动校准 | `seg_stat.py` |
| **深度学习分词** | 字嵌入 → 双向 LSTM → 线性分类头 → Viterbi 约束解码；**PyTorch 实现（有 GPU 自动用 CUDA）**，无 torch 时自动回退纯 NumPy 手写反向传播版本；监督信号为词表约束伪标注 | `seg_dl.py` / `seg_dl_numpy.py` |
| **大模型分词** | DeepSeek Chat few-shot 提示直接生成切分，`/` 连接 + 字符级一致性校验 + 结果缓存 + 失败重试 | `seg_llm.py` |

辅助对照：`seg_stat.py` 内另实现 **BEMS-HMM**（Baum-Welch 无监督训练 + Viterbi），
用于说明「纯 HMM 无词典统计分词」的水平差异。

### 深度学习分词细节（`seg_dl.py`）

- 结构：`Embedding(128) → BiLSTM(128×2，pack_padded) → Dropout → Linear(4)`，标签集 `B/M/E/S`
- 训练：mini-batch 64，Adam（lr 2e-3，梯度裁剪 5.0），20 轮，训练/验证 9:1，交叉熵忽略 padding
- 设备：`cuda`（RTX 4060 Laptop）自动检测，训练 1.5 s；无 GPU/无 torch 时回退 CPU（`seg_dl_numpy`）
- 解码：Viterbi + 合法转移约束（B→M/E，M→M/E，E→B/S，S→B/S）、句首/句末约束、可校准单字词惩罚
- NumPy 回退实现的 BPTT 已通过数值梯度校验（`tests/test_dl_grad.py`，相对误差 3e-6）

## 5. 评价协议

- **样本**：默认 120 条，按来源分层采样（两来源各 60 条），随机种子 `20240501` 可复现。
- **评价协议**：只对「词」打分——先剔除纯标点 token，再比较切分位置；
  这样「某方法是否输出标点 token」不会影响 F1（大模型倾向输出标点，
  规则/统计/深度模型丢弃标点，不归一会系统性低估大模型）。
- **参照切分**（无人工标注，故用多路互为参照）：
  1. 规则分词（baseline）
  2. 四方法共识切分（同一位置被 ≥3 个方法切出才采纳）
  3. 统计分词
  4. 大模型裁判给出的「理想切分」
- **指标**：边界 P/R/F1、词语 P/R/F1、完全一致率、近似一致率（词 F1 ≥ 0.9）、
  平均词长、单字词占比、压缩率、OOV 率、两两一致性矩阵。
- **大模型裁判（LLM-as-a-judge）**：先让模型给出理想切分，再对四方法按
  边界合理性 / 表意完整 / 专名完整 / 无碎词 / 总体 五个维度打 0-10 分。
  **这是机器评判，不是人工标注**，报告中已明确标注，仅作近似参照。

## 6. 自动校准

无监督方法（统计分词）的切分粒度、深度学习解码的单字词倾向都需要调参，
脚本用一份**与评价样本不重叠**的开发集（`calib_texts`，默认 100 条）自动校准：

- 统计分词：在 `(λ, β)` 网格上以「与开发集参照切分的边界 F1」为准则，
  先按「平均词长接近中文经验值 2.1±0.45」筛掉退化切分（全切单字 / 整句成一个词）；
- 深度学习：在「单字词惩罚」网格上做同样的校准；
- 校准结果写入 `seg_report.json` 并在报告中列表展示。

## 7. 模块清单

| 文件 | 职责 |
| --- | --- |
| `clean_pipeline.py` | 主入口：清洗 → 四方法分词 → 评价 → 报告 |
| `config.py` | 路径、标签体系、停用词、四方法超参、凭据 |
| `clean_utils.py` | 清洗规则库（正则、全角半角、简繁、去重键） |
| `seg_rule.py` | 规则分词 |
| `seg_stat.py` | 统计分词（LLR 词发现 + 一元模型 + Viterbi）与 BEMS-HMM 对照 |
| `seg_dl.py` | BiLSTM 序列标注分词（PyTorch 后端，含伪标注构造、Viterbi 解码、后端选择） |
| `seg_dl_numpy.py` | 纯 NumPy 手写反向传播的回退实现（接口与 `seg_dl.py` 完全一致） |
| `seg_llm.py` | 大模型分词客户端（缓存 / 重试 / 字符级校验） |
| `seg_judge.py` | 大模型裁判（理想切分 + 多维度打分） |
| `seg_metrics.py` | 边界/词语 F1、共识参照、描述统计 |

自检脚本（均在 `tests/`，可直接用 venv 运行）：

| 脚本 | 作用 |
| --- | --- |
| `tests/selfcheck.py` | 模块导入 + 清洗规则 + 评价指标 + 深度学习梯度校验 |
| `tests/test_dl_torch.py` | PyTorch 后端冒烟（后端选择 / 训练 / 解码 / 保存加载） |
| `tests/test_dl_grad.py` | 纯 NumPy 后端数值梯度校验 + 小规模训练冒烟 |
| `tests/test_llm.py` | 大模型分词连通性（真实调用 API，8 条） |
| `tests/test_judge.py` | 大模型裁判连通性（真实调用 API，6 条） |

```powershell
& $PY .\tests\selfcheck.py
& $PY .\tests\test_dl_torch.py
```

## 9. 默认配置下的实测结果

清洗：输入 826 条（携程 526 + 大众点评 300）→ 输出 **811 条**（保留率 98.18%）；
精确去重 12 条、近似去重 1 条、过短 2 条、超长截断 3 条；
弱标签分布：正面 739 / 中性 45 / 负面 19 / 无法判断 8。

分词对比（120 条分层采样，2026-10-03 运行）：

| 方法 | 平均词长 | 单字词占比 | 与规则分词边界F1 | 与共识切分边界F1 | 与规则分词词语F1 | 单次切分耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| 规则分词 | 1.713 | 0.3921 | — | 0.7416 | — | 0.010 s |
| 统计分词 | 1.969 | 0.1553 | 0.8304 | 0.7085 | 0.6176 | 0.005 s |
| 深度学习分词 | 1.641 | 0.3961 | 0.8382 | 0.7313 | 0.6106 | 0.054 s |
| 大模型分词 | 1.765 | 0.3283 | **0.9453** | **0.7432** | **0.8554** | 0.0 s（缓存）/ 3~4 s per 8 条（联网） |

> 深度学习分词在 RTX 4060 上训练 20 轮仅 **1.5 s**（验证集标签准确率 0.852），
> 同一模型用纯 NumPy 后端训练需要 ~55 s；两者切分质量基本持平（与共识切分边界 F1
> 0.731 vs 0.728），说明瓶颈在伪标注质量而不在算力。

大模型裁判（20 条，五维 0-10 分）：

| 方法 | 边界合理 | 表意完整 | 专名完整 | 无碎词 | 总体 |
| --- | --- | --- | --- | --- | --- |
| 规则分词 | 7.95 | 7.60 | 8.30 | 7.90 | 7.70 |
| 统计分词 | 6.65 | 6.15 | 7.40 | 6.25 | 6.20 |
| 深度学习分词 | 5.65 | 5.15 | 7.30 | 5.25 | 5.25 |
| 大模型分词 | **9.40** | **9.35** | **9.80** | **9.40** | **9.40** |

结论要点：大模型分词质量最好但成本最高；规则分词在领域词典加持下性价比最高；
统计分词完全无监督、单字词最少但会把「导/游」类搭配切开；
深度学习分词（伪标注监督）略优于纯统计，但受伪标签质量限制，是四者中最有改进空间的一环。

## 10. 依赖

- 推荐环境 `C:\Users\Windy\Documents\Code\Python\WHR\.venv`：`torch 2.10.0+cu130`（CUDA 可用）、
  `jieba`、`pandas`、`numpy`、`opencc`、`scipy`
- 备用环境 `C:\Users\Windy\Desktop\7NLP作业9.23\.venv`：无 torch，深度学习自动走 NumPy 回退
- 大模型部分只用标准库 `urllib`，不需要 `openai` 包

```powershell
# 在任意环境补齐依赖
& $PY -m pip install jieba opencc-python-reimplemented
```

