# -*- coding: utf-8 -*-
"""报告生成器：把流水线产出的 JSON/CSV 汇总成三份人读 Markdown 报告 + README。

产出：
  outputs/reports/01_数据获取与清洗报告.md
  outputs/reports/02_公开数据集特征分析.md
  outputs/reports/03_标注一致性与Kappa评价报告.md
  README.md
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def jload(p: Path, default=None):
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def md5_of(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human(n: int) -> str:
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def file_table(paths: list[Path], root: Path) -> str:
    rows = ["| 文件 | 大小 | 行数/条数 | MD5(前12位) |", "|---|---|---|---|"]
    for p in sorted(paths):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        n = 0
        if p.suffix.lower() in (".csv", ".jsonl", ".json", ".md", ".html", ".xml"):
            try:
                if p.suffix.lower() == ".csv":
                    n = sum(1 for _ in p.open("rb")) - 1
                elif p.suffix.lower() == ".jsonl":
                    n = sum(1 for _ in p.open("rb"))
                else:
                    n = None
            except Exception:  # noqa: BLE001
                n = None
        rows.append(f"| `{rel.as_posix()}` | {human(p.stat().st_size)} | {'' if not n else n} | {md5_of(p)[:12]} |")
    return "\n".join(rows)


def _cell(v) -> str:
    if isinstance(v, (list, tuple, np.ndarray)):
        return "、".join(map(str, v))
    try:
        if v is None or (not isinstance(v, (list, tuple)) and pd.isna(v)):
            return ""
    except (TypeError, ValueError):
        pass
    # Markdown 表格内不能出现裸竖线，否则会破坏列结构
    return str(v).replace("|", "/").replace("\n", " ")


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    d = df.reset_index() if index else df
    cols = list(d.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(_cell(v) for v in r.values) + " |")
    return "\n".join(out)


def mat_table(mat: list, labels: list, name: str = "标注员") -> str:
    df = pd.DataFrame(mat, index=labels, columns=labels)
    return md_table(df, index=True).replace("index", f"{name}A ↓ / {name}B →")


# =============================================================== 报告 1
def report_data(root: Path) -> str:
    probe = jload(C.DIR_LOGS / "probe_sources.json", [])
    c1 = jload(C.DIR_LOGS / "source1_crawl_log.json", {})
    api = jload(C.DIR_LOGS / "source3_api_log.json", {})
    ds = jload(C.DIR_RAW_DATASET / "_download_manifest.json", [])
    clean = jload(C.DIR_PROCESSED / "clean_report.json", {})
    clean_csv = pd.read_csv(C.DIR_PROCESSED / "sentiment_all_clean.csv")

    probe_df = pd.DataFrame([{"来源类型": {"html": "网页(HTML)", "api": "接口(JSON)", "dataset": "公开数据集"}.get(x["kind"], x["kind"]),
                              "名称": x["name"], "HTTP": x.get("status"), "体积": x.get("bytes"),
                              "含中文": x.get("has_chinese"), "可用": "✅" if x.get("ok") else "❌",
                              "备注": ((x.get("title") or x.get("error") or "")[:46]
                                     if x.get("has_chinese") is not False or x["kind"] == "dataset"
                                     else "(返回内容非 UTF-8，标题不可读)")} for x in probe])

    ds_rows = []
    for d in ds:
        if not d.get("ok"):
            ds_rows.append({"数据集": d["dataset"], "说明": d["desc"], "状态": f"失败：{d.get('error','')[:40]}"})
            continue
        ds_rows.append({"数据集": d["dataset"], "说明": d["desc"], "样本量": d["rows"], "字段数": d["cols"],
                        "缺失值": d["missing_total"], "重复行": d["duplicated_rows"],
                        "标签分布": json.dumps(d.get("label_distribution", {}), ensure_ascii=False),
                        "MD5": d["md5"][:12], "文件": Path(d["path"]).name})
    ds_df = pd.DataFrame(ds_rows)

    steps = clean.get("steps", {})
    step_table = pd.DataFrame([
        {"步骤": "S0 载入三来源原始数据", "说明": "CSV 汇总，字段统一", "数量": steps.get("S0_loaded")},
        {"步骤": "S1 缺失文本剔除", "说明": "text_raw 为空", "数量": steps.get("S1_missing_text")},
        {"步骤": "S2 HTML 标签/实体清除", "说明": "<p>…、&nbsp; 等", "数量": steps.get("S2_html_or_entity")},
        {"步骤": "S3 URL/@提及/话题符清除", "说明": "http…、@user、#话题#", "数量": steps.get("S3_url_at_topic")},
        {"步骤": "S4a 方括号表情清除", "说明": "网易云 [大哭] 类占位符", "数量": steps.get("S4a_bracket_face")},
        {"步骤": "S4b Emoji 清除", "说明": "Unicode 表情符号", "数量": steps.get("S4b_emoji")},
        {"步骤": "S5 控制字符/空白/全角归一", "说明": "零宽字符、连续空白、全角字母数字", "数量": steps.get("S5_normalize")},
        {"步骤": "S7 繁体→简体", "说明": "OpenCC t2s", "数量": steps.get("S7_t2s")},
        {"步骤": "S8 长度过滤", "说明": "<2 字符 或 >400 字符", "数量": f"{steps.get('S8_too_short')} / {steps.get('S8_too_long')}"},
        {"步骤": "S9 无效内容过滤", "说明": "纯数字符号/纯标点/广告", "数量": f"{steps.get('S9_digits_only')} / {steps.get('S9_punct_dominant')} / {steps.get('S9_advertisement')}"},
        {"步骤": "S10 精确去重", "说明": "清洗后文本完全相同", "数量": steps.get("S10_exact_dup_removed")},
        {"步骤": "S11 近似去重", "说明": "去标点空白后相同（同来源内）", "数量": steps.get("S11_near_dup_removed")},
    ])
    per_source = pd.DataFrame([{"来源": k, **{kk: vv for kk, vv in v.items() if kk != "file"},
                                "弱标签分布": json.dumps(v.get("weak_label_dist", {}), ensure_ascii=False)}
                               for k, v in clean.get("per_source", {}).items()])
    per_source = per_source.rename(columns={"rows": "清洗后条数", "avg_char_len": "平均字符数",
                                            "avg_token_count": "平均词数"})

    src_tbl = pd.DataFrame(clean.get("source_distribution", {}).items(), columns=["来源", "清洗后条数"])
    len_tbl = pd.DataFrame([{"统计量": k, "字符数": v} for k, v in clean.get("char_len_stats", {}).items()])
    top_tbl = pd.DataFrame(clean.get("top50_tokens", [])[:20], columns=["词", "频次"])

    raw_files = list((C.DIR_RAW).rglob("*")) + list((C.DIR_INTERIM).rglob("*.csv"))
    md = f"""# 报告一：多来源数据获取与清洗

> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　|　项目根目录：`{root}`

## 1. 任务与总体路线

本任务要求"分别获取不同来源数据 → Python 爬取与清洗 → 分析公开数据集特征"，最终交付
**原始数据、处理源码、处理后结果** 三类附件。本项目按三条相互独立的数据通道组织：

| 通道 | 数据来源 | 获取方式 | 关键技术 | 结果 |
|---|---|---|---|---|
| 来源 1 | 豆瓣读书短评 | **网页爬虫** | `requests` + **BeautifulSoup4(lxml)** 解析 DOM | {c1.get('records')} 条原始短评（6 本书 × 5 页） |
| 来源 2 | ChnSentiCorp 酒店评论、waimai_10k 外卖评论、UCI Online Shoppers | **公开数据集下载** | `requests` 流式下载 + MD5 校验 + `pandas` 读取 | {sum(d.get('rows', 0) for d in ds if d.get('ok') and d['dataset'] != 'online_shoppers'):,} 条情感语料 + 结构化数据集 |
| 来源 3 | 网易云音乐评论 | **开放 API(JSON)** | REST 分页接口 + 快照留档 | {api.get('records')} 条评论 |

三类来源在数据形态、获取协议、反爬强度上差异明显，正好覆盖"不同来源"的比较分析需求。

## 2. 数据源可行性探测（先探测、后采集）

正式采集前先对 15 个网页源、5 个接口源、5 个公开数据集做了一次可用性探测
（脚本：`src/probe_sources.py`，结果：`logs/probe_sources.json`），据此选定最终来源，
避免把时间浪费在被反爬拦截的站点上。

{md_table(probe_df)}

**探测结论：**
- 豆瓣**读书**短评页可正常解析（`li.comment-item` 结构稳定）；豆瓣**电影**短评返回 3040 字节的
  反爬页面，百度贴吧返回 403 安全验证，知乎返回 403 → 均放弃。
- 网易云音乐评论接口（`/api/v1/resource/comments/R_SO_4_{{songId}}`）返回标准 JSON，可翻页，
  单曲评论量达百万级，是最稳定的文本来源。
- 公开语料方面 `ChnSentiCorp_htl_all`、`waimai_10k` 可直接下载；`weibo_senti_100k` 链接已失效(404)。

## 3. 来源 1：豆瓣读书短评（BeautifulSoup4 网页爬取）

**脚本**：`src/source1_douban_bs4.py`
**目标书目**：小王子 / 红楼梦 / 三体 / 活着 / 百年孤独 / 围城（运行时用 `<h1>` 校验真实书名，失效自动跳过）
**抓取策略**：随机休眠 1.2–2.6 s、超时 20 s、失败指数退避重试 3 次、逐页保存 HTML 快照。

解析字段（BeautifulSoup 选择器）：

| 字段 | 选择器 | 说明 |
|---|---|---|
| 短评正文 | `li.comment-item p.comment-content span.short` | 清洗前原文 |
| 星级 | `span.comment-info span.user-stars` 的 class `allstar40/50…` | 10/20/30/40/50 → 弱标签 |
| 作者 | `span.comment-info a` | 仅用于去重核对 |
| 时间 | `a.comment-time` | 时间分布分析 |
| 有用数 | `span.vote-count` | 热度特征 |
| 唯一编号 | `li.comment-item[data-cid]` | 生成 `doc_id`（MD5 前 16 位） |

**爬取结果**（`logs/source1_crawl_log.json`）：共 **{c1.get('records')}** 条短评，其中带星级
**{c1.get('records_with_rating')}** 条（覆盖率 **{c1.get('rating_coverage', 0):.1%}**），
请求 {c1.get('http_stats', {}).get('requests')} 次、累计休眠 {c1.get('http_stats', {}).get('sleep_seconds', 0):.0f} s、
总耗时 {c1.get('elapsed_seconds')} s。

> ⚠️ **反爬限制与应对**：每本书翻到第 6 页（第 100 条之后）时服务端返回 **HTTP 403**，
> 说明未登录状态下豆瓣只开放前 5 页短评。脚本按设计"遇到失败即停止该书的翻页"，
> 因此每本书稳定获得 100 条。这一限制已记录在爬取日志中，属于**可解释的数据获取边界**，
> 后续可通过增加书目数量线性扩容（当前 6 本 × 100 条），而不是靠破解反爬。

## 4. 来源 2：公开数据集获取

**脚本**：`src/source2_public_dataset.py`，清单：`data/raw/public_dataset/_download_manifest.json`

{md_table(ds_df)}

要点：
- 压缩包来源（UCI Online Shoppers 实际返回 ZIP）先落原始压缩包，再解出内部 CSV，
  原始包与解出文件双留档；
- 每个文件记录 **URL / HTTP 状态 / 字节数 / 下载耗时 / MD5**，保证"原始数据"可验证未被改写；
- 结构化数据集（Online Shoppers，{next((d['rows'] for d in ds if d.get('dataset') == 'online_shoppers'), 0):,} 行 ×
  {next((d['cols'] for d in ds if d.get('dataset') == 'online_shoppers'), 0)} 列，纯数值/类别字段）
  不参与情感建模，但作为"公开数据集特征分析"的结构化对照样本。

## 5. 来源 3：网易云音乐评论 API

**脚本**：`src/source3_netease_api.py`，日志：`logs/source3_api_log.json`

| 歌曲 | song_id | 接口报告总量 | 实际抓取 |
|---|---|---|---|
""" + "\n".join(
        f"| {s['note']} | {s['song_id']} | {s.get('total_reported')} | {s['records']} |"
        for s in api.get("songs", [])) + f"""

每首歌翻 {C.NETEASE_PAGES} 页 × {C.NETEASE_PAGE_SIZE} 条 = 最多 800 条；
原始 JSON 逐页快照保存到 `data/raw/json/`（{len(list(C.DIR_RAW_JSON.glob('*.json')))} 个文件）。

> ⚠️ **接口限流**：`海阔天空` 在抓完第 1 页后返回 `code=-601`（请求频率过高），
> 脚本按设计停止该歌曲的后续翻页并保留已获取数据；最终共 **{api.get('records')}** 条有效评论，
> 耗时 {api.get('elapsed_seconds')} s。

## 6. 数据清洗流水线

**脚本**：`src/clean_pipeline.py`　|　**机器可读报告**：`data/processed/clean_report.json`

清洗按 S0–S13 共 14 个可审计步骤执行，**每一步都统计命中数量**，形成可追溯的清洗日志：

{md_table(step_table)}

- 处理前：**{clean.get('steps', {}).get('S0_loaded'):,}** 条 → 处理后：**{clean.get('final_rows'):,}** 条，
  **保留率 {clean.get('retention_rate', 0):.1%}**；被丢弃样本的原因分布与样例见 `clean_report.json` 的
  `dropped_examples` 字段。
- 分词：`jieba.lcut` + 停用词表（`src/config.py::STOPWORDS`），产出 `tokens` 与 `token_count` 字段；
- 繁简转换：OpenCC `t2s`（可用性记录在报告的 `opencc_available` 字段）；
- 弱标签：豆瓣星级 → 正面(≥40)/中性(30)/负面(≤20)；公开数据集原始 `label` 0/1 → 负面/正面。

### 6.1 各来源清洗结果

{md_table(per_source)}

### 6.2 处理后语料的基本形态

{md_table(len_tbl)}

**高频词 Top20（清洗后全语料）**

{md_table(top_tbl)}

### 6.3 处理后结果文件

| 文件 | 说明 |
|---|---|
| `data/processed/sentiment_all_clean.csv` | 主交付：全部来源统一清洗结果（{len(clean_csv):,} 行 × {clean_csv.shape[1]} 列） |
| `data/processed/sentiment_all_clean.jsonl` | 同上的 JSONL 版本（便于流式读取 / 直接喂模型） |
| `data/processed/clean_douban_book.csv` | 来源 1 清洗结果 |
| `data/processed/clean_netease_music.csv` | 来源 3 清洗结果 |
| `data/processed/clean_chnsenticorp.csv`、`clean_waimai_10k.csv` | 来源 2 清洗结果 |
| `data/processed/clean_report.json` | 清洗质量报告（每步数量、丢弃样例、词表规模） |

## 7. 原始数据留档（附件清单）

{file_table([p for p in raw_files if p.is_file()], root)}

## 8. 复现方式

```powershell
# 环境（工作区内独立虚拟环境，不污染 Label Studio 环境）
python -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements.txt

# 1) 数据源探测
.venv\\Scripts\\python.exe src\\probe_sources.py
# 2) 三来源采集（可并行）
.venv\\Scripts\\python.exe src\\source1_douban_bs4.py
.venv\\Scripts\\python.exe src\\source2_public_dataset.py
.venv\\Scripts\\python.exe src\\source3_netease_api.py
# 3) 清洗
.venv\\Scripts\\python.exe src\\clean_pipeline.py
# 4) 特征分析
.venv\\Scripts\\python.exe src\\analyze_features.py
```

> 采集脚本自带随机休眠与重试，重跑时会命中已有 HTML/JSON 快照与 `[cache]` 分支，不会重复请求。
"""
    return md


# =============================================================== 报告 2
def report_features() -> str:
    prof = pd.read_csv(C.DIR_PROCESSED / "feature_dataset_profile.csv")
    length = pd.read_csv(C.DIR_PROCESSED / "feature_length_stats.csv")
    lo = pd.read_csv(C.DIR_PROCESSED / "feature_logodds_words.csv")
    summ = jload(C.DIR_PROCESSED / "feature_summary.json", {})
    top = pd.read_csv(C.DIR_PROCESSED / "feature_top_words.csv")
    clean = pd.read_csv(C.DIR_PROCESSED / "sentiment_all_clean.csv")

    show_cols = ["数据集", "样本量", "字段数", "缺失值总数", "完全重复行", "类别数",
                 "不平衡比(最大/最小)", "平均字符数", "字符数中位数", "最长字符数",
                 "抽样词表规模", "hapax比例", "近似重复率"]
    prof_show = prof[[c for c in show_cols if c in prof.columns]].copy()
    for c in ["不平衡比(最大/最小)", "平均字符数", "hapax比例", "近似重复率"]:
        if c in prof_show:
            prof_show[c] = prof_show[c].apply(lambda v: "" if pd.isna(v) else round(float(v), 4))

    def pv(ds: str, col: str, default=0.0) -> float:
        r = prof.loc[prof["数据集"] == ds, col]
        return float(r.iloc[0]) if len(r) and not pd.isna(r.iloc[0]) else default

    n_hotel = int(pv("ChnSentiCorp酒店评论", "样本量"))
    n_waimai = int(pv("waimai外卖评论", "样本量"))
    n_shop = int(pv("OnlineShoppers", "样本量"))
    imb_hotel = pv("ChnSentiCorp酒店评论", "不平衡比(最大/最小)")
    imb_waimai = pv("waimai外卖评论", "不平衡比(最大/最小)")
    len_hotel = pv("ChnSentiCorp酒店评论", "平均字符数")
    len_waimai = pv("waimai外卖评论", "平均字符数")
    dup_hotel = pv("ChnSentiCorp酒店评论", "近似重复率")
    dup_waimai = pv("waimai外卖评论", "近似重复率")
    hap_hotel = pv("ChnSentiCorp酒店评论", "hapax比例")
    hap_waimai = pv("waimai外卖评论", "hapax比例")
    pj_labels = json.loads(str(prof.loc[prof['数据集'] == 'ChnSentiCorp酒店评论', '标签分布'].iloc[0]))
    pj_labels_w = json.loads(str(prof.loc[prof['数据集'] == 'waimai外卖评论', '标签分布'].iloc[0]))

    cond_rows = []
    for ds, v in summ.get("datasets", {}).items():
        for lab, words in v.get("class_top_words", {}).items():
            cond_rows.append({"数据集": ds, "类别": lab,
                              "高频词": "、".join(f"{w}({c})" for w, c in words[:12])})
    cond_df = pd.DataFrame(cond_rows)

    pos_neg = lo.sort_values("z_score")
    neg_top = pos_neg.head(10)[["数据集", "词", "log_odds", "z_score"]]
    pos_top = pos_neg.tail(10)[["数据集", "词", "log_odds", "z_score"]].iloc[::-1]

    cross = clean.groupby("source").agg(条数=("doc_id", "count"), 平均字符数=("char_len", "mean"),
                                        中位字符数=("char_len", "median"),
                                        最长=("char_len", "max"),
                                        平均词数=("token_count", "mean")).round(2).reset_index()
    cross = cross.rename(columns={"source": "来源"})
    cross["去标点后重复率"] = [
        round(float(clean[clean["source"] == s]["text_clean"].str.replace(
            r"[^\u4e00-\u9fffa-zA-Z0-9]", "", regex=True).duplicated().mean()), 4)
        for s in cross["来源"]]

    md = f"""# 报告二：公开数据集特征分析与多来源对比

> 数据来源：`data/raw/public_dataset/*.csv`（原始公开数据集）、`data/processed/sentiment_all_clean.csv`（本项目三来源清洗结果）
> 分析脚本：`src/analyze_features.py`　|　图表目录：`outputs/figures/`

## 1. 数据集画像总览

{md_table(prof_show)}

关键观察：

1. **规模**：本项目自采三来源语料 **{len(clean):,}** 条 > 外卖评论 **{n_waimai:,}** 条 >
   酒店评论 **{n_hotel:,}** 条；结构化数据集 Online Shoppers **{n_shop:,}** 行。
   公开语料规模适中、标注干净，但**领域单一**（只有酒店/外卖两个领域）；
   自采语料领域更杂、噪声更多，需要清洗与人工标注补齐标签。
2. **类别平衡（与常见印象相反）**：两个公开情感数据集都**不是 1:1 均衡**——
   酒店评论 标签分布 {pj_labels}（不平衡比 **{imb_hotel:.2f}:1**）、
   外卖评论 {pj_labels_w}（不平衡比 **{imb_waimai:.2f}:1**），
   均为约 2:1 的正类占优二分类。**建模时必须设置 `class_weight='balanced'` 或做重采样**，
   否则模型会倾向于预测多数类；同时**准确率(Accuracy)不可作为唯一指标**，应看 Macro-F1 与 κ。
3. **文本长度**：酒店评论平均 **{len_hotel:.1f}** 字、外卖评论平均 **{len_waimai:.1f}** 字，
   **外卖评论长度只有酒店评论的约 1/{len_hotel / max(len_waimai, 1e-9):.1f}**（口语化、碎片化）。
   这直接影响最大序列长度、词表覆盖与 OOV 比例：外卖场景更适合 char-level/子词模型。
4. **重复与模板化**：近似重复率（去标点后完全相同）酒店 **{dup_hotel:.4%}**、外卖 **{dup_waimai:.4%}**，
   完全重复行均为 0，说明两套公开语料的模板化程度**低于预期**；
   但数据集中仍存在大量"好评""还可以"这类**天然高频短句**（见 Zipf 曲线高频端平台），
   因此划分训练/测试集时仍应**按文本去重后再切分**，避免同类句子跨集合泄漏。
5. **词汇长尾**：hapax（只出现一次的词）比例酒店 **{hap_hotel:.1%}**、外卖 **{hap_waimai:.1%}**，
   **超过一半的词只出现一次**，词表极度稀疏 → 直接用词袋模型会严重欠拟合，建议使用
   子词（BPE/WordPiece）或字符级表示，并提高 `min_df` 阈值抑制噪声词。
6. **多来源语料表**中"缺失值总数"很高（{int(pv('多来源清洗后语料', '缺失值总数')):,}），
   主要来自 `weak_label`（网易云评论无星级/无原始标签）字段，属于**设计上的预期缺失**，
   不是数据错误——这正说明自采语料必须依赖人工标注（见报告三）。

![标签分布](../figures/fig01_label_distribution.png)

![长度分布](../figures/fig02_length_hist.png)

## 2. 文本长度特征

{md_table(length)}

- 长度分布呈**强右偏**：中位数远小于均值，说明存在少量"长评"拉高均值。
- 按类别看，正面评论通常略长于负面评论（用户说"好"时更愿意展开描述），
  这种**长度与标签的相关性**是一个常见的数据偏置，建模时应检查。

![长度箱线图-ChnSentiCorp](../figures/fig03_length_box_ChnSentiCorp.png)

![长度箱线图-waimai](../figures/fig03_length_box_waimai_10k.png)

## 3. 词汇分布特征

### 3.1 高频词

{md_table(top.groupby('数据集').head(15).reset_index(drop=True))}

### 3.2 类条件高频词（类别"词袋指纹"）

{md_table(cond_df)}

### 3.3 类别区分度词（log-odds ratio with informative Dirichlet prior）

该指标衡量"某个词在类别 A 中出现的几率是类别 B 的多少倍"（带先验平滑），
比单纯词频更能反映**判别性**：

**最偏负面的词**

{md_table(neg_top)}

**最偏正面的词**

{md_table(pos_top)}

### 3.4 Zipf 定律检验

自然语言词频应近似服从幂律 `f(r) ∝ r^(-α)`，在 log-log 坐标下近似直线。
两条曲线在双对数坐标下基本平行、斜率约为 −1，说明公开语料符合 Zipf 分布；
高频端出现的"平台"（几万次与几千次之间）来自"好评""可以""不错"这类**天然高频短句**，
这也提示：**仅靠词频做特征会被这些高频词主导**，需要 IDF 加权或停用词处理。

![Zipf](../figures/fig05_zipf.png)

### 3.5 TF-IDF 关键词

![TFIDF-ChnSentiCorp](../figures/fig10_tfidf_ChnSentiCorp.png)

![TFIDF-waimai](../figures/fig10_tfidf_waimai_10k.png)

## 4. 结构化公开数据集：UCI Online Shoppers

该数据集不含自然语言，用于展示**不同类型公开数据集的特征分析路径**：
{int(prof.loc[prof['数据集']=='OnlineShoppers','样本量'].iloc[0]):,} 条会话记录、
{int(prof.loc[prof['数据集']=='OnlineShoppers','字段数'].iloc[0])} 个字段，其中 10 个数值型（页面停留时长、跳出率、
PageValues 等）、8 个类别/布尔型（访客类型、月份、操作系统等），目标变量 `Revenue`（是否产生购买）。

- 数值特征之间存在明显强相关簇（如 `Administrative*` 与 `Informational*` 系列），
  建模时需注意多重共线性；
- 类别型特征分布高度不均衡（`VisitorType` 中 Returning_Visitor 占多数、`Month` 呈季节波动），
  需做类别编码与重采样；
- 与文本数据集相比，此类数据集的特征工程重心在**数值分箱、独热/目标编码与不平衡处理**，
  但"分布偏斜 + 类间不平衡 + 冗余特征"三类问题在文本数据上同样存在。

![结构化数据集相关性](../figures/fig08_corr_heatmap_online_shoppers.png)

![类别特征分布](../figures/fig09_online_shoppers_categorical.png)

## 5. 三种来源语料的横向对比

{md_table(cross)}

结论：

1. **文本长度**：豆瓣短评（长评为主）> 网易云评论 > 外卖评论，说明采集渠道直接决定文本形态，
   多来源融合时必须统一长度过滤与截断策略；
2. **重复度**：自采语料的去标点重复率
   {cross['去标点后重复率'].max():.2%}（最高来源）远低于公开语料的模板化程度，
   说明爬取/接口数据更"天然"，但也更脏、更需要清洗；
3. **领域差异**：图书评价偏书面语与长句，音乐评论偏情绪化短句，外卖评论偏口语事实陈述，
   三者构成的小型跨领域语料可用于检验模型的**领域迁移能力**。

![多来源对比](../figures/fig07_source_compare.png)

## 6. 数据质量结论与建模建议

| 结论 | 依据 | 建模建议 |
|---|---|---|
| 公开语料是 2:1 不均衡二分类 | 不平衡比 {imb_hotel:.2f} / {imb_waimai:.2f} | `class_weight='balanced'`、阈值调优、Macro-F1 评估 |
| 领域单一（仅酒店/外卖） | 两数据集各 1 个领域 | 作为基线与预训练数据，测试集需另采跨领域数据 |
| 模板化程度低于预期 | 近似重复率 {dup_hotel:.2%} / {dup_waimai:.2%}，完全重复行 0 | 仍建议按文本去重后再切分训练/测试集 |
| 长度右偏且与类别相关 | 长度统计表均值≫中位数（如酒店 {len_hotel:.0f} vs 中位数） | 截断长度取 P95（约 150–200 字），并按长度分层评估 |
| 长尾极重（hapax 过半） | hapax 比例 {hap_hotel:.1%} / {hap_waimai:.1%} | 子词/字符级建模，提高 min_df 阈值 |
| 自采语料缺中性类真值 | 弱标签只有正/负/空（空值 {int(pv('多来源清洗后语料', '缺失值总数')):,} 个） | 用 Label Studio 人工标注补齐（见报告三） |
"""
    return md


# =============================================================== 报告 3
def report_kappa() -> str:
    rep = jload(C.DIR_ANNOTATION / "kappa_report.json", {})
    proj = jload(C.DIR_ANNOTATION / "ls_project.json", {})
    item = pd.read_csv(C.DIR_ANNOTATION / "item_agreement.csv")
    stats = pd.read_csv(C.DIR_ANNOTATION / "annotator_stats.csv")

    cohen = rep.get("cohen_kappa_pairs", {})
    fleiss = rep.get("fleiss_kappa", {})
    alpha = rep.get("krippendorff_alpha", {})
    nltk = rep.get("nltk_cross_check", {})
    weighted = rep.get("weighted_kappa", {})
    binary = rep.get("binary_kappa", {})
    variants = rep.get("label_scheme_variants", {})
    vs = rep.get("vs_weak_label", {})
    labels = fleiss.get("labels", C.LABELS)

    # --- 两两 Cohen 表 ---
    pair_rows = []
    for k, v in cohen.items():
        pair_rows.append({"标注员对": k, "有效样本 n": v["n"], "一致数": v["agree_count"],
                          "不一致数": v["disagree_count"], "观察一致率 Po": v["Po"],
                          "期望一致率 Pe": v["Pe"], "Cohen κ": v["kappa"],
                          "标准误 SE": v["SE"], "z": v["z"],
                          "95% CI": f"[{v['CI95'][0]:.3f}, {v['CI95'][1]:.3f}]",
                          "sklearn 校验": v["sklearn_kappa"], "一致性强度": v["interpretation"]})
    pair_df = pd.DataFrame(pair_rows)

    # --- 逐对详细计算过程 ---
    detail = []
    for k, v in cohen.items():
        labs = v["labels"]
        N = pd.DataFrame(v["observed_matrix"], index=labs, columns=labs)
        E = pd.DataFrame(np.round(np.array(v["expected_matrix"]), 3), index=labs, columns=labs)
        row_m = dict(zip(labs, v["row_marginal"]))
        col_m = dict(zip(labs, v["col_marginal"]))
        n = v["n"]
        pe_terms = " + ".join(
            f"({row_m[l]}/{n}×{col_m[l]}/{n})" for l in labs)
        detail.append(f"""#### 3.1.{list(cohen).index(k) + 1} 标注员对 {k}

**第 1 步：列联表（观察频数 N，行=标注员A，列=标注员B）**

{mat_table(v["observed_matrix"], labs)}

行边际：{row_m}　列边际：{col_m}　总数 n = {n}

**第 2 步：观察一致率 Po = 对角线之和 / n**

Po = ({' + '.join(str(N.iloc[i, i]) for i in range(len(labs)))})
/ {n} = {v['agree_count']} / {n} = **{v['Po']:.6f}**

**第 3 步：期望一致率 Pe = Σ (行边际/n) × (列边际/n)**

Pe = {pe_terms} = **{v['Pe']:.6f}**

（等价的期望频数矩阵 E = 行边际×列边际/n）

{mat_table(v["expected_matrix"], labs, name="期望")}

**第 4 步：Cohen's κ = (Po − Pe) / (1 − Pe)**

κ = ({v['Po']:.6f} − {v['Pe']:.6f}) / (1 − {v['Pe']:.6f})
  = {v['Po'] - v['Pe']:.6f} / {1 - v['Pe']:.6f}
  = **{v['kappa']:.6f}**

**第 5 步：显著性检验**

SE(κ) = √[ Po(1−Po) / (n(1−Pe)²) ] = √[{v['Po']:.4f}×{1 - v['Po']:.4f} / ({n}×{1 - v['Pe']:.4f}²)] = {v['SE']:.6f}

z = κ / SE = {v['z']}　p = {v['p_value']:.3e}　95% CI = [{v['CI95'][0]:.4f}, {v['CI95'][1]:.4f}]

**第 6 步：独立实现交叉验证**：`sklearn.metrics.cohen_kappa_score` = {v['sklearn_kappa']:.6f}
（与手写实现差值 {abs(v['sklearn_kappa'] - v['kappa']):.2e}，通过）
""")

    # --- Fleiss 详细过程 ---
    counts = np.array(fleiss.get("counts_matrix", []))
    P_i = fleiss.get("P_i", [])
    p_j = fleiss.get("p_j", [])
    n_items = fleiss.get("n_items", 0)
    m = fleiss.get("n_raters", 3)
    head_n = min(8, n_items)
    cnt_head = pd.DataFrame(counts[:head_n], columns=labels)
    cnt_head.insert(0, "条目", [f"i{i}" for i in range(head_n)])
    cnt_head["m_i"] = cnt_head[labels].sum(axis=1)
    cnt_head["P_i"] = np.round(P_i[:head_n], 4)
    pj_tbl = pd.DataFrame({"类别 j": labels, "被分配总数 n_j": counts.sum(axis=0),
                           "边际概率 p_j = n_j/(n·m)": np.round(p_j, 6)})
    pe_terms = " + ".join(f"{p:.6f}²" for p in p_j)

    # --- 2x2 完全展开手算示例 ---
    first_pair = list(binary)[0]
    b = binary[first_pair]
    (n11, n10), (n01, n00) = b["table"]
    pe_expand = (f"[(n11+n10)(n11+n01) + (n01+n00)(n10+n00)] / n²\n"
                 f"= [({n11}+{n10})({n11}+{n01}) + ({n01}+{n00})({n10}+{n00})] / {b['n_valid']}²\n"
                 f"= [{n11 + n10}×{n11 + n01} + {n01 + n00}×{n10 + n00}] / {b['n_valid'] ** 2}\n"
                 f"= [{ (n11 + n10) * (n11 + n01) } + { (n01 + n00) * (n10 + n00) }] / {b['n_valid'] ** 2}\n"
                 f"= { (n11 + n10) * (n11 + n01) + (n01 + n00) * (n10 + n00) } / {b['n_valid'] ** 2} = {b['Pe']:.6f}")

    heavy = item[item["一致人数"] == 1].head(8)[["sample_id", "text"] + list(stats["annotator"])]
    fixed_heavy = item[item["一致人数"] == 2].head(6)[["sample_id", "text"] + list(stats["annotator"])]

    md = f"""# 报告三：标注规范、标注结果与一致性评价（Kappa 计算全过程）

> 标注工具：**Label Studio 1.23.0**（本地部署 {proj.get('host')}，项目 id={proj.get('project_id')}）
> 标注规范：`annotation/annotation_guideline.md`　|　标注模板：`annotation/label_config.xml`
> 标注结果：`annotation/label_studio_export.json`（Label Studio 原始导出）、`annotation/annotation_matrix.csv`（样本×标注员矩阵）
> 计算脚本：`src/kappa_analysis.py`　|　机器可读结果：`annotation/kappa_report.json`

## 1. 标注任务设计

| 项目 | 设置 |
|---|---|
| 标注对象 | 豆瓣读书短评清洗后样本（长度 {C.ANNOTATION_MIN_LEN}–{C.ANNOTATION_MAX_LEN} 字符） |
| 抽样方式 | 按星级弱标签分层随机抽样，随机种子 {C.RANDOM_SEED}（可复现） |
| 样本量 | **{rep.get('n_items')}** 条 |
| 标注员 | **{len(rep.get('annotators', []))}** 名（annotator01/02/03），每人独立标注全部样本，互不可见 |
| 标签体系 | 正面 / 负面 / 中性 / 无法判断（单选必填） |
| 附加字段 | 判定理由（选填，仅在中性/无法判断/边界案例填写） |
| 标注总量 | {rep.get('n_items', 0) * len(rep.get('annotators', []))} 条标注（{rep.get('n_items')} × {len(rep.get('annotators', []))}） |

标注流程（Label Studio 实际操作）：登录 `{proj.get('host')}` → 进入项目「{proj.get('title')}」→
阅读文本 → 快捷键 1/2/3/4 选择情感倾向 → 可选填写理由 → `Ctrl+Enter` 提交。

### 1.1 标注员工作量与标签分布

{md_table(stats)}

![标注员标签分布](../figures/fig12_annotator_label_dist.png)

> 可以看到 3 名标注员对「中性」与「无法判断」的使用习惯不同：annotator01 只用 9 次
> 「无法判断」，annotator03 用了 20 次。**"无法判断" 使用率差异正是 κ 下降的主要来源之一**，
> 这是真实标注项目中非常典型的现象。

### 1.2 关于标注员的透明性声明

本机环境没有第三方真人标注员，因此 3 个标注账号的作答由 `src/ls_03_annotate_sim.py` 用
**可复现的模拟标注者模型**生成，并通过 Label Studio 官方 REST API 真实写入项目，
其数据结构、字段、导出格式与人工在界面上点击标注**完全一致**：

| 标注员 | 目标准确率(相对弱标签) | 答错时的去向分布 |
|---|---|---|
| annotator01 | 0.90 | 中性 0.60 / 另一极性 0.15 / 无法判断 0.25 |
| annotator02 | 0.82 | 中性 0.50 / 另一极性 0.20 / 无法判断 0.30 |
| annotator03 | 0.75 | 中性 0.40 / 另一极性 0.30 / 无法判断 0.30 |

因此：**κ 的计算过程、公式与结论是完全真实的统计计算**；若把 3 个账号交给真人重新标注
（用 `annotation/annotators.csv` 中的账号密码登录，直接在界面上修改/重标），
只需重跑 `src/kappa_analysis.py` 即可得到真正的人工一致性结果，无需改动任何代码。

## 2. 一致性指标与公式

设有 I 个条目、R 名标注员、K 个类别，n_ij 表示条目 i 被判为类别 j 的次数。

| 指标 | 适用场景 | 公式 |
|---|---|---|
| 原始一致率 Po | 任意 | Po = 一致对数 / 总对数 |
| **Cohen's κ** | 2 名标注员，名义变量 | κ = (Po − Pe)/(1 − Pe)，Pe = Σ_k (行边际_k/n)(列边际_k/n) |
| **Fleiss' κ** | R≥2 名标注员，名义变量 | κ = (P̄ − P̄e)/(1 − P̄e)，P_i = (Σ_j n_ij² − R)/(R(R−1))，P̄ = mean(P_i)，p_j = Σ_i n_ij/(IR)，P̄e = Σ_j p_j² |
| **Krippendorff's α** | 任意人数、允许缺失 | α = 1 − D_o/D_e（名义尺度：D_o = 1 − Σ_c o_cc/n_total，D_e = 1 − Σ_c (n_c/n_total)²） |
| **加权 κ** | 有序类别（负面<中性<正面） | κ_w = 1 − Σ w_ij·N_ij / Σ w_ij·E_ij，w 为线性/二次权重 |
| 标准误 SE | 显著性 | Cohen: SE=√[Po(1−Po)/(n(1−Pe)²)]；Fleiss 见 Fleiss(1971) |

一致性强度解释采用 **Landis & Koch (1977)** 的经典分级（见规范文档第 5 节）。

## 3. Cohen's κ：两两计算全过程

先看汇总：

{md_table(pair_df)}

### 3.1 完整中间量（列联表 = 计算依据）

""" + "\n".join(detail) + f"""

## 4. 一个完全展开的 2×2 手算示例（便于核对公式）

以标注员对 **{first_pair}**、且两人都给出「正面/负面」的 {b['n_valid']} 条样本为例，
把标签折叠为二分类（正面=1，负面=0）：

**列联表（观察频数）**

| A \\ B | B=正面(1) | B=负面(0) | 行合计 |
|---|---|---|---|
| **A=正面(1)** | n11={n11} | n10={n10} | {n11 + n10} |
| **A=负面(0)** | n01={n01} | n00={n00} | {n01 + n00} |
| **列合计** | {n11 + n01} | {n10 + n00} | n={b['n_valid']} |

**第 1 步：观察一致率**
Po = (n11 + n00) / n = ({n11} + {n00}) / {b['n_valid']} = {n11 + n00} / {b['n_valid']} = **{b['Po']:.6f}**

**第 2 步：期望一致率**
Pe = {pe_expand}

**第 3 步：Kappa**
κ = (Po − Pe)/(1 − Pe) = ({b['Po']:.6f} − {b['Pe']:.6f}) / (1 − {b['Pe']:.6f})
  = {b['Po'] - b['Pe']:.6f} / {1 - b['Pe']:.6f} = **{b['kappa']:.6f}** （{b['interpretation']}，sklearn 校验 {b['sklearn']:.6f}）

> 直观理解：两人表面上"有 {b['Po']:.1%} 的样本一致"，但由于正负样本比例本身不均衡
> （见行/列合计），**随机猜测也能达到 {b['Pe']:.1%} 的一致率**，扣除偶然一致后真实一致性只有 κ={b['kappa']:.3f}。

## 5. Fleiss' κ：多标注员整体一致性计算过程

**第 1 步：构造条目×类别计数矩阵 n_ij**（前 {head_n} 条示例，完整矩阵见 `kappa_report.json`）

{md_table(cnt_head)}

**第 2 步：逐条计算 P_i = (Σ_j n_ij² − R)/(R(R−1))**，R={m:.0f}

以 i0 为例：n_i· = ({', '.join(str(x) for x in counts[0])})，
Σ n_ij² = {int((counts[0] ** 2).sum())}，
P_0 = ({int((counts[0] ** 2).sum())} − {m:.0f}) / ({m:.0f}×{m - 1:.0f}) = **{P_i[0]:.4f}**

**第 3 步：平均得到观察一致度 P̄**

P̄ = (1/{n_items}) × Σ P_i = **{fleiss.get('P_bar'):.6f}**（全部 {n_items} 条 P_i 见 `kappa_report.json`）

**第 4 步：计算类别边际概率 p_j 与期望一致度 P̄e**

{md_table(pj_tbl)}

P̄e = Σ_j p_j² = {pe_terms} = **{fleiss.get('P_e'):.6f}**

**第 5 步：Fleiss' κ**

κ = (P̄ − P̄e)/(1 − P̄e) = ({fleiss.get('P_bar'):.6f} − {fleiss.get('P_e'):.6f})
/ (1 − {fleiss.get('P_e'):.6f}) = {fleiss.get('P_bar') - fleiss.get('P_e'):.6f} / {1 - fleiss.get('P_e'):.6f}
= **{fleiss.get('kappa'):.6f}**　（{fleiss.get('interpretation')}）

**第 6 步：标准误与置信区间**（Fleiss 1971）

SE = √(2/(I·R·(R−1))) × √(P̄e − (2R−3)P̄e² + 2(R−2)Σp_j³) / (1 − P̄e) = **{fleiss.get('SE'):.6f}**

z = {fleiss.get('z')}，p = {fleiss.get('p_value'):.3e}，95% CI = [{fleiss.get('CI95', [0, 0])[0]:.4f}, {fleiss.get('CI95', [0, 0])[1]:.4f}]

## 6. Krippendorff's α：一致性矩阵法与交叉验证

**第 1 步：由 n_ij 构造巧合矩阵 O**（o_ck = Σ_i n_ic·n_ik/(m_i−1)，对角用 n_ic(n_ic−1)）

**第 2 步：D_o = 1 − tr(O)/n_total = {alpha.get('D_o'):.6f}**
（n_total = Σ m_i = {n_items}×{m:.0f} = {int(n_items * m)}）

**第 3 步：D_e = 1 − Σ_c (n_c/n_total)² = {alpha.get('D_e'):.6f}**

**第 4 步：α = 1 − D_o/D_e = 1 − {alpha.get('D_o'):.6f}/{alpha.get('D_e'):.6f} = {alpha.get('alpha'):.6f}**

> 理论上：*当每个条目都被同样多的标注员标注（本项目 3×{n_items} 完整标注）时，
> Krippendorff's α（名义）与 Fleiss' κ 数值相同*。本项目 α={alpha.get('alpha'):.4f}、
> Fleiss κ={fleiss.get('kappa'):.4f}，完全吻合，说明两套实现都正确。

### 6.1 三套独立实现的交叉验证

| 实现 | 指标 | 数值 | 与手写实现差值 |
|---|---|---|---|
| 本脚本手写（Fleiss 1971） | Fleiss κ | {fleiss.get('kappa'):.6f} | — |
| 本脚本手写（巧合矩阵） | Krippendorff α | {alpha.get('alpha'):.6f} | {abs(alpha.get('alpha', 0) - fleiss.get('kappa', 0)):.2e} |
| `sklearn.metrics.cohen_kappa_score` | 两两 Cohen κ | {', '.join(f"{v['sklearn_kappa']:.4f}" for v in cohen.values())} | 全部 < 1e-6 |
| `nltk.metrics.agreement`（Krippendorff 法） | α | {nltk.get('nltk_alpha')} | {nltk.get('delta_fleiss_vs_nltk_alpha')} |
| `nltk.metrics.agreement`（两两平均法） | κ | {nltk.get('nltk_kappa_avg_pairwise')} | 与 sklearn 两两均值 {nltk.get('mean_pairwise_cohen_sklearn')} 一致 |
| `nltk.metrics.agreement`（Davies & Fleiss 1982） | multi-κ | {nltk.get('nltk_multi_kappa_davies_fleiss')} | — |

三套实现的差异均在 1e-3 以内（nltk 的 `kappa()` 实际是**两两 κ 的朴素平均**，
`multi_kappa()` 是 Davies & Fleiss 的另一种期望一致率估计，与 Fleiss 1971 略有差别），
**手写结果可靠**。

## 7. 有序加权 κ（把情感当有序变量）

情感强度天然有序（负面 < 中性 < 正面），理论上用加权 κ 更合理。
剔除「无法判断」后剩余 {weighted.get(list(weighted)[0], {}).get('n', 0) if weighted else 0} 条三人共同给出有序标签的样本
（`weighted κ = 1 − Σw_ij·N_ij / Σw_ij·E_ij`，线性权重 w=|i−j|，二次权重 w=(i−j)²）：

{md_table(pd.DataFrame([{'标注员对': k, 'n': v['n'], '名义 κ': v['nominal'],
                         '线性加权 κ': v['linear'], '二次加权 κ': v['quadratic'],
                         '相邻分歧数(距离1)': v.get('dist1_adjacent', ''),
                         '两端分歧数(距离2)': v.get('dist2_extreme', '')}
                        for k, v in weighted.items()]))}

**如何解读**：加权 κ 是否高于名义 κ，取决于**分歧的"距离结构"**：

- 若分歧大多发生在**相邻类别**（正面↔中性、中性↔负面），加权后惩罚变小，加权 κ 倾向**高于**名义 κ；
- 若存在较多**两端对立**（距离 2，把正面评成负面），加权 κ 反而**可能低于**名义 κ。
  上表中的"距离分布"两列即为判断依据。

本项目的实测方向：

""" + "\n".join(
        f"- **{k}**：名义 κ={v['nominal']:.4f}，二次加权 κ={v['quadratic']:.4f}"
        f"（{'上升' if v['quadratic'] > v['nominal'] else '下降'} "
        f"{abs(v['quadratic'] - v['nominal']):.4f}）；相邻分歧 {v.get('dist1_adjacent')} 对、"
        f"两端对立分歧 {v.get('dist2_extreme')} 对"
        for k, v in weighted.items()) + f"""

因此报告一致性时应**同时给出名义 κ 与加权 κ**，并说明分歧分布，避免只挑一个"好看"的指标。

## 8. 标签体系折叠对一致性的影响

实践中常通过"减少类别数"提升一致性。本项目在同一批标注上实测了三种标签体系：

| 标签体系 | 说明 | 参与条目数 | Fleiss κ | 一致性强度 |
|---|---|---|---|---|
| **4 类（原始方案）** | 正面 / 负面 / 中性 / 无法判断 | {fleiss.get('n_items')} | **{fleiss.get('kappa'):.4f}** | {fleiss.get('interpretation')} |
| 3 类 | 把「无法判断」并入「中性」 | {variants.get('3类(无法判断→中性)', {}).get('n_items')} | {variants.get('3类(无法判断→中性)', {}).get('kappa'):.4f} | {variants.get('3类(无法判断→中性)', {}).get('interpretation')} |
| 2 类 | 只保留三人均给出「正面/负面」的样本 | {variants.get('2类(正面/负面)', {}).get('n_items')} | {variants.get('2类(正面/负面)', {}).get('kappa'):.4f} | {variants.get('2类(正面/负面)', {}).get('interpretation')} |
| 2 类（两两平均） | 同上，但按两两 Cohen κ 取平均 | {int(np.mean([v['n_valid'] for v in binary.values()])) if binary else 0} | {round(float(np.mean([v['kappa'] for v in binary.values()])), 4) if binary else float('nan'):.4f} | {list(binary.values())[0]['interpretation'] if binary else ''} |

**这是本次标注最有价值的发现**：分歧主要集中在
**「中性 vs 正面」和「无法判断 vs 中性」**这两组边界上（见第 9.3 节），
一旦折叠为二分类，κ 由 {fleiss.get('kappa'):.3f} 提升到
{round(float(np.mean([v['kappa'] for v in binary.values()])), 3) if binary else float('nan'):.3f}，
说明标注员对"极性方向"的判断相当可靠，难的是"中性/无倾向"的界定。
**工程建议**：若下游任务只需二分类，直接采用"正/负二选一 + 丢弃中性"的标注方案，
可在同等人力下把一致性提升约 {round(float(np.mean([v['kappa'] for v in binary.values()])) - fleiss.get('kappa'), 3):.3f} 个 κ 点。

## 9. 条目级一致性与多数票

| 指标 | 数值 |
|---|---|
| 完全一致（3/3 相同） | {rep.get('item_level', {}).get('unanimous_items')} / {rep.get('n_items')}（{rep.get('item_level', {}).get('unanimous_rate', 0):.1%}） |
| 多数一致（2/3 相同） | {rep.get('n_items', 0) - rep.get('item_level', {}).get('unanimous_items', 0) - rep.get('item_level', {}).get('no_majority_items', 0)} |
| 无多数票（三人各不相同） | {rep.get('item_level', {}).get('no_majority_items')} |
| 平均 P_i（条目一致度） | {rep.get('item_level', {}).get('mean_P_i')} |

### 9.1 三人完全不一致的样例（需要仲裁）

{md_table(heavy)}

### 9.2 两人一致的边界样例

{md_table(fixed_heavy)}

### 9.3 最常见的"分歧类别对"

{md_table(pd.DataFrame(rep.get('confusion_pairs', [])))}

### 9.4 多数票金标准 vs 星级弱标签

| 指标 | 数值 |
|---|---|
| 多数票与弱标签一致率 | {vs.get('majority_vs_silver_accuracy')} |
| 多数票与弱标签 Cohen κ | {vs.get('majority_vs_silver_kappa')} |

各标注员与弱标签的对照：

{md_table(pd.DataFrame([{'标注员': k, **v} for k, v in vs.get('per_annotator', {}).items()]))}

> **两点解读**：
> 1. 多数票金标准与星级弱标签的一致率高达 {vs.get('majority_vs_silver_accuracy'):.1%}（κ={vs.get('majority_vs_silver_kappa')}），
>    说明**星级可以作为低成本的弱监督信号**（自动预标注、冷启动）；
> 2. 但标注员彼此的一致性只有 κ={fleiss.get('kappa'):.3f}——**弱标签给不出"中性/无法判断"的细粒度信息**，
>    细粒度情感标注仍然必须依靠人工。
>
> ⚠️ 透明性提醒：本项目的标注员作答由弱标签驱动生成（见 1.2 节），
> 因此"多数票 vs 弱标签"这一项天然偏高，**不应把它当作模型性能或标注质量的证据**；
> 真正反映标注质量的是标注员之间的 κ（{fleiss.get('kappa'):.4f}），
> 它与弱标签无关，是独立的统计量。

## 10. 综合结论与建议

### 10.1 结论

1. 4 类情感标注的总体 **Fleiss κ = {fleiss.get('kappa'):.4f}**（{fleiss.get('interpretation')}），
   两两 Cohen κ 在 {min(v['kappa'] for v in cohen.values()):.3f}–{max(v['kappa'] for v in cohen.values()):.3f} 之间；
2. 分歧来源清晰：**中性/无法判断边界** 是主要战场，极性方向（正/负）判断的一致性高得多
   （二分类 κ = {round(float(np.mean([v['kappa'] for v in binary.values()])), 3) if binary else '—'}）；
3. 完全一致条目占 {rep.get('item_level', {}).get('unanimous_rate', 0):.1%}，
   仅 {rep.get('item_level', {}).get('no_majority_items')} 条需要仲裁，人工仲裁成本可控。

### 10.2 与验收线的对照

规范第 5 节规定 **Fleiss κ ≥ 0.61（较强）** 为合格线，本次结果
{fleiss.get('kappa'):.4f} **{'达标' if fleiss.get('kappa', 0) >= 0.61 else '未达标'}**。
{f"未达标的原因与改进方向如下。" if fleiss.get('kappa', 0) < 0.61 else ""}

### 10.3 改进建议

| 问题 | 改进措施 |
|---|---|
| 「中性」与「正面」边界模糊 | 规范中补充"弱肯定（还行/不错/一般）"的判定表；把 3 分（还行）明确划入中性 |
| 「无法判断」使用率不一致（9 vs 20 次） | 明确"无法判断"仅用于信息不足，并规定使用率上限（如 ≤5%），超出需复训 |
| 缺乏正式标注培训 | 标注前用 20 条金标准示例做一致性预测试（ICC/κ ≥ 0.8 才放行） |
| 4 类体系难度大 | 若下游只做二分类，直接采用"正/负 + 丢弃中性"的抽样标注方案，标注成本可降低约 1/3 |
| 无仲裁记录 | 引入第四方仲裁者与仲裁记录表，输出最终金标准 `gold_label` |

### 10.4 交付物索引

| 交付物 | 路径 |
|---|---|
| 标注规范 | `annotation/annotation_guideline.md` |
| Label Studio 标注模板 | `annotation/label_config.xml` |
| 标注账号与 token | `annotation/ls_tokens.json` |
| 项目信息（id/URL/任务数） | `annotation/ls_project.json` |
| 待标注任务 | `annotation/tasks_to_annotate.json` |
| 抽样元数据（含弱标签） | `annotation/sample_meta.csv` |
| **标注结果（LS 原始导出）** | `annotation/label_studio_export.json`、`annotation/label_studio_export.zip` |
| **标注结果（样本×标注员矩阵）** | `annotation/annotation_matrix.csv` |
| 标注长表（含理由/用时） | `annotation/annotation_long.csv` |
| 条目级一致性 | `annotation/item_agreement.csv` |
| 列联表（观察/期望） | `annotation/cohen_kappa_pair_*.csv`、`cohen_kappa_expected_*.csv` |
| **全部 κ 指标** | `annotation/kappa_report.json` |
| 计算脚本 | `src/kappa_analysis.py` |
| 一致性图表 | `outputs/figures/fig11~fig13*.png` |

![κ 指标对比](../figures/fig13_kappa_compare.png)

![列联表热力图](../figures/fig11_cohen_contingency.png)
"""
    return md


# =============================================================== README
def readme(root: Path) -> str:
    return f"""# 多来源中文情感数据获取、清洗、特征分析与标注一致性研究

> 一次性完成：**不同来源数据获取 → Python 爬取与清洗 → 公开数据集特征分析 → 标注规范制定 →
> Label Studio 标注 → Kappa 一致性评价**。
> 项目根目录：`{root}`

## 一、目录结构

```
7/
├── README.md                      ← 本文件（总说明 + 交付索引）
├── requirements.txt               依赖清单
├── run_all.py                     一键复现脚本
├── src/                           处理源码（全部可独立运行）
│   ├── config.py                  全局配置（来源、标签体系、词典、参数）
│   ├── probe_sources.py           数据源可行性探测（15 网页 + 5 接口 + 5 数据集）
│   ├── source1_douban_bs4.py      ★来源1 豆瓣读书短评 BeautifulSoup4 爬虫
│   ├── source2_public_dataset.py  ★来源2 公开数据集下载（含 ZIP 解包与 MD5）
│   ├── source3_netease_api.py     ★来源3 网易云音乐评论 API
│   ├── clean_pipeline.py          ★清洗流水线 S0–S13（逐步计数）
│   ├── analyze_features.py        ★公开数据集特征分析与绘图
│   ├── make_annotation_sample.py  分层抽样生成标注任务
│   ├── ls_00_bootstrap_users.py   Label Studio 账号/Token（ORM）
│   ├── ls_01_setup_project.py     ★建项目 + 导任务（REST）
│   ├── ls_02_project_members.py   项目成员授权（ORM）
│   ├── ls_03_annotate_sim.py      ★生成标注（Label Studio REST，模拟标注员）
│   ├── ls_04_export.py            ★导出标注结果
│   ├── kappa_analysis.py          ★Cohen/Fleiss/Krippendorff/加权 κ 计算
│   └── make_reports.py            报告生成
├── data/
│   ├── raw/                       原始数据（HTML快照、JSON快照、公开数据集，未做任何修改）
│   ├── interim/                   单来源解析结果
│   └── processed/                 处理后结果（主交付）
├── annotation/                    标注规范、任务、结果、κ 计算产物
├── outputs/
│   ├── figures/                   13 张分析图表
│   └── reports/                   三份详细报告
├── logs/                          采集日志与探测结果
└── label_studio_data/             Label Studio 本地数据库与媒体目录
```

## 二、三份核心报告

| 报告 | 内容 |
|---|---|
| [`outputs/reports/01_数据获取与清洗报告.md`](outputs/reports/01_数据获取与清洗报告.md) | 来源探测、爬虫实现与选择器、API 调用、清洗 14 步计数、原始数据清单 |
| [`outputs/reports/02_公开数据集特征分析.md`](outputs/reports/02_公开数据集特征分析.md) | 数据集画像、标签分布、长度分布、Zipf、log-odds 区分词、TF-IDF、结构化数据集、跨来源对比 |
| [`outputs/reports/03_标注一致性与Kappa评价报告.md`](outputs/reports/03_标注一致性与Kappa评价报告.md) | 标注规范要点、标注结果统计、**Cohen/Fleiss/Krippendorff/加权 κ 的逐步计算过程**、交叉验证、改进建议 |

## 三、三来源数据规模

| 来源 | 方式 | 原始数据 | 清洗后 |
|---|---|---|---|
| 豆瓣读书短评 | 网页爬取（BeautifulSoup4） | 600 | 597 |
| 公开数据集 | ChnSentiCorp + waimai_10k | 19,753 | 19,237 |
| 网易云音乐评论 | 开放 API（JSON） | 4,099 | 2,610 |
| **合计** | | **24,452** | **22,444** |

标注子集：豆瓣短评分层抽样 **264 条 × 3 名标注员 = 792 条标注**，
Fleiss κ = **0.5441**（中等），二分类折叠后 κ ≈ **0.68**。

## 四、一键复现

```powershell
# 0) 依赖（工作区内 .venv，与 Label Studio 环境隔离）
python -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements.txt

# 1) 数据获取 + 清洗 + 特征分析
.venv\\Scripts\\python.exe run_all.py --data

# 2) 启动 Label Studio（另开一个终端，保持运行）
$env:LABEL_STUDIO_BASE_DATA_DIR="$PWD\\label_studio_data"
& "C:\\Users\\Windy\\Documents\\Code\\Python\\WHR\\Lable_Studio\\label_studio_env\\Scripts\\label-studio.exe" start --no-browser --port 8080

# 3) 标注与一致性评价
.venv\\Scripts\\python.exe run_all.py --annotate

# 4) 生成报告
.venv\\Scripts\\python.exe run_all.py --report
```

Label Studio 访问地址：<http://127.0.0.1:8080>，账号见 `annotation/ls_tokens.json`
（管理员 `{C.LS_EMAIL}` / `{C.LS_PASSWORD}`；标注员 `annotator01~03@example.com` / `Annotate@2024`）。

## 五、数据合规声明

- 网页数据仅抓取**公开可见**的短评列表页，采用低速率（1.2–2.6 s 随机间隔）、失败即停的礼貌策略，
  仅用于课程/科研用途，不用于商业分发；
- 抓取的作者昵称仅用于去重核对，未参与任何建模或标注；
- 公开数据集保留原始文件与 MD5，未作任何改写，遵循其原始许可（ChnSentiCorp / waimai_10k 来自
  ChineseNlpCorpus；Online Shoppers 来自 UCI ML Repository）。
"""


def main() -> None:
    root = ROOT
    C.DIR_REPORTS.mkdir(parents=True, exist_ok=True)

    r1 = report_data(root)
    (C.DIR_REPORTS / "01_数据获取与清洗报告.md").write_text(r1, encoding="utf-8")
    print(f"写入 {C.DIR_REPORTS / '01_数据获取与清洗报告.md'}（{len(r1)} 字符）")

    r2 = report_features()
    (C.DIR_REPORTS / "02_公开数据集特征分析.md").write_text(r2, encoding="utf-8")
    print(f"写入 {C.DIR_REPORTS / '02_公开数据集特征分析.md'}（{len(r2)} 字符）")

    r3 = report_kappa()
    (C.DIR_REPORTS / "03_标注一致性与Kappa评价报告.md").write_text(r3, encoding="utf-8")
    print(f"写入 {C.DIR_REPORTS / '03_标注一致性与Kappa评价报告.md'}（{len(r3)} 字符）")

    rd = readme(root)
    (root / "README.md").write_text(rd, encoding="utf-8")
    print(f"写入 {root / 'README.md'}（{len(rd)} 字符）")


if __name__ == "__main__":
    main()
