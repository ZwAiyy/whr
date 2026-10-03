# -*- coding: utf-8 -*-
"""数据源可行性探测：在正式爬取前，快速判断各候选来源的可访问性与可解析性。

运行环境：Label Studio 自带 venv（含 requests）
输出：logs/probe_sources.json + 控制台表格
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

# ---------------- 候选来源清单 ----------------
HTML_SOURCES = {
    "douban_movie_comments": "https://movie.douban.com/subject/1292052/comments?status=P&sort=new_score",
    "douban_book_comments": "https://book.douban.com/subject/1084336/comments/",
    "douban_music_comments": "https://music.douban.com/subject/1394107/comments/",
    "tieba_post": "https://tieba.baidu.com/p/8503785845",
    "autohome_koubei": "https://k.autohome.com.cn/314/",
    "smzdm_post": "https://post.smzdm.com/",
    "ithome_news": "https://www.ithome.com/",
    "chinanews": "https://www.chinanews.com.cn/",
    "thepaper": "https://www.thepaper.cn/",
    "huxiu": "https://www.huxiu.com/",
    "36kr": "https://36kr.com/",
    "zhihu_question": "https://www.zhihu.com/question/19551147",
    "dgtle": "https://www.dgtle.com/",
    "netease_news": "https://news.163.com/",
    "maoyan_movie": "https://www.maoyan.com/films/1200486",
}

API_SOURCES = {
    "netease_music_comments": "https://music.163.com/api/v1/resource/comments/R_SO_4_186016?limit=20&offset=0",
    "netease_music_hot": "https://music.163.com/api/v1/resource/comments/R_SO_4_1330348068?limit=20&offset=0",
    "bilibili_reply": "https://api.bilibili.com/x/v2/reply?type=1&oid=170001&sort=2&ps=20&pn=1",
    "github_api": "https://api.github.com/search/repositories?q=sentiment+analysis&sort=stars&per_page=5",
    "hn_api": "https://hacker-news.firebaseio.com/v0/topstories.json",
}

DATASET_SOURCES = {
    "ChnSentiCorp_htl_all": "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv",
    "weibo_senti_100k": "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/weibo_senti_100k/weibo_senti_100k.csv",
    "waimai_10k": "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv",
    "online_shoppers": "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
    "imdb_50k": "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
}


def probe(kind: str, name: str, url: str, timeout: int = 15) -> dict:
    rec = {"kind": kind, "name": name, "url": url}
    t0 = time.time()
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        rec["status"] = r.status_code
        rec["bytes"] = len(r.content)
        rec["elapsed"] = round(time.time() - t0, 2)
        rec["ctype"] = r.headers.get("Content-Type", "")
        text = r.text
        rec["has_chinese"] = bool(re.search(r"[\u4e00-\u9fff]", text))
        rec["title"] = ""
        m = re.search(r"<title[^>]*>(.*?)</title>", text, re.S | re.I)
        if m:
            rec["title"] = m.group(1).strip()[:80]
        rec["ok"] = r.status_code == 200 and len(r.content) > 500
    except Exception as e:  # noqa: BLE001
        rec.update({"status": None, "ok": False, "error": f"{type(e).__name__}: {e}"[:160], "elapsed": round(time.time() - t0, 2)})
    return rec


def main() -> None:
    results = []
    for kind, group in (("html", HTML_SOURCES), ("api", API_SOURCES), ("dataset", DATASET_SOURCES)):
        for name, url in group.items():
            rec = probe(kind, name, url)
            results.append(rec)
            flag = "OK " if rec.get("ok") else "XX "
            print(f"[{flag}] {kind:8s} {name:26s} status={rec.get('status')} bytes={rec.get('bytes')} cn={rec.get('has_chinese')} {rec.get('title','')[:40]}")
            sys.stdout.flush()

    out = LOG_DIR / "probe_sources.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
