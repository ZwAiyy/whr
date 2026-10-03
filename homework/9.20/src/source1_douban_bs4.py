# -*- coding: utf-8 -*-
"""来源 1：网页爬取 —— 豆瓣读书短评（BeautifulSoup4 解析）

输出：
  data/raw/html/douban_book_<id>_p<n>.html   原始 HTML 快照（可复现证据）
  data/interim/source1_douban_bs4_raw.csv    解析后的原始记录（未清洗）
  logs/source1_crawl_log.json                爬取日志（请求数/失败/耗时）

说明：
  * 使用 requests 发请求、BeautifulSoup4 解析 DOM，符合"爬虫使用 beautifulsoup4"的要求；
  * 采用低速率 + 随机休眠 + 失败退避的礼貌爬取策略，仅抓取公开可见的短评列表页；
  * 豆瓣部分短评不显示星级，因此星级仅作为"弱标注"用于抽样分层，不作为最终标签。
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

COMMENT_URL = "https://book.douban.com/subject/{sid}/comments/?start={start}&limit=20&status=P&sort=new_score"
SUBJECT_URL = "https://book.douban.com/subject/{sid}/"

FIELDS = ["doc_id", "source", "source_type", "site", "domain", "doc_url", "author",
          "rating_star", "rating_title", "comment_time", "votes", "text_raw",
          "crawl_time", "page"]


def make_doc_id(source: str, key: str) -> str:
    return hashlib.md5(f"{source}::{key}".encode("utf-8")).hexdigest()[:16]


class PoliteSession:
    """带随机休眠与指数退避重试的 requests 会话。"""

    def __init__(self) -> None:
        self.s = requests.Session()
        self.s.headers.update(C.BASE_HEADERS)
        self.stats = {"requests": 0, "ok": 0, "failed": 0, "http_error": [], "sleep_seconds": 0.0}

    def get(self, url: str, referer: str | None = None) -> requests.Response | None:
        for attempt in range(1, C.CRAWL_RETRY + 1):
            try:
                self.stats["requests"] += 1
                h = {"Referer": referer} if referer else None
                r = self.s.get(url, timeout=C.CRAWL_TIMEOUT, headers=h)
                if r.status_code == 200:
                    self.stats["ok"] += 1
                    self._sleep()
                    return r
                self.stats["http_error"].append({"url": url, "status": r.status_code})
                print(f"    [warn] HTTP {r.status_code} {url} (第 {attempt} 次)")
            except Exception as e:  # noqa: BLE001
                print(f"    [warn] {type(e).__name__}: {str(e)[:100]} (第 {attempt} 次)")
            time.sleep(2.0 * attempt)  # 退避
        self.stats["failed"] += 1
        return None

    def _sleep(self) -> None:
        t = random.uniform(*C.CRAWL_DELAY)
        self.stats["sleep_seconds"] += t
        time.sleep(t)


def parse_comment_list(html: str, sid: int, book_title: str, page: int, url: str) -> list[dict]:
    """BeautifulSoup4 解析短评列表页 → 记录列表。"""
    soup = BeautifulSoup(html, "lxml")
    records: list[dict] = []
    for li in soup.select("li.comment-item"):
        cid = li.get("data-cid", "")
        content_el = li.select_one("p.comment-content span.short") or li.select_one("p.comment-content")
        if content_el is None:
            continue
        text = content_el.get_text("\n", strip=True)
        info = li.select_one("span.comment-info")
        author, t_el, star_el = "", None, None
        if info is not None:
            a = info.find("a")
            author = a.get_text(strip=True) if a else ""
            t_el = info.select_one("a.comment-time")
            star_el = info.select_one("span.user-stars")
        star = None
        star_title = ""
        if star_el is not None:
            m = re.search(r"allstar(\d+)", " ".join(star_el.get("class", [])))
            if m:
                star = int(m.group(1))
            star_title = star_el.get("title", "") or ""
        votes = ""
        v = li.select_one("span.vote-count")
        if v is not None:
            votes = v.get_text(strip=True)
        records.append({
            "doc_id": make_doc_id("douban_book", f"{sid}:{cid}"),
            "source": "豆瓣读书短评",
            "source_type": "web_crawl_beautifulsoup4",
            "site": "book.douban.com",
            "domain": book_title,
            "doc_url": f"https://book.douban.com/subject/{sid}/comments/",
            "author": author,
            "rating_star": star if star is not None else "",
            "rating_title": star_title,
            "comment_time": t_el.get_text(strip=True) if t_el is not None else "",
            "votes": votes,
            "text_raw": text,
            "crawl_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "page": page,
        })
    return records


def crawl_book(sess: PoliteSession, sid: int, expect_title: str) -> list[dict]:
    print(f"\n[书目] {expect_title} (subject_id={sid})")
    r = sess.get(SUBJECT_URL.format(sid=sid))
    if r is None:
        print("  [skip] 详情页不可访问")
        return []
    soup = BeautifulSoup(r.text, "lxml")
    h1 = soup.find("h1")
    real_title = h1.get_text(" ", strip=True) if h1 else ""
    if not real_title:
        print("  [skip] 未取到书名，疑似反爬页面")
        return []
    print(f"  校验书名：{real_title}")

    out: list[dict] = []
    for page in range(1, C.DOUBAN_PAGES_PER_BOOK + 1):
        url = COMMENT_URL.format(sid=sid, start=(page - 1) * 20)
        resp = sess.get(url, referer=SUBJECT_URL.format(sid=sid))
        if resp is None:
            print(f"  [stop] 第 {page} 页请求失败")
            break
        snap = C.DIR_RAW_HTML / f"douban_book_{sid}_p{page:02d}.html"
        snap.write_text(resp.text, encoding="utf-8")
        recs = parse_comment_list(resp.text, sid, real_title, page, url)
        if not recs:
            print(f"  [stop] 第 {page} 页无短评（可能已到末页或被限流）")
            break
        out.extend(recs)
        print(f"  p{page:02d}: +{len(recs):3d} 条，累计 {len(out)} 条")
    return out


def main() -> None:
    t0 = time.time()
    sess = PoliteSession()
    all_recs: list[dict] = []
    for sid, title in C.DOUBAN_BOOKS:
        all_recs.extend(crawl_book(sess, sid, title))

    # 页内去重（同一 cid 可能重复出现）
    seen, uniq = set(), []
    for r in all_recs:
        if r["doc_id"] in seen:
            continue
        seen.add(r["doc_id"])
        uniq.append(r)

    out_csv = C.DIR_INTERIM / "source1_douban_bs4_raw.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(uniq)

    with_html_rating = sum(1 for r in uniq if r["rating_star"] != "")
    log = {
        "source": "豆瓣读书短评 (BeautifulSoup4)",
        "crawled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - t0, 1),
        "books": [{"subject_id": s, "title": t} for s, t in C.DOUBAN_BOOKS],
        "records": len(uniq),
        "records_with_rating": with_html_rating,
        "rating_coverage": round(with_html_rating / max(len(uniq), 1), 4),
        "http_stats": sess.stats,
        "output": str(out_csv),
    }
    (C.DIR_LOGS / "source1_crawl_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成：{len(uniq)} 条原始短评 → {out_csv}")
    print(f"其中带星级 {with_html_rating} 条（覆盖率 {log['rating_coverage']:.1%}），"
          f"耗时 {log['elapsed_seconds']}s，请求 {sess.stats['requests']} 次")


if __name__ == "__main__":
    main()
