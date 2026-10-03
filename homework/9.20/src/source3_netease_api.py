# -*- coding: utf-8 -*-
"""来源 3：开放 API 获取 —— 网易云音乐评论 API（JSON 接口，非 HTML 解析）

接口：https://music.163.com/api/v1/resource/comments/R_SO_4_{song_id}?limit=&offset=
输出：
  data/raw/json/netease_song_<id>_p<n>.json   原始 JSON 快照
  data/interim/source3_netease_api_raw.csv    解析后的原始记录
  logs/source3_api_log.json                   接口调用日志
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

API = "https://music.163.com/api/v1/resource/comments/R_SO_4_{sid}"
FIELDS = ["doc_id", "source", "source_type", "site", "domain", "doc_url", "author",
          "rating_star", "rating_title", "comment_time", "votes", "text_raw",
          "crawl_time", "page"]


def ts(ms) -> str:
    try:
        return datetime.fromtimestamp(int(ms) / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:  # noqa: BLE001
        return ""


def fetch_song(sess: requests.Session, sid: int, note: str) -> tuple[list[dict], dict]:
    """抓取单首歌的多页评论；返回 (记录, 日志)。"""
    print(f"\n[歌曲] {note} (song_id={sid})")
    recs: list[dict] = []
    calls = 0
    total = None
    for page in range(1, C.NETEASE_PAGES + 1):
        offset = (page - 1) * C.NETEASE_PAGE_SIZE
        params = {"limit": C.NETEASE_PAGE_SIZE, "offset": offset}
        try:
            r = sess.get(API.format(sid=sid), params=params, timeout=C.CRAWL_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            calls += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] 第 {page} 页失败 {type(e).__name__}: {str(e)[:80]}")
            break
        snap = C.DIR_RAW_JSON / f"netease_song_{sid}_p{page:02d}.json"
        snap.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        if data.get("code") != 200:
            print(f"  [stop] 接口返回 code={data.get('code')}")
            break
        total = data.get("total", total)
        comments = data.get("comments") or []
        if not comments:
            print(f"  [stop] 第 {page} 页无评论")
            break
        for c in comments:
            cid = c.get("commentId")
            user = (c.get("user") or {}).get("nickname", "")
            recs.append({
                "doc_id": hashlib.md5(f"netease::{sid}:{cid}".encode()).hexdigest()[:16],
                "source": "网易云音乐评论",
                "source_type": "open_api_json",
                "site": "music.163.com",
                "domain": note,
                "doc_url": f"https://music.163.com/#/song?id={sid}",
                "author": user,
                "rating_star": "",
                "rating_title": "",
                "comment_time": ts(c.get("time")),
                "votes": c.get("likedCount", ""),
                "text_raw": (c.get("content") or "").strip(),
                "crawl_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "page": page,
            })
        print(f"  p{page:02d}: +{len(comments):3d} 条，累计 {len(recs)} 条 (total={total})")
        time.sleep(random.uniform(0.6, 1.4))
    return recs, {"song_id": sid, "note": note, "api_calls": calls, "total_reported": total,
                  "records": len(recs)}


def main() -> None:
    t0 = time.time()
    sess = requests.Session()
    sess.headers.update({**C.BASE_HEADERS, "Referer": "https://music.163.com/"})
    all_recs: list[dict] = []
    logs = []
    for sid, note in C.NETEASE_SONGS:
        recs, lg = fetch_song(sess, sid, note)
        all_recs.extend(recs)
        logs.append(lg)

    seen, uniq = set(), []
    for r in all_recs:
        if r["doc_id"] in seen or not r["text_raw"]:
            continue
        seen.add(r["doc_id"])
        uniq.append(r)

    out = C.DIR_INTERIM / "source3_netease_api_raw.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(uniq)

    log = {"source": "网易云音乐评论 API", "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "elapsed_seconds": round(time.time() - t0, 1), "songs": logs,
           "records": len(uniq), "output": str(out)}
    (C.DIR_LOGS / "source3_api_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成：{len(uniq)} 条 API 评论 → {out}（耗时 {log['elapsed_seconds']}s）")


if __name__ == "__main__":
    main()
