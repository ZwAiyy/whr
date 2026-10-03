# -*- coding: utf-8 -*-
"""第二轮探测：确认 API 来源的具体资源 ID 是否有效（网易云音乐歌曲、B站视频）。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
H = {"User-Agent": UA, "Referer": "https://music.163.com/", "Accept-Language": "zh-CN,zh;q=0.9"}

SONGS = [186016, 347230, 1330348068, 287035, 202369, 5238999, 108392, 418603077, 347231, 27591499]
AIDS = [170001, 2, 455876751, 883959229, 311800980, 431664465, 590893156, 758615761]

print("=== 网易云音乐 ===")
for sid in SONGS:
    try:
        r = requests.get(f"https://music.163.com/api/v1/resource/comments/R_SO_4_{sid}",
                         params={"limit": 20, "offset": 0}, headers=H, timeout=15)
        j = r.json()
        n = len(j.get("comments") or [])
        hot = len(j.get("hotComments") or [])
        sample = (j.get("comments") or [{}])[0].get("content", "")[:30]
        print(f"song {sid}: code={j.get('code')} comments={n} hot={hot} total={j.get('total')} sample={sample!r}")
    except Exception as e:  # noqa: BLE001
        print(f"song {sid}: FAIL {type(e).__name__} {str(e)[:80]}")

print("\n=== B站视频 ===")
for aid in AIDS:
    try:
        r = requests.get("https://api.bilibili.com/x/v2/reply",
                         params={"type": 1, "oid": aid, "sort": 2, "ps": 20, "pn": 1},
                         headers={"User-Agent": UA, "Referer": f"https://www.bilibili.com/video/av{aid}"}, timeout=15)
        j = r.json()
        rep = (j.get("data") or {}).get("replies") or []
        sample = rep[0]["content"]["message"][:30] if rep else ""
        title = ((j.get("data") or {}).get("page") or {}).get("count", "")
        print(f"aid {aid}: code={j.get('code')} replies={len(rep)} msg={str(j.get('message'))[:30]} sample={sample!r}")
    except Exception as e:  # noqa: BLE001
        print(f"aid {aid}: FAIL {type(e).__name__} {str(e)[:80]}")
