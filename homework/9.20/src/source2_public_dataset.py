# -*- coding: utf-8 -*-
"""来源 2：公开数据集获取（公开语料下载 + 完整性校验）

输出：
  data/raw/public_dataset/<filename>        原始数据文件（保持原样，不改写）
  data/raw/public_dataset/_download_manifest.json  下载清单（URL/大小/MD5/形状/字段）
  data/interim/source2_public_raw.csv       统一字段后的合并原始记录
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

FIELDS = ["doc_id", "source", "source_type", "site", "domain", "doc_url", "author",
          "rating_star", "rating_title", "comment_time", "votes", "text_raw",
          "crawl_time", "page"]


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path) -> dict:
    t0 = time.time()
    r = requests.get(url, headers=C.BASE_HEADERS, timeout=120, stream=True)
    r.raise_for_status()
    with dest.open("wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)
    return {
        "url": url,
        "path": str(dest),
        "bytes": dest.stat().st_size,
        "md5": md5_of(dest),
        "http_status": r.status_code,
        "elapsed_seconds": round(time.time() - t0, 2),
        "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def load_dataset(name: str, path: Path) -> pd.DataFrame:
    """读取公开数据集为 DataFrame（统一为 text/label 两列）。"""
    if name == "online_shoppers":
        df = pd.read_csv(path)
        return df
    df = pd.read_csv(path)
    return df


def to_unified(name: str, meta: dict, df: pd.DataFrame) -> list[dict]:
    """把公开数据集映射到与其他来源一致的最小字段集合。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    recs: list[dict] = []
    if name == "online_shoppers":
        # 结构化数据集没有自然语言文本，只登记元信息（用于特征分析对照）
        return []
    text_col = "review" if "review" in df.columns else df.columns[1]
    label_col = "label" if "label" in df.columns else df.columns[0]
    for i, row in df.iterrows():
        text = "" if pd.isna(row[text_col]) else str(row[text_col])
        lab = row[label_col]
        recs.append({
            "doc_id": hashlib.md5(f"{name}::{i}".encode()).hexdigest()[:16],
            "source": name,
            "source_type": "public_dataset",
            "site": "GitHub/ChineseNlpCorpus",
            "domain": meta["domain"],
            "doc_url": meta["url"],
            "author": "",
            "rating_star": "",
            "rating_title": int(lab) if not pd.isna(lab) else "",
            "comment_time": "",
            "votes": "",
            "text_raw": text,
            "crawl_time": now,
            "page": "",
        })
    return recs


def main() -> None:
    manifest = []
    all_recs: list[dict] = []
    for name, meta in C.PUBLIC_DATASETS.items():
        dest = C.DIR_RAW_DATASET / meta["filename"]
        is_zip_source = meta["url"].lower().endswith(".zip") or name == "online_shoppers"
        archive = C.DIR_RAW_DATASET / f"{name}_archive.zip"
        info: dict = {}
        if is_zip_source:
            # 压缩包来源：先落原始压缩包，再解出内部 CSV（原始包保留作为证据）
            if not (archive.exists() and archive.stat().st_size > 1000):
                print(f"[down ] {name} <- {meta['url']} (zip)")
                try:
                    info = download(meta["url"], archive)
                except Exception as e:  # noqa: BLE001
                    print(f"  [fail] {type(e).__name__}: {e}")
                    manifest.append({"dataset": name, "desc": meta["desc"], "ok": False,
                                     "error": str(e)[:200]})
                    continue
            else:
                print(f"[cache] {name} (zip)")
                info = {"url": meta["url"], "bytes": archive.stat().st_size,
                        "md5": md5_of(archive), "http_status": "cached", "elapsed_seconds": 0.0,
                        "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            with zipfile.ZipFile(archive) as z:
                inner = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not inner:
                    print("  [fail] 压缩包内没有 CSV")
                    manifest.append({"dataset": name, "desc": meta["desc"], "ok": False,
                                     "error": "no csv in zip"})
                    continue
                with z.open(inner[0]) as f:
                    dest.write_bytes(f.read())
                info["archive"] = str(archive)
                info["extracted_from"] = inner[0]
                print(f"  [unzip] {inner[0]} -> {dest.name}")
        elif dest.exists() and dest.stat().st_size > 1000:
            info = {"url": meta["url"], "bytes": dest.stat().st_size, "md5": md5_of(dest),
                    "http_status": "cached", "elapsed_seconds": 0.0,
                    "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            print(f"[cache] {name}")
        else:
            print(f"[down ] {name} <- {meta['url']}")
            try:
                info = download(meta["url"], dest)
            except Exception as e:  # noqa: BLE001
                print(f"  [fail] {type(e).__name__}: {e}")
                manifest.append({"dataset": name, "desc": meta["desc"], "ok": False,
                                 "error": str(e)[:200]})
                continue
        info["path"] = str(dest)
        info["csv_md5"] = md5_of(dest)
        try:
            df = load_dataset(name, Path(info["path"]))
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] 读取失败：{e}")
            manifest.append({"dataset": name, "desc": meta["desc"], "ok": False, "error": str(e)[:200]})
            continue
        recs = to_unified(name, meta, df)
        all_recs.extend(recs)
        entry = {
            "dataset": name, "desc": meta["desc"], "domain": meta["domain"], "ok": True,
            "rows": int(df.shape[0]), "cols": int(df.shape[1]),
            "columns": [str(c) for c in df.columns][:30],
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()} if df.shape[1] <= 25 else {},
            "missing_total": int(df.isna().sum().sum()),
            "duplicated_rows": int(df.duplicated().sum()),
            "unified_records": len(recs),
            **info,
        }
        if name != "online_shoppers" and "label" in df.columns:
            vc = df["label"].value_counts().to_dict()
            entry["label_distribution"] = {str(k): int(v) for k, v in vc.items()}
        manifest.append(entry)
        print(f"  OK rows={entry['rows']} cols={entry['cols']} md5={info['md5'][:10]}…")

    (C.DIR_RAW_DATASET / "_download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    out = C.DIR_INTERIM / "source2_public_raw.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_recs)
    print(f"\n完成：公开语料 {len(all_recs)} 条 → {out}")
    print(f"清单：{C.DIR_RAW_DATASET / '_download_manifest.json'}")


if __name__ == "__main__":
    main()
