# -*- coding: utf-8 -*-
"""从 Label Studio 导出标注结果（JSON），并整理成分析用表格。

输出：
  annotation/label_studio_export.zip     Label Studio 原始导出包（证据留档）
  annotation/label_studio_export.json    解包后的标注 JSON（任务 + 标注）
  annotation/annotation_matrix.csv       样本 × 标注员 的标签矩阵（供 κ 计算）
  annotation/annotation_long.csv         长表（sample_id, annotator, label, 用时等）
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def label_of(ann: dict) -> str:
    for r in ann.get("result", []):
        if r.get("from_name") == "sentiment":
            ch = r.get("value", {}).get("choices") or []
            if ch:
                return ch[0]
    return ""


def reason_of(ann: dict) -> str:
    for r in ann.get("result", []):
        if r.get("from_name") == "reason":
            t = r.get("value", {}).get("text") or []
            if t:
                return t[0]
    return ""


def main() -> None:
    tokens = json.loads((C.DIR_ANNOTATION / "ls_tokens.json").read_text(encoding="utf-8"))
    proj = json.loads((C.DIR_ANNOTATION / "ls_project.json").read_text(encoding="utf-8"))
    pid = proj["project_id"]
    admin = tokens[C.LS_EMAIL]
    h = {"Authorization": f"Token {admin['token']}"}

    r = requests.get(f"{C.LS_URL}/api/projects/{pid}/export",
                     params={"exportType": "JSON"}, headers=h, timeout=300)
    r.raise_for_status()
    zip_path = C.DIR_ANNOTATION / "label_studio_export.zip"
    zip_path.write_bytes(r.content)
    print(f"导出包：{zip_path}（{len(r.content)} bytes）")

    if r.content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".json")]
            raw = z.read(names[0])
            print(f"  包内文件：{names}")
        data = json.loads(raw.decode("utf-8"))
    else:
        # Label Studio 1.23 在只有 JSON 结果时可能直接返回 JSON 文本
        data = r.json()
        print("  直接返回 JSON（非 zip 包）")
    if isinstance(data, dict):
        data = data.get("results", data.get("tasks", []))
    (C.DIR_ANNOTATION / "label_studio_export.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    email_by_id = {v["id"]: k.split("@")[0] for k, v in tokens.items()}
    rows, wide = [], []
    for item in data:
        sid = item.get("data", {}).get("sample_id", "")
        text = item.get("data", {}).get("text", "")
        rec = {"sample_id": sid, "task_id": item.get("id"), "text": text}
        for ann in item.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            who = email_by_id.get(ann.get("completed_by"), str(ann.get("completed_by")))
            lab = label_of(ann)
            rec[who] = lab
            rows.append({"sample_id": sid, "task_id": item.get("id"), "annotator": who,
                         "label": lab, "reason": reason_of(ann),
                         "lead_time": ann.get("lead_time"),
                         "created_at": ann.get("created_at")})
        wide.append(rec)

    long_df = pd.DataFrame(rows)
    wide_df = pd.DataFrame(wide)
    long_df.to_csv(C.DIR_ANNOTATION / "annotation_long.csv", index=False, encoding="utf-8-sig")
    wide_df.to_csv(C.DIR_ANNOTATION / "annotation_matrix.csv", index=False, encoding="utf-8-sig")

    print(f"标注长表：{len(long_df)} 条；宽表：{len(wide_df)} 个样本")
    print("每个标注员标注量：")
    print(long_df.groupby("annotator")["label"].count().to_string())
    print("\n标签分布（全部标注）：")
    print(long_df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
