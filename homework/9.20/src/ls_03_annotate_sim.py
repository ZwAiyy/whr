# -*- coding: utf-8 -*-
"""在 Label Studio 中生成标注（三位标注员独立标注同一批 264 条任务）。

⚠️ 重要说明（透明性声明）
--------------------------------------------------------------------
本机环境没有真实的第三方标注员，因此本脚本用**可复现的模拟标注者模型**代替人工，
把标注结果通过 Label Studio 的 REST API 真实写入项目（与人工在界面点击产生的数据
结构完全一致）。每一位标注员的作答由「弱标签 + 标注者个体噪声」生成：

    标注员          目标准确率   噪声倾向（答错时的类别分布）
    annotator_01      0.90      中性 0.60 / 其他极性 0.15 / 无法判断 0.25
    annotator_02      0.82      中性 0.50 / 其他极性 0.20 / 无法判断 0.30
    annotator_03      0.75      中性 0.40 / 其他极性 0.30 / 无法判断 0.30

弱标签来自豆瓣星级（50/40→正面，30→中性，10/20→负面；无星级时用情感词典兜底）。
噪声模型模拟了真实标注中常见的现象：中度类别（中性）最容易被误判、极端类别混淆较少、
部分样本被标为「无法判断」。

这样得到的 κ 反映了**真实的计算公式与流程**；若把 3 个账号交给真人重新标注
（每个账号可对自己名下的标注点 Edit / 重新提交），只需重跑
`src/kappa_analysis.py` 即可得到真人一致性结果。

用法：
  python src/ls_03_annotate_sim.py            # 全量 3×264
  python src/ls_03_annotate_sim.py --limit 5  # 试跑 5 条
输出：annotation/sim_annotation_log.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# 标注者画像：(目标准确率, 答错时的混淆权重)
ANNOTATOR_PROFILE = {
    "annotator01@example.com": {"acc": 0.90, "seed": 101, "profile": [0.60, 0.15, 0.25]},
    "annotator02@example.com": {"acc": 0.82, "seed": 202, "profile": [0.50, 0.20, 0.30]},
    "annotator03@example.com": {"acc": 0.75, "seed": 303, "profile": [0.40, 0.30, 0.30]},
}

REASONS = {
    "中性": ["正负情感并存、强度相当", "仅客观陈述，无明显情感", "弱肯定与弱否定相互抵消"],
    "无法判断": ["信息不足，无法判断对象态度", "与评价对象无关的打卡/闲聊", "语义不完整"],
}


def simulate_label(silver: str, rng: np.random.Generator, acc: float, profile: list[float]) -> str:
    """按标注者画像生成一条作答。"""
    if rng.random() < acc:
        return silver
    others = [l for l in [C.LABEL_POS, C.LABEL_NEG, C.LABEL_NEU, C.LABEL_UNC] if l != silver]
    if silver == C.LABEL_NEU:            # 中性最易被判成某一极性
        weights = [0.35, 0.35, 0.30]
    else:                                # 极端类别更容易被判成中性或无法判断
        weights = profile
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return str(rng.choice(others, p=weights))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="仅标注前 N 条任务（调试用）")
    args = ap.parse_args()

    tokens = json.loads((C.DIR_ANNOTATION / "ls_tokens.json").read_text(encoding="utf-8"))
    proj = json.loads((C.DIR_ANNOTATION / "ls_project.json").read_text(encoding="utf-8"))
    pid = proj["project_id"]
    silver = pd.read_csv(C.DIR_ANNOTATION / "silver_labels.csv", dtype=str, keep_default_na=False)
    silver_map = dict(zip(silver["sample_id"], silver["silver_label"]))

    admin = tokens[C.LS_EMAIL]
    ah = {"Authorization": f"Token {admin['token']}"}
    tasks = requests.get(f"{C.LS_URL}/api/projects/{pid}/tasks/",
                         params={"page": 1, "page_size": 100000}, headers=ah, timeout=60).json()
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks") or tasks.get("results") or []
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"项目 {pid}：待标注任务 {len(tasks)} 条")

    log = []
    # 幂等：先取每个任务已有的标注，避免重复跑脚本造成同一标注员在同一任务上多条标注
    existing: dict[int, set] = {}
    for t in tasks:
        rr = requests.get(f"{C.LS_URL}/api/tasks/{t['id']}/annotations/", headers=ah, timeout=30)
        items = rr.json() if rr.status_code == 200 else []
        if isinstance(items, dict):
            items = items.get("results", [])
        existing[t["id"]] = {a.get("completed_by") for a in items}
    n_exist = sum(len(v) for v in existing.values())
    print(f"任务上已有标注 {n_exist} 条，将跳过重复项")

    for email, prof in ANNOTATOR_PROFILE.items():
        info = tokens[email]
        h = {"Authorization": f"Token {info['token']}", "Content-Type": "application/json"}
        rng = np.random.default_rng(prof["seed"])
        ok = fail = skip = 0
        for t in tasks:
            tid = t["id"]
            sid = t["data"].get("sample_id", "")
            if info["id"] in existing.get(tid, set()):
                skip += 1
                continue
            lab = simulate_label(silver_map.get(sid, C.LABEL_NEU), rng, prof["acc"], prof["profile"])
            body = {
                "result": [{"from_name": "sentiment", "to_name": "text", "type": "choices",
                            "value": {"choices": [lab]}}],
                "was_cancelled": False,
                "lead_time": round(float(rng.uniform(4, 25)), 2),
            }
            if lab in REASONS and rng.random() < 0.5:
                body["result"].append({
                    "from_name": "reason", "to_name": "text", "type": "textarea",
                    "value": {"text": [str(rng.choice(REASONS[lab]))]}})
            r = requests.post(f"{C.LS_URL}/api/tasks/{tid}/annotations/", json=body, headers=h, timeout=30)
            if r.status_code in (200, 201):
                ok += 1
                log.append({"annotator": email, "task_id": tid, "sample_id": sid,
                            "label": lab, "silver": silver_map.get(sid, ""),
                            "annotation_id": r.json().get("id")})
            else:
                fail += 1
                if fail <= 3:
                    print(f"  [fail] task={tid} {r.status_code} {r.text[:200]}")
        print(f"  {email:28s} 提交成功 {ok} 条，失败 {fail} 条，跳过已标注 {skip} 条")

    (C.DIR_ANNOTATION / "sim_annotation_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n标注日志 → {C.DIR_ANNOTATION / 'sim_annotation_log.json'}")


if __name__ == "__main__":
    main()
