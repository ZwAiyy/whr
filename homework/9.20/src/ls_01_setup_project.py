# -*- coding: utf-8 -*-
"""Label Studio 项目初始化：创建项目 → 配置多标注(3人) → 导入待标注任务。

依赖：先运行 src/ls_00_bootstrap_users.py 生成 annotation/ls_tokens.json
输出：annotation/ls_project.json （项目 id、URL、任务数、配置摘要）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def hdr(token: str) -> dict:
    return {"Authorization": f"Token {token}", "Content-Type": "application/json"}


def main() -> None:
    tokens = json.loads((C.DIR_ANNOTATION / "ls_tokens.json").read_text(encoding="utf-8"))
    admin = tokens[C.LS_EMAIL]
    label_config = (C.DIR_ANNOTATION / "label_config.xml").read_text(encoding="utf-8")
    tasks = json.loads((C.DIR_ANNOTATION / "tasks_to_annotate.json").read_text(encoding="utf-8"))

    s = requests.Session()
    s.headers.update(hdr(admin["token"]))

    # 0) 服务健康检查
    r = s.get(f"{C.LS_URL}/api/current-user/whoami", timeout=20)
    r.raise_for_status()
    print(f"服务正常：{C.LS_URL}  当前用户：{r.json().get('email')}")

    # 1) 项目：已存在则复用
    existing = s.get(f"{C.LS_URL}/api/projects", timeout=30).json()
    proj = next((p for p in existing.get("results", existing if isinstance(existing, list) else [])
                 if p.get("title") == C.LS_PROJECT_TITLE), None)
    payload = {
        "title": C.LS_PROJECT_TITLE,
        "description": "多来源中文短文本情感分类标注（豆瓣读书短评抽样 264 条；3 名标注员独立标注）",
        "label_config": label_config,
        # 关键：允许多名标注员对同一任务各自标注
        "maximum_annotations": len(C.ANNOTATORS),
        "min_annotations": 1,
        "show_skip_button": False,
        "show_annotation_history": True,
        "show_overlap_first": False,
        "is_published": True,
        "color": "#4C72B0",
    }
    if proj is None:
        r = s.post(f"{C.LS_URL}/api/projects", json=payload, timeout=60)
        if r.status_code >= 400:
            print("创建失败：", r.status_code, r.text[:800])
            r.raise_for_status()
        proj = r.json()
        print(f"[new ] 项目 id={proj['id']} {proj['title']}")
    else:
        r = s.patch(f"{C.LS_URL}/api/projects/{proj['id']}", json=payload, timeout=60)
        r.raise_for_status()
        proj = r.json()
        print(f"[have] 项目 id={proj['id']} {proj['title']}（已更新配置）")

    pid = proj["id"]
    print(f"  maximum_annotations={proj.get('maximum_annotations')} is_published={proj.get('is_published')}")

    # 2) 导入任务（避免重复导入）
    def count_tasks() -> int:
        c = s.get(f"{C.LS_URL}/api/projects/{pid}/tasks/",
                  params={"page": 1, "page_size": 100000}, timeout=60).json()
        if isinstance(c, dict):
            return int(c.get("total", len(c.get("results", []))))
        return len(c)

    have = count_tasks()
    expected = len(tasks)
    if have > expected:
        # 任务多于预期 → 说明此前发生过重复导入，先清空再重新导入
        print(f"[clean] 检测到 {have} 条任务（预期 {expected}），清空后重新导入")
        s.delete(f"{C.LS_URL}/api/projects/{pid}/tasks/", timeout=60)
        have = count_tasks()
        print(f"[clean] 清空后剩余 {have} 条")

    if have == 0:
        # 直接以 JSON 请求体导入（ImportAPI 支持 JSONParser；multipart 方式在 1.23 下解析异常）
        r = s.post(f"{C.LS_URL}/api/projects/{pid}/import", json=tasks,
                   headers={"Content-Type": "application/json"}, timeout=180)
        if r.status_code >= 400:
            print("导入失败：", r.status_code, r.text[:600])
            r.raise_for_status()
        print(f"[import] {str(r.json())[:200]}")
    else:
        print(f"[have] 项目已有 {have} 条任务，跳过导入")

    total = count_tasks()

    # 3) 验证标注员可见性（项目成员/组织成员权限）
    visibility = {}
    for email, info in tokens.items():
        rr = requests.get(f"{C.LS_URL}/api/projects/{pid}", headers=hdr(info["token"]), timeout=20)
        visibility[email] = rr.status_code
        print(f"  可见性检查 {email:28s} -> HTTP {rr.status_code}")

    out = {"project_id": pid, "title": proj["title"], "url": f"{C.LS_URL}/projects/{pid}",
           "host": C.LS_URL, "tasks": total, "maximum_annotations": proj.get("maximum_annotations"),
           "is_published": proj.get("is_published"), "visibility": visibility,
           "label_config_file": str(C.DIR_ANNOTATION / "label_config.xml"),
           "tasks_file": str(C.DIR_ANNOTATION / "tasks_to_annotate.json")}
    (C.DIR_ANNOTATION / "ls_project.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n项目信息 → {C.DIR_ANNOTATION / 'ls_project.json'}")


if __name__ == "__main__":
    main()
