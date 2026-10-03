# -*- coding: utf-8 -*-
"""把标注员加入 Label Studio 项目（ProjectMember）。

Label Studio 1.23 中「组织成员」不等于「项目成员」：非管理员账号只有在成为
项目成员后才能看到该项目（否则 REST 返回 404、Web 端项目列表为空）。
本脚本通过 Django ORM 为 3 名标注员建立项目成员关系。

运行：
  label_studio_env\\Scripts\\python.exe src\\ls_02_project_members.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LS_PKG = Path(r"C:\Users\Windy\Documents\Code\Python\WHR\Lable_Studio\label_studio_env\Lib\site-packages\label_studio")

os.environ["DJANGO_SETTINGS_MODULE"] = "core.settings.label_studio"
os.environ["LABEL_STUDIO_BASE_DATA_DIR"] = str(ROOT / "label_studio_data")
os.environ.setdefault("DJANGO_ALLOW_ASYNC_UNSAFE", "true")
sys.path.insert(0, str(LS_PKG))

import django  # noqa: E402

django.setup()

from projects.models import Project, ProjectMember  # noqa: E402
from users.models import User  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402


def main() -> None:
    proj_info = json.loads((C.DIR_ANNOTATION / "ls_project.json").read_text(encoding="utf-8"))
    tokens = json.loads((C.DIR_ANNOTATION / "ls_tokens.json").read_text(encoding="utf-8"))
    pid = proj_info["project_id"]
    project = Project.objects.get(pk=pid)
    print(f"项目：{project.title} (id={pid})")

    for email, info in tokens.items():
        if info["role"] != "annotator":
            continue
        u = User.objects.get(pk=info["id"])
        pm, created = ProjectMember.objects.get_or_create(user=u, project=project,
                                                          defaults={"enabled": True})
        if not created and not pm.enabled:
            pm.enabled = True
            pm.save(update_fields=["enabled"])
        print(f"  {'[new ]' if created else '[have]'} {email} → 项目成员(enabled={pm.enabled})")

    print("\n当前项目成员：")
    for m in project.members.select_related("user"):
        print(f"  {m.user.email:28s} enabled={m.enabled}")


if __name__ == "__main__":
    main()
