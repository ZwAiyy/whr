# -*- coding: utf-8 -*-
"""Label Studio 用户/组织初始化（ORM 方式）

为什么用 ORM 而不是 REST：
  Label Studio 的 /user/signup/、/user/login/ 视图带有 @enforce_csrf_checks，
  纯脚本登录需要额外处理 CSRF；而 DRF Token 认证的接口更适合自动化。
  本脚本直接用 Django ORM 在本项目的 SQLite 库中创建账号并读取其 API Token，
  之后所有 REST 调用统一使用 `Authorization: Token <key>`，无需会话与 CSRF。

运行（必须使用 Label Studio 自带 venv）：
  label_studio_env\\Scripts\\python.exe src\\ls_00_bootstrap_users.py

输出：annotation/ls_tokens.json  (email -> token，供后续脚本使用)
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
os.environ.setdefault("LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK", "false")
os.environ.setdefault("DJANGO_ALLOW_ASYNC_UNSAFE", "true")
sys.path.insert(0, str(LS_PKG))

import django  # noqa: E402

django.setup()

from organizations.models import Organization  # noqa: E402
from rest_framework.authtoken.models import Token  # noqa: E402
from users.models import User  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402

SUPERUSER = {"email": C.LS_EMAIL, "password": C.LS_PASSWORD, "name": "项目管理员"}
ANNOTATORS = [
    {"email": "annotator01@example.com", "password": "Annotate@2024", "name": "标注员01"},
    {"email": "annotator02@example.com", "password": "Annotate@2024", "name": "标注员02"},
    {"email": "annotator03@example.com", "password": "Annotate@2024", "name": "标注员03"},
]


def ensure_user(email: str, password: str, name: str, org: Organization, is_super: bool) -> User:
    u = User.objects.filter(email=email).first()
    if u is None:
        if is_super:
            u = User.objects.create_superuser(email=email, password=password, first_name=name)
        else:
            u = User.objects.create_user(email=email, password=password, first_name=name)
        print(f"[new ] {email} (id={u.id})")
    else:
        print(f"[have] {email} (id={u.id})")
    org.add_user(u)                       # 同一组织，保证可见同一批项目
    # 关键：设置「当前活动组织」，否则 Project.objects.for_user() 取不到项目（REST 返回 404）
    if u.active_organization_id != org.id:
        u.active_organization = org
        u.save(update_fields=["active_organization"])
        print(f"       └ 设置 active_organization = {org.id}")
    Token.objects.get_or_create(user=u)   # 保证存在 API Token
    return u


def main() -> None:
    org = Organization.objects.first()
    if org is None:
        raise SystemExit("未找到组织，请确认 Label Studio 已启动并初始化数据库")
    print(f"组织：{org.title} (id={org.id})")

    # Label Studio 1.23 默认关闭「legacy Token 认证」，只允许 JWT。
    # 为便于脚本化调用 REST API（并发压测/批量标注），这里显式开启 legacy token。
    jwtset = org.jwt
    if not jwtset.legacy_api_tokens_enabled:
        jwtset.legacy_api_tokens_enabled = True
        jwtset.save(update_fields=["legacy_api_tokens_enabled"])
        print("[cfg ] 已开启组织级 legacy API token 认证")
    else:
        print("[cfg ] legacy API token 认证已开启")

    tokens = {}
    su = ensure_user(SUPERUSER["email"], SUPERUSER["password"], SUPERUSER["name"], org, True)
    tokens[su.email] = {"token": su.get_token().key, "id": su.id, "role": "admin",
                        "password": SUPERUSER["password"], "name": SUPERUSER["name"]}
    for a in ANNOTATORS:
        u = ensure_user(a["email"], a["password"], a["name"], org, False)
        tokens[u.email] = {"token": u.get_token().key, "id": u.id, "role": "annotator",
                           "password": a["password"], "name": a["name"]}

    out = ROOT / "annotation" / "ls_tokens.json"
    out.write_text(json.dumps(tokens, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n账号与 Token 已写入 {out}")
    for e, v in tokens.items():
        print(f"  {v['role']:9s} {e:28s} id={v['id']} token={v['token'][:12]}…")


if __name__ == "__main__":
    main()
