"""从 FastAPI 路由 + Pydantic 请求模型自动生成 Markdown 接口文档。

用法（仓库根目录）：
    .venv/bin/python scripts/gen_api_docs.py [--out docs/api_http_generated.md]

生成物覆盖：HTTP 方法 / 路径 / 路径与查询参数 / 请求体模型字段（必填与默认值）
/ 处理器 docstring 首行。保持 docs/接口说明书.md 的 HTTP 章节永不落后：
该文档指向本生成物，路由或模型一变，重跑本脚本即可同步。
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any, get_type_hints

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DEFAULT = ROOT / "docs" / "api_http_generated.md"


def _first_doc_lines(fn: Any, n: int = 1) -> str:
    doc = inspect.getdoc(fn) or ""
    lines = [ln.strip() for ln in doc.splitlines() if ln.strip()]
    return " ".join(lines[:n]) if lines else ""


def _describe_model(model: type) -> str:
    """渲染 Pydantic 请求模型为字段表（类型 / 必填 / 默认）。"""
    rows: list[str] = []
    for name, field in model.model_fields.items():
        req = "必填" if field.is_required() else "可选"
        default = "" if field.is_required() else f"（默认 {field.default!r}）"
        ann = getattr(field, "annotation", None)
        tname = getattr(ann, "__name__", str(ann)).replace("NoneType", "null")
        rows.append(f"`{name}` {tname} · {req}{default}")
    return "；".join(rows) if rows else "—"


def _body_model(route: Any) -> type | None:
    """定位请求体 Pydantic 模型。

    FastAPI 内部 ModelField 包装不暴露 type_/annotation，改从处理器函数
    签名注解解析（def handler(req: DownloadDataRequest, ...)）。
    """
    try:
        hints = get_type_hints(route.endpoint)
    except Exception:  # noqa: BLE001 解析失败退回已知字段路径
        hints = {}
    for name, t in hints.items():
        if name == "return":
            continue
        # 解析 Optional[X] / 别名：取注解本体判断
        base = getattr(t, "__origin__", t)
        if hasattr(base, "model_fields") or hasattr(t, "model_fields"):
            return t
    return None


def _required_field(p: Any) -> bool:
    """兼容 fastapi 内部 ModelField 的 v1/v2 差异。"""
    for attr in ("is_required", "required"):
        val = getattr(p, attr, None)
        if callable(val):
            return bool(val())
        if val is not None:
            return bool(val)
    d = getattr(p, "default", inspect.Parameter.empty)
    return d is inspect.Parameter.empty


def _describe_query_params(route: Any) -> str:
    dependant = getattr(route, "dependant", None)
    items = []
    for p in (dependant.query_params if dependant else []):
        name = getattr(p, "name", "?")
        items.append(f"`{name}`{'（必填）' if _required_field(p) else ''}")
    return ", ".join(items) if items else "—"


def build_markdown() -> str:
    from web import app as webapp

    app = webapp.app
    lines: list[str] = []
    lines.append("# HTTP 接口参考（自动生成）\n")
    lines.append(
        "> 本文件由 `scripts/gen_api_docs.py` 从 FastAPI 路由 + Pydantic 模型自动生成，"
        "请勿手改；路由/模型变化后重跑 `python scripts/gen_api_docs.py` 同步。\n"
    )
    lines.append(f"- 服务入口：`run_web.py`，默认 `http://127.0.0.1:8765`\n")
    lines.append("- 返回均为 JSON（导出接口为二进制文件）；错误 `{\"detail\": \"...\"}`\n")

    routes = []
    for route in app.routes:
        methods = sorted(getattr(route, "methods", None) or [])
        path = getattr(route, "path", "")
        if not methods or path in ("/favicon.ico",):
            continue
        routes.append((methods[0], path, route))

    # 按路径分组并保持与 说明书 一致的模块顺序
    order = {
        "/api/health": 0, "/api/routes": 1, "/api/settings": 2, "/api/config": 3,
        "/api/overview": 4, "/api/data": 5, "/api/data-file": 6, "/api/strategy-file": 7,
        "/api/training": 8, "/api/strategies": 9, "/api/symbols": 10, "/api/backtest": 11,
        "/api/realtime": 12, "/api/ai": 13, "/api/debug": 14, "/api/symbols": 15,
    }
    routes.sort(key=lambda r: (next((v for k, v in order.items() if r[1].startswith(k)), 99), r[1]))

    current_group: str | None = None
    for method, path, route in routes:
        group = path.split("/")[2] if path.startswith("/api/") else "other"
        if group != current_group:
            current_group = group
            lines.append(f"\n## `{group}`\n")
        body = "—"
        model = _body_model(route)
        if model is not None:
            body = _describe_model(model)
        query = _describe_query_params(route)
        doc = _first_doc_lines(route.endpoint)
        lines.append(f"| `{method}` | `{path}` |")
        lines.append(f"|:--|:--|")
        if body != "—":
            lines.append(f"  - **请求体**：{body}")
        if query != "—":
            lines.append(f"  - **查询参数**：{query}")
        if doc:
            lines.append(f"  - {doc}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_markdown(), encoding="utf-8")
    print(f"已生成 {out}（{len(out.read_text(encoding='utf-8').splitlines())} 行）")


if __name__ == "__main__":
    main()
