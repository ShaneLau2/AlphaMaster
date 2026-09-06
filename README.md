# AlphaMaster · Web 界面渲染（GitHub Pages 镜像）

AlphaMaster（量化因子挖掘中心）Web 前端的**静态渲染镜像**，托管于
https://shanelau2.github.io/alphamaster-web/ 。GitHub Pages 只服务静态文件，
无法运行 Python/FastAPI 后端，故本仓库内容为只读快照：

- `index.html` — 界面导览（各页签真实渲染截图 + 本版特性）
- `axis.html` — 「三轴联合回测」数据页（最近一次完整运行的全网格静态渲染）
- `shots/` — 渲染截图（headless Chromium 取自本机实时后端）
- `app/` — 前端静态外壳副本（index.html + static/，路径已改写为相对，适配子路径）

每次版本更新用 `web/static` 重新构建后推 main 即可自动发布（Pages source = main / root）。
完整功能请在本机运行 `run_web.py`。
