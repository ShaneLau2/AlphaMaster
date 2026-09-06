#!/usr/bin/env bash
# =========================================================================
# AlphaMaster → GitHub 同步脚本（两个模式，默认 dry-run，--push 才写入远端）。
#
#   GitHub 仓库: https://github.com/ShaneLau2/AlphaMaster   (公开镜像)
#
# ── 模式 A: gh-pages 站点同步（默认）────────────────────────────────────
#   Pages 来源 : 分支 gh-pages, 路径 /   站点: https://shanelau2.github.io/AlphaMaster/
#   站点布局   : 根着陆页 index.html / shots/ / axis.html（绝不改动）+
#                app/index.html + app/static/*（由 web/static 同步生成）
#   · app/index.html  ← web/static/index.html 改写 "/static/" → "./static/"
#   · app/static/*    ← web/static/* 其余文件
#
# ── 模式 B: main sanitized 镜像同步（--sync-main）──────────────────────
#   main 分支 = 本地 HEAD 的净化快照（单 commit）：git archive HEAD 去掉
#   results/ 运行产物后重建为镜像 commit。推送到 main 是 force-push（历史被
#   替换为新的单 commit —— 与现有镜像形态一致），gh-pages 分支不受影响。
#   镜像不包含任何训练数据/检查点/结果产物。
#
# 为什么这样部署:
#   · AlphaMaster 本体是研究/回测/训练项目, 含大体积数据与检查点; GitHub Pages
#     只能托管静态文件、跑不了 Python 后端 (127.0.0.1:8765), 所以 Pages 只承载
#     「UI 外壳快照」。站点此前在独立仓库 alphamaster-web（已删除）, 现统一在
#     AlphaMaster 的 gh-pages 分支; main 分支只作公开净化镜像。
#
# 安全边界（脚本内断言强制, 违反即中止 exit 3）:
#   · gh-pages 模式: 只允许改动 app/ 内的路径; 仓库根（着陆页/shots/axis.html/
#     README/.nojekyll）出现任何暂存改动立即中止。
#   · main 模式: 镜像文件集必须 == 本地 tracked − results/（双向完全相等）;
#     results/ 或疑似敏感路径（token/secret/.env/凭证/密钥文件）出现即中止
#     （允许例: .env.example、tests/unit/test_api_token.py 测试桩）。
#
# 依赖:  ~/.agents/bin/github-cli 已配置 PAT (仅 --push 需要; token 只进一次性 URL,
#        不写入 git config / 任何文件)。
# 用法:
#   scripts/export_pages_site.sh                      # A: gh-pages dry-run
#   scripts/export_pages_site.sh --push               # A: 提交 gh-pages 并推送
#   scripts/export_pages_site.sh --sync-main          # B: main 镜像 dry-run
#   scripts/export_pages_site.sh --sync-main --push   # B: 重建并 force-push main
# =========================================================================
set -euo pipefail

OWNER="ShaneLau2"
REPO="AlphaMaster"
BRANCH="gh-pages"
CLONE_URL="https://github.com/${OWNER}/${REPO}.git"
PUSH_URL_PREFIX="https://x-access-token"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLI="${FREEBUFF_GITHUB_CLI:-$HOME/.agents/bin/github-cli}"
SITE="https://shanelau2.github.io/${REPO}/app/"
IDENT=( -c user.name="ShaneLau2" -c user.email="148608035+ShaneLau2@users.noreply.github.com" )
STATIC_SRC="$ROOT/web/static"
MIRROR_MSG="Public mirror of AlphaMaster — RL interpretable-factor discovery / backtest / realtime / paper trading"

PUSH=0
MODE="pages"
for arg in "$@"; do
  case "$arg" in
    --push) PUSH=1 ;;
    --sync-main) MODE="main" ;;
    -h|--help) grep -E '^# ' "$0" | sed -E 's/^# ?//'; exit 0 ;;
    *) echo "✗ 未知参数: $arg (支持: --push, --sync-main)" >&2; exit 2 ;;
  esac
done

[[ -f "$CLI" ]] || { echo "✗ 找不到 $CLI (请先配置 github-cli + PAT)" >&2; exit 1; }
if [[ "$MODE" == "pages" ]]; then
  [[ -f "$STATIC_SRC/index.html" ]] || { echo "✗ 未找到 web/static/index.html (在 $STATIC_SRC)" >&2; exit 1; }
fi
if [[ "$PUSH" == 1 ]]; then
  TOKEN_VAL="$("$CLI" token get)"   # 只在本进程使用, 不写入任何文件/配置
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# 无变化时收尾（两模式共用）
finish_no_change() {
  echo "    无变化 — 远端已是本地最新"
  echo "→ 完成 (未做任何写入/推送)"
  exit 0
}

# 有改动时 dry-run 收尾（两模式共用）
finish_dryrun() {
  local nch="$1"
  echo "    将改动 ${nch} 个文件:"
  git diff --cached --stat
  echo "→ dry-run 完成: 以上是加 --push 时将提交并推送的内容; 未提交 / 未推送"
  exit 0
}

# ─────────────────────────── 模式 A: gh-pages ───────────────────────────
if [[ "$MODE" == "pages" ]]; then
  echo "→ 1/4 clone ${OWNER}/${REPO}#${BRANCH} (全新, 保证与远端一致)"
  git clone -q --branch "$BRANCH" --single-branch "$CLONE_URL" "$WORK/site" || {
    echo "✗ 克隆失败: $CLONE_URL (分支 $BRANCH 是否存在?)" >&2
    exit 1
  }

  echo "→ 2/4 同步 app/ ← web/static"
  # 只替换 app/ 下我们拥有的路径; 仓库根内容原样保留
  git -C "$WORK/site" rm -rq --ignore-unmatch app/index.html app/static 2>/dev/null || true
  mkdir -p "$WORK/site/app/static"
  cp "$STATIC_SRC/index.html" "$WORK/site/app/index.html"
  for f in "$STATIC_SRC"/*; do
    [[ -f "$f" && "$(basename "$f")" != "index.html" ]] && cp "$f" "$WORK/site/app/static/"
  done
  # 适配 /<repo>/app 子路径: 绝对 /static/ 引用 → 相对 ./static/
  sed -i '' -E 's#(href|src)="/static/#\1="./static/#g' "$WORK/site/app/index.html"

  cd "$WORK/site"
  git add -A

  # —— 安全断言: 除 app/ 外不得有任何暂存改动 (保护根着陆页 / shots/ / axis.html / README) ——
  OUTSIDE="$(git diff --cached --name-only | awk '!/^app\//' || true)"
  if [[ -n "$OUTSIDE" ]]; then
    echo "✗ 安全断言失败: 转换产生了 app/ 之外的改动, 已中止 (绝不触碰仓库根):" >&2
    echo "$OUTSIDE" >&2
    exit 3
  fi

  NCH="$(git diff --cached --name-only | wc -l | tr -d ' ')"
  [[ "$NCH" == "0" ]] && finish_no_change
  [[ "$PUSH" == 0 ]] && finish_dryrun "$NCH"

  echo "→ 3/4 提交 (gh-pages: app/ 同步)"
  git "${IDENT[@]}" commit -q -m "chore(pages): sync app/ UI mirror from web/static (auto)"

  echo "→ 4/4 推送 gh-pages + 等待 Pages 上线"
  git push -q "${PUSH_URL_PREFIX}:${TOKEN_VAL}@github.com/${OWNER}/${REPO}.git" HEAD:"$BRANCH"
  for i in $(seq 1 40); do
    code="$(curl -s -o /dev/null -w '%{http_code}' "$SITE" || true)"
    [[ "$code" == "200" ]] && { echo "✓ 已上线: $SITE (约 $((i*3))s 后)"; exit 0; }
    sleep 3
  done
  echo "⚠ 页面尚未返回 200, 稍后手动访问: $SITE" >&2
  exit 1
fi

# ──────────────────────── 模式 B: main sanitized 镜像 ───────────────────
echo "→ 1/4 从本地 HEAD 重建 sanitized 镜像 (git archive HEAD − results/)"
HEAD_SHORT="$(git -C "$ROOT" rev-parse --short HEAD)"
mkdir -p "$WORK/snap"
git -C "$ROOT" archive HEAD | tar -x -C "$WORK/snap"
rm -rf "$WORK/snap/results"
cd "$WORK/snap"
git init -q -b main
git add -A

# —— 安全断言 1: 镜像文件集必须 == 本地 tracked − results/（双向完全相等）——
LOCAL_SET="$(git -C "$ROOT" ls-files | grep -v '^results/' | sort)"
SNAP_SET="$(git ls-files | sort)"
if [[ "$LOCAL_SET" != "$SNAP_SET" ]]; then
  echo "✗ 安全断言失败: 镜像文件集 ≠ 本地 tracked − results/ (diff 如下):" >&2
  diff <(echo "$LOCAL_SET") <(echo "$SNAP_SET") | head -20 >&2 || true
  exit 3
fi

# —— 安全断言 2: 不得含 results/ 或疑似敏感路径（允许例: .env.example / token 测试桩）——
BAD="$(git ls-files | grep '^results/' || true)"
if [[ -n "$BAD" ]]; then
  echo "✗ 安全断言失败: 镜像含 results/ 路径:" >&2
  echo "$BAD" >&2
  exit 3
fi
BAD="$(git ls-files | grep -iE '(^|/)(token|secret|web_token|credential|\.env$|\.env\.|\.pem$|\.key$)' \
        | grep -v '\.env\.example' | grep -v 'test_api_token' || true)"
if [[ -n "$BAD" ]]; then
  echo "✗ 安全断言失败: 镜像含疑似敏感路径:" >&2
  echo "$BAD" >&2
  exit 3
fi

echo "→ 2/4 fetch 远端 main (用于计算将推送的 diff; 树哈希对比)"
git remote add origin "$CLONE_URL" 2>/dev/null || git remote set-url origin "$CLONE_URL"
git fetch -q origin main || {
  echo "✗ 获取远端失败: $CLONE_URL (分支 main 是否存在?)" >&2
  exit 1
}

SNAP_TREE="$(git write-tree)"
REMOTE_TREE="$(git rev-parse FETCH_HEAD^{tree})"
if [[ "$SNAP_TREE" == "$REMOTE_TREE" ]]; then
  finish_no_change
fi
NCH="$(git diff --name-only FETCH_HEAD "$SNAP_TREE" | wc -l | tr -d ' ')"
if [[ "$PUSH" == 0 ]]; then
  echo "    将改动 ${NCH} 个文件 (vs 远端 main):"
  git diff --stat FETCH_HEAD "$SNAP_TREE"
  echo "→ dry-run 完成: 以上是加 --push 时将重建并 force-push 的内容; 未提交 / 未推送"
  exit 0
fi

echo "→ 3/4 提交 (main: sanitized 镜像, 来自 $HEAD_SHORT)"
git "${IDENT[@]}" commit -q -m "$MIRROR_MSG" -m "Sanitized snapshot synced from local HEAD $HEAD_SHORT ($(date +%F)): excludes results/ run artifacts."

echo "→ 4/4 force-push main (仅 main; gh-pages 不受影响)"
git push -f -q "${PUSH_URL_PREFIX}:${TOKEN_VAL}@github.com/${OWNER}/${REPO}.git" HEAD:main
echo "✓ main 已更新为 sanitized 镜像 ($HEAD_SHORT)"