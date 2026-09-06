"""训练 ETA 估算：按近期 (wall 时钟, step) 样本外推剩余步数与预计结束时刻。

纯函数模块（不读文件/不碰时钟），便于单测：
- eta_from_samples(step, elapsed_seconds, now, history_total_seconds=None)
- sample_regression(samples)：对样本做最小二乘回归 → (seconds_per_step, r2)

口径说明：
- 每步耗时用**近期窗口回归斜率**而非全程均值：训练越久越贴近当前真实速度
  （重启/预热/早停抖动都被远端样本稀释）；
- 历史总耗时（web/training_time 记录的同品种历史会话）作为冷启动先验：
  活跃会话样本不足（<2）时用它估 pace，样本足够后自动切换为实时回归；
- 样本越少越不可信，`confidence` 字段直接给 UI 提示（低/中/高）。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

MIN_STEP_DELTA = 1  # 至少推进 1 步才成对采样
CONFIDENCE_SAMPLES = (3, 6)  # 有效样本 <3 个=low，≥6 个=high，其余 medium


def sample_regression(samples: list[tuple[float, int]]) -> tuple[float | None, float | None]:
    """最小二乘拟合 step = a + b·elapsed → 返回 (seconds_per_step=b, r²)。

    样本 <2 对、时间跨度 0、步数无推进 → (None, None)。
    """
    if len(samples) < 2:
        return None, None
    pts = [(t, s) for t, s in samples]
    t0 = pts[0][0]
    xs = [t - t0 for t, _ in pts]
    ys = [s for _, s in pts]
    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    if dx <= 0 or dy < MIN_STEP_DELTA:
        return None, None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0:
        return None, None
    b = sxy / sxx  # steps per second
    if b <= 0:
        return None, None
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return 1.0 / b, max(0.0, min(1.0, r2))


def eta_from_samples(
    step: int | None,
    elapsed_seconds: float | None,
    train_steps: int | None = None,
    *,
    samples: list[tuple[float, int]] | None = None,
    history_total_seconds: float | None = None,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """计算 ETA 概要。

    - step/train_steps：当前步与总步（总步未知时只给 pace 不给剩余）；
    - elapsed_seconds：本次会话已用秒（与 step 同源，构成最新样本）；
    - samples：历史上采集的 (wall_ts, step) 样本（旧→新，可含当前）；
    - history_total_seconds：同品种历史会话总耗时（冷启动先验）；
    - now：可注入时钟（测试用）；默认 UTC now。
    """
    if step is None or step < 0:
        return None
    now = now or datetime.now(timezone.utc)

    pace: float | None = None
    r2: float | None = None
    src = "unknown"
    used = list(samples or [])
    if elapsed_seconds is not None:
        # 当前读数永远作为最新样本参与回归（若调用方未并入）
        used = [(t, s) for t, s in used if s != step] + [(now.timestamp(), step)]
    pace, r2 = sample_regression(used)
    if pace is not None:
        src = "session_regression"
    if pace is None and elapsed_seconds and step > 0:
        pace = elapsed_seconds / step  # 会话均值回退
        src = "session_mean"
    if pace is None and history_total_seconds and step > 0:
        pace = history_total_seconds / step  # 历史先验
        src = "history_prior"
    if pace is None or pace <= 0:
        return None

    span = len(used) if used and elapsed_seconds is not None else 0
    confidence = "low"
    if span >= CONFIDENCE_SAMPLES[1]:
        confidence = "high"
    elif span >= CONFIDENCE_SAMPLES[0]:
        confidence = "medium"

    out: dict[str, Any] = {
        "current_step": step,
        "seconds_per_step": round(pace, 3),
        "pace_source": src,
        "confidence": confidence,
    }
    if train_steps and train_steps > 0:
        remaining = max(0, train_steps - step)
        remaining_seconds = remaining * pace
        finish = now.timestamp() + remaining_seconds
        out.update(
            {
                "train_steps": train_steps,
                "remaining_steps": remaining,
                "remaining_seconds": round(remaining_seconds),
                "estimated_finish_local": datetime.fromtimestamp(
                    finish, tz=timezone.utc
                ).astimezone().strftime("%Y-%m-%d %H:%M"),
            }
        )
    return out
