#!/usr/bin/env python3
"""PWA用のアイコンを生成する。

    python tools/make_icons.py

出力（docs/icons/）:
    icon-192.png              Android・Chrome 用
    icon-512.png              スプラッシュ画面やストア表示用
    apple-touch-icon-180.png  iOS 用（iOSは manifest のアイコンを使わない）

文字は入れない。ホーム画面ではアイコンの下にアプリ名が出るうえ、
文字を入れるとフォントに依存して他の環境で再生成できなくなるため。

Pillow だけで描く（SVG変換ツールがこの環境に無いため）。
なめらかに見せるため4倍で描いてから縮小している。
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "docs" / "icons"

BG = (13, 27, 42)        # #0d1b2a  サイトの背景色と合わせる
FG = (234, 242, 251)     # #eaf2fb  サイトの文字色と合わせる

SS = 4                   # スーパーサンプリング倍率


def draw_snowflake(size: int) -> Image.Image:
    """雪の結晶（6方向・各腕に枝2対）を描く"""
    n = size * SS
    img = Image.new("RGB", (n, n), BG)
    d = ImageDraw.Draw(img)

    cx = cy = n / 2
    # マスカブルアイコンの安全域（中央80%）に収まるよう、腕の長さは半径の32%に抑える
    arm = n * 0.32
    width = max(1, int(n * 0.022))

    def line(a, b):
        d.line([a, b], fill=FG, width=width)

    for i in range(6):
        th = math.radians(60 * i)
        dx, dy = math.cos(th), math.sin(th)
        tip = (cx + dx * arm, cy + dy * arm)
        line((cx, cy), tip)

        # 枝は腕の 45% と 72% の位置から、左右に 60度ずつ
        for pos, blen in ((0.45, 0.28), (0.72, 0.20)):
            bx, by = cx + dx * arm * pos, cy + dy * arm * pos
            for sign in (+1, -1):
                bth = th + sign * math.radians(60)
                line((bx, by),
                     (bx + math.cos(bth) * arm * blen,
                      by + math.sin(bth) * arm * blen))

    # 中心の丸
    r = n * 0.035
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=FG)

    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for size, name in [(192, "icon-192.png"), (512, "icon-512.png"),
                       (180, "apple-touch-icon-180.png")]:
        path = OUT_DIR / name
        draw_snowflake(size).save(path, "PNG", optimize=True)
        print(f"{path.relative_to(BASE_DIR)}  {size}x{size}  {path.stat().st_size} バイト")


if __name__ == "__main__":
    main()
