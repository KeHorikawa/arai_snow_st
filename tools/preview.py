#!/usr/bin/env python3
"""静的サイト（docs/）を手元のPCで開いて確かめるための、小さなWebサーバー。

    python tools/preview.py              # http://localhost:8765/ で開く
    python tools/preview.py --port 9000  # ポートを変える

使いどころ：PRをマージする前に、そのブランチの画面をPCのブラウザで確かめる。
GitHub Pages は main しか公開しないので、ブランチの画面はこの方法で見る。

    git switch <ブランチ名>
    python tools/preview.py
    # → ブラウザで http://localhost:8765/ を開く。終わったら Ctrl+C

表示されるデータは、そのブランチの docs/data/snow_data.json（手元のファイル）。

■ Service Worker について
localhost でも Service Worker は本番と同じように働く（キャッシュ優先で画面を出す）。
そのため、ファイルを直して開き直しても、古い画面が出ることがある。確かめるときは：

  - シークレットウィンドウで開く（閉じればキャッシュごと消える。いちばん確実）
  - または Chrome の開発者ツール → Application → Service workers →
    「Update on reload」にチェックを入れて再読み込み

本番（kehorikawa.github.io）とは別の場所（オリジン）なので、
ここで何をしても、ふだん使っているスマホやPCのアプリには影響しない。

■ このサーバーがしていること
ブラウザのHTTPキャッシュを使わせない（Cache-Control: no-store）。
直したファイルが、開き直すたびに確実に届くようにするため。
"""

from __future__ import annotations

import argparse
import functools
import http.server
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="docs/ を手元で配信して確かめる")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    handler = functools.partial(NoCacheHandler, directory=str(DOCS_DIR))
    # 127.0.0.1 だけで待ち受ける（同じネットワークの他の機器からは見えない）
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"配信中: http://localhost:{args.port}/  （{DOCS_DIR}）")
        print("シークレットウィンドウで開くと、Service Worker のキャッシュに惑わされない。終了は Ctrl+C")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n終了しました")


if __name__ == "__main__":
    main()
