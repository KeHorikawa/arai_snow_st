/* 妙高の雪 — Service Worker
 *
 * 通信を横取りして、保存してあるものを返す番人。
 * 「速さ」と「新しさ」は競合するので、ファイルの種類ごとに戦略を分けている。
 *
 *   シェル（HTML・Chart.js・アイコン・manifest） … キャッシュ優先
 *       毎朝同じものを読むだけなので、ネットワークを待たずに即出す。起動を速くするため。
 *
 *   data/snow_data.json                        … ネットワーク優先、失敗したらキャッシュ
 *       数字は新しいほうがよい。ただし圏外でも昨日の数値が出てほしい。
 *
 * パスはすべて相対で書く。GitHub Pages のプロジェクトサイトは
 * /arai_snow_st/ 配下に置かれるため、/ から始めると404になる。
 */

// 配信物を更新したらここを上げる。古いキャッシュは activate で消える。
const VERSION = "v1";
const SHELL_CACHE = `miyoko-snow-shell-${VERSION}`;
const DATA_CACHE = `miyoko-snow-data-${VERSION}`;

const SHELL_ASSETS = [
  "./",
  "./index.html",
  "./manifest.json",
  "./vendor/chart.umd.min.js",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
  "./icons/apple-touch-icon-180.png"
];

const DATA_PATH = "data/snow_data.json";

self.addEventListener("install", (event) => {
  event.waitUntil(
    Promise.all([
      caches.open(SHELL_CACHE).then((cache) => cache.addAll(SHELL_ASSETS)),
      // データも install の時点で入れておく。
      // 初回アクセスでは、ページがデータを取りに行く時点でまだ Service Worker が
      // 通信を見張っていない。ここで入れておかないと、
      // 「一度も online で開き直していない状態で圏外になる」と何も出せない。
      caches.open(DATA_CACHE).then((cache) => cache.add("./" + DATA_PATH))
    ])
      // 新しい版をすぐ使う。毎朝1回しか開かない道具なので、
      // 古い版に張り付いたまま気づかない状態のほうが困る。
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((names) => Promise.all(
        names
          .filter((n) => n !== SHELL_CACHE && n !== DATA_CACHE)
          .map((n) => caches.delete(n))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;

  // GET 以外と、他サイトへの通信には手を出さない
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  // ① データ：ネットワーク優先
  if (url.pathname.endsWith(DATA_PATH)) {
    event.respondWith(
      fetch(req)
        .then((res) => {
          const copy = res.clone();
          caches.open(DATA_CACHE).then((c) => c.put(req, copy));
          return res;
        })
        .catch(() => caches.match(req).then((hit) => hit || caches.match("./" + DATA_PATH)))
    );
    return;
  }

  // ② 画面遷移：キャッシュ優先（圏外でも開けるように）
  if (req.mode === "navigate") {
    event.respondWith(
      caches.match("./index.html").then((hit) => hit || fetch(req))
    );
    return;
  }

  // ③ シェル：キャッシュ優先。取れたものは次回のために保存しておく
  event.respondWith(
    caches.match(req).then((hit) => {
      if (hit) return hit;
      return fetch(req).then((res) => {
        if (res && res.status === 200 && res.type === "basic") {
          const copy = res.clone();
          caches.open(SHELL_CACHE).then((c) => c.put(req, copy));
        }
        return res;
      });
    })
  );
});
