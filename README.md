# boatrace_bot

ボートレースの直前情報からAIモデルで予測し、期待値のある舟券をDiscordに通知するボット。

## 自動実行の仕組み

GitHub Actionsの`schedule`トリガーは実行タイミングが大きく遅延・間引きされ信頼できないため、外部サービス [cron-job.org](https://cron-job.org) から `workflow_dispatch` をHTTPで叩く方式で動かしている。

実測（2026-07-13、6時間分）では、`cron: '3-59/15 * * * *'`（15分おき）を指定しても`schedule`は**2回しか発火しなかった**（本来24回）。さらに稀に発火した回が`concurrency`の`cancel-in-progress`によってcron-job.orgからの本来の実行をキャンセルする事故も起きていたため、**`schedule`トリガーはワークフローから完全に削除した**。定期実行はcron-job.orgに一本化している。

- cron-job.orgが7:00〜21:00 JSTの間、15分おきに以下を呼び出す
  ```
  POST https://api.github.com/repos/kibidanngo3/boatrace_bot/actions/workflows/run_bot.yml/dispatches
  Headers:
    Authorization: Bearer <Personal Access Token>
    Accept: application/vnd.github+json
    Content-Type: application/json
  Body: {"ref":"main"}
  ```
- main.py側にも7:00〜22:00 JST以外はスキャンをスキップするガードを入れてある（cron-job.org側の設定ミスや手動実行に対する保険）

## Discord通知チャンネルの分割

通知は用途ごとに別チャンネルへ振り分けられる。GitHubのSecretsに以下を登録する（Settings → Secrets and variables → Actions）。

| Secret名 | 流れる通知 |
|---|---|
| `DISCORD_WEBHOOK_URL_PREDICT` | 投資チャンス（買い目）の通知 |
| `DISCORD_WEBHOOK_URL_STATS` | 日次・週次・月次の成績サマリー |
| `DISCORD_WEBHOOK_URL_SYSTEM` | クラッシュ・スケジュール取得失敗・モデル更新 |
| `DISCORD_WEBHOOK_URL_SCHEDULE` | 締切順ダイジェスト |

**未設定のものは既存の `DISCORD_WEBHOOK_URL` に自動でフォールバックする**ので、分割したいチャンネルだけ登録すればよい。全て未設定なら従来どおり1チャンネルに全部流れる。

Webhookの発行方法: Discordで対象チャンネルを右クリック → 連携サービス → ウェブフック → 新しいウェブフック → URLをコピー。

### 締切順ダイジェストについて

Botは15分おきに走り、EV条件を満たさず見送ったレースは「通知済み」にならない。そのため後の実行でオッズが動いて条件を満たすと、**すでに通知したレースより締切が早いレース**が後から通知され、タイムライン上で締切が前後して見える。

これを補うため、予想を出した実行の最後に、その時点で未締切の予想レースを締切順に並べた一覧を投稿する。

```
⏰ 締切スケジュール（未締切 3件）
13:06 蒲郡 5R｜6分後｜2点 ¥300
13:14 江戸川 11R｜14分後｜1点 ¥200
13:40 津 7R｜40分後｜3点 ¥400
```

残り時間はDiscordの相対タイムスタンプ記法 `<t:UNIX時刻:R>` で埋め込んでいる。これは**閲覧しているDiscordクライアントが現在時刻から描画する**ため、投稿から時間が経っても「6分後」が固まったまま残らず、常に今から見た残り時間が表示される。同じ理由で個別の予想通知にも締切カウントダウンを入れてある。

## Personal Access Token (PAT) の期限と再設定方法

cron-job.orgに登録しているGitHubのFine-grained PATは **2026年10月6日に失効する**。失効するとcron-job.orgからの`workflow_dispatch`が401エラーで失敗し、自動実行が静かに止まるので注意。

### 再設定手順

1. https://github.com/settings/tokens?type=beta にアクセスし、既存の`boatrace_bot`トークンを開く
2. 「Regenerate token」で新しいトークンを発行する（有効期限を再度設定。Repository access・Permissionsは維持される）
   - もしくは新規発行する場合は以下を設定する:
     - Repository access: "Only select repositories" → `kibidanngo3/boatrace_bot`
     - Permissions → Repository permissions → **Actions: Read and write**
3. 発行された新しいトークンをコピー（画面を閉じると二度と表示されない）
4. [cron-job.org](https://cron-job.org) にログインし、該当ジョブの編集画面を開く
5. Headers の `Authorization` の値を `Bearer <新しいトークン>` に書き換えて保存
6. ジョブの「Test run」を実行し、GitHub Actions側（Actionsタブ）で新しい実行が`workflow_dispatch`イベントとして開始されることを確認する

## 賭け金の計算 (ケリー基準)

各買い目の賭け金は、複数の排反な買い目に同時に賭ける場合のケリー基準 (Thorpの一般化式) で計算している。

- `main.py`の定数（変更する場合はここを編集する）
  - `STARTING_BANKROLL = 10000`: 初期バンクロール（円）
  - `KELLY_FRACTION = 0.25`: 1/4ケリー（モデル誤差を考慮して保守的に）
  - `MAX_RACE_STAKE_RATIO = 0.10`: 1レースあたりの賭け金上限（バンクロールの10%）
  - `DAILY_LOSS_LIMIT_RATIO = 0.20`: その日の損失がバンクロールの20%に達したら以降の新規ベットを停止（決済・サマリー送信は継続）
- 現在のバンクロール・その日の開始時点バンクロールは`bot_state.json`に保存される
- バンクロールを手動でリセットしたい場合は、後述の`bot-state`ブランチ上の`bot_state.json`を直接編集する

## 状態の永続化 (predictions.csv / bot_state.json)

これらは`main`ブランチにはコミットされず（`.gitignore`対象）、専用の**`bot-state`ブランチ**に毎回コミットされる形で永続化している。GitHub Actionsの`actions/cache`はビルドキャッシュ用途のものでいつ消えても文句を言えない仕組みのため、金銭に関わる状態の保存先としては使っていない。

- ワークフロー実行のたびに`bot-state`ブランチから最新の状態を取得 → 実行 → 更新分を`bot-state`ブランチにコミット&プッシュ、という流れ
- 過去の状態はすべて`bot-state`ブランチのコミット履歴として残るので、`git log`でバンクロールの推移などを後から追跡できる
- 手元で最新の`predictions.csv`を見たい場合は以下で取得できる
  ```
  git fetch origin bot-state
  git show origin/bot-state:predictions.csv > predictions.csv
  ```

## 成績ダッシュボード

`predictions.csv`から自己完結型のHTML（資金推移グラフ・日別/会場別/戦略別の集計・全履歴・最大ドローダウン）を生成し、Bot実行のたびに`gh-pages`ブランチへ自動公開している。

- 公開URL: https://kibidanngo3.github.io/boatrace_bot/
- 手元で生成する場合: `python scripts/build_dashboard.py` → `docs/index.html`

## ヘルスチェック（サイレント故障の検知）

**GitHub Actionsが`success`でも中身が壊れている**ことがあり、実際に以下の障害を長期間気づかず踏んでいた。

- `bot-state`ブランチが一度も作られず、実績が全く蓄積されていなかった（`git add -A`が`.gitignore`で無視されていた）
- 連敗カウントが古い値のまま残り、ケリー係数が半減しっぱなしだった

このためBotは毎回以下を自己点検し、異常があれば`#システム`チャンネルに警告する（正常時は何も送らない。同じ警告は6時間に1回まで）。

| 検知内容 | 意味 |
|---|---|
| 予想ログの行数が前回より減った | 状態の復元に失敗し実績が失われている疑い |
| 稼働時間中に6時間以上ベットが出ない | スクレイピング破損・モデル異常の兆候（相場次第で正常なこともある） |
| 締切から3時間過ぎた未決着が3件以上 | 結果ページの取得に失敗し決着処理が回っていない疑い |
| バンクロールが最小賭け金を下回った | 資金枯渇でベット不能 |

なお、スケジュール取得の連続失敗とクラッシュは以前から別途通知している。

## バックテスト

蓄積した`predictions.csv`を使って成績・EV閾値ごとの回収率を確認できる。上記の方法で`bot-state`ブランチから最新の`predictions.csv`を取得してから実行する。

```
python scripts/backtest.py
python scripts/backtest.py --strategy FOCUS
python scripts/backtest.py --ev-thresholds 1.0 1.5 2.0
```
