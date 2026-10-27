# coinrun-DT

Procgen CoinRun のランダムなステージで Decision Transformer (DT) を学習し、その DT を使ってステージ（level seed）ごとの難易度を評価する。

```
PPO 学習 ──▶ 複数チェックポイントで軌跡収集 ──▶ DT 学習 ──▶ ステージ難易度評価・相関分析
ppo.train      data.collect                   dt.train     difficulty.score
```

## 研究室の CPU サーバで一括実行

```bash
git clone https://github.com/arataendo/coinrun-DT.git
cd coinrun-DT
bash scripts/setup_server.sh                                 # 初回のみ: .venv を作って依存を入れ、テストする
nohup bash scripts/run_pipeline.sh > pipeline.log 2>&1 &     # configs/cpu で全工程を実行
tail -f pipeline.log
```

- `setup_server.sh` は Python 3.10 を必要とする（procgen 0.10.7 は 3.10 以下でしか入らない）。`python3.10` がなければ `PYTHON=/path/to/python3.10` で指定する。`uv` があれば自動で使う。
- torch のスレッド数は、既定でコア数と 16 の小さい方。変えるときは `NUM_THREADS=8 nohup bash scripts/run_pipeline.sh ...` のように指定する。
- 止まっても、同じコマンドで再実行すれば続きから再開する。
  - 終わった工程は `runs/cpu/.done/` の印を見て飛ばす。
  - PPO と DT は最新のチェックポイントから続ける。
  - 難易度評価は、評価済みのステージを飛ばす。
  - 軌跡収集だけは最初からやり直す。
- 特定の工程からやり直すときは `FROM=dt bash scripts/run_pipeline.sh` とする。この工程以降の印が消える。
- 結果は `runs/cpu/difficulty/` に出る（`difficulty.csv`, `correlation_spearman.csv`, `scatter.png`）。学習ログは `runs/cpu/*/metrics.jsonl` と TensorBoard（`tensorboard --logdir runs/cpu`）に出る。
- ディスクは、軌跡データと DT 用キャッシュで約 15GB を見込んでおく。

## 設定

| ディレクトリ | 用途 | 規模 |
|---|---|---|
| `configs/small/` | 動作確認（このリポジトリのテスト用、数分） | PPO 5 万ステップ、48 エピソード、10 ステージ |
| `configs/cpu/` | 研究室の CPU サーバ | PPO 1000 万ステップ、6 チェックポイント × 512 エピソード、200 ステージ |
| `configs/gpu/` | GPU サーバ | PPO 2500 万ステップ、6 × 2048 エピソード、1000 ステージ |

`bash scripts/run_pipeline.sh configs/gpu` のように、設定のディレクトリを引数で渡す。各工程は単体でも動かせる。そのときは `--set key=value` で設定を上書きできる。

```bash
python -m coinrun_dt.ppo.train        --config configs/cpu/ppo.yaml --set total_steps=5000000
python -m coinrun_dt.data.collect     --config configs/cpu/collect.yaml
python -m coinrun_dt.dt.train         --config configs/cpu/dt.yaml
python -m coinrun_dt.difficulty.score --config configs/cpu/difficulty.yaml
python -m coinrun_dt.difficulty.analyze --csv runs/cpu/difficulty/difficulty.csv   # 相関分析だけやり直す
```

## 手元（Windows）でのセットアップ

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install setuptools wheel
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install --no-build-isolation gym==0.26.2   # ビルド分離では失敗するため
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e . --no-deps --no-build-isolation
.venv\Scripts\python -m pytest
.\scripts\smoke_test.ps1     # 全工程を small 設定で通す
```

## 難易度評価関数

`coinrun_dt.difficulty.score.evaluate_levels(dt_ckpt, level_seeds, cfg)` が、ステージごとの指標を DataFrame で返す。

**A. ロールアウトベース**（`rollout_metrics.py`）
各ステージで、目標リターン（既定 10 / 5 / 0）ごとに DT を `rollouts_per_level` 回動かす。

| 列 | 意味 |
|---|---|
| `success_rate` | 目標リターン 10 のときの成功率 |
| `mean_steps_to_success` | 成功したエピソードの平均ステップ数 |
| `rtg_sensitivity` | 目標リターン最大と最小のときの成功率の差 |
| `difficulty_rollout` | `1 - success_rate × (1 - steps_weight × 成功時ステップ数/1000)`。範囲は [0, 1] |

**B. DT 内部指標**（`model_metrics.py`）
ステージごとに参照軌跡を1本だけ用意し（最終 PPO 方策の最短成功軌跡、または DT の greedy 実行）、その軌跡に沿って DT の行動分布を調べる。

| 列 | 意味 |
|---|---|
| `mean_action_entropy`, `p95_action_entropy` | DT の行動分布のエントロピー（迷い） |
| `expert_nll` | 参照軌跡の行動に対する負の対数尤度 |
| `rtg_kl` | 目標リターンを高くしたときと低くしたときの行動分布の KL 距離 |
| `difficulty_model` | 評価した全ステージで z-score 化した指標の重み付き和（重みは `model_metric_weights`。既定は `expert_nll` 単体） |

**参照値**: `ppo_ckpt` を指定すると、最終 PPO 方策のステージごとの成功率 `ppo_ref_success_rate` も出る。相関分析で各指標と比べられる。

**重みを変えて合成し直す**: `difficulty_model` は、評価をやり直さずに `difficulty.csv` から重みを変えて作り直せる。結果は `runs/cpu/difficulty/reweighted/` に出て、元の CSV は書き換えない。重みは負にもできる。

```bash
python -m coinrun_dt.difficulty.analyze --csv runs/cpu/difficulty/difficulty.csv \
    --weights expert_nll=1 rtg_kl=1
```

**重みの組み合わせを一括で比べる**: `configs/cpu/weights.yaml` の `weight_sets` に並べた組み合わせと、各指標単体を一度に比べる。結果は `results/cpu/` に CSV で出る。`results/` は git add できる。

```bash
bash scripts/compare_weights.sh            # configs/cpu/weights.yaml
git add results/cpu && git commit -m "重みの比較結果" && git push
```

| ファイル | 中身 |
|---|---|
| `weight_comparison.csv` | 1行が1つの指標または合成。`rho_vs_ppo_success`（PPO 成功率との Spearman 相関）の小さい順に並ぶ |
| `levels.csv` | ステージごとの全指標と、各合成の値（`model_<名前>` 列） |
| `weights.yaml` | 比べた重みの組み合わせ |

`rho_vs_ppo_success` は −1 に近いほど良い（PPO が苦戦するステージを難しいと判定できている）。`strength` 列はその目安で、strong（≦ −0.7）、moderate（≦ −0.4）、weak（< 0）、reversed（≧ 0。向きが逆）の4段階。

**ホールドアウト検証**: 重みを選ぶのに使ったステージ（seed 0〜199）と別のステージ（seed 1000〜1199）で評価をやり直し、同じ比較表を作る。同じステージで選んで評価すると相関がよく見えすぎるので、ここでの数字を最終的な性能とする。サーバで約1時間かかる。

```bash
nohup bash scripts/run_holdout.sh > holdout.log 2>&1 &     # 結果は results/cpu_holdout/
git add results/cpu_holdout && git commit -m "ホールドアウト検証の結果" && git push
```

**別のエージェントでの検証**: 参照値 `ppo_ref_success_rate` は、DT の学習データを作った PPO 自身の成功率なので循環がある。そこで次の3つのエージェントを用意し、同じステージ（seed 0〜199 と 1000〜1199）の成功率を取り直して、各指標と比べる。

- seed だけ違う、独立に学習した PPO（`ppo_seed1.yaml`）
- DQN（`dqn.yaml`。Double DQN + Dueling + 3-step、環境ごとに違う ε）。PPO とは別の種類のエージェント。CPU で 1000 万ステップでどこまで学習できるかは未確認なので、`runs/cpu/logs/dqn.log` の `success_rate_eps<0.01`（ほぼ greedy な環境の成功率）を見て、ランダム（約 0.35）を上回っているか確かめる
- ランダム方策。学習しないので、ステージの構造だけで決まる難しさを表す

指標は評価し直さず、`results/cpu/` と `results/cpu_holdout/` の `levels.csv` を使う。サーバでは、2つの学習を並列にすると約5時間、その後の評価に約30分かかる。

```bash
nohup bash scripts/run_reference.sh > reference.log 2>&1 &   # 学習ログは runs/cpu/logs/
git add results/cpu_reference && git commit -m "別のエージェントでの検証結果" && git push
```

| ファイル | 中身 |
|---|---|
| `comparison.csv` | 指標 × 参照エージェント（`ppo_base` が元の PPO、`ref_<名前>` が新しいもの）× ステージの組（train / holdout / all）ごとの Spearman 相関 |
| `agreement.csv` | 参照エージェント同士の Spearman 相関。エージェント同士でもこの程度しか一致しないという、指標が届きうる上限の目安 |
| `levels.csv` | ステージごとの指標と、各エージェントの成功率 |

DQN の学習を最初からやり直すときは `checkpoints/cpu/dqn` と `runs/cpu/.done/dqn` を消す。評価をやり直すときは `runs/cpu/reference/` を消す。

## 構成

```
src/coinrun_dt/
  envs.py          環境生成（gym3 API。num_levels=0 でランダムなステージ、make_level_env で特定のステージ）
  runner.py        ベクトル化環境でエピソードを集める共通ループ（PPO と DT の両方で使う）
  ppo/             IMPALA CNN の Actor-Critic と PPO
  dqn/             DQN（参照値を別の種類のエージェントで取るため）
  data/            軌跡収集、DT 用データセット（メモリマップのキャッシュ）
  dt/              Decision Transformer、オンライン実行（DTPolicy）、軌跡に沿ったロジット計算
  difficulty/      難易度指標、評価関数、相関分析、別のエージェントでの検証（reference.py）
scripts/
  setup_server.sh  サーバの初回セットアップ
  run_pipeline.sh  全工程の一括実行（再開対応）
  compare_weights.sh  難易度指標の重みの組み合わせを比べ、results/ に CSV で保存
  run_holdout.sh   別のステージでの評価と比較（ホールドアウト検証）
  run_reference.sh 別のエージェント（PPO seed 違い、DQN、ランダム）での参照値と比較
  common.sh        上のスクリプトが共通で読み込む設定（Python の場所、スレッド数、工程の実行）
  smoke_test.ps1   Windows での動作確認
```

## メモ

- CoinRun の報酬は、コインを取ったときの 10 だけ。DT の RTG は `reward_scale=0.1` で 1.0 に正規化する。
- 学習時のステージは 2^31 の空間からランダムに引かれる。評価に使う seed（既定 0〜199 / 0〜999）との重なりは事実上ない。
- 評価用のステージを学習から完全に外したいときは、PPO と収集の `start_level` / `num_levels` で学習側の範囲を指定する。
