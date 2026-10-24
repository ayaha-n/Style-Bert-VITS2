# JetPack 5 の Jetson で editor サーバーを GPU で動かす (Docker)

reComputer Mini J4012 (Jetson Orin NX 16GB) など、JetPack 5 (L4T r35, Ubuntu 20.04) の
Jetson で `server_editor.py --onnx` を GPU で動かすための手順です。
reComputer Mini J4012 (L4T r35.5.0, JetPack 5.1.3) で動作を確認しました。

使うファイルは [`docs/jetson_jp5/`](jetson_jp5/) にあります。

| ファイル | 役割 |
|---|---|
| `install_py310.sh` | Python 3.10 をソースからビルドする (コンテナ内で使う) |
| `build_ort.sh` | onnxruntime-gpu 1.18.1 (Python 3.10, CUDA) の wheel をビルドする (コンテナ内で使う) |
| `patch_antialias.py` | onnxruntime を CUDA 11.4 でビルドするためのソースの修正 (`build_ort.sh` が使う) |
| `Dockerfile` | サーバー用のイメージ `sbv2-jp5` |
| `run_server.sh` | サーバーを起動する (ホストで使う) |

## なぜ Docker か

- Style-Bert-VITS2 は Python 3.9 以上が必要ですが、JetPack 5 の Python は 3.8 です。
  NVIDIA が配布する Jetson 用の PyTorch や onnxruntime-gpu も JetPack 5 向けは Python 3.8 用が中心です。
- JetPack 6 に上げれば Python 3.10 になりますが、ROS noetic などホスト側の環境を変えたくないので、
  サーバーだけを r35 系のコンテナに入れ、その中で Python 3.10 を使います。
  ホストに入れるものはなく、作られるのは Docker イメージと作業ディレクトリ (`~/ort-build`) だけです。
- r36 系 (JetPack 6) のイメージは JetPack 5 のホストでは動きません。r35 系を使います。

## このフォークでの変更

### `--onnx` では PyTorch の CUDA を見ない

ONNX モデルの推論 (`style_bert_vits2/models/infer_onnx.py` と ONNX 版 BERT) は onnxruntime だけで行い、
PyTorch は使いません。ところが `server_editor.py` は `--device cuda` のとき
`torch.cuda.is_available()` が偽だと黙って CPU に切り替えていました。
Python 3.10 で使える CUDA 対応の PyTorch は JetPack 5 では手に入りにくいため、
`--onnx` のときは onnxruntime に `CUDAExecutionProvider` があるかで判断するようにしました。
起動時に `Device: cuda (onnx=True)` のように使うデバイスをログに出します。

これにより、コンテナには **CPU 版の PyTorch と、CUDA 対応の onnxruntime-gpu** があれば足ります。
GPU が必要なのは onnxruntime-gpu だけです。

### ピッチと抑揚の調整を速くする

`pitchScale` か `intonationScale` が 1.0 でないとき、合成した音声を pyworld で作り直します
(`style_bert_vits2/voice.py`)。この処理は CPU で動き、f0 の推定に使っていた `harvest` が遅く、
Jetson では GPU での合成本体より時間がかかっていました。`dio` + `stonemask` に変えました。
聞き比べて違いは分かりませんでした。

## 0. ホストの確認

```bash
cat /etc/nv_tegra_release        # 例: R35 (release), REVISION: 5.0 → L4T r35.5.0
sudo docker info | grep -i runtime    # nvidia ランタイムがあること
```

JetPack を SDK Manager などで入れていれば `nvidia-container-runtime` も入っています。
コンテナのイメージは `nvcr.io/nvidia/l4t-jetpack:r35.4.1` を使います
(r35.5.0 のホストで動きました。r35.x のコンテナは r35.x の別リビジョンのホストでもおおむね動きます)。

モデル (pooh_body の `new_model2`) は Git LFS で管理されています。中身が 134 バイトの
ポインタのままなら、LFS をそのリポジトリだけで有効にして取得します
(`--local` を付けないと `~/.gitconfig` が変わり、LFS を使うほかのリポジトリにも効きます)。

```bash
sudo apt install git-lfs
cd ~/catkin_ws/src/modular_robot_model_zoo
git lfs install --local
git lfs pull --include "pooh_body/resources/Style-Bert-VITS2/**"
```

## 1. onnxruntime-gpu (Python 3.10) の wheel を作る

[Jetson Zoo](https://elinux.org/Jetson_Zoo#ONNX_Runtime) の JetPack 5 向け wheel は Python 3.8 用なので、
Python 3.10 用を自分でビルドします。

```bash
mkdir -p ~/ort-build
cp -a ~/Style-Bert-VITS2/docs/jetson_jp5/. ~/ort-build/
sudo docker run -it --rm --runtime nvidia -v ~/ort-build:/out -e DEBIAN_FRONTEND=noninteractive \
  nvcr.io/nvidia/l4t-jetpack:r35.4.1 /out/build_ort.sh $(id -u) $(id -g)
ls ~/ort-build/onnxruntime_gpu-*.whl
```

- `--parallel 2` で数時間かかります。Orin NX 16GB (CPU 4 コアで動作) では、スワップを足さずにメモリは足りました。
  途中で切れないように byobu や tmux の中で実行します。
- ログは `~/ort-build/build.log` に残ります。
- ソースは `~/ort-build/src` に置くので、途中で止まっても同じコマンドで続きからビルドできます。
- 終わったら、`~/ort-build` の中身の所有者はホストのユーザーに戻ります。

## 2. サーバー用のイメージを作る

```bash
cp ~/Style-Bert-VITS2/requirements-infer.txt ~/ort-build/
cd ~/ort-build && sudo docker build -t sbv2-jp5 .
sudo docker run --rm --runtime nvidia sbv2-jp5 \
  python -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
# 1.18.1 ['CUDAExecutionProvider', 'CPUExecutionProvider'] と出ること
```

`requirements-infer.txt` から onnxruntime を外しているのは、PyPI の `onnxruntime` が入ると
自前の `onnxruntime-gpu` と衝突するためです。

## 3. BERT とモデルを置く

`--onnx` のサーバーが使う BERT は、日本語の ONNX 版 `model_fp16.onnx` だけです。
`initialize.py --only_infer` はほかの言語や PyTorch 版も取って数 GB になるので、これだけを取得します
(取得済みの PC から scp で持ってきてもかまいません)。

```bash
cd ~/Style-Bert-VITS2
sudo docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:$PWD -w $PWD sbv2-jp5 \
  python -c "from huggingface_hub import hf_hub_download; hf_hub_download('tsukumijima/deberta-v2-large-japanese-char-wwm-onnx', 'model_fp16.onnx', local_dir='bert/deberta-v2-large-japanese-char-wwm-onnx')"
ln -s ~/catkin_ws/src/modular_robot_model_zoo/pooh_body/resources/Style-Bert-VITS2/model_assets/new_model2 model_assets/
```

## 4. 起動する

```bash
~/Style-Bert-VITS2/docs/jetson_jp5/run_server.sh cuda 8000
```

起動ログに `Device: cuda (onnx=True)` と `Uvicorn running on http://0.0.0.0:8000` が出れば動いています。
最初の合成のあと `Model loaded successfully from ... to CUDAExecutionProvider` と出れば、GPU で合成しています。

ロボットで常に使うときは `--daemon` を付けます。コンテナ `sbv2-server` としてバックグラウンドで起動し、
以後はホストの起動時に Docker が自動で起動します (sudo が要るのは最初の 1 回だけ)。

```bash
~/Style-Bert-VITS2/docs/jetson_jp5/run_server.sh --daemon cuda 8000
sudo docker logs -f sbv2-server     # ログ
sudo docker rm -f sbv2-server       # 止めて自動起動もやめる
```

`pooh_bringup.launch` はサーバーを起動せず、TTS ノード (`pooh_body/scripts/style_bert_vits2_tts.py`) だけを起動します。
TTS ノードはそのまま `localhost:8000` のこのサーバーを使います。
別のホストのサーバーを使う場合は TTS ノードの `_server_url` で指定します。

## 速さ

TTS ノードの既定値 (`speed` 0.7、`pitchScale` 0.8、`intonationScale` 1.2) で、
音声 6〜8 秒の文を合成した時間です。

| 環境 | 合成時間 / 音声の長さ |
|---|---|
| ThinkPad の CPU | 約 1.0 |
| Jetson のコンテナの CPU (`run_server.sh cpu`) | 約 3.3 |
| Jetson の GPU (`harvest`) | 約 1.0 |
| Jetson の GPU (`dio`、現在) | 約 0.6 |
| Jetson の GPU (`pitchScale`、`intonationScale` とも 1.0) | 約 0.2 |

GPU での合成本体は 1〜1.5 秒で、残りはピッチと抑揚の調整 (CPU) の時間です。
ピッチと抑揚を 1.0 にしても同じ声になるモデル (調整済みの音声で学習し直すなど) にすれば、
さらに速くなります。

## うまくいかなかったこと

`build_ort.sh` などで対処済みです。onnxruntime の版を変えるときの参考に残します。

| 症状 | 原因 | 対処 |
|---|---|---|
| `Unable to locate package python3.10-dev` | deadsnakes PPA が focal 向けの配布をやめた | `install_py310.sh` でソースからビルド |
| eigen の `does not match expected value` | GitLab がアーカイブを作り直してハッシュが変わった (中身は同じコミット) | `cmake/deps.txt` のハッシュを `32b145f5…` に変更 |
| `pytorch_clog` などの `does not match expected value` | ダウンロードの一時的な失敗 | `--update` を 3 回まで繰り返す |
| `detected dubious ownership in repository` | `/out` がホストのユーザーの所有で、コンテナの git は root | コンテナ内の git だけで `safe.directory` を設定 |
| `The compiler doesn't support BFLOAT16!!!` | onnxruntime 1.18 は aarch64 で bf16 対応のコンパイラが必要。gcc 9 は非対応 | gcc-10 を使う (`CC`、`CXX`、`CUDAHOSTCXX`) |
| `'CUDA_R_8F_E4M3' was not declared` | FP8 型は CUDA 11.8 以上 | `--disable_types float8` |
| `missing return statement at end of non-void function` (`_IsInf`) | CUDA 11.4 の nvcc が `if constexpr` で誤った警告を出し、それがエラー扱いになる | `--compile_no_warning_as_error` |
| `Call parameter type does not match function signature!` (`resize_antialias_impl.cu`) | CUDA 11.4 の nvcc が `std::tuple` を値渡しするカーネルで内部エラーになる (gcc の版によらない) | `patch_antialias.py` で `cuda::std::tuple` に置き換え |
| サーバー起動時に `接続が閉じられました` (`update_dict`) | pyopenjtalk が初回に辞書をパッケージのディレクトリに入れようとするが、root 以外では書けない | イメージを作るときに辞書を入れる (`Dockerfile`) |
| `NvRmMemInitNvmap failed with Permission denied`、`CUDA failure 801` で CPU に切り替わる | Jetson の GPU のデバイスは video グループでないと開けない | `--group-add video` |
| `Directory 'static' does not exist` | `--skip_static_files` でも `static/` を読み込む | `static/` を作る (`run_server.sh`) |

次のメッセージは出ますが、動作に影響はありません。

- `pthread_setaffinity_np failed ... mask: {5, }`: 電源モードで CPU が 4 コアしか動いていないため。
- `[transformers] Disabling PyTorch because PyTorch >= 2.6 is required`: `--onnx` では PyTorch を使わない。
- `Specified provider 'DmlExecutionProvider' is not in available provider names`: Windows 用。
