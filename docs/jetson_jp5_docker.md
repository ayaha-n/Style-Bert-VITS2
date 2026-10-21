# JetPack 5 の Jetson で editor サーバーを GPU で動かす (Docker)

reComputer Mini J4012 (Jetson Orin NX 16GB) など、JetPack 5 (L4T r35, Ubuntu 20.04) の
Jetson で `server_editor.py --onnx` を GPU で動かすための手順です。
**まだ実機では試していません。** 試して分かったことはこのファイルに書き足してください。

## なぜ Docker か

- Style-Bert-VITS2 は Python 3.9 以上が必要ですが、JetPack 5 の Python は 3.8 です。
  NVIDIA が配布する Jetson 用の PyTorch や onnxruntime-gpu も JetPack 5 向けは Python 3.8 用が中心です。
- JetPack 6 に上げれば Python 3.10 になりますが、ROS noetic などホスト側の環境を変えたくないので、
  サーバーだけを r35 系のコンテナに入れ、その中で Python 3.10 を使います。
- r36 系 (JetPack 6) のイメージは JetPack 5 のホストでは動きません。r35 系を使います。

## このフォークでの変更: `--onnx` では PyTorch の CUDA を見ない

ONNX モデルの推論 (`style_bert_vits2/models/infer_onnx.py` と ONNX 版 BERT) は onnxruntime だけで行い、
PyTorch は使いません。ところが `server_editor.py` は `--device cuda` のとき
`torch.cuda.is_available()` が偽だと黙って CPU に切り替えていました。
Python 3.10 で使える CUDA 対応の PyTorch は JetPack 5 では手に入りにくいため、
`--onnx` のときは onnxruntime に `CUDAExecutionProvider` があるかで判断するようにしました。
起動時に `Device: cuda (onnx=True)` のように使うデバイスをログに出します。

これにより、コンテナには **CPU 版の PyTorch と、CUDA 対応の onnxruntime-gpu** があれば足ります。
GPU が必要なのは onnxruntime-gpu だけです。

## 0. ホストの確認

```bash
cat /etc/nv_tegra_release        # 例: R35 (release), REVISION: 4.1 → L4T r35.4.1
docker info | grep -i runtime    # nvidia ランタイムがあること
```

JetPack を SDK Manager などで入れていれば `nvidia-container-runtime` も入っています。
以下の `r35.4.1` は、ホストの L4T に合わせて読み替えてください
(r35.x のコンテナは r35.x の別リビジョンのホストでもおおむね動きます)。

onnxruntime のビルドはメモリを多く使うので、スワップを足しておくと安全です。

```bash
sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```

## 1. onnxruntime-gpu (Python 3.10) の wheel を作る

[Jetson Zoo](https://elinux.org/Jetson_Zoo#ONNX_Runtime) の JetPack 5 向け wheel は Python 3.8 用なので、
Python 3.10 用を自分でビルドします。Jetson Zoo に JetPack 5 向けの 1.18 系があるので、
CUDA 11.4 / cuDNN 8.6 でビルドできる見込みの v1.18.1 を使います。
Orin の GPU アーキテクチャは `87` です。

ビルド用のコンテナに入ります (`l4t-jetpack` は CUDA、cuDNN、TensorRT の開発用ファイルを含みます)。

```bash
mkdir -p ~/ort-build
docker run -it --rm --runtime nvidia -v ~/ort-build:/out nvcr.io/nvidia/l4t-jetpack:r35.4.1 bash
```

コンテナの中で:

```bash
apt-get update && apt-get install -y software-properties-common git build-essential
add-apt-repository -y ppa:deadsnakes/ppa && apt-get update
apt-get install -y python3.10 python3.10-dev python3.10-venv
python3.10 -m ensurepip --upgrade
python3.10 -m pip install -U pip "cmake<4" "numpy<2" packaging wheel setuptools

git clone --recursive --branch v1.18.1 https://github.com/microsoft/onnxruntime /src/onnxruntime
cd /src/onnxruntime
# build.sh は python3 (3.8) を使うので、build.py を python3.10 で直接呼ぶ
python3.10 tools/ci_build/build.py --build_dir build/Linux \
  --config Release --update --build --parallel 4 --build_wheel \
  --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu \
  --skip_tests --allow_running_as_root \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=87 onnxruntime_BUILD_UNIT_TESTS=OFF
cp build/Linux/Release/dist/onnxruntime_gpu-*.whl /out/
```

数時間かかります。`--parallel` を増やすと速くなりますが、メモリ不足で落ちやすくなります。

## 2. サーバー用のイメージを作る

`~/ort-build/` に、次の `Dockerfile` と、1 で作った wheel を置きます。

```dockerfile
FROM nvcr.io/nvidia/l4t-jetpack:r35.4.1

RUN apt-get update && apt-get install -y software-properties-common build-essential cmake git \
 && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update \
 && apt-get install -y python3.10 python3.10-dev python3.10-venv \
 && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip install -U pip wheel

# PyTorch は CPU 版 (PyPI の aarch64 wheel) で足りる。GPU は onnxruntime-gpu だけが使う
COPY requirements-infer.txt /tmp/
RUN grep -v '^onnxruntime' /tmp/requirements-infer.txt > /tmp/req.txt \
 && pip install -r /tmp/req.txt
COPY onnxruntime_gpu-*.whl /tmp/
RUN pip install /tmp/onnxruntime_gpu-*.whl
```

```bash
cp ~/Style-Bert-VITS2/requirements-infer.txt ~/ort-build/
cd ~/ort-build && docker build -t sbv2-jp5 .
```

`requirements-infer.txt` から onnxruntime を外しているのは、PyPI の `onnxruntime` が入ると
自前の `onnxruntime-gpu` と衝突するためです。
`pyopenjtalk-dict` などは aarch64 の wheel がなくソースからビルドされることがあります
(そのために `build-essential` と `cmake` を入れています)。

## 3. 起動する

リポジトリ (モデル、BERT、ユーザー辞書を含む) はイメージに入れず、マウントします。
`model_assets/new_model2` は pooh_body のリソースへのシンボリックリンクなので、
リンク先も同じパスでマウントします。

```bash
docker run --rm --runtime nvidia --network host \
  -v $HOME/Style-Bert-VITS2:$HOME/Style-Bert-VITS2 \
  -v $HOME/catkin_ws/src/modular_robot_model_zoo:$HOME/catkin_ws/src/modular_robot_model_zoo \
  -w $HOME/Style-Bert-VITS2 sbv2-jp5 \
  python server_editor.py --device cuda --onnx --port 8000 --skip_static_files
```

BERT の ONNX モデルがまだなければ、先に同じコンテナで
`python initialize.py --only_infer` を実行して取得します。
TTS ノード (`pooh_body/scripts/style_bert_vits2_tts.py`) 側は変更不要です。
別のホストで動かす場合は `_server_url` で指定します。

## 4. GPU で動いているか確かめる

```bash
docker run --rm --runtime nvidia sbv2-jp5 \
  python -c "import onnxruntime as o; print(o.get_available_providers())"
# CUDAExecutionProvider が含まれていること
```

- サーバーの起動ログに `Device: cuda (onnx=True)` と出ること。
- 最初の合成のあと、`Model loaded successfully from ... to CUDAExecutionProvider` と出ること
  (`CPUExecutionProvider` なら GPU は使われていません)。
- 速さは、同じ文を CPU 版 (`--device cpu`) と GPU 版で合成し、音声の長さに対する合成時間で比べます。
  CPU ではほぼ音声と同じ長さかかっていました。

## うまくいかないとき

- onnxruntime のビルドが CUDA 11.4 で通らない場合は、少し古い版 (v1.17.x、v1.16.x) を試します。
- ONNX モデルが新しい opset を要求して読み込めない場合は、onnxruntime の版を上げるか、
  モデルを低い opset で変換し直します。
- それでも難しければ、GPU 付きの x86 の VM (Azure の NC 系など) でサーバーを動かし、
  TTS ノードの `_server_url` をそちらに向けます。x86 なら PyPI の `onnxruntime-gpu` と CUDA 版 PyTorch が
  そのまま使えます。ネットワークの往復の分だけ遅れが増える点に注意します。
