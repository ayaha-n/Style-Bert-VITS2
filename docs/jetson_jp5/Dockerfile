FROM nvcr.io/nvidia/l4t-jetpack:r35.4.1

# deadsnakes PPA は focal (20.04) 向けの配布をやめたので、Python 3.10 はソースからビルドする
ENV DEBIAN_FRONTEND=noninteractive
COPY install_py310.sh /tmp/
RUN apt-get update && apt-get install -y build-essential cmake git \
 && /tmp/install_py310.sh \
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

# pyopenjtalk は初回に辞書を自分のパッケージのディレクトリにダウンロードする。
# サーバーは root 以外のユーザーで動かし、そこには書き込めないので、イメージを作るときに入れておく
RUN python -c "import pyopenjtalk; print(pyopenjtalk.g2p('こんにちは'))"
