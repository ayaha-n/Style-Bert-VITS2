#!/bin/bash
# Python 3.10 をソースからビルドして /usr/local/bin/python3.10 に入れる (コンテナ内で実行する)
# deadsnakes PPA が focal (20.04) 向けの配布をやめたため
set -euxo pipefail
PY_VERSION=3.10.14
command -v python3.10 && exit 0

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential wget ca-certificates \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev \
  uuid-dev libncursesw5-dev libgdbm-dev
cd /tmp
wget -q https://www.python.org/ftp/python/${PY_VERSION}/Python-${PY_VERSION}.tgz
tar xzf Python-${PY_VERSION}.tgz
cd Python-${PY_VERSION}
./configure --prefix=/usr/local --with-ensurepip=install
make -j"$(nproc)"
make altinstall   # python3 (3.8) は置き換えない
cd / && rm -rf /tmp/Python-${PY_VERSION}*
python3.10 -c "import ssl, sqlite3, lzma, bz2, ctypes; print('python3.10 ok')"
