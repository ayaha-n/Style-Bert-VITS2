#!/bin/bash
# onnxruntime-gpu (Python 3.10, CUDA, sm_87) をビルドする。l4t-jetpack:r35.4.1 コンテナ内で実行する
# 使い方: build_ort.sh <ホストの uid> <ホストの gid>
set -euxo pipefail
# 出力を /out/build.log にも残す
exec > >(tee /out/build.log) 2>&1
HOST_UID=${1:-1000}
HOST_GID=${2:-1000}
# 失敗しても、ホストのユーザーで扱えるように所有者を戻す
trap 'chown -R "$HOST_UID:$HOST_GID" /out' EXIT

apt-get update && apt-get install -y git build-essential gcc-10 g++-10
# onnxruntime 1.18 は aarch64 で bf16 対応のコンパイラが必要。gcc 9 は非対応なので gcc 10 を使う (nvcc のホストコンパイラも)
export CC=gcc-10 CXX=g++-10 CUDAHOSTCXX=g++-10
# /out はホストのユーザーの所有なので、コンテナの root の git に信頼させる (コンテナ内の設定なのでホストには影響しない)
git config --global --add safe.directory '*'
/out/install_py310.sh
python3.10 -m pip install -U pip "cmake<4" "numpy<2" packaging wheel setuptools psutil

# ソースは /out/src に置く (途中で止まっても再開できるように)
if [ ! -d /out/src/onnxruntime ]; then
  git clone --recursive --branch v1.18.1 https://github.com/microsoft/onnxruntime /out/src/onnxruntime
fi
cd /out/src/onnxruntime
# GitLab が eigen のアーカイブを作り直してハッシュが変わったので合わせる (中身は同じコミット)
sed -i 's/be8be39fdbc6e60e94fa7870b280707069b5b81a/32b145f525a8308d7ab1c09388b2e288312d8eba/' cmake/deps.txt
# CUDA 11.4 の nvcc が std::tuple を値渡しするカーネルで内部エラーになるので、cuda::std::tuple に置き換える
python3.10 /out/patch_antialias.py onnxruntime/core/providers/cuda/tensor/resize_antialias_impl.cu
# build.sh は python3 (3.8) を使うので、build.py を python3.10 で直接呼ぶ
# スワップを足さないので --parallel 2 にしている
ORT_ARGS=(--build_dir build/Linux --config Release --parallel 2 --build_wheel
  --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu
  --skip_tests --allow_running_as_root
  --disable_types float8  # FP8 型は CUDA 11.8 以上が必要。JetPack 5 は CUDA 11.4
  --compile_no_warning_as_error  # CUDA 11.4 の nvcc が if constexpr で誤った警告を出し、それがエラー扱いになるため
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=87 onnxruntime_BUILD_UNIT_TESTS=OFF)
# --update (cmake) は依存のダウンロードで一時的に失敗することがあるので 3 回まで試す
for i in 1 2 3; do
  python3.10 tools/ci_build/build.py "${ORT_ARGS[@]}" --update && break
  [ "$i" = 3 ] && exit 1
  sleep 10
done
python3.10 tools/ci_build/build.py "${ORT_ARGS[@]}" --build
cp build/Linux/Release/dist/onnxruntime_gpu-*.whl /out/
