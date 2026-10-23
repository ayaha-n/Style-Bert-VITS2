#!/bin/bash
# sbv2-jp5 イメージで server_editor.py --onnx を起動する (ホストで実行する)
# 使い方: run_server.sh [cuda|cpu] [ポート]   例: run_server.sh cuda 8000
set -euo pipefail
DEVICE=${1:-cuda}
PORT=${2:-8000}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
# model_assets/new_model2 は pooh_body のリソースへのシンボリックリンクなので、リンク先も同じパスでマウントする
MODEL_ZOO=${MODEL_ZOO:-$HOME/catkin_ws/src/modular_robot_model_zoo}

# --skip_static_files でも static/ がないと起動時に落ちる (static/ は .gitignore 済み)
mkdir -p "$REPO/static"

# --user: リポジトリに root 所有のファイルを残さない
# --group-add video: Jetson の GPU (/dev/nvmap など) は video グループでないと開けない
exec sudo docker run --rm --runtime nvidia --network host \
  --user "$(id -u):$(id -g)" --group-add "$(getent group video | cut -d: -f3)" -e HOME=/tmp \
  -v "$REPO:$REPO" -v "$MODEL_ZOO:$MODEL_ZOO:ro" -w "$REPO" sbv2-jp5 \
  python server_editor.py --device "$DEVICE" --onnx --port "$PORT" --skip_static_files
