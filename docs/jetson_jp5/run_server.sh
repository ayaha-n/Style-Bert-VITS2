#!/bin/bash
# sbv2-jp5 イメージで server_editor.py --onnx を起動する (ホストで実行する)
# 使い方: run_server.sh [--daemon] [cuda|cpu] [ポート]   例: run_server.sh cuda 8000
#   --daemon: コンテナ sbv2-server としてバックグラウンドで起動し、ホストの起動時にも自動で起動する
#             (ログ: sudo docker logs -f sbv2-server、停止: sudo docker rm -f sbv2-server)
set -euo pipefail
DAEMON=false
if [ "${1:-}" = "--daemon" ]; then
  DAEMON=true
  shift
fi
DEVICE=${1:-cuda}
PORT=${2:-8000}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
# model_assets/new_model2 は pooh_body のリソースへのシンボリックリンクなので、リンク先も同じパスでマウントする
MODEL_ZOO=${MODEL_ZOO:-$HOME/catkin_ws/src/modular_robot_model_zoo}

# --skip_static_files でも static/ がないと起動時に落ちる (static/ は .gitignore 済み)
mkdir -p "$REPO/static"

# --user: リポジトリに root 所有のファイルを残さない
# --group-add video: Jetson の GPU (/dev/nvmap など) は video グループでないと開けない
ARGS=(--runtime nvidia --network host
  --user "$(id -u):$(id -g)" --group-add "$(getent group video | cut -d: -f3)" -e HOME=/tmp
  -v "$REPO:$REPO" -v "$MODEL_ZOO:$MODEL_ZOO:ro" -w "$REPO" sbv2-jp5
  python server_editor.py --device "$DEVICE" --onnx --port "$PORT" --skip_static_files)

if $DAEMON; then
  sudo docker rm -f sbv2-server >/dev/null 2>&1 || true
  sudo docker run -d --restart unless-stopped --name sbv2-server "${ARGS[@]}"
else
  exec sudo docker run --rm "${ARGS[@]}"
fi
