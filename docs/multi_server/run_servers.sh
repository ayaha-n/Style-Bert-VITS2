#!/bin/bash
# server_editor.py を複数起動し、その前にユーザー権限の nginx を置いて 1 つのポートで受ける
# (同時リクエストが多いとき用。tmux セッション sbv2 の中で動く)
# 使い方: run_servers.sh [プロセス数] [ポート]   例: run_servers.sh 2 8000
#   nginx がポートで受け、ポート+1 から プロセス数 個の server_editor.py に振り分ける
#   起動後に各プロセスへ MODEL (既定 new_model3) で 1 回合成してモデルを読み込ませておく
#   1 つ落ちても nginx が残りに振り分ける。落ちたものを戻す: tmux respawn-window -k -t sbv2:tts8002
#   様子を見る: tmux attach -t sbv2   止める: tmux kill-session -t sbv2
# 詳しくは docs/multi_server.md
set -euo pipefail
N=${1:-2}
PORT=${2:-8000}
MODEL=${MODEL:-new_model3}
DEVICE=${DEVICE:-cuda}
SESSION=sbv2
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${XDG_RUNTIME_DIR:-/tmp}/sbv2-nginx

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux セッション $SESSION がすでにあります (止める: tmux kill-session -t $SESSION)" >&2
  exit 1
fi

# nginx の pid・ログ・一時ファイルを /var 以下でなく $RUN に置き、root なしで動かす
mkdir -p "$RUN/tmp" "$REPO/static"
BACKENDS=""
for i in $(seq 1 "$N"); do
  BACKENDS+="    server 127.0.0.1:$((PORT + i)) max_fails=1 fail_timeout=10s;"$'\n'
done
sed -e "s|@RUN@|$RUN|g" -e "s|@PORT@|$PORT|g" \
  -e "/@BACKENDS@/{r /dev/stdin" -e "d}" \
  "$REPO/docs/multi_server/nginx.conf.in" <<<"$BACKENDS" >"$RUN/nginx.conf"
nginx -t -p "$RUN" -c "$RUN/nginx.conf" 2>&1 | grep -v "could not open error log file" >&2

# server_editor.py は 0.0.0.0 で待ち受ける。pyopenjtalk の worker は最初の 1 つが起動し、全プロセスで共有する
tmux new-session -d -s "$SESSION" -n nginx -c "$REPO" \
  "nginx -p '$RUN' -c '$RUN/nginx.conf' -g 'daemon off;'"
for i in $(seq 1 "$N"); do
  tmux new-window -d -t "$SESSION" -n "tts$((PORT + i))" -c "$REPO" \
    "venv/bin/python server_editor.py --device $DEVICE --port $((PORT + i)) --skip_static_files; read -p 'exited; press Enter'"
done

for i in $(seq 1 "$N"); do
  P=$((PORT + i))
  until curl -sf -o /dev/null "http://127.0.0.1:$P/api/version"; do sleep 1; done
  "$REPO/venv/bin/python" - "$P" "$MODEL" <<'EOF'
import sys, time, requests
port, model = sys.argv[1], sys.argv[2]
url = f"http://127.0.0.1:{port}"
text = "こんにちは。"
tones = requests.post(url + "/api/g2p", json={"text": text}).json()
start = time.time()
r = requests.post(url + "/api/synthesis", json={
    "model": model, "modelFile": f"model_assets/{model}/{model}.safetensors",
    "speaker": model, "text": text, "moraToneList": tones})
r.raise_for_status()
print(f"port {port}: {model} loaded ({time.time() - start:.1f} s)")
EOF
done
echo "ready: http://127.0.0.1:$PORT -> ${N} processes on $((PORT + 1))-$((PORT + N))"
