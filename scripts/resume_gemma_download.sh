#!/bin/bash
# Resume gemma3-27b download using curl (bypasses broken hf/huggingface-cli tool)
HF_TOKEN=$(python3 -c "from huggingface_hub import get_token; print(get_token())")
BASE="https://huggingface.co/google/gemma-3-27b-it/resolve/main"
DEST="/home/neo/.cache/huggingface/hub/gemma3-27b"
LOG="/home/neo/projects/no-harm-vllm_jury_v3/logs/download_gemma3_27b.log"

SHARDS=(
  "model-00004-of-00012.safetensors"
  "model-00007-of-00012.safetensors"
  "model-00009-of-00012.safetensors"
  "model-00010-of-00012.safetensors"
  "model-00011-of-00012.safetensors"
)

echo "[$(date)] Starting curl-based resume download" | tee -a "$LOG"

for shard in "${SHARDS[@]}"; do
  SIZE=$(stat -c%s "$DEST/$shard" 2>/dev/null || echo 0)
  echo "[$(date)] Resuming $shard from byte $SIZE" | tee -a "$LOG"
  curl -L -C - \
    -H "Authorization: Bearer $HF_TOKEN" \
    -o "$DEST/$shard" \
    --progress-bar \
    "$BASE/$shard" 2>&1 | tee -a "$LOG"
  echo "[$(date)] Done: $shard" | tee -a "$LOG"
done

echo "[$(date)] All shards complete" | tee -a "$LOG"
