#!/bin/bash
# Upload the static app (index.html + precomputed tables) to a free static Hugging Face Space.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPACE="${1:-nebadan/amharic-next-word}"
HF="$ROOT/.venv/bin/hf"
"$HF" repos create "$SPACE" --type space --sdk static --public --exist-ok
"$HF" upload "$SPACE" "$ROOT/static_app/space" . --repo-type space --commit-message "Deploy Amharic next-word suggestion (static)"
echo "https://huggingface.co/spaces/$SPACE"
