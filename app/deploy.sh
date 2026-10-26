#!/bin/bash
# Stage the Space (app + library + model) and upload it to Hugging Face.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPACE="${1:-nebadan/amharic-next-word}"
STAGE="$(mktemp -d)"
cp "$ROOT"/app/{app.py,requirements.txt,README.md} "$STAGE"/
mkdir -p "$STAGE/amharic_lm" "$STAGE/model"
cp "$ROOT"/amharic_lm/{__init__.py,text.py,ngram.py} "$STAGE/amharic_lm/"
cp "$ROOT"/app/model/amharic_4gram_kn.npz "$STAGE/model/"
HF="$ROOT/.venv/bin/hf"
"$HF" repos create "$SPACE" --type space --sdk gradio --public --exist-ok
"$HF" upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy Amharic next-word suggestion app"
rm -rf "$STAGE"
echo "https://huggingface.co/spaces/$SPACE"
