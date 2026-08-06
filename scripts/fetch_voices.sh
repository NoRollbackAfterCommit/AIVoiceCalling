#!/usr/bin/env bash
# Download Piper voices for the languages this platform targets.
#
#   bash scripts/fetch_voices.sh                 # English + Hindi
#   bash scripts/fetch_voices.sh all             # every voice below
#
# Voices are MIT/CC licensed and roughly 20-60 MB each. Once downloaded the
# platform never touches the network for speech again, which is what makes an
# air-gapped deployment possible.
set -euo pipefail

DIR="${VAANI_TTS_VOICES_DIR:-./models/piper}"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"

mkdir -p "$DIR"

# name|path under $BASE
CORE=(
  "en_US-lessac-medium|en/en_US/lessac/medium/en_US-lessac-medium"
  "hi_IN-pratham-medium|hi/hi_IN/pratham/medium/hi_IN-pratham-medium"
)
EXTRA=(
  "en_GB-alba-medium|en/en_GB/alba/medium/en_GB-alba-medium"
  "hi_IN-priyamvada-medium|hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium"
  "ne_NP-google-medium|ne/ne_NP/google/medium/ne_NP-google-medium"
  "ar_JO-kareem-medium|ar/ar_JO/kareem/medium/ar_JO-kareem-medium"
  "fr_FR-siwis-medium|fr/fr_FR/siwis/medium/fr_FR-siwis-medium"
  "es_ES-davefx-medium|es/es_ES/davefx/medium/es_ES-davefx-medium"
)

VOICES=("${CORE[@]}")
if [[ "${1:-}" == "all" ]]; then
  VOICES+=("${EXTRA[@]}")
fi

for entry in "${VOICES[@]}"; do
  name="${entry%%|*}"
  path="${entry##*|}"
  if [[ -f "$DIR/$name.onnx" ]]; then
    echo "  have $name"
    continue
  fi
  echo "  fetching $name"
  curl -fSL --progress-bar -o "$DIR/$name.onnx"      "$BASE/$path.onnx"
  curl -fSL --progress-bar -o "$DIR/$name.onnx.json" "$BASE/$path.onnx.json"
done

echo
echo "Voices in $DIR:"
ls -1 "$DIR"/*.onnx 2>/dev/null | xargs -n1 basename || echo "  (none)"
echo
echo "Set VAANI_TTS_PROVIDER=piper and VAANI_TTS_VOICE=<name> to use them."
