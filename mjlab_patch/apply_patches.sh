#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="$ROOT/mjlab_patch/mjlab"
SITE="$(python -c "import mjlab, pathlib; print(pathlib.Path(mjlab.__file__).parent)")"

echo "Applying mjlab patches to: $SITE"

cp "$PATCH_DIR/managers/observation_manager.py" \
  "$SITE/managers/observation_manager.py"

cp "$PATCH_DIR/sim/sim.py" \
  "$SITE/sim/sim.py"

echo "Done."
