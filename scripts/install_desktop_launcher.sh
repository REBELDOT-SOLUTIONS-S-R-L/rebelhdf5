#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPLICATIONS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESKTOP_FILE="$APPLICATIONS_DIR/rebelhdf5.desktop"
LAUNCHER="$ROOT_DIR/scripts/launch_desktop.sh"
ICON="$ROOT_DIR/public/favicon.ico"

mkdir -p "$APPLICATIONS_DIR"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=rebelHDF5
Comment=Local HDF5 viewer and processing app
Exec="$LAUNCHER"
Icon=$ICON
Terminal=false
Categories=Science;Development;
StartupWMClass=rebelHDF5
EOF

chmod 0644 "$DESKTOP_FILE"

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$APPLICATIONS_DIR" >/dev/null 2>&1 || true
fi

echo "Installed $DESKTOP_FILE"
echo "Open the Dash and search for rebelHDF5."
