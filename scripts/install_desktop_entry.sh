#!/bin/bash
# Install Bubby desktop entry for Linux application menus.
# Run: bash scripts/install_desktop_entry.sh
set -e

BUBBY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DESKTOP_DIR="${HOME}/.local/share/applications"
DESKTOP_FILE="${DESKTOP_DIR}/bubby.desktop"
ICON_SRC="${BUBBY_DIR}/assets/bubby_icon.png"
ICON_DST="${HOME}/.local/share/icons/bubby.png"

echo "=== Bubby Desktop Integration ==="
echo "Project root: ${BUBBY_DIR}"

# Generate icon if missing
if [ ! -f "${ICON_SRC}" ]; then
    echo "Generating app icon..."
    python3 "${BUBBY_DIR}/scripts/generate_icon.py"
fi

# Install icon
mkdir -p "$(dirname "${ICON_DST}")"
cp "${ICON_SRC}" "${ICON_DST}"
echo "Icon installed: ${ICON_DST}"

# Create .desktop file
mkdir -p "${DESKTOP_DIR}"
cat > "${DESKTOP_FILE}" << EOF
[Desktop Entry]
Type=Application
Name=Bubby
Comment=Desktop companion — a friendly slime that lives on your screen
Exec=bash "${BUBBY_DIR}/run_bubby.sh"
Icon=${ICON_DST}
Terminal=false
Categories=Utility;
StartupNotify=false
X-GNOME-Autostart-enabled=true
EOF

chmod +x "${DESKTOP_FILE}"

# Refresh desktop database
if command -v update-desktop-database &>/dev/null; then
    update-desktop-database "${DESKTOP_DIR}"
fi

echo ""
echo "=== Desktop entry installed ==="
echo "  Desktop file: ${DESKTOP_FILE}"
echo "  Icon:         ${ICON_DST}"
echo ""
echo "You can now launch Bubby from your application menu or run:"
echo "  gtk-launch bubby"
echo ""
echo "To uninstall: rm ${DESKTOP_FILE} ${ICON_DST}"