#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="Face Swap Studio"
BUNDLE_IDENTIFIER="local.face-swap-studio.app"

DIST_DIR="$PROJECT_ROOT/dist"
APP_DIR="$DIST_DIR/$APP_NAME.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
BUNDLED_PROJECT_DIR="$RESOURCES_DIR/project"

ICON_SOURCE="$PROJECT_ROOT/assets/icons/logo.png"
ICONSET_DIR="$DIST_DIR/AppIcon.iconset"
ICNS_PATH="$RESOURCES_DIR/AppIcon.icns"

cd "$PROJECT_ROOT"

if [[ ! -f ".venv/bin/python" ]]; then
    echo "Не найдено окружение: $PROJECT_ROOT/.venv"
    exit 1
fi

if [[ ! -f "$ICON_SOURCE" ]]; then
    echo "Не найдена иконка: $ICON_SOURCE"
    exit 1
fi

echo "Проверяю pywebview..."
if ! ".venv/bin/python" -c "import webview" >/dev/null 2>&1; then
    echo "pywebview не найден. Устанавливаю в .venv..."
    ".venv/bin/python" -m pip install pywebview
fi

echo "Очищаю старую сборку..."
rm -rf "$APP_DIR"
rm -rf "$ICONSET_DIR"

mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"
mkdir -p "$BUNDLED_PROJECT_DIR"
mkdir -p "$ICONSET_DIR"

echo "Генерирую .icns иконку..."
sips -z 16 16 "$ICON_SOURCE" --out "$ICONSET_DIR/icon_16x16.png" >/dev/null
sips -z 32 32 "$ICON_SOURCE" --out "$ICONSET_DIR/icon_16x16@2x.png" >/dev/null
sips -z 32 32 "$ICON_SOURCE" --out "$ICONSET_DIR/icon_32x32.png" >/dev/null
sips -z 64 64 "$ICON_SOURCE" --out "$ICONSET_DIR/icon_32x32@2x.png" >/dev/null
sips -z 128 128 "$ICON_SOURCE" --out "$ICONSET_DIR/icon_128x128.png" >/dev/null
sips -z 256 256 "$ICON_SOURCE" --out "$ICONSET_DIR/icon_128x128@2x.png" >/dev/null
sips -z 256 256 "$ICON_SOURCE" --out "$ICONSET_DIR/icon_256x256.png" >/dev/null
sips -z 512 512 "$ICON_SOURCE" --out "$ICONSET_DIR/icon_256x256@2x.png" >/dev/null
sips -z 512 512 "$ICON_SOURCE" --out "$ICONSET_DIR/icon_512x512.png" >/dev/null
sips -z 1024 1024 "$ICON_SOURCE" --out "$ICONSET_DIR/icon_512x512@2x.png" >/dev/null

iconutil -c icns "$ICONSET_DIR" -o "$ICNS_PATH"
rm -rf "$ICONSET_DIR"

echo "Копирую проект внутрь .app..."
rsync -a \
    --exclude ".git" \
    --exclude "dist" \
    --exclude "__pycache__" \
    --exclude ".pytest_cache" \
    --exclude ".ruff_cache" \
    --exclude ".mypy_cache" \
    --exclude ".DS_Store" \
    "$PROJECT_ROOT/" \
    "$BUNDLED_PROJECT_DIR/"

echo "Создаю executable launcher..."
cat > "$MACOS_DIR/$APP_NAME" <<'EOF'
#!/bin/bash

set -euo pipefail

APP_EXECUTABLE="$0"
MACOS_DIR="$(cd "$(dirname "$APP_EXECUTABLE")" && pwd)"
CONTENTS_DIR="$(cd "$MACOS_DIR/.." && pwd)"
PROJECT_ROOT="$CONTENTS_DIR/Resources/project"

export FACE_SWAP_STUDIO_PROJECT_ROOT="$PROJECT_ROOT"

PYTHON_PATH="$PROJECT_ROOT/.venv/bin/python"
LAUNCHER_PATH="$PROJECT_ROOT/scripts/macos_app_launcher.py"

if [[ ! -f "$PYTHON_PATH" ]]; then
    osascript -e 'display alert "Face Swap Studio" message "Bundled Python environment was not found."'
    exit 1
fi

if [[ ! -f "$LAUNCHER_PATH" ]]; then
    osascript -e 'display alert "Face Swap Studio" message "macOS app launcher was not found."'
    exit 1
fi

exec "$PYTHON_PATH" "$LAUNCHER_PATH"
EOF

chmod +x "$MACOS_DIR/$APP_NAME"

echo "Создаю Info.plist..."
cat > "$CONTENTS_DIR/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "https://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>

    <key>CFBundleDisplayName</key>
    <string>$APP_NAME</string>

    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_IDENTIFIER</string>

    <key>CFBundleVersion</key>
    <string>1.0.0</string>

    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>

    <key>CFBundlePackageType</key>
    <string>APPL</string>

    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>

    <key>CFBundleIconFile</key>
    <string>AppIcon</string>

    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>

    <key>NSHighResolutionCapable</key>
    <true/>

    <key>LSApplicationCategoryType</key>
    <string>public.app-category.photography</string>
</dict>
</plist>
EOF

echo "Удаляю quarantine flag, если он есть..."
xattr -cr "$APP_DIR" || true

echo "Готово:"
echo "$APP_DIR"
echo ""
echo "Для zip-архива выполни:"
echo "ditto -c -k --sequesterRsrc --keepParent \"$APP_DIR\" \"$DIST_DIR/$APP_NAME.zip\""