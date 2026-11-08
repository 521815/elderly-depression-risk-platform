#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP_NAME="老年抑郁风险筛查桌面版"
APP_PATH="$HOME/Desktop/${APP_NAME}.app"
BUILD_DIR="${TMPDIR:-/tmp}/elderly_desktop_build"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/Contents/MacOS" "$BUILD_DIR/Contents/Resources"

clang -O2 -Wall -Wextra -o "$BUILD_DIR/Contents/MacOS/launcher" \
  "$ROOT/scripts/native_app/launcher.c"

cp "$ROOT/scripts/native_app/Info.plist" "$BUILD_DIR/Contents/Info.plist"
cp "$ROOT/output/AppIcon.icns" "$BUILD_DIR/Contents/Resources/AppIcon.icns"

rm -rf "$APP_PATH"
ditto "$BUILD_DIR" "$APP_PATH"

codesign --force --deep --sign - "$APP_PATH"
codesign --verify --verbose=2 "$APP_PATH"

echo "Built $APP_PATH"
