#!/bin/bash
# Ensure git exists
if ! command -v git &> /dev/null; then
  echo "Git command not found, skipping version update."
  exit 0
fi

mkdir -p bot
VERSION=$(git describe --always --dirty 2>/dev/null)
if [ -z "$VERSION" ]; then
  VERSION="unknown"
fi
echo "__version__ = '$VERSION'" > bot/version.py
echo "Updated version to: $VERSION"