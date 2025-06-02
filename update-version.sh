#!/bin/bash
# Ensure git exists
if ! command -v git &> /dev/null; then
  echo "Git command not found, skipping version update."
  exit 0
fi

mkdir -p bot
VERSION=$(git describe --always --dirty)
echo "__version__ = '$VERSION'" > bot/version.py
echo "Updated version to: $VERSION"