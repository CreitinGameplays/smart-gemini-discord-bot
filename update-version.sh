#!/bin/bash
VERSION=$(git describe --always --dirty)
echo "__version__ = '$VERSION'" > bot/version.py
echo "Updated version to: $VERSION"