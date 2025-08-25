#!/usr/bin/env bash
set -euo pipefail
TAG="${1:-v0.1.0}"
mkdir -p assets && cd assets
echo "Downloading release assets for tag: $TAG"
# Requires GitHub CLI: gh auth login (once)
gh release download "$TAG" -R vipinmenon1989/CRISPR-webserver
echo "Done. Unpack as needed, e.g.: tar -xzf models_i.tar.gz -C .."
